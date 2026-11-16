"""
CETT特征提取器 - 使用预提取的答案词元
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
import hashlib
import os
import time
from tqdm import tqdm
import logging

from configs.config import cett_config, model_config, training_config

logger = logging.getLogger(__name__)

class CETTExtractor:
    """CETT特征提取器 - 使用预提取的答案词元"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # 激活值存储
        self.activations = {}
        self.hooks = []
        
        # 缓存
        os.makedirs(cett_config.cache_dir, exist_ok=True)
        
        # 注册钩子
        self._register_hooks()
        
        logger.info(f"CETT提取器初始化完成，监控 {len(cett_config.layers_to_extract)} 层")
    
    def _register_hooks(self):
        """注册前向钩子以捕获中间激活"""
        # 清除之前的钩子
        self._remove_hooks()
        
        # 获取模型层
        model_layers = self.model.model.layers
        
        for layer_idx in cett_config.layers_to_extract:
            if layer_idx >= len(model_layers):
                logger.warning(f"层 {layer_idx} 超出范围，模型只有 {len(model_layers)} 层")
                continue
            
            layer = model_layers[layer_idx]
            mlp = layer.mlp
            
            # 创建钩子函数
            def make_hook(l_idx):
                def hook(module, input, output):
                    # 获取中间激活值
                    hidden_states = input[0]  # [batch, seq_len, hidden_size]
                    
                    # Qwen2.5使用SwiGLU: SwiGLU(x) = silu(gate_proj(x)) * up_proj(x)
                    gate = module.gate_proj(hidden_states)
                    up = module.up_proj(hidden_states)
                    
                    # 计算中间激活 z = silu(gate) * up
                    z = nn.functional.silu(gate) * up  # [batch, seq_len, intermediate_size]
                    
                    # 存储激活值
                    self.activations[f"layer_{l_idx}"] = z
                    
                    return output
                
                return hook
            
            # 注册钩子
            hook = mlp.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)
        
        logger.debug(f"注册了 {len(self.hooks)} 个钩子")
    
    def _remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _generate_cache_key(self, question: str, model_answer: str, answer_token_indices: List[int]) -> str:
        """生成缓存键"""
        content = f"{question}|||{model_answer}|||{answer_token_indices}|||{cett_config.layers_to_extract}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _load_cached_features(self, cache_key: str) -> Optional[Dict]:
        """加载缓存的CETT特征"""
        if not cett_config.use_cache:
            return None
        
        cache_file = os.path.join(cett_config.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                # 检查缓存是否过期
                if cett_config.cache_expire_days > 0:
                    file_age = time.time() - os.path.getmtime(cache_file)
                    if file_age > cett_config.cache_expire_days * 24 * 3600:
                        logger.debug(f"缓存过期: {cache_file}")
                        return None
                
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
                
                logger.debug(f"使用缓存特征: {cache_file}")
                return features
                
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        return None
    
    def _save_cached_features(self, cache_key: str, features: Dict):
        """保存特征到缓存"""
        if not cett_config.use_cache:
            return
        
        cache_file = os.path.join(cett_config.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            
            logger.debug(f"保存特征到缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _compute_cett_contributions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                                  answer_token_indices: List[int]) -> Dict[str, Dict[str, float]]:
        """
        按照论文公式(2)计算CETT贡献度
        
        CETT_j,t = ||h_j_t|| / ||h_t||
        where h_j_t = z_j,t * W_down[:,j]
              h_t = sum_j h_j_t
        """
        # 清空之前的激活值
        self.activations.clear()
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        batch_size, seq_len = input_ids.shape
        
        # 准备结果字典
        features = {
            'answer': {},  # 答案词元的平均CETT
            'other': {}    # 非答案词元的平均CETT
        }
        
        # 所有非答案词元位置
        all_indices = list(range(seq_len))
        other_indices = [i for i in all_indices if i not in answer_token_indices]
        
        # 处理每一层的激活
        for layer_key, z in self.activations.items():
            # z shape: [1, seq_len, intermediate_size]
            layer_idx = int(layer_key.split('_')[1])
            layer = self.model.model.layers[layer_idx]
            mlp = layer.mlp
            
            # 获取down投影权重 W_down
            W_down = mlp.down_proj.weight  # [hidden_size, intermediate_size]
            intermediate_size = W_down.shape[1]
            
            # 计算该层的总隐藏状态 h = W_down * z
            h = mlp.down_proj(z)  # [1, seq_len, hidden_size]
            
            # 计算每个token位置的总隐藏状态范数 ||h_t||
            h_norms = torch.norm(h[0], dim=1, p=2)  # [seq_len]
            h_norms = h_norms + cett_config.epsilon  # 避免除零
            
            # 为每个神经元计算贡献
            for neuron_idx in range(intermediate_size):
                # 该神经元的激活值 z_j,t
                z_neuron = z[0, :, neuron_idx]  # [seq_len]
                
                # 对应的down投影权重列 w_j = W_down[:,j]
                w_neuron = W_down[:, neuron_idx]  # [hidden_size]
                
                # 计算单个神经元的贡献向量：h_j_t = z_j,t * w_j
                h_j = z_neuron.unsqueeze(1) * w_neuron.unsqueeze(0)  # [seq_len, hidden_size]
                
                # 计算贡献范数 ||h_j_t||_2
                h_j_norms = torch.norm(h_j, dim=1, p=2)  # [seq_len]
                
                # 计算CETT值：||h_j_t|| / ||h_t||
                cett_values = h_j_norms / h_norms  # [seq_len]
                
                # 计算答案词元的平均CETT
                if answer_token_indices:
                    answer_cett = cett_values[answer_token_indices].mean().item()
                else:
                    answer_cett = 0.0
                
                # 计算非答案词元的平均CETT
                if other_indices:
                    other_cett = cett_values[other_indices].mean().item()
                else:
                    other_cett = 0.0
                
                # 存储特征
                neuron_key = f"{layer_key}_neuron_{neuron_idx}"
                features['answer'][neuron_key] = answer_cett
                features['other'][neuron_key] = other_cett
        
        return features
    
    def extract_features_for_sample(self, sample: Dict, use_cache: bool = True) -> Optional[Dict]:
        """
        提取单个样本的CETT特征 - 使用预提取的答案词元
        
        参数:
            sample: 包含question, model_answer, answer_token_indices, label的字典
            use_cache: 是否使用缓存
        
        返回:
            Dict: 包含CETT特征和其他信息的字典
        """
        try:
            question = sample['question']
            model_answer = sample['model_answer']
            answer_token_indices = sample.get('answer_token_indices', [])
            
            if not answer_token_indices:
                logger.warning(f"样本没有答案词元索引: {question[:50]}...")
                # 如果没有预提取的答案词元，使用整个回答
                full_tokens = self.tokenizer.encode(model_answer, add_special_tokens=False)
                answer_token_indices = list(range(len(full_tokens)))
            
            # 生成缓存键
            cache_key = self._generate_cache_key(question, model_answer, answer_token_indices)
            
            # 检查缓存
            if use_cache and cett_config.use_cache:
                cached_features = self._load_cached_features(cache_key)
                if cached_features is not None:
                    # 添加样本信息
                    cached_features['question'] = question
                    cached_features['model_answer'] = model_answer
                    cached_features['label'] = sample.get('label', 0)
                    cached_features['answer_token_indices'] = answer_token_indices
                    return cached_features
            
            logger.debug(f"提取特征: {question[:50]}...")
            
            # 构建输入
            # 论文中是在生成答案时提取特征，所以我们需要模拟生成过程
            # 这里我们使用完整的问题+回答作为输入
            prompt = f"问题：{question}\n回答：{model_answer}"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=model_config.max_seq_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 计算答案词元在输入中的位置
            # 我们需要找到"回答："之后的部分
            prompt_tokens = self.tokenizer.encode(f"问题：{question}\n回答：", add_special_tokens=False)
            
            # 调整答案词元索引
            adjusted_indices = []
            for idx in answer_token_indices:
                adjusted_idx = idx + len(prompt_tokens)
                if adjusted_idx < input_ids.shape[1]:
                    adjusted_indices.append(adjusted_idx)
            
            if not adjusted_indices:
                logger.warning(f"调整后的答案词元索引为空，使用默认位置")
                adjusted_indices = [len(prompt_tokens)]
            
            # 计算CETT特征
            cett_features = self._compute_cett_contributions(
                input_ids, attention_mask, adjusted_indices
            )
            
            # 构建结果
            result = {
                'question': question,
                'model_answer': model_answer,
                'answer_token_indices': adjusted_indices,
                'original_answer_token_indices': answer_token_indices,
                'features_answer': cett_features['answer'],
                'features_other': cett_features['other'],
                'label': sample.get('label', 0)
            }
            
            # 保存到缓存
            if use_cache and cett_config.use_cache:
                self._save_cached_features(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"提取样本特征失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def extract_features_batch(self, samples: List[Dict], batch_size: int = 1) -> List[Dict]:
        """
        批量提取CETT特征
        
        参数:
            samples: 样本列表（必须包含answer_token_indices字段）
            batch_size: 批次大小
        
        返回:
            List[Dict]: 特征列表
        """
        logger.info(f"开始批量提取特征，共 {len(samples)} 个样本")
        
        all_features = []
        failed_count = 0
        
        # 验证样本是否包含答案词元索引
        valid_samples = []
        for sample in samples:
            if 'answer_token_indices' not in sample:
                logger.warning(f"样本缺少answer_token_indices字段: {sample.get('question', '')[:50]}...")
                failed_count += 1
            else:
                valid_samples.append(sample)
        
        logger.info(f"有效样本: {len(valid_samples)}/{len(samples)}")
        
        # 使用tqdm显示进度
        for i in tqdm(range(0, len(valid_samples), batch_size), desc="提取CETT特征"):
            batch_samples = valid_samples[i:i+batch_size]
            
            for sample in batch_samples:
                features = self.extract_features_for_sample(sample)
                
                if features:
                    all_features.append(features)
                else:
                    failed_count += 1
                
                # 定期保存进度
                if len(all_features) % training_config.save_frequency == 0:
                    logger.info(f"已处理 {len(all_features)} 个样本，失败 {failed_count} 个")
        
        logger.info(f"特征提取完成: 成功 {len(all_features)}，失败 {failed_count}")
        
        return all_features
    
    def __del__(self):
        """清理钩子"""
        self._remove_hooks()