"""
H-Neurons测试推理管道 - PCA降维版本
使用PCA降维模型进行推理
"""

import os
import json
import pickle
import torch
import logging
import time
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import gc

# ==================== 1. 全局配置 ====================
class GlobalConfig:
    # 路径配置
    model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    classifier_path = "/mnt/project/H-Neurous/output/pca_results/h_neurons_pca_demo_20260118_051018/pca_classifier.pkl"  # 修改为PCA模型路径
    input_data_path = "/mnt/project/H-Neurous/data/test/test_inference_chat.json"  # 改为已生成的数据文件
    output_path = "/mnt/project/H-Neurous/data/test/test_inference_chat_pca.json"
    
    # PCA推理参数
    max_seq_length = 1024  # 与训练保持一致
    use_4bit = True  # 与训练保持一致
    extract_all_layers = True  # 与训练保持一致
    threshold = 0.5  # 判定阈值

# 日志监控
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/mnt/project/H-Neurous/logs/test_inference_pca.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 2. CETT特征提取器（保持不变） ====================
class HybridCETTExtractor:
    """
    CETT特征提取器 - 混合优化版本
    保持字符映射对齐逻辑，但使用训练代码的向量化计算和列表格式
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 获取模型信息
        self.num_layers = len(model.model.layers)
        
        # 确定要提取的层 - 按照论文，提取所有层
        self.layers_to_extract = list(range(self.num_layers))
        
        self.logger.info(f"模型总层数: {self.num_layers}")
        self.logger.info(f"提取特征层数: {len(self.layers_to_extract)}")
        
        # 预计算所有层的权重范数（与训练代码一致，用于向量化计算）
        self.logger.info("预计算权重范数...")
        self.precomputed_w_norms = []
        for layer_idx in self.layers_to_extract:
            mlp = self.model.model.layers[layer_idx].mlp
            w_down = mlp.down_proj.weight
            
            # 反量化处理
            if hasattr(w_down, 'quant_state'):
                w_down_dequant = bnb.functional.dequantize_4bit(w_down.data, w_down.quant_state)
            else:
                w_down_dequant = w_down
            
            # 计算范数 ||W_down[:, j]||_2
            w_norm = torch.norm(w_down_dequant.float(), dim=0, p=2).cpu()
            self.precomputed_w_norms.append(w_norm)
            
            # 清理临时变量
            del w_down_dequant
        
        torch.cuda.empty_cache()
        self.logger.info("权重范数预计算完成")
        
        # 记录神经元数量信息
        if self.layers_to_extract:
            first_layer = self.model.model.layers[0].mlp
            self.neurons_per_layer = first_layer.up_proj.weight.shape[1]  # intermediate_size
            self.total_neurons = len(self.layers_to_extract) * self.neurons_per_layer
            self.logger.info(f"每层神经元数: {self.neurons_per_layer}")
            self.logger.info(f"总神经元数: {self.total_neurons}")

    def _validate_with_unicode_tolerance(self, orig_text, cett_text, max_tolerate=3):
        """终极乱码容忍校验逻辑（保持不变）"""
        # 1. 清洗掉所有空白符（空格、换行等）
        s1 = "".join(orig_text.split())
        s2 = "".join(cett_text.split())

        # 2. 长度校验：如果去掉空格后长度都不一样，说明 Token 映射逻辑肯定错了
        if len(s1) != len(s2):
            return False, f"长度不一 (预期{len(s1)} vs 实际{len(s2)})"

        # 3. 逐字比对
        mismatches = 0
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                # 如果不相等，检查 c2 是不是乱码符号 \ufffd ()
                if c2 == '' or c2 == '\ufffd':
                    mismatches += 1
                else:
                    # 发现真正的字符不匹配（不是乱码导致的），直接失败
                    return False, f"字符冲突 ('{c1}' vs '{c2}')"

        # 4. 容忍度校验：如果乱码位太多，可能也不是我们要的
        if mismatches > max_tolerate:
            return False, f"乱码过多 ({mismatches}处)"

        return True, "对齐成功"
    
    def _compute_cett_vectorized(self, z: torch.Tensor, layer_idx: int, 
                               h_norm: torch.Tensor, answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化计算CETT特征 - 完全使用训练代码的实现
        """
        # 获取预计算的权重范数
        w_norm = self.precomputed_w_norms[layer_idx].to(z.device).to(z.dtype)
        
        # 取batch中的第一个样本
        z_abs = torch.abs(z[0])  # [seq_len, num_neurons]
        
        # 计算分母：||h_t||_2 已经计算好
        # 计算分子：|z_j,t| * ||W_down[:, j]||_2
        scaled_z = z_abs / h_norm.unsqueeze(-1)
        
        # 创建所有索引的集合
        seq_len = z_abs.shape[0]
        all_indices = set(range(seq_len))
        valid_answer_indices = [idx for idx in answer_indices if idx < seq_len]
        other_indices = list(all_indices - set(valid_answer_indices))
        
        # 向量化聚合 - 直接计算均值
        answer_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
        other_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
        
        if valid_answer_indices:
            answer_cett = torch.mean(scaled_z[valid_answer_indices], dim=0) * w_norm
        
        if other_indices:
            other_cett = torch.mean(scaled_z[other_indices], dim=0) * w_norm
        
        # 转换为numpy并清理GPU内存
        answer_np = answer_cett.detach().cpu().numpy().astype(np.float32)
        other_np = other_cett.detach().cpu().numpy().astype(np.float32)
        
        # 清理
        del z_abs, scaled_z, answer_cett, other_cett
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return answer_np, other_np

    def extract_features_for_sample(self, question: str, model_answer: str, answer_text: str) -> Optional[Dict]:
        """
        为单个样本提取CETT特征 - 混合版本
        保持字符映射对齐逻辑，但使用向量化计算和列表格式
        """
        try:
            self.logger.info(f"开始提取特征 - 问题: {question[:50]}...")
            
            # 步骤1: 先单独tokenize model_answer，获取精确的答案词元
            if not answer_text or answer_text not in model_answer:
                self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
                self.logger.info("回退: 使用完整回答作为答案词元")
                answer_text = model_answer
            
            # 找到答案在model_answer中的精确字符位置
            start_char = model_answer.find(answer_text)
            end_char = start_char + len(answer_text)
            
            if start_char == -1:
                self.logger.warning(f"无法找到答案文本: {answer_text[:50]}...")
                self.logger.info("回退: 使用完整回答作为答案词元")
                answer_text = model_answer
                start_char = 0
                end_char = len(model_answer)
            
            # 对model_answer进行tokenization，获取offset mapping
            self.logger.info(f"单独tokenize model_answer，长度: {len(model_answer)}")
            model_answer_enc = self.tokenizer(
                model_answer,
                return_offsets_mapping=True,
                add_special_tokens=False,
                return_tensors='pt'
            )
            model_answer_offsets = model_answer_enc['offset_mapping'][0].cpu().numpy()
            
            # 基于字符位置找到model_answer中的答案token索引
            answer_token_indices_in_model = []
            for token_idx, (char_start, char_end) in enumerate(model_answer_offsets):
                # token完全在答案区域内，或者部分重叠但主要部分在答案内
                if (char_start >= start_char and char_end <= end_char) or \
                (char_start <= start_char and char_end > start_char + (end_char - start_char) / 2):
                    answer_token_indices_in_model.append(token_idx)
            
            if not answer_token_indices_in_model:
                self.logger.warning(f"无法找到model_answer中的答案token: {answer_text[:50]}...")
                self.logger.info("回退: 使用完整回答作为答案词元")
                # 回退：使用所有token作为答案词元
                answer_token_indices_in_model = list(range(len(model_answer_offsets)))
            
            # 获取原始答案词元文本
            seq_len = model_answer_enc['input_ids'].size(1)
            model_answer_answer_tokens = [
                model_answer_enc['input_ids'][0][i].item() 
                for i in answer_token_indices_in_model
                if i < seq_len
            ]
            orig_answer_text = self.tokenizer.decode(
                model_answer_answer_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # 步骤1: 构建与训练代码完全相同的Chat消息格式
            messages = [
                {
                    "role": "system",
                    "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
                },
                {
                    "role": "user",
                    "content": f"请回答下面的问题：\n\n问题：{question}"
                },
                {
                    "role": "assistant",
                    "content": model_answer
                }
            ]
            
            # 步骤2: 应用Chat模板，与训练代码完全一致
            full_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 步骤3: 对完整文本进行分词，获取offset mapping
            self.logger.info(f"使用Chat模板构建prompt，长度: {len(full_prompt)}")
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True
            )
            
            input_ids = inputs['input_ids'].to(self.model.device)
            attention_mask = inputs['attention_mask'].to(self.model.device)
            full_offsets = inputs['offset_mapping'][0].cpu().numpy()
            
            # 步骤4: 精确对齐 - 找到答案文本在完整prompt中的位置
            # 在完整prompt中找到model_answer的起始位置
            ans_start_in_prompt = full_prompt.rfind(model_answer)
            if ans_start_in_prompt == -1:
                self.logger.warning("无法在完整prompt中找到model_answer，使用启发式定位")
                # 回退：尝试查找"assistant"内容
                ans_start_in_prompt = full_prompt.find("assistant") + len("assistant") + 5
                if ans_start_in_prompt <= len("assistant") + 5:
                    ans_start_in_prompt = len(full_prompt) - len(model_answer)
            
            # 找到答案文本在model_answer中的相对位置
            rel_start_char = model_answer.find(answer_text)
            if rel_start_char == -1:
                self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
                # 回退：使用完整回答作为答案文本
                answer_text = model_answer
                rel_start_char = 0
            
            # 计算答案文本在完整prompt中的绝对位置
            abs_start = ans_start_in_prompt + rel_start_char
            abs_end = abs_start + len(answer_text)
            
            # 步骤5: 基于字符偏移映射找到对应的token索引
            adjusted_answer_indices = []
            for token_idx, (c_start, c_end) in enumerate(full_offsets):
                # 跳过padding或特殊token
                if c_start == 0 and c_end == 0:
                    continue
                
                # 检查token是否与答案文本重叠
                if not (c_end <= abs_start or c_start >= abs_end):
                    adjusted_answer_indices.append(token_idx)
            
            if not adjusted_answer_indices:
                self.logger.warning("未找到匹配的答案词元索引，回退到完整回答")
                # 回退：将整个回答作为答案词元
                for token_idx, (c_start, c_end) in enumerate(full_offsets):
                    if c_start == 0 and c_end == 0:
                        continue
                    if c_start >= ans_start_in_prompt:
                        adjusted_answer_indices.append(token_idx)
            
            # 获取CETT对齐后的答案词元
            prompt_answer_tokens = [
                input_ids[0][i].item()
                for i in adjusted_answer_indices
                if i < input_ids.size(1)
            ]
            cett_answer_decoded = self.tokenizer.decode(
                prompt_answer_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # 验证对齐结果
            orig_answer_text = answer_text
            is_match, match_msg = self._validate_with_unicode_tolerance(
                orig_answer_text, cett_answer_decoded
            )
            
            self.logger.warning(
                "\n"
                "================ CETT FULL DEBUG =================\n"
                f"[问题]\n{question}\n\n"
                f"[原始答案文本]\n{model_answer}\n\n"
                f"[原始答案词元]\n{orig_answer_text}\n\n"
                f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
                f"[对齐结果]\n{match_msg}\n"
                "=================================================="
            )
            
            if not is_match:
                self.logger.warning(f"对齐校验不通过: {match_msg}")
                self.logger.info("回退: 使用完整回答作为答案词元")
                # 重新调整答案索引
                adjusted_answer_indices = []
                for token_idx, (c_start, c_end) in enumerate(full_offsets):
                    if c_start == 0 and c_end == 0:
                        continue
                    if c_start >= ans_start_in_prompt:
                        adjusted_answer_indices.append(token_idx)
                
                # 重新解码
                prompt_answer_tokens = [
                    input_ids[0][i].item()
                    for i in adjusted_answer_indices
                    if i < input_ids.size(1)
                ]
                cett_answer_decoded = self.tokenizer.decode(
                    prompt_answer_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                self.logger.info(f"回退后答案词元: {cett_answer_decoded[:100]}...")
            else:
                self.logger.info(f"✓ 答案词元成功对齐: {match_msg}")

            # 一次性前向传播获取所有层的隐藏状态
            self.logger.info("开始前向传播...")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = outputs.hidden_states  # 所有层的隐藏状态
            
            # 向量化提取所有层的特征（使用训练代码的方法）
            all_answer_features = []
            all_other_features = []
            
            self.logger.info(f"向量化计算CETT特征，共{len(self.layers_to_extract)}层")
            for i, layer_idx in enumerate(self.layers_to_extract):
                layer = self.model.model.layers[layer_idx]
                
                # 获取归一化后的隐藏状态
                normed_hs = layer.post_attention_layernorm(hidden_states[layer_idx])
                
                # 计算MLP激活值z
                mlp = layer.mlp
                gate = mlp.gate_proj(normed_hs)
                up = mlp.up_proj(normed_hs)
                z = torch.nn.functional.silu(gate) * up
                
                # 计算分母 ||h_t||_2
                h_total = mlp.down_proj(z)
                h_norm = torch.norm(h_total[0].float(), dim=-1, p=2)
                h_norm = torch.clamp(h_norm, min=1e-10)
                
                # 向量化计算CETT特征（使用训练代码的向量化函数）
                answer_vec, other_vec = self._compute_cett_vectorized(
                    z, i, h_norm, adjusted_answer_indices
                )
                
                # 收集该层的特征
                all_answer_features.append(answer_vec)
                all_other_features.append(other_vec)
                
                # 清理中间变量
                del gate, up, z, h_total, h_norm, normed_hs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 拼接所有层的特征（向量化操作）
            final_answer_features = np.concatenate(all_answer_features, axis=0)
            final_other_features = np.concatenate(all_other_features, axis=0)
            
            # 转换为列表格式（更高效的JSON序列化）
            features_answer_list = final_answer_features.tolist()
            features_other_list = final_other_features.tolist()
            
            self.logger.info(f"特征提取成功，特征数量: {len(features_answer_list)}")
            
            # 清理前向传播结果
            del outputs, hidden_states
            torch.cuda.empty_cache()
            
            # 修复JSON序列化问题：确保所有数据类型都是Python原生类型
            return {
                'features_answer_list': features_answer_list,  # 列表格式，与训练代码一致
                'features_other_list': features_other_list,    # 列表格式，与训练代码一致
                'aligned_tokens': cett_answer_decoded,
                'original_tokens': orig_answer_text,
                'answer_indices': [int(idx) for idx in adjusted_answer_indices],  # 转换为Python int
                'is_aligned': bool(is_match),  # 确保是Python bool
                'total_features': int(len(features_answer_list))  # 转换为Python int
            }
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU内存不足，跳过此样本")
            gc.collect()
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            self.logger.error(f"提取特征失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

# ==================== 3. PCA特征处理器 ====================
class PCAFeatureProcessor:
    """PCA特征处理器 - 用于处理PCA降维后的模型"""
    
    def __init__(self, pca_model_data):
        # 加载PCA模型组件
        self.scaler = pca_model_data.get('scaler', None)
        self.pca = pca_model_data.get('pca', None)
        self.classifier = pca_model_data.get('classifier', None)
        self.results = pca_model_data.get('results', {})
        self.config = pca_model_data.get('config', {})
        
        self.logger = logging.getLogger(__name__)
        
        # 检查必要组件
        if self.pca is None or self.classifier is None:
            raise ValueError("PCA模型或分类器未找到")
        
        # 记录模型信息
        self.original_dim = self.pca.n_features_in_ if hasattr(self.pca, 'n_features_in_') else "未知"
        self.pca_dim = self.pca.n_components_ if hasattr(self.pca, 'n_components_') else "未知"
        
        self.logger.info(f"PCA模型信息:")
        self.logger.info(f"  原始特征维度: {self.original_dim}")
        self.logger.info(f"  PCA降维维度: {self.pca_dim}")
        self.logger.info(f"  分类器类型: {type(self.classifier).__name__}")
        
        if hasattr(self.pca, 'explained_variance_ratio_'):
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            self.logger.info(f"  解释方差比例: {explained_variance:.4f}")
    
    def process_features(self, features_list: List[float]) -> np.ndarray:
        """处理特征：标准化 -> PCA降维"""
        if features_list is None:
            return None
        
        # 转换为numpy数组
        features = np.array(features_list, dtype=np.float32).reshape(1, -1)
        
        # 检查特征维度
        if hasattr(self.pca, 'n_features_in_') and features.shape[1] != self.pca.n_features_in_:
            self.logger.error(f"特征维度不匹配! 预期: {self.pca.n_features_in_}, 实际: {features.shape[1]}")
            return None
        
        # 1. 标准化（如果训练时使用了标准化）
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # 2. PCA降维
        features_pca = self.pca.transform(features_scaled)
        
        # 3. 检查并清理特征
        if np.any(np.isnan(features_pca)) or np.any(np.isinf(features_pca)):
            features_pca = np.nan_to_num(features_pca, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.logger.info(f"特征处理完成: {features.shape[1]} → {features_pca.shape[1]}")
        
        return features_pca
    
    def predict(self, features_list: List[float]) -> Tuple[float, bool]:
        """预测：处理特征后分类"""
        features_pca = self.process_features(features_list)
        if features_pca is None:
            return 0.0, False
        
        # 使用分类器预测
        prob = self.classifier.predict_proba(features_pca)[0][1]
        is_hallucination = prob > GlobalConfig.threshold
        
        # 获取最重要的PCA成分（用于调试）
        if hasattr(self.classifier, 'coef_'):
            top_indices = np.argsort(np.abs(self.classifier.coef_[0]))[-5:][::-1]
            self.logger.info(f"最重要的PCA成分: {top_indices}")
        
        return float(prob), bool(is_hallucination)

# ==================== 4. 模型加载器 ====================
class ModelLoader:
    """模型和分词器加载器"""
    
    @staticmethod
    def load_model_and_tokenizer(config):
        """加载模型和分词器"""
        logger.info(f"加载模型: {config.model_path}")
        
        bnb_config = None
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        logger.info("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        model.eval()
        logger.info(f"模型加载完成")
        
        return model, tokenizer

# ==================== 5. 辅助函数：Python类型转换 ====================
def convert_to_python_types(obj):
    """
    递归地将numpy/pandas类型转换为Python原生类型
    解决JSON序列化问题
    """
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_python_types(obj.tolist())
    elif hasattr(obj, 'item'):  # 其他可能的numpy标量类型
        try:
            return obj.item()
        except:
            return obj
    else:
        return obj

# ==================== 6. 主程序逻辑 ====================
def main():
    config = GlobalConfig()
    
    # Step A: 加载PCA分类器组件
    logger.info(f"正在加载PCA分类器包: {config.classifier_path}")
    try:
        with open(config.classifier_path, 'rb') as f:
            pca_model_data = pickle.load(f)
        
        logger.info(f"PCA分类器加载成功")
        
        # 检查关键组件
        required_components = ['classifier', 'pca']
        missing = [c for c in required_components if c not in pca_model_data]
        if missing:
            logger.error(f"缺少关键组件: {missing}")
            return
        
        # 初始化PCA特征处理器
        feature_processor = PCAFeatureProcessor(pca_model_data)
        
        # 记录配置信息
        if 'config' in pca_model_data:
            logger.info(f"模型配置: {json.dumps(pca_model_data['config'], indent=2)}")
        
    except Exception as e:
        logger.error(f"加载PCA分类器失败: {e}")
        return

    # Step B: 加载模型
    logger.info(f"正在加载模型: {config.model_path}")
    try:
        model, tokenizer = ModelLoader.load_model_and_tokenizer(config)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # Step C: 初始化CETT特征提取器
    cett_extractor = HybridCETTExtractor(model, tokenizer, config)

    # Step D: 读取数据
    logger.info(f"正在读取测试数据: {config.input_data_path}")
    try:
        with open(config.input_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载 {len(data)} 条测试数据")
    except Exception as e:
        logger.error(f"加载测试数据失败: {e}")
        return
    
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    logger.info(f"开始逐条检测，结果将保存到: {config.output_path}")

    # 流式写入JSON文件
    with open(config.output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("[\n")  # JSON数组开始
        
        processed_count = 0
        failed_count = 0
        alignment_failures = 0
        
        for idx, sample in enumerate(tqdm(data, desc="处理测试样本")):
            question = sample.get("question", "").strip()
            source_id = sample.get("source_id", f"sample_{idx}")
            reference_answer = sample.get("reference_answer", "")
            model_answer = sample.get("model_answer", "")
            extracted_part = sample.get("extracted_part", "")
            
            if not question or not model_answer: 
                logger.warning(f"样本 {idx} 问题或回答为空，跳过")
                continue
            
            try:
                logger.info(f"\n=== 处理样本 {idx+1}/{len(data)}: {source_id} ===")
                
                # 使用已有的extracted_part字段
                answer_text = extracted_part
                if not answer_text or answer_text not in model_answer:
                    logger.warning(f"extracted_part不在model_answer中，尝试使用完整回答: {answer_text[:50]}...")
                    # 回退：使用完整回答作为答案词元
                    answer_text = model_answer
                
                logger.info(f"模型回答: {model_answer[:100]}...")
                logger.info(f"提取部分: {answer_text[:100]}...")

                # 提取CETT特征（使用混合优化版本）
                logger.info("开始CETT特征提取...")
                features_result = cett_extractor.extract_features_for_sample(question, model_answer, answer_text)
                
                if features_result is None:
                    logger.error(f"样本 {idx} 特征提取失败，跳过")
                    failed_count += 1
                    continue
                
                if not features_result.get('is_aligned', False):
                    alignment_failures += 1
                    logger.warning(f"对齐失败，使用完整回答作为答案词元")
                
                # 处理特征并预测（使用PCA降维）
                logger.info("处理特征并预测...")
                features_list = features_result['features_answer_list']
                prob, is_hallu = feature_processor.predict(features_list)
                
                logger.info(f"预测结果 - 概率: {prob:.4f}, 是否幻觉: {is_hallu}")

                # 整合结果并确保Python原生类型
                res_item = convert_to_python_types({
                    "source_id": source_id,
                    "question": question,
                    "reference_answer": reference_answer,
                    "model_answer": model_answer,
                    "extracted_part": answer_text,
                    "hallucination_prob": round(float(prob), 4),
                    "is_hallucination": is_hallu,
                    "model_type": "pca",  # 标识模型类型
                    "pca_dim": feature_processor.pca_dim,
                    "original_dim": feature_processor.original_dim
                })
                
                # 流式写入文件
                if processed_count > 0: 
                    f_out.write(",\n")
                f_out.write(json.dumps(res_item, ensure_ascii=False, indent=2))
                f_out.flush()
                
                processed_count += 1

                # 定期清理内存
                if (idx + 1) % 20 == 0:
                    logger.info(f"已处理 {idx+1} 个样本，清理内存...")
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"处理样本 {idx} 失败: {e}")
                failed_count += 1
                # 写入错误结果
                if processed_count > 0: 
                    f_out.write(",\n")
                error_item = convert_to_python_types({
                    "source_id": source_id,
                    "question": question,
                    "reference_answer": reference_answer,
                    "model_answer": model_answer,
                    "extracted_part": extracted_part,
                    "hallucination_prob": -1.0,
                    "is_hallucination": False,
                    "error": str(e),
                    "model_type": "pca"
                })
                f_out.write(json.dumps(error_item, ensure_ascii=False, indent=2))
                f_out.flush()
                continue
        
        f_out.write("\n]")  # JSON数组结束
    
    # 生成汇总报告
    logger.info(f"\n✨ 检测完成！最终结果保存至: {config.output_path}")
    
    # 读取结果生成统计报告
    try:
        with open(config.output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        summary_path = config.output_path.replace('.json', '_summary.json')
        summary = convert_to_python_types({
            "total_samples": len(data),
            "processed_samples": len(successful_results),
            "failed_samples": len(failed_results),
            "alignment_failures": alignment_failures,
            "hallucination_count": sum(1 for r in successful_results if r["is_hallucination"]),
            "hallucination_rate": sum(1 for r in successful_results if r["is_hallucination"]) / len(successful_results) if successful_results else 0,
            "avg_hallucination_prob": np.mean([r["hallucination_prob"] for r in successful_results]) if successful_results else 0,
            "model_info": {
                "model_type": "pca",
                "pca_dimension": feature_processor.pca_dim,
                "original_dimension": feature_processor.original_dim,
                "explained_variance": np.sum(feature_processor.pca.explained_variance_ratio_) if hasattr(feature_processor.pca, 'explained_variance_ratio_') else "未知"
            },
            "threshold": config.threshold,
            "model_path": config.model_path,
            "classifier_path": config.classifier_path,
            "processing_time": datetime.now().isoformat()
        })
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("PCA降维模型检测统计报告")
        print(f"{'='*60}")
        print(f"总样本数: {summary['total_samples']}")
        print(f"成功处理: {summary['processed_samples']}")
        print(f"处理失败: {summary['failed_samples']}")
        print(f"对齐失败: {summary['alignment_failures']}")
        print(f"幻觉数量: {summary['hallucination_count']}")
        print(f"幻觉比例: {summary['hallucination_rate']:.2%}")
        print(f"平均概率: {summary['avg_hallucination_prob']:.4f}")
        print(f"\n模型信息:")
        print(f"  模型类型: {summary['model_info']['model_type']}")
        print(f"  原始特征维度: {summary['model_info']['original_dimension']}")
        print(f"  PCA降维维度: {summary['model_info']['pca_dimension']}")
        if summary['model_info']['explained_variance'] != "未知":
            print(f"  解释方差: {summary['model_info']['explained_variance']:.4f}")
        print(f"\n详细结果: {config.output_path}")
        print(f"统计报告: {summary_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"生成统计报告失败: {e}")

if __name__ == "__main__":
    print("="*60)
    print("H-Neurons幻觉检测测试（PCA降维版本）")
    print("="*60)
    
    # 运行主测试
    main()
    
    print("\n" + "="*60)
    print("批量测试完成!")
    print("="*60)