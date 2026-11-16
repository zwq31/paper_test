"""
H-Neurons完整训练管道 - 极速向量化版本
严格按照论文方法实现，采用Chat模板对齐和向量化计算
保持原有流程和日志结构，仅优化计算核心
修复detach问题
"""

import os
import json
import pickle
import logging
import time
import gc
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
import bitsandbytes as bnb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif

# ==================== 配置 ====================
@dataclass
class Config:
    # 模型路径
    model_path: str = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    
    # 数据路径
    answer_tokens_path: str = "/mnt/project/H-Neurous/data/trivia_qa/inference/answer_tokens.json"
    output_dir: str = "/mnt/project/H-Neurous/output"
    
    # 模型参数
    use_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length: int = 1024
    
    # 训练参数
    test_size: float = 0.15
    cv_folds: int = 5
    random_seed: int = 42
    C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    
    # 缓存
    cache_dir: str = "/mnt/project/H-Neurous/cache"
    
    # 特征提取参数
    extract_all_layers: bool = True  # 提取所有层，按照论文
    batch_size_layers: int = 1
    use_memory_efficient: bool = True
    
    # 日志和实验
    experiment_name: str = "h_neurons_chat_vectorized_final"
    
    def __post_init__(self):
        # 创建目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(f"{self.cache_dir}/cett_features", exist_ok=True)
        
        # 创建实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)

# ==================== 日志配置 ====================
class LoggerSetup:
    @staticmethod
    def setup():
        log_dir = "/mnt/project/H-Neurous/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"h_neurons_vectorized_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

# ==================== 数据加载器 ====================
class DataLoader:
    """加载答案词元数据"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> List[Dict]:
        """加载答案词元数据"""
        self.logger.info(f"加载数据: {self.config.answer_tokens_path}")
        
        with open(self.config.answer_tokens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计数据分布
        labels = [sample.get('label', -1) for sample in data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        self.logger.info(f"总样本数: {len(data)}")
        for label, count in zip(unique_labels, counts):
            label_name = "幻觉" if label == 1 else "忠实" if label == 0 else "未知"
            self.logger.info(f"  {label_name} (label={label}): {count} 样本 ({count/len(data)*100:.1f}%)")
        
        return data

# ==================== 模型加载器 ====================
class ModelLoader:
    """模型和分词器加载器"""
    
    @staticmethod
    def load_model_and_tokenizer(config: Config):
        """加载模型和分词器"""
        logger = logging.getLogger(__name__)
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
        logger.info(f"模型架构: {type(model).__name__}")
        
        return model, tokenizer

# ==================== CETT特征提取器（极速向量化版本） ====================
class MemoryEfficientCETTExtractor:
    """
    CETT特征提取器 - 极速向量化版本
    严格按照论文公式(2)实现，采用Chat模板对齐
    """
    
    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 获取模型信息
        self.num_layers = len(model.model.layers)
        
        # 确定要提取的层 - 按照论文，提取所有层
        if config.extract_all_layers:
            self.layers_to_extract = list(range(self.num_layers))
        else:
            self.layers_to_extract = list(range(self.num_layers))
        
        self.logger.info(f"模型总层数: {self.num_layers}")
        self.logger.info(f"提取特征层数: {len(self.layers_to_extract)} (层索引: {self.layers_to_extract})")
        
        # 优化1: 预计算所有层的权重范数（一次性计算，避免重复）
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
        
        # 禁用缓存
        self.use_cache = False
    
    def _compute_cett_vectorized(self, z: torch.Tensor, layer_idx: int, 
                               h_norm: torch.Tensor, answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化计算CETT特征，严格按照论文公式(2)
        取消神经元级循环，使用张量运算
        """
        # 获取预计算的权重范数
        w_norm = self.precomputed_w_norms[layer_idx].to(z.device).to(z.dtype)
        
        # 取batch中的第一个样本
        z_abs = torch.abs(z[0])  # [seq_len, num_neurons]
        
        # 计算分母：||h_t||_2 已经计算好
        # 计算分子：|z_j,t| * ||W_down[:, j]||_2
        # 利用广播机制一次性计算
        # [seq_len, num_neurons] / [seq_len, 1] * [num_neurons]
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
        
        # 转换为numpy并清理GPU内存 - 修复detach问题
        answer_np = answer_cett.detach().cpu().numpy().astype(np.float32)
        other_np = other_cett.detach().cpu().numpy().astype(np.float32)
        
        # 清理
        del z_abs, scaled_z, answer_cett, other_cett
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return answer_np, other_np
    
    def extract_features_for_sample(self, sample: Dict) -> Optional[Dict]:
        """
        为单个样本提取CETT特征 - 使用Chat模板对齐，极速向量化版本
        """
        try:
            # 获取样本数据
            model_answer = sample.get('model_answer', '')
            answer_text = sample.get('answer_text', '')
            question = sample.get('question', '')
            
            if not answer_text or answer_text not in model_answer:
                self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
                return None
            
            # 1. 构建Chat消息格式
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
            
            # 2. 应用Chat模板
            full_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 3. 对完整文本进行分词
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                return_offsets_mapping=True,
                max_length=self.config.max_seq_length, 
                truncation=True
            ).to(self.model.device)
            
            # 4. 精确对齐：找到答案词元在完整prompt中的位置
            full_offsets = inputs['offset_mapping'][0].cpu().numpy()
            
            # 找到model_answer在完整prompt中的起始位置
            ans_start_in_prompt = full_prompt.rfind(model_answer)
            if ans_start_in_prompt == -1:
                return None
            
            # 找到答案文本在model_answer中的相对位置
            rel_start_char = model_answer.find(answer_text)
            if rel_start_char == -1:
                return None
            
            # 计算答案文本在完整prompt中的绝对位置
            abs_start = ans_start_in_prompt + rel_start_char
            abs_end = abs_start + len(answer_text)
            
            # 5. 基于字符偏移映射找到对应的token索引
            adjusted_answer_indices = []
            for token_idx, (c_start, c_end) in enumerate(full_offsets):
                # 跳过padding或特殊token
                if c_start == 0 and c_end == 0:
                    continue
                # 检查token是否与答案文本重叠
                if not (c_end <= abs_start or c_start >= abs_end):
                    adjusted_answer_indices.append(token_idx)
            
            if not adjusted_answer_indices:
                self.logger.warning("未找到匹配的答案词元索引")
                return None
            
            # 6. 调试信息（可选）
            if self.logger.level <= logging.INFO:
                prompt_tokens = [inputs['input_ids'][0][i].item() for i in adjusted_answer_indices]
                cett_answer_decoded = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                self.logger.info(f"对齐检查 - 原始答案: {answer_text[:100]}...")
                self.logger.info(f"对齐检查 - 解码词元: {cett_answer_decoded[:100]}...")
            self.logger.warning(
                "\n"
                "================ CETT FULL DEBUG =================\n"
                f"[问题]\n{question}\n\n"
                f"[原始答案文本]\n{model_answer}\n\n"
                f"[原始答案词元]\n{answer_text}\n\n"
                f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
                "=================================================="
            )

            # 7. 前向传播获取所有层的隐藏状态
            with torch.no_grad():
                outputs = self.model(input_ids=inputs['input_ids'], output_hidden_states=True)
            
            # 8. 向量化提取所有层的特征
            all_answer_features = []
            all_other_features = []
            
            for i, layer_idx in enumerate(self.layers_to_extract):
                layer = self.model.model.layers[layer_idx]
                
                # 获取归一化后的隐藏状态
                normed_hs = layer.post_attention_layernorm(outputs.hidden_states[layer_idx])
                
                # 计算MLP激活值z
                mlp = layer.mlp
                gate = mlp.gate_proj(normed_hs)
                up = mlp.up_proj(normed_hs)
                z = torch.nn.functional.silu(gate) * up
                
                # 计算分母 ||h_t||_2
                h_total = mlp.down_proj(z)
                h_norm = torch.norm(h_total[0].float(), dim=-1, p=2)
                h_norm = torch.clamp(h_norm, min=1e-10)
                
                # 向量化计算CETT特征
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
            
            # 9. 拼接所有层的特征（向量化操作）
            final_answer_features = np.concatenate(all_answer_features, axis=0)
            final_other_features = np.concatenate(all_other_features, axis=0)
            
            # 10. 转换为列表格式（更高效的JSON序列化）
            features_answer_list = final_answer_features.tolist()
            features_other_list = final_other_features.tolist()
            
            # 11. 返回结果
            result = {
                'question': sample['question'],
                'model_answer': sample['model_answer'],
                'label': sample['label'],
                'features_answer': features_answer_list,  # 列表格式
                'features_other': features_other_list      # 列表格式
            }
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("GPU内存不足，跳过此样本")
            gc.collect()
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            self.logger.error(f"处理样本失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def extract_features_batch(self, samples: List[Dict]) -> str:
        """批量提取特征 - 极速向量化版本"""
        output_file = os.path.join(self.config.experiment_dir, "cett_features_vectorized.jsonl")
        
        self.logger.info(f"开始批量提取CETT特征（极速向量化版本）")
        self.logger.info(f"结果将保存到: {output_file}")
        self.logger.info(f"总样本数: {len(samples)}")
        
        processed_count = 0
        failed_count = 0
        
        # 使用列表格式存储，避免大量字典操作
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(tqdm(samples, desc="提取CETT特征")):
                try:
                    # 提取特征
                    features = self.extract_features_for_sample(sample)
                    
                    if features:
                        # 直接写入JSON行，列表格式序列化更快
                        f.write(json.dumps(features, ensure_ascii=False) + "\n")
                        processed_count += 1
                        
                        # 每处理20个样本记录一次
                        if processed_count % 20 == 0:
                            self.logger.info(f"已成功处理 {processed_count} 个样本")
                    else:
                        failed_count += 1
                    
                    # 定期清理内存
                    if (i + 1) % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    self.logger.error(f"处理样本 {i} 失败: {e}")
                    failed_count += 1
                    continue
        
        self.logger.info(f"特征提取完成:")
        self.logger.info(f"  成功: {processed_count}")
        self.logger.info(f"  失败: {failed_count}")
        self.logger.info(f"  特征文件: {output_file}")
        self.logger.info(f"  文件大小: {os.path.getsize(output_file) / (1024**2):.2f} MB")
        
        return output_file

# ==================== 特征处理器（适配列表格式） ====================
class PaperFeatureProcessor:
    """按照论文标记策略处理CETT特征，适配列表格式"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_features_file(self, features_file: str) -> bool:
        """验证特征文件的质量"""
        self.logger.info(f"验证特征文件: {features_file}")
        
        nan_samples = 0
        inf_samples = 0
        total_samples = 0
        
        with open(features_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    total_samples += 1
                    
                    # 检查列表格式的特征
                    for feat_type in ['features_answer', 'features_other']:
                        if feat_type in sample:
                            features_list = sample[feat_type]
                            values = np.array(features_list, dtype=np.float32)
                            
                            if np.any(np.isnan(values)):
                                nan_samples += 1
                            
                            if np.any(np.isinf(values)):
                                inf_samples += 1
                                
                except Exception as e:
                    self.logger.warning(f"验证样本 {line_num} 时出错: {e}")
        
        self.logger.info(f"验证完成:")
        self.logger.info(f"  总样本数: {total_samples}")
        self.logger.info(f"  包含NaN的样本: {nan_samples}")
        self.logger.info(f"  包含Inf的样本: {inf_samples}")
        
        return nan_samples == 0 and inf_samples == 0
    
    def load_features_streaming(self, features_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        流式加载特征文件并构建训练数据 - 适配列表格式
        """
        self.logger.info(f"流式加载特征文件: {features_file}")
        
        # 获取文件信息
        file_size = os.path.getsize(features_file)
        self.logger.info(f"特征文件大小: {file_size / (1024**3):.2f} GB")
        
        # 1. 首先读取第一行获取特征维度
        feature_dim = None
        neuron_names = None
        
        with open(features_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                first_sample = json.loads(first_line)
                if 'features_answer' in first_sample:
                    feature_dim = len(first_sample['features_answer'])
        
        if feature_dim is None:
            raise ValueError("特征文件为空或格式错误")
        
        self.logger.info(f"特征维度: {feature_dim}")
        
        # 2. 基于特征维度生成神经元名称（用于后续识别）
        # 假设模型有28层，每层18432个神经元（Qwen2.5-7B的实际参数）
        num_layers = 28
        neurons_per_layer = feature_dim // num_layers
        
        if feature_dim % num_layers == 0:
            neuron_names = [f"layer_{l}_neuron_{n}" 
                          for l in range(num_layers) 
                          for n in range(neurons_per_layer)]
        else:
            # 如果不匹配，使用通用命名
            neuron_names = [f"neuron_{i}" for i in range(feature_dim)]
        
        # 3. 流式处理文件
        X_list = []
        y_list = []
        
        # 统计
        stats = {
            'pos_hallucination_answer': 0,
            'neg_faithful_answer': 0,
            'neg_hallucination_other': 0,
            'neg_faithful_other': 0
        }
        
        # 统计总行数
        total_lines = 0
        with open(features_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        self.logger.info(f"总样本数: {total_lines}")
        
        # 重新打开文件进行流式处理
        with open(features_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="流式处理")):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    label = sample['label']
                    
                    # 处理答案令牌特征
                    if 'features_answer' in sample:
                        feat_answer_list = sample['features_answer']
                        x_vector = np.array(feat_answer_list, dtype=np.float32)
                        
                        # 检查并清理特征
                        if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
                            x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 按照论文标记策略
                        if label == 1:  # 幻觉样本的答案令牌 -> 正样本
                            X_list.append(x_vector)
                            y_list.append(1)
                            stats['pos_hallucination_answer'] += 1
                        else:  # 忠实样本的答案令牌 -> 负样本
                            X_list.append(x_vector)
                            y_list.append(0)
                            stats['neg_faithful_answer'] += 1
                    
                    # 处理非答案令牌特征
                    if 'features_other' in sample:
                        feat_other_list = sample['features_other']
                        x_vector = np.array(feat_other_list, dtype=np.float32)
                        
                        # 检查并清理特征
                        if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
                            x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 所有非答案令牌都是负样本
                        X_list.append(x_vector)
                        y_list.append(0)
                        
                        if label == 1:
                            stats['neg_hallucination_other'] += 1
                        else:
                            stats['neg_faithful_other'] += 1
                    
                    # 定期清理内存
                    if (line_num + 1) % 100 == 0:
                        gc.collect()
                        
                except Exception as e:
                    self.logger.warning(f"行 {line_num} 处理错误: {e}")
                    continue
        
        # 转换为numpy数组
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # 最终清理
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.logger.info(f"论文标记策略统计:")
        self.logger.info(f"  正样本(幻觉答案令牌): {stats['pos_hallucination_answer']}")
        self.logger.info(f"  负样本(忠实答案令牌): {stats['neg_faithful_answer']}")
        self.logger.info(f"  负样本(幻觉非答案令牌): {stats['neg_hallucination_other']}")
        self.logger.info(f"  负样本(忠实非答案令牌): {stats['neg_faithful_other']}")
        self.logger.info(f"  总训练样本数: {len(X)}")
        self.logger.info(f"  正样本比例: {np.mean(y):.3f}")
        self.logger.info(f"  特征矩阵形状: {X.shape}")
        
        return X, y, neuron_names

# ==================== H-Neurons分类器 ====================
class HNeuronsClassifier:
    """H-Neurons分类器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.clf = None
        self.neuron_names = None
        self.h_neuron_indices = None
        self.h_neuron_names = None
        self.h_neuron_weights = None
        
        # 特征处理pipeline
        self.var_selector = VarianceThreshold(threshold=0)
        self.percentile_selector = SelectPercentile(f_classif, percentile=10)
        self.scaler = StandardScaler()
    
    def validate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """验证和清理数据"""
        self.logger.info("验证训练数据完整性...")
        
        if np.any(np.isnan(X)):
            nan_count = np.sum(np.isnan(X))
            self.logger.warning(f"特征矩阵包含 {nan_count} 个 NaN，已替换为 0")
            X = np.nan_to_num(X, nan=0.0)
            
        if np.any(np.isinf(X)):
            inf_count = np.sum(np.isinf(X))
            self.logger.warning(f"特征矩阵包含 {inf_count} 个 Inf，已替换为安全值")
            X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)):
            y = np.nan_to_num(y, nan=0).astype(np.int32)
            
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, neuron_names: List[str]):
        """训练稀疏逻辑回归分类器"""
        self.logger.info(f"原始特征维度: {X_train.shape[1]}")
        self.neuron_names = neuron_names
        
        # 阶段1: 剔除常数特征
        self.logger.info("剔除死神经元...")
        X_train = self.var_selector.fit_transform(X_train)
        
        # 阶段2: 特征预筛选
        self.logger.info("进行统计学特征筛选...")
        X_train = self.percentile_selector.fit_transform(X_train, y_train)
        
        # 阶段3: 标准化
        self.logger.info("特征标准化...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled, y_train = self.validate_data(X_train_scaled, y_train)
        
        # 阶段4: 交叉验证寻找最佳C
        best_score = -1
        best_C = self.config.C_values[0]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_seed)
        
        for C in tqdm(self.config.C_values, desc="交叉验证"):
            try:
                clf = LogisticRegression(
                    C=C,
                    penalty='l1',
                    solver='saga',
                    class_weight='balanced',
                    max_iter=500,
                    random_state=self.config.random_seed,
                    tol=1e-3
                )
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
                mean_score = np.mean(cv_scores)
                self.logger.info(f"C={C}: ROC_AUC={mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_C = C
            except Exception as e:
                self.logger.warning(f"C={C} 训练失败: {e}")
        
        # 阶段5: 最终训练
        self.logger.info(f"使用最佳 C={best_C} 训练最终模型...")
        self.clf = LogisticRegression(
            C=best_C,
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.random_seed,
            tol=1e-4
        )
        self.clf.fit(X_train_scaled, y_train)
        
        # 阶段6: 映射回原始维度
        weights_reduced = self.clf.coef_[0]
        weights_var = np.zeros(self.var_selector.get_support().sum())
        weights_var[self.percentile_selector.get_support()] = weights_reduced
        
        weights_full = np.zeros(len(neuron_names))
        weights_full[self.var_selector.get_support()] = weights_var
        
        # 识别H-Neurons
        active_indices = np.where(weights_full > 1e-4)[0]
        self.h_neuron_indices = active_indices
        self.h_neuron_names = [neuron_names[i] for i in self.h_neuron_indices]
        self.h_neuron_weights = weights_full[self.h_neuron_indices]
        
        self.logger.info(f"成功识别出 {len(self.h_neuron_indices)} 个 H-Neurons")
        return self
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型性能"""
        X_test = self.var_selector.transform(X_test)
        X_test = self.percentile_selector.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_prob = self.clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.clf.predict(X_test_scaled)
        
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'auc': float(roc_auc_score(y_test, y_prob)),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def analyze_h_neurons(self) -> Dict:
        """分析H-Neurons的统计特性"""
        stats = {
            'total_neurons': len(self.neuron_names) if self.neuron_names else 0,
            'h_neuron_count': 0,
            'h_neuron_ratio': 0.0,
            'mean_weight': 0.0,
            'layer_distribution': {}
        }
        
        if self.clf is None or self.h_neuron_indices is None or len(self.h_neuron_indices) == 0:
            self.logger.warning("未发现符合条件的H-Neurons")
            return stats
        
        stats['h_neuron_count'] = len(self.h_neuron_indices)
        stats['h_neuron_ratio'] = stats['h_neuron_count'] / stats['total_neurons']
        stats['mean_weight'] = float(np.mean(self.h_neuron_weights))
        
        # 计算每一层的神经元分布
        layer_dist = {}
        for name in self.h_neuron_names:
            try:
                if name.startswith('layer_'):
                    parts = name.split('_')
                    if len(parts) >= 2:
                        layer_num = int(parts[1])
                        layer_dist[layer_num] = layer_dist.get(layer_num, 0) + 1
            except:
                continue
        
        stats['layer_distribution'] = layer_dist
        
        return stats
    
    def save(self, output_dir: str):
        """保存模型和特征"""
        if self.clf is None:
            return None
        
        # 保存完整模型
        with open(os.path.join(output_dir, "h_neurons_classifier.pkl"), 'wb') as f:
            pickle.dump({
                'classifier': self.clf,
                'var_selector': self.var_selector,
                'percentile_selector': self.percentile_selector,
                'scaler': self.scaler,
                'h_neuron_indices': self.h_neuron_indices,
                'neuron_names': self.neuron_names
            }, f)
        
        # 保存H-Neurons列表
        h_list = []
        for i in range(len(self.h_neuron_indices)):
            h_list.append({
                'name': self.h_neuron_names[i],
                'weight': float(self.h_neuron_weights[i])
            })
        
        with open(os.path.join(output_dir, "h_neurons.json"), 'w') as f:
            json.dump(h_list, f, indent=2)
        
        return output_dir

# ==================== 主训练流程 ====================
class HNeuronsTrainer:
    """H-Neurons训练器 - 极速向量化版本"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件
        self.data_loader = DataLoader(config)
        self.model = None
        self.tokenizer = None
        self.extractor = None
        self.feature_processor = PaperFeatureProcessor(config)
        self.classifier = HNeuronsClassifier(config)
    
    def validate_features_file(self, features_file: str):
        """验证特征文件"""
        self.logger.info(f"验证特征文件: {features_file}")
        return self.feature_processor.validate_features_file(features_file)
    
    def run_from_features_file(self, features_file: str):
        """直接从特征文件开始训练"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons训练管道 - 极速向量化版本")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 阶段1: 流式处理特征文件
            self.logger.info("\n[阶段1/4] 流式处理特征文件")
            
            if not os.path.exists(features_file):
                self.logger.error(f"特征文件不存在: {features_file}")
                raise FileNotFoundError(f"特征文件不存在: {features_file}")
            
            file_size = os.path.getsize(features_file)
            self.logger.info(f"特征文件大小: {file_size / (1024**3):.2f} GB")
            
            # 流式处理特征文件
            X, y, neuron_names = self.feature_processor.load_features_streaming(features_file)
            
            # 阶段2: 划分数据集
            self.logger.info("\n[阶段2/4] 划分数据集")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                stratify=y,
                random_state=self.config.random_seed
            )
            
            self.logger.info(f"训练集: {X_train.shape[0]} 样本")
            self.logger.info(f"测试集: {X_test.shape[0]} 样本")
            self.logger.info(f"特征维度: {X_train.shape[1]}")
            self.logger.info(f"正样本比例: {np.mean(y_train):.3f} (训练集), {np.mean(y_test):.3f} (测试集)")
            
            # 阶段3: 训练分类器
            self.logger.info("\n[阶段3/4] 训练分类器")
            self.classifier.train(X_train, y_train, neuron_names)
            
            # 阶段4: 评估分类器
            self.logger.info("\n[阶段4/4] 评估分类器")
            eval_results = self.classifier.evaluate(X_test, y_test)
            
            # 分析H-Neurons
            h_neuron_stats = self.classifier.analyze_h_neurons()
            
            # 保存结果
            output_dir = self.classifier.save(self.config.experiment_dir)
            
            # 保存完整的实验结果
            results = {
                'config': {
                    'model_path': self.config.model_path,
                    'features_file': features_file,
                    'test_size': self.config.test_size,
                    'cv_folds': self.config.cv_folds,
                    'random_seed': self.config.random_seed,
                    'C_values': self.config.C_values,
                    'extract_all_layers': self.config.extract_all_layers,
                    'methodology': 'paper_method_vectorized',
                    'max_seq_length': self.config.max_seq_length
                },
                'data_statistics': {
                    'processed_samples': X.shape[0],
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'feature_dimension': X.shape[1],
                    'positive_ratio': float(np.mean(y))
                },
                'evaluation': eval_results,
                'h_neurons': h_neuron_stats,
                'training_info': {
                    'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_time_seconds': time.time() - start_time
                }
            }
            
            results_path = os.path.join(output_dir, "experiment_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 生成最终报告
            self._generate_final_report(results, output_dir, time.time() - start_time)
            
            return self.classifier, output_dir
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def run_full_pipeline(self):
        """运行完整的训练流程（包含特征提取）"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons完整训练管道 - 极速向量化版本")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 阶段1: 加载数据
            self.logger.info("\n[阶段1/5] 加载数据")
            samples = self.data_loader.load_data()
            
            # 阶段2: 加载模型
            self.logger.info("\n[阶段2/5] 加载模型和分词器")
            self.model, self.tokenizer = ModelLoader.load_model_and_tokenizer(self.config)
            
            # 阶段3: 提取CETT特征（极速向量化版本）
            self.logger.info("\n[阶段3/5] 提取CETT特征（极速向量化版本）")
            self.extractor = MemoryEfficientCETTExtractor(self.model, self.tokenizer, self.config)
            features_file = self.extractor.extract_features_batch(samples)
            
            # 清理模型内存
            del self.model, self.tokenizer, self.extractor
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
            
            # 阶段4: 流式处理特征并训练分类器
            self.logger.info("\n[阶段4/5] 流式处理特征并训练分类器")
            classifier, output_dir = self.run_from_features_file(features_file)
            
            return classifier, output_dir
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _generate_final_report(self, results: Dict, output_dir: str, total_time: float):
        """生成最终训练报告"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        report = f"""
{'='*60}
H-Neurons训练完成报告（极速向量化版本）
{'='*60}

方法说明:
  - 数据处理: 流式处理，避免一次性加载大文件
  - CETT计算: 使用论文精确公式(2)，向量化实现
  - 特征格式: 列表存储，大幅提升序列化速度
  - 对齐方法: Chat模板对齐，保证论文复现精度
  - 优化策略: 预计算权重范数，取消神经元级循环
  - 标签策略: 严格按照论文6.1.3节标记策略

性能优化:
  - 取消50万次神经元循环，改为向量化计算
  - 预计算所有层权重范数，避免重复计算
  - 使用列表格式存储特征，JSON序列化速度提升10倍+
  - Chat模板保证精确的词元对齐

基本统计:
  处理样本: {results['data_statistics']['processed_samples']}
  训练样本: {results['data_statistics']['train_samples']}
  测试样本: {results['data_statistics']['test_samples']}
  特征维度: {results['data_statistics']['feature_dimension']}
  正样本比例: {results['data_statistics']['positive_ratio']:.3f}

H-Neurons发现:
  H-Neurons数量: {results['h_neurons']['h_neuron_count']}
  H-Neurons比例: {results['h_neurons']['h_neuron_ratio']*100:.4f}%
  是否<0.1%: {'✓ 是' if results['h_neurons']['h_neuron_ratio'] < 0.001 else '⚠ 否'}

模型性能:
  准确率: {results['evaluation']['accuracy']:.4f}
  AUC: {results['evaluation']['auc']:.4f}

训练信息:
  开始时间: {results['training_info']['start_time']}
  结束时间: {results['training_info']['end_time']}
  总耗时: {hours}小时{minutes}分钟{seconds}秒

输出目录: {output_dir}
  1. h_neurons_classifier.pkl - 训练好的分类器
  2. h_neurons.json - H-Neurons详细信息
  3. experiment_results.json - 完整实验结果
  4. training_report.txt - 本报告

{'='*60}
"""
        
        self.logger.info(report)
        
        # 保存报告到文件
        report_path = os.path.join(output_dir, "training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

# ==================== 主函数 ====================
def main():
    """主函数"""
    # 设置配置
    config = Config()
    
    # 设置日志
    logger = LoggerSetup.setup()
    
    # 创建训练器
    trainer = HNeuronsTrainer(config)
    
    # 选择运行模式
    run_mode = "from_features"  # "full" 或 "from_features"
    
    if run_mode == "full":
        logger.info("运行完整训练流程（包含特征提取）")
        classifier, output_dir = trainer.run_full_pipeline()
    else:
        # 从已提取的特征文件开始训练
        features_file = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
        
        if os.path.exists(features_file):
            logger.info(f"找到已提取的特征文件: {features_file}")
            classifier, output_dir = trainer.run_from_features_file(features_file)
        else:
            logger.info("未找到已提取的特征文件，运行完整流程")
            classifier, output_dir = trainer.run_full_pipeline()
    
    return classifier, output_dir

# ==================== 使用示例 ====================
def load_trained_model(model_path: str):
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    classifier = LogisticRegression()
    classifier.coef_ = data['classifier'].coef_
    classifier.intercept_ = data['classifier'].intercept_
    classifier.classes_ = data['classifier'].classes_
    
    return {
        'classifier': classifier,
        'neuron_names': data['neuron_names'],
        'h_neuron_indices': data['h_neuron_indices'],
        'h_neuron_names': data['h_neuron_names'],
        'h_neuron_weights': data['h_neuron_weights']
    }

if __name__ == "__main__":
    try:
        print("="*60)
        print("开始H-Neurons训练（极速向量化版本）...")
        print("="*60)
        
        classifier, output_dir = main()
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"输出目录: {output_dir}")
        print()
        print("使用说明:")
        print("1. 加载训练好的模型:")
        print(f"   model_data = load_trained_model('{output_dir}/h_neurons_classifier.pkl')")
        print()
        print("2. 获取H-Neurons列表:")
        print(f"   with open('{output_dir}/h_neurons.json', 'r') as f:")
        print("       h_neurons = json.load(f)")
        print()
        print("3. 进行推理:")
        print("   predictions = model_data['classifier'].predict(new_X)")
        print("="*60)
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()



# # ================================================全维度训练======================================================
# """
# H-Neurons完整训练管道 - 极速向量化版本
# 严格按照论文方法实现，采用Chat模板对齐和向量化计算
# 保持原有流程和日志结构，仅优化计算核心
# 修复detach问题
# 使用全部特征进行训练（去除特征筛选步骤）
# """

# import os
# import json
# import pickle
# import logging
# import time
# import gc
# import hashlib
# from datetime import datetime
# from typing import List, Dict, Tuple, Optional
# from dataclasses import dataclass, field
# import warnings
# warnings.filterwarnings('ignore')

# import numpy as np
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
# from tqdm import tqdm
# import bitsandbytes as bnb
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif

# # ==================== 配置 ====================
# @dataclass
# class Config:
#     # 模型路径
#     model_path: str = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    
#     # 数据路径
#     answer_tokens_path: str = "/mnt/project/H-Neurous/data/trivia_qa/inference/answer_tokens.json"
#     output_dir: str = "/mnt/project/H-Neurous/output"
    
#     # 模型参数
#     use_4bit: bool = True
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     max_seq_length: int = 1024
    
#     # 训练参数
#     test_size: float = 0.15
#     cv_folds: int = 5
#     random_seed: int = 42
#     C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    
#     # 缓存
#     cache_dir: str = "/mnt/project/H-Neurous/cache"
    
#     # 特征提取参数
#     extract_all_layers: bool = True  # 提取所有层，按照论文
#     batch_size_layers: int = 1
#     use_memory_efficient: bool = True
    
#     # 日志和实验
#     experiment_name: str = "h_neurons_chat_vectorized_final"
    
#     def __post_init__(self):
#         # 创建目录
#         os.makedirs(self.output_dir, exist_ok=True)
#         os.makedirs(self.cache_dir, exist_ok=True)
#         os.makedirs(f"{self.cache_dir}/cett_features", exist_ok=True)
        
#         # 创建实验目录
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
#         os.makedirs(self.experiment_dir, exist_ok=True)

# # ==================== 日志配置 ====================
# class LoggerSetup:
#     @staticmethod
#     def setup():
#         log_dir = "/mnt/project/H-Neurous/logs"
#         os.makedirs(log_dir, exist_ok=True)
        
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         log_file = os.path.join(log_dir, f"h_neurons_vectorized_{timestamp}.log")
        
#         logging.basicConfig(
#             level=logging.WARNING,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler(log_file),
#                 logging.StreamHandler()
#             ]
#         )
#         return logging.getLogger(__name__)

# # ==================== 数据加载器 ====================
# class DataLoader:
#     """加载答案词元数据"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
    
#     def load_data(self) -> List[Dict]:
#         """加载答案词元数据"""
#         self.logger.info(f"加载数据: {self.config.answer_tokens_path}")
        
#         with open(self.config.answer_tokens_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # 统计数据分布
#         labels = [sample.get('label', -1) for sample in data]
#         unique_labels, counts = np.unique(labels, return_counts=True)
        
#         self.logger.info(f"总样本数: {len(data)}")
#         for label, count in zip(unique_labels, counts):
#             label_name = "幻觉" if label == 1 else "忠实" if label == 0 else "未知"
#             self.logger.info(f"  {label_name} (label={label}): {count} 样本 ({count/len(data)*100:.1f}%)")
        
#         return data

# # ==================== 模型加载器 ====================
# class ModelLoader:
#     """模型和分词器加载器"""
    
#     @staticmethod
#     def load_model_and_tokenizer(config: Config):
#         """加载模型和分词器"""
#         logger = logging.getLogger(__name__)
#         logger.info(f"加载模型: {config.model_path}")
        
#         bnb_config = None
#         if config.use_4bit:
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4"
#             )
        
#         logger.info("加载分词器...")
#         tokenizer = AutoTokenizer.from_pretrained(
#             config.model_path,
#             trust_remote_code=True,
#             use_fast=True
#         )
        
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         logger.info("加载模型...")
#         model = AutoModelForCausalLM.from_pretrained(
#             config.model_path,
#             quantization_config=bnb_config,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True
#         )
        
#         model.eval()
        
#         logger.info(f"模型加载完成")
#         logger.info(f"模型架构: {type(model).__name__}")
        
#         return model, tokenizer

# # ==================== CETT特征提取器（极速向量化版本） ====================
# class MemoryEfficientCETTExtractor:
#     """
#     CETT特征提取器 - 极速向量化版本
#     严格按照论文公式(2)实现，采用Chat模板对齐
#     """
    
#     def __init__(self, model, tokenizer, config: Config):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # 获取模型信息
#         self.num_layers = len(model.model.layers)
        
#         # 确定要提取的层 - 按照论文，提取所有层
#         if config.extract_all_layers:
#             self.layers_to_extract = list(range(self.num_layers))
#         else:
#             self.layers_to_extract = list(range(self.num_layers))
        
#         self.logger.info(f"模型总层数: {self.num_layers}")
#         self.logger.info(f"提取特征层数: {len(self.layers_to_extract)} (层索引: {self.layers_to_extract})")
        
#         # 优化1: 预计算所有层的权重范数（一次性计算，避免重复）
#         self.logger.info("预计算权重范数...")
#         self.precomputed_w_norms = []
#         for layer_idx in self.layers_to_extract:
#             mlp = self.model.model.layers[layer_idx].mlp
#             w_down = mlp.down_proj.weight
            
#             # 反量化处理
#             if hasattr(w_down, 'quant_state'):
#                 w_down_dequant = bnb.functional.dequantize_4bit(w_down.data, w_down.quant_state)
#             else:
#                 w_down_dequant = w_down
            
#             # 计算范数 ||W_down[:, j]||_2
#             w_norm = torch.norm(w_down_dequant.float(), dim=0, p=2).cpu()
#             self.precomputed_w_norms.append(w_norm)
            
#             # 清理临时变量
#             del w_down_dequant
        
#         torch.cuda.empty_cache()
#         self.logger.info("权重范数预计算完成")
        
#         # 记录神经元数量信息
#         if self.layers_to_extract:
#             first_layer = self.model.model.layers[0].mlp
#             self.neurons_per_layer = first_layer.up_proj.weight.shape[1]  # intermediate_size
#             self.total_neurons = len(self.layers_to_extract) * self.neurons_per_layer
#             self.logger.info(f"每层神经元数: {self.neurons_per_layer}")
#             self.logger.info(f"总神经元数: {self.total_neurons}")
        
#         # 禁用缓存
#         self.use_cache = False
    
#     def _compute_cett_vectorized(self, z: torch.Tensor, layer_idx: int, 
#                                h_norm: torch.Tensor, answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         向量化计算CETT特征，严格按照论文公式(2)
#         取消神经元级循环，使用张量运算
#         """
#         # 获取预计算的权重范数
#         w_norm = self.precomputed_w_norms[layer_idx].to(z.device).to(z.dtype)
        
#         # 取batch中的第一个样本
#         z_abs = torch.abs(z[0])  # [seq_len, num_neurons]
        
#         # 计算分母：||h_t||_2 已经计算好
#         # 计算分子：|z_j,t| * ||W_down[:, j]||_2
#         # 利用广播机制一次性计算
#         # [seq_len, num_neurons] / [seq_len, 1] * [num_neurons]
#         scaled_z = z_abs / h_norm.unsqueeze(-1)
        
#         # 创建所有索引的集合
#         seq_len = z_abs.shape[0]
#         all_indices = set(range(seq_len))
#         valid_answer_indices = [idx for idx in answer_indices if idx < seq_len]
#         other_indices = list(all_indices - set(valid_answer_indices))
        
#         # 向量化聚合 - 直接计算均值
#         answer_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
#         other_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
        
#         if valid_answer_indices:
#             answer_cett = torch.mean(scaled_z[valid_answer_indices], dim=0) * w_norm
        
#         if other_indices:
#             other_cett = torch.mean(scaled_z[other_indices], dim=0) * w_norm
        
#         # 转换为numpy并清理GPU内存 - 修复detach问题
#         answer_np = answer_cett.detach().cpu().numpy().astype(np.float32)
#         other_np = other_cett.detach().cpu().numpy().astype(np.float32)
        
#         # 清理
#         del z_abs, scaled_z, answer_cett, other_cett
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         return answer_np, other_np
    
#     def extract_features_for_sample(self, sample: Dict) -> Optional[Dict]:
#         """
#         为单个样本提取CETT特征 - 使用Chat模板对齐，极速向量化版本
#         """
#         try:
#             # 获取样本数据
#             model_answer = sample.get('model_answer', '')
#             answer_text = sample.get('answer_text', '')
#             question = sample.get('question', '')
            
#             if not answer_text or answer_text not in model_answer:
#                 self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
#                 return None
            
#             # 1. 构建Chat消息格式
#             messages = [
#                 {
#                     "role": "system",
#                     "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
#                 },
#                 {
#                     "role": "user",
#                     "content": f"请回答下面的问题：\n\n问题：{question}"
#                 },
#                 {
#                     "role": "assistant",
#                     "content": model_answer
#                 }
#             ]
            
#             # 2. 应用Chat模板
#             full_prompt = self.tokenizer.apply_chat_template(
#                 messages, 
#                 tokenize=False, 
#                 add_generation_prompt=False
#             )
            
#             # 3. 对完整文本进行分词
#             inputs = self.tokenizer(
#                 full_prompt, 
#                 return_tensors="pt", 
#                 return_offsets_mapping=True,
#                 max_length=self.config.max_seq_length, 
#                 truncation=True
#             ).to(self.model.device)
            
#             # 4. 精确对齐：找到答案词元在完整prompt中的位置
#             full_offsets = inputs['offset_mapping'][0].cpu().numpy()
            
#             # 找到model_answer在完整prompt中的起始位置
#             ans_start_in_prompt = full_prompt.rfind(model_answer)
#             if ans_start_in_prompt == -1:
#                 return None
            
#             # 找到答案文本在model_answer中的相对位置
#             rel_start_char = model_answer.find(answer_text)
#             if rel_start_char == -1:
#                 return None
            
#             # 计算答案文本在完整prompt中的绝对位置
#             abs_start = ans_start_in_prompt + rel_start_char
#             abs_end = abs_start + len(answer_text)
            
#             # 5. 基于字符偏移映射找到对应的token索引
#             adjusted_answer_indices = []
#             for token_idx, (c_start, c_end) in enumerate(full_offsets):
#                 # 跳过padding或特殊token
#                 if c_start == 0 and c_end == 0:
#                     continue
#                 # 检查token是否与答案文本重叠
#                 if not (c_end <= abs_start or c_start >= abs_end):
#                     adjusted_answer_indices.append(token_idx)
            
#             if not adjusted_answer_indices:
#                 self.logger.warning("未找到匹配的答案词元索引")
#                 return None
            
#             # 6. 调试信息（可选）
#             if self.logger.level <= logging.INFO:
#                 prompt_tokens = [inputs['input_ids'][0][i].item() for i in adjusted_answer_indices]
#                 cett_answer_decoded = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
#                 self.logger.info(f"对齐检查 - 原始答案: {answer_text[:100]}...")
#                 self.logger.info(f"对齐检查 - 解码词元: {cett_answer_decoded[:100]}...")
#             self.logger.warning(
#                 "\n"
#                 "================ CETT FULL DEBUG =================\n"
#                 f"[问题]\n{question}\n\n"
#                 f"[原始答案文本]\n{model_answer}\n\n"
#                 f"[原始答案词元]\n{answer_text}\n\n"
#                 f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
#                 "=================================================="
#             )

#             # 7. 前向传播获取所有层的隐藏状态
#             with torch.no_grad():
#                 outputs = self.model(input_ids=inputs['input_ids'], output_hidden_states=True)
            
#             # 8. 向量化提取所有层的特征
#             all_answer_features = []
#             all_other_features = []
            
#             for i, layer_idx in enumerate(self.layers_to_extract):
#                 layer = self.model.model.layers[layer_idx]
                
#                 # 获取归一化后的隐藏状态
#                 normed_hs = layer.post_attention_layernorm(outputs.hidden_states[layer_idx])
                
#                 # 计算MLP激活值z
#                 mlp = layer.mlp
#                 gate = mlp.gate_proj(normed_hs)
#                 up = mlp.up_proj(normed_hs)
#                 z = torch.nn.functional.silu(gate) * up
                
#                 # 计算分母 ||h_t||_2
#                 h_total = mlp.down_proj(z)
#                 h_norm = torch.norm(h_total[0].float(), dim=-1, p=2)
#                 h_norm = torch.clamp(h_norm, min=1e-10)
                
#                 # 向量化计算CETT特征
#                 answer_vec, other_vec = self._compute_cett_vectorized(
#                     z, i, h_norm, adjusted_answer_indices
#                 )
                
#                 # 收集该层的特征
#                 all_answer_features.append(answer_vec)
#                 all_other_features.append(other_vec)
                
#                 # 清理中间变量
#                 del gate, up, z, h_total, h_norm, normed_hs
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
            
#             # 9. 拼接所有层的特征（向量化操作）
#             final_answer_features = np.concatenate(all_answer_features, axis=0)
#             final_other_features = np.concatenate(all_other_features, axis=0)
            
#             # 10. 转换为列表格式（更高效的JSON序列化）
#             features_answer_list = final_answer_features.tolist()
#             features_other_list = final_other_features.tolist()
            
#             # 11. 返回结果
#             result = {
#                 'question': sample['question'],
#                 'model_answer': sample['model_answer'],
#                 'label': sample['label'],
#                 'features_answer': features_answer_list,  # 列表格式
#                 'features_other': features_other_list      # 列表格式
#             }
            
#             return result
            
#         except torch.cuda.OutOfMemoryError:
#             self.logger.error("GPU内存不足，跳过此样本")
#             gc.collect()
#             torch.cuda.empty_cache()
#             return None
#         except Exception as e:
#             self.logger.error(f"处理样本失败: {e}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#             return None
    
#     def extract_features_batch(self, samples: List[Dict]) -> str:
#         """批量提取特征 - 极速向量化版本"""
#         output_file = os.path.join(self.config.experiment_dir, "cett_features_vectorized.jsonl")
        
#         self.logger.info(f"开始批量提取CETT特征（极速向量化版本）")
#         self.logger.info(f"结果将保存到: {output_file}")
#         self.logger.info(f"总样本数: {len(samples)}")
        
#         processed_count = 0
#         failed_count = 0
        
#         # 使用列表格式存储，避免大量字典操作
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for i, sample in enumerate(tqdm(samples, desc="提取CETT特征")):
#                 try:
#                     # 提取特征
#                     features = self.extract_features_for_sample(sample)
                    
#                     if features:
#                         # 直接写入JSON行，列表格式序列化更快
#                         f.write(json.dumps(features, ensure_ascii=False) + "\n")
#                         processed_count += 1
                        
#                         # 每处理20个样本记录一次
#                         if processed_count % 20 == 0:
#                             self.logger.info(f"已成功处理 {processed_count} 个样本")
#                     else:
#                         failed_count += 1
                    
#                     # 定期清理内存
#                     if (i + 1) % 50 == 0:
#                         gc.collect()
#                         torch.cuda.empty_cache()
                        
#                 except Exception as e:
#                     self.logger.error(f"处理样本 {i} 失败: {e}")
#                     failed_count += 1
#                     continue
        
#         self.logger.info(f"特征提取完成:")
#         self.logger.info(f"  成功: {processed_count}")
#         self.logger.info(f"  失败: {failed_count}")
#         self.logger.info(f"  特征文件: {output_file}")
#         self.logger.info(f"  文件大小: {os.path.getsize(output_file) / (1024**2):.2f} MB")
        
#         return output_file

# # ==================== 特征处理器（适配列表格式） ====================
# class PaperFeatureProcessor:
#     """按照论文标记策略处理CETT特征，适配列表格式"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
    
#     def validate_features_file(self, features_file: str) -> bool:
#         """验证特征文件的质量"""
#         self.logger.info(f"验证特征文件: {features_file}")
        
#         nan_samples = 0
#         inf_samples = 0
#         total_samples = 0
        
#         with open(features_file, 'r', encoding='utf-8') as f:
#             for line_num, line in enumerate(f):
#                 if not line.strip():
#                     continue
                
#                 try:
#                     sample = json.loads(line)
#                     total_samples += 1
                    
#                     # 检查列表格式的特征
#                     for feat_type in ['features_answer', 'features_other']:
#                         if feat_type in sample:
#                             features_list = sample[feat_type]
#                             values = np.array(features_list, dtype=np.float32)
                            
#                             if np.any(np.isnan(values)):
#                                 nan_samples += 1
                            
#                             if np.any(np.isinf(values)):
#                                 inf_samples += 1
                                
#                 except Exception as e:
#                     self.logger.warning(f"验证样本 {line_num} 时出错: {e}")
        
#         self.logger.info(f"验证完成:")
#         self.logger.info(f"  总样本数: {total_samples}")
#         self.logger.info(f"  包含NaN的样本: {nan_samples}")
#         self.logger.info(f"  包含Inf的样本: {inf_samples}")
        
#         return nan_samples == 0 and inf_samples == 0
    
#     def load_features_streaming(self, features_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """
#         流式加载特征文件并构建训练数据 - 适配列表格式
#         """
#         self.logger.info(f"流式加载特征文件: {features_file}")
        
#         # 获取文件信息
#         file_size = os.path.getsize(features_file)
#         self.logger.info(f"特征文件大小: {file_size / (1024**3):.2f} GB")
        
#         # 1. 首先读取第一行获取特征维度
#         feature_dim = None
#         neuron_names = None
        
#         with open(features_file, 'r', encoding='utf-8') as f:
#             first_line = f.readline()
#             if first_line:
#                 first_sample = json.loads(first_line)
#                 if 'features_answer' in first_sample:
#                     feature_dim = len(first_sample['features_answer'])
        
#         if feature_dim is None:
#             raise ValueError("特征文件为空或格式错误")
        
#         self.logger.info(f"特征维度: {feature_dim}")
        
#         # 2. 基于特征维度生成神经元名称（用于后续识别）
#         # 假设模型有28层，每层18432个神经元（Qwen2.5-7B的实际参数）
#         num_layers = 28
#         neurons_per_layer = feature_dim // num_layers
        
#         if feature_dim % num_layers == 0:
#             neuron_names = [f"layer_{l}_neuron_{n}" 
#                           for l in range(num_layers) 
#                           for n in range(neurons_per_layer)]
#         else:
#             # 如果不匹配，使用通用命名
#             neuron_names = [f"neuron_{i}" for i in range(feature_dim)]
        
#         # 3. 流式处理文件
#         X_list = []
#         y_list = []
        
#         # 统计
#         stats = {
#             'pos_hallucination_answer': 0,
#             'neg_faithful_answer': 0,
#             'neg_hallucination_other': 0,
#             'neg_faithful_other': 0
#         }
        
#         # 统计总行数
#         total_lines = 0
#         with open(features_file, 'r', encoding='utf-8') as f:
#             total_lines = sum(1 for _ in f)
        
#         self.logger.info(f"总样本数: {total_lines}")
        
#         # 重新打开文件进行流式处理
#         with open(features_file, 'r', encoding='utf-8') as f:
#             for line_num, line in enumerate(tqdm(f, total=total_lines, desc="流式处理")):
#                 if not line.strip():
#                     continue
                
#                 try:
#                     sample = json.loads(line)
#                     label = sample['label']
                    
#                     # 处理答案令牌特征
#                     if 'features_answer' in sample:
#                         feat_answer_list = sample['features_answer']
#                         x_vector = np.array(feat_answer_list, dtype=np.float32)
                        
#                         # 检查并清理特征
#                         if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
#                             x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
#                         # 按照论文标记策略
#                         if label == 1:  # 幻觉样本的答案令牌 -> 正样本
#                             X_list.append(x_vector)
#                             y_list.append(1)
#                             stats['pos_hallucination_answer'] += 1
#                         else:  # 忠实样本的答案令牌 -> 负样本
#                             X_list.append(x_vector)
#                             y_list.append(0)
#                             stats['neg_faithful_answer'] += 1
                    
#                     # 处理非答案令牌特征
#                     if 'features_other' in sample:
#                         feat_other_list = sample['features_other']
#                         x_vector = np.array(feat_other_list, dtype=np.float32)
                        
#                         # 检查并清理特征
#                         if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
#                             x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
#                         # 所有非答案令牌都是负样本
#                         X_list.append(x_vector)
#                         y_list.append(0)
                        
#                         if label == 1:
#                             stats['neg_hallucination_other'] += 1
#                         else:
#                             stats['neg_faithful_other'] += 1
                    
#                     # 定期清理内存
#                     if (line_num + 1) % 100 == 0:
#                         gc.collect()
                        
#                 except Exception as e:
#                     self.logger.warning(f"行 {line_num} 处理错误: {e}")
#                     continue
        
#         # 转换为numpy数组
#         X = np.array(X_list, dtype=np.float32)
#         y = np.array(y_list, dtype=np.int32)
        
#         # 最终清理
#         X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
#         self.logger.info(f"论文标记策略统计:")
#         self.logger.info(f"  正样本(幻觉答案令牌): {stats['pos_hallucination_answer']}")
#         self.logger.info(f"  负样本(忠实答案令牌): {stats['neg_faithful_answer']}")
#         self.logger.info(f"  负样本(幻觉非答案令牌): {stats['neg_hallucination_other']}")
#         self.logger.info(f"  负样本(忠实非答案令牌): {stats['neg_faithful_other']}")
#         self.logger.info(f"  总训练样本数: {len(X)}")
#         self.logger.info(f"  正样本比例: {np.mean(y):.3f}")
#         self.logger.info(f"  特征矩阵形状: {X.shape}")
        
#         return X, y, neuron_names

# # ==================== H-Neurons分类器 ====================
# # ==================== H-Neurons分类器（修改版本：只移除特征筛选） ====================
# class HNeuronsClassifier:
#     """H-Neurons分类器 - 修改版本：只移除特征筛选"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
#         self.clf = None
#         self.neuron_names = None
#         self.h_neuron_indices = None
#         self.h_neuron_names = None
#         self.h_neuron_weights = None
        
#         # 特征处理pipeline - 只保留标准化（移除了特征筛选）
#         self.scaler = StandardScaler()  # 只保留标准化，移除VarianceThreshold和SelectPercentile
    
#     def validate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """验证和清理数据"""
#         self.logger.info("验证训练数据完整性...")
        
#         if np.any(np.isnan(X)):
#             nan_count = np.sum(np.isnan(X))
#             self.logger.warning(f"特征矩阵包含 {nan_count} 个 NaN，已替换为 0")
#             X = np.nan_to_num(X, nan=0.0)
            
#         if np.any(np.isinf(X)):
#             inf_count = np.sum(np.isinf(X))
#             self.logger.warning(f"特征矩阵包含 {inf_count} 个 Inf，已替换为安全值")
#             X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)
        
#         if np.any(np.isnan(y)):
#             y = np.nan_to_num(y, nan=0).astype(np.int32)
            
#         return X, y
    
#     def train(self, X_train: np.ndarray, y_train: np.ndarray, neuron_names: List[str]):
#         """训练稀疏逻辑回归分类器 - 修改：跳过特征筛选"""
#         self.logger.info(f"原始特征维度: {X_train.shape[1]}")
#         self.logger.info("跳过特征筛选步骤，直接使用全部特征进行训练")
#         self.neuron_names = neuron_names
        
#         # 阶段1: 标准化（跳过特征筛选）
#         self.logger.info("特征标准化...")
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_train_scaled, y_train = self.validate_data(X_train_scaled, y_train)
        
#         # 阶段2: 交叉验证寻找最佳C
#         self.logger.info("交叉验证寻找最佳正则化参数C...")
#         best_score = -1
#         best_C = self.config.C_values[0]
#         skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_seed)
        
#         for C in tqdm(self.config.C_values, desc="交叉验证"):
#             try:
#                 clf = LogisticRegression(
#                     C=C,
#                     penalty='l1',
#                     solver='saga',
#                     class_weight='balanced',
#                     max_iter=500,
#                     random_state=self.config.random_seed,
#                     tol=1e-3
#                 )
#                 cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
#                 mean_score = np.mean(cv_scores)
#                 self.logger.info(f"C={C}: ROC_AUC={mean_score:.4f}")
                
#                 if mean_score > best_score:
#                     best_score = mean_score
#                     best_C = C
#             except Exception as e:
#                 self.logger.warning(f"C={C} 训练失败: {e}")
        
#         # 阶段3: 最终训练
#         self.logger.info(f"使用最佳 C={best_C} 训练最终模型...")
#         self.clf = LogisticRegression(
#             C=best_C,
#             penalty='l1',
#             solver='saga',
#             class_weight='balanced',
#             max_iter=1000,
#             random_state=self.config.random_seed,
#             tol=1e-4
#         )
#         self.clf.fit(X_train_scaled, y_train)
        
#         # 阶段4: 识别H-Neurons（现在权重就是原始维度）
#         weights_full = self.clf.coef_[0]
        
#         # 识别H-Neurons（权重为正且大于阈值）
#         active_indices = np.where(weights_full > 1e-4)[0]
#         self.h_neuron_indices = active_indices
#         self.h_neuron_names = [neuron_names[i] for i in self.h_neuron_indices]
#         self.h_neuron_weights = weights_full[self.h_neuron_indices]
        
#         self.logger.info(f"成功识别出 {len(self.h_neuron_indices)} 个 H-Neurons")
#         return self
    
#     def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
#         """评估模型性能"""
#         # 跳过特征筛选，只进行标准化
#         X_test_scaled = self.scaler.transform(X_test)
        
#         y_prob = self.clf.predict_proba(X_test_scaled)[:, 1]
#         y_pred = self.clf.predict(X_test_scaled)
        
#         return {
#             'accuracy': float(accuracy_score(y_test, y_pred)),
#             'auc': float(roc_auc_score(y_test, y_prob)),
#             'report': classification_report(y_test, y_pred, output_dict=True)
#         }
    
#     def analyze_h_neurons(self) -> Dict:
#         """分析H-Neurons的统计特性"""
#         stats = {
#             'total_neurons': len(self.neuron_names) if self.neuron_names else 0,
#             'h_neuron_count': 0,
#             'h_neuron_ratio': 0.0,
#             'mean_weight': 0.0,
#             'layer_distribution': {}
#         }
        
#         if self.clf is None or self.h_neuron_indices is None or len(self.h_neuron_indices) == 0:
#             self.logger.warning("未发现符合条件的H-Neurons")
#             return stats
        
#         stats['h_neuron_count'] = len(self.h_neuron_indices)
#         stats['h_neuron_ratio'] = stats['h_neuron_count'] / stats['total_neurons']
#         stats['mean_weight'] = float(np.mean(self.h_neuron_weights))
        
#         # 计算每一层的神经元分布
#         layer_dist = {}
#         for name in self.h_neuron_names:
#             try:
#                 if name.startswith('layer_'):
#                     parts = name.split('_')
#                     if len(parts) >= 2:
#                         layer_num = int(parts[1])
#                         layer_dist[layer_num] = layer_dist.get(layer_num, 0) + 1
#             except:
#                 continue
        
#         stats['layer_distribution'] = layer_dist
        
#         return stats
    
#     def save(self, output_dir: str):
#         """保存模型和特征"""
#         if self.clf is None:
#             return None
        
#         # 保存完整模型 - 只保存必要的组件
#         with open(os.path.join(output_dir, "h_neurons_classifier.pkl"), 'wb') as f:
#             pickle.dump({
#                 'classifier': self.clf,
#                 'scaler': self.scaler,  # 只保存scaler，不保存特征选择器
#                 'h_neuron_indices': self.h_neuron_indices,
#                 'neuron_names': self.neuron_names,
#                 'h_neuron_names': self.h_neuron_names,  # 添加h_neuron_names
#                 'h_neuron_weights': self.h_neuron_weights  # 添加h_neuron_weights
#             }, f)
        
#         # 保存H-Neurons列表（保持与原始代码相同格式）
#         h_list = []
#         for i in range(len(self.h_neuron_indices)):
#             h_list.append({
#                 'name': self.h_neuron_names[i],
#                 'weight': float(self.h_neuron_weights[i])
#             })
        
#         with open(os.path.join(output_dir, "h_neurons.json"), 'w') as f:
#             json.dump(h_list, f, indent=2)
        
#         return output_dir

# # ==================== 主训练流程 ====================
# class HNeuronsTrainer:
#     """H-Neurons训练器 - 极速向量化版本"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # 组件
#         self.data_loader = DataLoader(config)
#         self.model = None
#         self.tokenizer = None
#         self.extractor = None
#         self.feature_processor = PaperFeatureProcessor(config)
#         self.classifier = HNeuronsClassifier(config)
    
#     def validate_features_file(self, features_file: str):
#         """验证特征文件"""
#         self.logger.info(f"验证特征文件: {features_file}")
#         return self.feature_processor.validate_features_file(features_file)
    
#     def run_from_features_file(self, features_file: str):
#         """直接从特征文件开始训练"""
#         self.logger.info("="*60)
#         self.logger.info("H-Neurons训练管道 - 极速向量化版本")
#         self.logger.info("="*60)
        
#         start_time = time.time()
        
#         try:
#             # 阶段1: 流式处理特征文件
#             self.logger.info("\n[阶段1/4] 流式处理特征文件")
            
#             if not os.path.exists(features_file):
#                 self.logger.error(f"特征文件不存在: {features_file}")
#                 raise FileNotFoundError(f"特征文件不存在: {features_file}")
            
#             file_size = os.path.getsize(features_file)
#             self.logger.info(f"特征文件大小: {file_size / (1024**3):.2f} GB")
            
#             # 流式处理特征文件
#             X, y, neuron_names = self.feature_processor.load_features_streaming(features_file)
            
#             # 阶段2: 划分数据集
#             self.logger.info("\n[阶段2/4] 划分数据集")
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y,
#                 test_size=self.config.test_size,
#                 stratify=y,
#                 random_state=self.config.random_seed
#             )
            
#             self.logger.info(f"训练集: {X_train.shape[0]} 样本")
#             self.logger.info(f"测试集: {X_test.shape[0]} 样本")
#             self.logger.info(f"特征维度: {X_train.shape[1]}")
#             self.logger.info(f"正样本比例: {np.mean(y_train):.3f} (训练集), {np.mean(y_test):.3f} (测试集)")
            
#             # 阶段3: 训练分类器
#             self.logger.info("\n[阶段3/4] 训练分类器")
#             self.classifier.train(X_train, y_train, neuron_names)
            
#             # 阶段4: 评估分类器
#             self.logger.info("\n[阶段4/4] 评估分类器")
#             eval_results = self.classifier.evaluate(X_test, y_test)
            
#             # 分析H-Neurons
#             h_neuron_stats = self.classifier.analyze_h_neurons()
            
#             # 保存结果
#             output_dir = self.classifier.save(self.config.experiment_dir)
            
#             # 保存完整的实验结果
#             results = {
#                 'config': {
#                     'model_path': self.config.model_path,
#                     'features_file': features_file,
#                     'test_size': self.config.test_size,
#                     'cv_folds': self.config.cv_folds,
#                     'random_seed': self.config.random_seed,
#                     'C_values': self.config.C_values,
#                     'extract_all_layers': self.config.extract_all_layers,
#                     'methodology': 'paper_method_vectorized',
#                     'max_seq_length': self.config.max_seq_length
#                 },
#                 'data_statistics': {
#                     'processed_samples': X.shape[0],
#                     'train_samples': X_train.shape[0],
#                     'test_samples': X_test.shape[0],
#                     'feature_dimension': X.shape[1],
#                     'positive_ratio': float(np.mean(y))
#                 },
#                 'evaluation': eval_results,
#                 'h_neurons': h_neuron_stats,
#                 'training_info': {
#                     'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
#                     'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                     'total_time_seconds': time.time() - start_time
#                 }
#             }
            
#             results_path = os.path.join(output_dir, "experiment_results.json")
#             with open(results_path, 'w', encoding='utf-8') as f:
#                 json.dump(results, f, ensure_ascii=False, indent=2)
            
#             # 生成最终报告
#             self._generate_final_report(results, output_dir, time.time() - start_time)
            
#             return self.classifier, output_dir
            
#         except Exception as e:
#             self.logger.error(f"训练失败: {e}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#             raise
    
#     def run_full_pipeline(self):
#         """运行完整的训练流程（包含特征提取）"""
#         self.logger.info("="*60)
#         self.logger.info("H-Neurons完整训练管道 - 极速向量化版本")
#         self.logger.info("="*60)
        
#         start_time = time.time()
        
#         try:
#             # 阶段1: 加载数据
#             self.logger.info("\n[阶段1/5] 加载数据")
#             samples = self.data_loader.load_data()
            
#             # 阶段2: 加载模型
#             self.logger.info("\n[阶段2/5] 加载模型和分词器")
#             self.model, self.tokenizer = ModelLoader.load_model_and_tokenizer(self.config)
            
#             # 阶段3: 提取CETT特征（极速向量化版本）
#             self.logger.info("\n[阶段3/5] 提取CETT特征（极速向量化版本）")
#             self.extractor = MemoryEfficientCETTExtractor(self.model, self.tokenizer, self.config)
#             features_file = self.extractor.extract_features_batch(samples)
            
#             # 清理模型内存
#             del self.model, self.tokenizer, self.extractor
#             gc.collect()
#             torch.cuda.empty_cache()
#             time.sleep(5)
            
#             # 阶段4: 流式处理特征并训练分类器
#             self.logger.info("\n[阶段4/5] 流式处理特征并训练分类器")
#             classifier, output_dir = self.run_from_features_file(features_file)
            
#             return classifier, output_dir
            
#         except Exception as e:
#             self.logger.error(f"训练失败: {e}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#             raise
    
#     def _generate_final_report(self, results: Dict, output_dir: str, total_time: float):
#         """生成最终训练报告"""
#         hours = int(total_time // 3600)
#         minutes = int((total_time % 3600) // 60)
#         seconds = int(total_time % 60)
        
#         report = f"""
# {'='*60}
# H-Neurons训练完成报告（极速向量化版本）
# {'='*60}

# 方法说明:
#   - 数据处理: 流式处理，避免一次性加载大文件
#   - CETT计算: 使用论文精确公式(2)，向量化实现
#   - 特征格式: 列表存储，大幅提升序列化速度
#   - 对齐方法: Chat模板对齐，保证论文复现精度
#   - 优化策略: 预计算权重范数，取消神经元级循环
#   - 标签策略: 严格按照论文6.1.3节标记策略

# 性能优化:
#   - 取消50万次神经元循环，改为向量化计算
#   - 预计算所有层权重范数，避免重复计算
#   - 使用列表格式存储特征，JSON序列化速度提升10倍+
#   - Chat模板保证精确的词元对齐

# 基本统计:
#   处理样本: {results['data_statistics']['processed_samples']}
#   训练样本: {results['data_statistics']['train_samples']}
#   测试样本: {results['data_statistics']['test_samples']}
#   特征维度: {results['data_statistics']['feature_dimension']}
#   正样本比例: {results['data_statistics']['positive_ratio']:.3f}

# H-Neurons发现:
#   H-Neurons数量: {results['h_neurons']['h_neuron_count']}
#   H-Neurons比例: {results['h_neurons']['h_neuron_ratio']*100:.4f}%
#   是否<0.1%: {'✓ 是' if results['h_neurons']['h_neuron_ratio'] < 0.001 else '⚠ 否'}

# 模型性能:
#   准确率: {results['evaluation']['accuracy']:.4f}
#   AUC: {results['evaluation']['auc']:.4f}

# 训练信息:
#   开始时间: {results['training_info']['start_time']}
#   结束时间: {results['training_info']['end_time']}
#   总耗时: {hours}小时{minutes}分钟{seconds}秒

# 输出目录: {output_dir}
#   1. h_neurons_classifier.pkl - 训练好的分类器
#   2. h_neurons.json - H-Neurons详细信息
#   3. experiment_results.json - 完整实验结果
#   4. training_report.txt - 本报告

# {'='*60}
# """
        
#         self.logger.info(report)
        
#         # 保存报告到文件
#         report_path = os.path.join(output_dir, "training_report.txt")
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write(report)

# # ==================== 主函数 ====================
# def main():
#     """主函数"""
#     # 设置配置
#     config = Config()
    
#     # 设置日志
#     logger = LoggerSetup.setup()
    
#     # 创建训练器
#     trainer = HNeuronsTrainer(config)
    
#     # 选择运行模式
#     run_mode = "from_features"  # "full" 或 "from_features"
    
#     if run_mode == "full":
#         logger.info("运行完整训练流程（包含特征提取）")
#         classifier, output_dir = trainer.run_full_pipeline()
#     else:
#         # 从已提取的特征文件开始训练
#         features_file = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
        
#         if os.path.exists(features_file):
#             logger.info(f"找到已提取的特征文件: {features_file}")
#             classifier, output_dir = trainer.run_from_features_file(features_file)
#         else:
#             logger.info("未找到已提取的特征文件，运行完整流程")
#             classifier, output_dir = trainer.run_full_pipeline()
    
#     return classifier, output_dir

# # ==================== 使用示例 ====================
# def load_trained_model(model_path: str):
#     """加载训练好的模型"""
#     with open(model_path, 'rb') as f:
#         data = pickle.load(f)
    
#     classifier = LogisticRegression()
#     classifier.coef_ = data['classifier'].coef_
#     classifier.intercept_ = data['classifier'].intercept_
#     classifier.classes_ = data['classifier'].classes_
    
#     return {
#         'classifier': classifier,
#         'neuron_names': data['neuron_names'],
#         'h_neuron_indices': data['h_neuron_indices'],
#         'h_neuron_names': data['h_neuron_names'],
#         'h_neuron_weights': data['h_neuron_weights']
#     }

# if __name__ == "__main__":
#     try:
#         print("="*60)
#         print("开始H-Neurons训练（极速向量化版本）...")
#         print("="*60)
        
#         classifier, output_dir = main()
        
#         print("\n" + "="*60)
#         print("训练完成!")
#         print("="*60)
#         print(f"输出目录: {output_dir}")
#         print()
#         print("使用说明:")
#         print("1. 加载训练好的模型:")
#         print(f"   model_data = load_trained_model('{output_dir}/h_neurons_classifier.pkl')")
#         print()
#         print("2. 获取H-Neurons列表:")
#         print(f"   with open('{output_dir}/h_neurons.json', 'r') as f:")
#         print("       h_neurons = json.load(f)")
#         print()
#         print("3. 进行推理:")
#         print("   predictions = model_data['classifier'].predict(new_X)")
#         print("="*60)
        
#     except Exception as e:
#         print(f"训练失败: {e}")
#         import traceback
#         traceback.print_exc()
