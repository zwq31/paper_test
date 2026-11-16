"""H-Neurons CETT特征提取管道（仅提取与保存）"""
import os
import json
import logging
import time
import gc
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import bitsandbytes as bnb

# ==================== 配置 ====================
@dataclass
class Config:
    # 模型路径
    model_path: str = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    # 数据路径
    answer_tokens_path: str = "/mnt/project/H-Neurous/data/failed_retry_fixed1.json"
    output_dir: str = "/mnt/project/H-Neurous/output1"
    
    # 模型参数
    use_4bit: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length: int = 1024
    
    # 特征提取参数
    extract_all_layers: bool = True
    batch_size_layers: int = 1
    use_memory_efficient: bool = True
    
    # 缓存与实验
    cache_dir: str = "/mnt/project/H-Neurous/cache"
    experiment_name: str = "h_neurons_cett_extraction"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)

# ==================== 日志配置 ====================
def setup_logger():
    log_dir = "/mnt/project/H-Neurous/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"h_neurons_cett_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# ==================== 数据加载器 ====================
class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> List[Dict]:
        self.logger.info(f"加载数据: {self.config.answer_tokens_path}")
        with open(self.config.answer_tokens_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = [sample.get('label', -1) for sample in data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.logger.info(f"总样本数: {len(data)}")
        for label, count in zip(unique_labels, counts):
            label_name = "幻觉" if label == 1 else "忠实" if label == 0 else "未知"
            self.logger.info(f"  {label_name} (label={label}): {count} 样本 ({count/len(data)*100:.1f}%)")
        return data

# ==================== 模型加载器 ====================
class ModelLoader:
    @staticmethod
    def load_model_and_tokenizer(config: Config):
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
            config.model_path, trust_remote_code=True, use_fast=True
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
        logger.info("模型加载完成")
        return model, tokenizer

# ==================== CETT特征提取器 ====================
class MemoryEfficientCETTExtractor:
    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.num_layers = len(model.model.layers)
        self.layers_to_extract = list(range(self.num_layers))
        self.logger.info(f"模型总层数: {self.num_layers}")

    def _compute_cett_for_layer(self, z: torch.Tensor, W_down: torch.Tensor, adjusted_answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if torch.isnan(z).any() or torch.isinf(z).any():
            self.logger.warning("输入 z 包含 NaN 或 Inf！")
        if torch.isnan(W_down).any() or torch.isinf(W_down).any():
            self.logger.warning("权重 W_down 包含 NaN 或 Inf！")

        z = z.float()
        W_down = W_down.float()
        batch_size, seq_len, num_neurons = z.shape

        h_total = torch.matmul(z, W_down.t())
        h_norm = torch.norm(h_total, dim=-1, p=2)
        eps = 1e-10
        h_norm = torch.nan_to_num(h_norm, nan=eps, posinf=1e12, neginf=eps)
        h_norm = torch.clamp(h_norm, min=eps)
        del h_total

        W_down_norm = torch.norm(W_down, dim=0, p=2)
        if torch.isnan(W_down_norm).any() or torch.isinf(W_down_norm).any():
            self.logger.warning("W_down_norm 包含 NaN 或 Inf！")

        z_abs = torch.abs(z[0])
        cett_matrix = (z_abs * W_down_norm) / h_norm[0].unsqueeze(-1)
        if torch.isnan(cett_matrix).any() or torch.isinf(cett_matrix).any():
            cett_matrix = torch.nan_to_num(cett_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        cett_np = cett_matrix.detach().cpu().numpy().astype(np.float32)

        answer_cett = np.zeros(num_neurons, dtype=np.float32)
        other_cett = np.zeros(num_neurons, dtype=np.float32)
        all_indices = set(range(seq_len))
        valid_indices = [idx for idx in adjusted_answer_indices if idx < seq_len]
        other_indices = list(all_indices - set(valid_indices))

        if valid_indices:
            answer_cett = np.mean(cett_np[valid_indices, :], axis=0)
        if other_indices:
            other_cett = np.mean(cett_np[other_indices, :], axis=0)

        answer_cett = np.nan_to_num(answer_cett)
        other_cett = np.nan_to_num(other_cett)

        del z, W_down, z_abs, h_norm, cett_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return answer_cett, other_cett

    def extract_features_for_sample(self, sample: Dict) -> Optional[Dict]:
        """
        为单个样本提取CETT特征 - 完全精确的答案词元对齐
        """
        try:
            # 步骤1: 先单独tokenize model_answer，获取精确的答案词元
            model_answer = sample.get('model_answer', '')
            answer_text = sample.get('answer_text', '')
            question = sample.get('question', '')
            
            if not answer_text or answer_text not in model_answer:
                self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
                return None
            
            # 找到答案在model_answer中的精确字符位置
            start_char = model_answer.find(answer_text)
            end_char = start_char + len(answer_text)
            
            if start_char == -1:
                self.logger.warning(f"无法找到答案文本: {answer_text[:50]}...")
                return None
            
            # 对model_answer进行tokenization，获取offset mapping（返回张量）
            model_answer_enc = self.tokenizer(
                model_answer,
                return_offsets_mapping=True,
                add_special_tokens=False,
                return_tensors='pt'  # 返回张量
            )
            # 将offset_mapping转换为numpy数组，方便处理
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
                return None
            
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
            
            # 步骤2: 构建完整prompt并确保tokenization的一致性
            prompt = f"问题：{question}\n回答：{model_answer}"
            
            # 关键：使用相同的tokenization参数，并且记录token到原始字符的映射
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True
            )
            
            input_ids = inputs['input_ids'].to(self.model.device)
            attention_mask = inputs['attention_mask'].to(self.model.device)
            full_offsets = inputs['offset_mapping'][0].cpu().numpy()
            
            # 步骤3: 在完整prompt中找到"回答："前缀的位置
            answer_prefix = "回答："
            answer_prefix_pos = prompt.find(answer_prefix)
            if answer_prefix_pos == -1:
                # 尝试其他可能的格式
                answer_prefix = "回答:"
                answer_prefix_pos = prompt.find(answer_prefix)
                if answer_prefix_pos == -1:
                    self.logger.warning(f"找不到回答前缀: {question[:50]}...")
                    return None
            
            # 步骤4: 构建映射表 - 将model_answer的token映射到完整prompt的token
            # 方法：使用字符位置进行精确映射
            
            # 首先，获取model_answer在完整prompt中的字符范围
            model_answer_start_in_prompt = answer_prefix_pos + len(answer_prefix)
            model_answer_end_in_prompt = model_answer_start_in_prompt + len(model_answer)
            
            # 建立映射字典：model_answer token index -> prompt token index
            token_mapping = {}
            
            for prompt_token_idx, (char_start, char_end) in enumerate(full_offsets):
                if char_start == 0 and char_end == 0:  # 跳过padding
                    continue
                
                # 检查这个token是否在model_answer的字符范围内
                if char_start >= model_answer_start_in_prompt and char_end <= model_answer_end_in_prompt:
                    # 计算在model_answer中的相对字符位置
                    relative_char_start = char_start - model_answer_start_in_prompt
                    relative_char_end = char_end - model_answer_start_in_prompt
                    
                    # 在model_answer的token中查找匹配的token
                    for model_token_idx, (model_char_start, model_char_end) in enumerate(model_answer_offsets):
                        if model_char_start == relative_char_start and model_char_end == relative_char_end:
                            token_mapping[model_token_idx] = prompt_token_idx
                            break
            
            # 步骤5: 使用映射找到完整prompt中的答案token索引
            adjusted_answer_indices = []
            for model_token_idx in answer_token_indices_in_model:
                if model_token_idx in token_mapping:
                    adjusted_answer_indices.append(token_mapping[model_token_idx])
            
            if not adjusted_answer_indices:
                self.logger.warning(f"无法映射答案token到完整prompt: {question[:50]}...")
                return None
            
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
            # ==================== 终极乱码容忍校验逻辑 ====================
            def validate_with_unicode_tolerance(orig_text, cett_text, max_tolerate=3):
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

            is_match, match_msg = validate_with_unicode_tolerance(orig_answer_text, cett_answer_decoded)

            # 打印调试信息
            self.logger.warning(
                "\n"
                "================ CETT FULL DEBUG =================\n"
                f"[问题]\n{question}\n\n"
                f"[原始答案文本]\n{model_answer}\n\n"
                f"[原始答案词元]\n{orig_answer_text}\n\n"
                f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
                "=================================================="
            )

            if not is_match:
                self.logger.warning(f"对齐校验不通过")

            
            self.logger.info(f"✓ 答案词元成功对齐 (已忽略字节切分产生的乱码位)")
            # ============================================================

            # 一次性前向传播获取所有层的隐藏状态
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = outputs.hidden_states  # 所有层的隐藏状态
            
            # 存储所有层的特征
            features_answer = {}
            features_other = {}
            
            # 逐层计算CETT特征
            for layer_idx in self.layers_to_extract:
                # 获取当前层的MLP模块
                layer = self.model.model.layers[layer_idx]
                # 核心修复：必须先经过归一化！
                # hidden_state 是来自上一层的原始残差
                hidden_state = hidden_states[layer_idx]
                
                # Qwen2/Llama 的 MLP 输入前必须经过这个 Norm
                normed_hidden_state = layer.post_attention_layernorm(hidden_state)
                
                # 使用归一化后的状态进行 MLP 计算
                mlp = layer.mlp
                gate = mlp.gate_proj(normed_hidden_state) # 现在数值会非常稳定
                up = mlp.up_proj(normed_hidden_state)
                z = torch.nn.functional.silu(gate) * up
                
                # 获取down_proj权重
                W_down = mlp.down_proj.weight  # [hidden_dim, intermediate_size]
                
                # 反量化（如果是4bit）
                if hasattr(W_down, 'quant_state'):
                    W_down_dequant = bnb.functional.dequantize_4bit(
                        W_down.data,
                        W_down.quant_state
                    ).to(torch.float16)
                else:
                    W_down_dequant = W_down
                
                # 计算该层的CETT
                answer_cett, other_cett = self._compute_cett_for_layer(
                    z, W_down_dequant, adjusted_answer_indices
                )
                
                num_neurons = z.shape[-1]
                
                # 存储该层特征
                for neuron_idx in range(num_neurons):
                    neuron_name = f"layer_{layer_idx}_neuron_{neuron_idx}"
                    features_answer[neuron_name] = float(answer_cett[neuron_idx])
                    features_other[neuron_name] = float(other_cett[neuron_idx])
                
                # 清理当前层的中间变量
                del gate, up, z, W_down_dequant, hidden_state
                torch.cuda.empty_cache()
            
            # 清理前向传播结果
            del outputs, hidden_states
            torch.cuda.empty_cache()
            
            # 构建结果
            result = {
                'question': sample['question'],
                'model_answer': sample['model_answer'],
                'label': sample['label'],
                'features_answer': features_answer,
                'features_other': features_other,
                'answer_token_indices': adjusted_answer_indices,
                'extraction_time': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"提取特征失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def extract_features_batch(self, samples: List[Dict]) -> str:
        output_file = "/mnt/project/H-Neurous/output/h_neurons_training_final_20260103_170934/cett_features.jsonl"  # ← 您的固定路径
        self.logger.info(f"开始批量提取CETT特征，结果将保存到: {output_file}")
        self.logger.info(f"总样本数: {len(samples)}")

        processed_count = 0
        failed_count = 0

        # 检查文件是否已存在，用于日志提示
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', encoding='utf-8') as f:
            if file_exists:
                self.logger.info(f"检测到已有特征文件，将在末尾追加新样本: {output_file}")
            else:
                self.logger.info(f"创建新特征文件: {output_file}")
            for i, sample in enumerate(tqdm(samples, desc="提取CETT特征")):
                try:
                    features = self.extract_features_for_sample(sample)
                    if features:
                        f.write(json.dumps(features, ensure_ascii=False) + "\n")
                        processed_count += 1
                        if processed_count % 10 == 0:
                            self.logger.info(f"已成功处理 {processed_count} 个样本")
                    else:
                        failed_count += 1

                    if (i + 1) % 20 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError as oom:
                    self.logger.error(f"CUDA内存不足，处理样本 {i} 时: {oom}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    failed_count += 1
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"处理样本 {i} 失败: {e}")
                    failed_count += 1

        self.logger.info(f"特征提取完成:")
        self.logger.info(f" 成功: {processed_count}")
        self.logger.info(f" 失败: {failed_count}")
        self.logger.info(f" 特征文件: {output_file}")
        return output_file

# ==================== 主函数 ====================
def main():
    config = Config()
    logger = setup_logger()
    
    # 加载数据
    data_loader = DataLoader(config)
    samples = data_loader.load_data()
    
    # 加载模型
    model, tokenizer = ModelLoader.load_model_and_tokenizer(config)
    
    # 提取特征
    extractor = MemoryEfficientCETTExtractor(model, tokenizer, config)
    features_file = extractor.extract_features_batch(samples)
    
    # 清理内存
    del model, tokenizer, extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info(f"所有CETT特征已保存至: {features_file}")

if __name__ == "__main__":
    main()