"""
H-Neurons测试推理管道 - 弹性网络版本
适用于弹性网络训练模型
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
from tqdm import tqdm
import gc
from dataclasses import dataclass, field

# ==================== 添加缺失的Config类定义 ====================
@dataclass
class Config:
    # 数据路径
    features_file: str = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
    output_dir: str = "/mnt/project/H-Neurous/output/elastic_net_reference"
    
    # 训练参数
    test_size: float = 0.15
    cv_folds: int = 5
    random_seed: int = 42
    
    # 弹性网络参数
    C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    l1_ratios: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    
    # 特征选择参数
    feature_selection_percentile: int = 10
    feature_selection_method: str = "f_classif"
    
    # 稳定性选择参数
    stability_iterations: int = 50
    stability_threshold: float = 0.8
    
    # 日志和实验
    experiment_name: str = "h_neurons_elastic_net_reference"
    
    def __post_init__(self):
        # 创建目录（推理时可能不需要，但保持兼容）
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)

# ==================== 全局配置 ====================
class GlobalConfig:
    # 路径配置
    model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    # 弹性网络模型路径
    classifier_path = "/mnt/project/H-Neurous/output/elastic_net_reference/h_neurons_elastic_net_reference_20260118_103645/elastic_net_model.pkl"
    input_data_path = "/mnt/project/H-Neurous/data/test/test_inference_chat.json"
    output_path = "/mnt/project/H-Neurous/data/test/test_inference_elastic_net.json"
    
    # 推理参数
    max_seq_length = 1024
    use_4bit = True
    extract_all_layers = True
    threshold = 0.5  # 判定阈值

# 日志监控
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/mnt/project/H-Neurous/logs/test_inference_elastic_net.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 2. CETT特征提取器（与训练一致） ====================
class ElasticNetCETTExtractor:
    """CETT特征提取器 - 弹性网络版本"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 获取模型信息
        self.num_layers = len(model.model.layers)
        self.layers_to_extract = list(range(self.num_layers))
        
        self.logger.info(f"模型总层数: {self.num_layers}")
        
        # 预计算权重范数
        self.logger.info("预计算权重范数...")
        self.precomputed_w_norms = []
        for layer_idx in self.layers_to_extract:
            mlp = self.model.model.layers[layer_idx].mlp
            w_down = mlp.down_proj.weight
            
            if hasattr(w_down, 'quant_state'):
                w_down_dequant = bnb.functional.dequantize_4bit(w_down.data, w_down.quant_state)
            else:
                w_down_dequant = w_down
            
            w_norm = torch.norm(w_down_dequant.float(), dim=0, p=2).cpu()
            self.precomputed_w_norms.append(w_norm)
            
            del w_down_dequant
        
        torch.cuda.empty_cache()
        
        # 记录神经元数量
        if self.layers_to_extract:
            first_layer = self.model.model.layers[0].mlp
            self.neurons_per_layer = first_layer.up_proj.weight.shape[1]
            self.total_neurons = len(self.layers_to_extract) * self.neurons_per_layer
            self.logger.info(f"每层神经元数: {self.neurons_per_layer}")
            self.logger.info(f"总神经元数: {self.total_neurons}")

    def _validate_with_unicode_tolerance(self, orig_text, cett_text, max_tolerate=3):
        """乱码容忍校验"""
        s1 = "".join(orig_text.split())
        s2 = "".join(cett_text.split())

        if len(s1) != len(s2):
            return False, f"长度不一 (预期{len(s1)} vs 实际{len(s2)})"

        mismatches = 0
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                if c2 == '' or c2 == '\ufffd':
                    mismatches += 1
                else:
                    return False, f"字符冲突 ('{c1}' vs '{c2}')"

        if mismatches > max_tolerate:
            return False, f"乱码过多 ({mismatches}处)"

        return True, "对齐成功"
    
    def _compute_cett_vectorized(self, z: torch.Tensor, layer_idx: int, 
                               h_norm: torch.Tensor, answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """向量化计算CETT特征"""
        w_norm = self.precomputed_w_norms[layer_idx].to(z.device).to(z.dtype)
        z_abs = torch.abs(z[0])
        scaled_z = z_abs / h_norm.unsqueeze(-1)
        
        seq_len = z_abs.shape[0]
        all_indices = set(range(seq_len))
        valid_answer_indices = [idx for idx in answer_indices if idx < seq_len]
        other_indices = list(all_indices - set(valid_answer_indices))
        
        answer_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
        other_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
        
        if valid_answer_indices:
            answer_cett = torch.mean(scaled_z[valid_answer_indices], dim=0) * w_norm
        
        if other_indices:
            other_cett = torch.mean(scaled_z[other_indices], dim=0) * w_norm
        
        answer_np = answer_cett.detach().cpu().numpy().astype(np.float32)
        other_np = other_cett.detach().cpu().numpy().astype(np.float32)
        
        del z_abs, scaled_z, answer_cett, other_cett
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return answer_np, other_np

    def extract_features_for_sample(self, question: str, model_answer: str, answer_text: str) -> Optional[Dict]:
        """提取CETT特征"""
        try:
            self.logger.info(f"开始提取特征 - 问题: {question[:50]}...")
            
            # 处理答案文本
            if not answer_text or answer_text not in model_answer:
                self.logger.warning("答案文本不在model_answer中，使用完整回答")
                answer_text = model_answer
            
            start_char = model_answer.find(answer_text)
            end_char = start_char + len(answer_text)
            
            if start_char == -1:
                answer_text = model_answer
                start_char = 0
                end_char = len(model_answer)
            
            # 构建Chat消息格式（与训练一致）
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
            
            full_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 分词处理
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
            
            # 精确对齐
            ans_start_in_prompt = full_prompt.rfind(model_answer)
            if ans_start_in_prompt == -1:
                ans_start_in_prompt = full_prompt.find("assistant") + len("assistant") + 5
                if ans_start_in_prompt <= len("assistant") + 5:
                    ans_start_in_prompt = len(full_prompt) - len(model_answer)
            
            rel_start_char = model_answer.find(answer_text)
            if rel_start_char == -1:
                answer_text = model_answer
                rel_start_char = 0
            
            abs_start = ans_start_in_prompt + rel_start_char
            abs_end = abs_start + len(answer_text)
            
            # 找到对应的token索引
            adjusted_answer_indices = []
            for token_idx, (c_start, c_end) in enumerate(full_offsets):
                if c_start == 0 and c_end == 0:
                    continue
                if not (c_end <= abs_start or c_start >= abs_end):
                    adjusted_answer_indices.append(token_idx)
            
            if not adjusted_answer_indices:
                for token_idx, (c_start, c_end) in enumerate(full_offsets):
                    if c_start == 0 and c_end == 0:
                        continue
                    if c_start >= ans_start_in_prompt:
                        adjusted_answer_indices.append(token_idx)
            
            # 验证对齐结果
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
            
            is_match, match_msg = self._validate_with_unicode_tolerance(
                answer_text, cett_answer_decoded
            )
            
            self.logger.warning(
                "\n================ CETT FULL DEBUG =================\n"
                f"[问题]\n{question}\n\n"
                f"[原始答案文本]\n{model_answer}\n\n"
                f"[原始答案词元]\n{answer_text}\n\n"
                f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
                f"[对齐结果]\n{match_msg}\n"
                "=================================================="
            )
            
            if not is_match:
                self.logger.warning(f"对齐校验不通过: {match_msg}")
                self.logger.info("回退: 使用完整回答作为答案词元")
                adjusted_answer_indices = []
                for token_idx, (c_start, c_end) in enumerate(full_offsets):
                    if c_start == 0 and c_end == 0:
                        continue
                    if c_start >= ans_start_in_prompt:
                        adjusted_answer_indices.append(token_idx)
            
            # 前向传播获取特征
            self.logger.info("开始前向传播...")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = outputs.hidden_states
            
            # 向量化提取特征
            all_answer_features = []
            all_other_features = []
            
            self.logger.info(f"向量化计算CETT特征，共{len(self.layers_to_extract)}层")
            for i, layer_idx in enumerate(self.layers_to_extract):
                layer = self.model.model.layers[layer_idx]
                normed_hs = layer.post_attention_layernorm(hidden_states[layer_idx])
                
                mlp = layer.mlp
                gate = mlp.gate_proj(normed_hs)
                up = mlp.up_proj(normed_hs)
                z = torch.nn.functional.silu(gate) * up
                
                h_total = mlp.down_proj(z)
                h_norm = torch.norm(h_total[0].float(), dim=-1, p=2)
                h_norm = torch.clamp(h_norm, min=1e-10)
                
                answer_vec, other_vec = self._compute_cett_vectorized(
                    z, i, h_norm, adjusted_answer_indices
                )
                
                all_answer_features.append(answer_vec)
                all_other_features.append(other_vec)
                
                del gate, up, z, h_total, h_norm, normed_hs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 拼接特征
            final_answer_features = np.concatenate(all_answer_features, axis=0)
            final_other_features = np.concatenate(all_other_features, axis=0)
            
            features_answer_list = final_answer_features.tolist()
            features_other_list = final_other_features.tolist()
            
            self.logger.info(f"特征提取成功，特征数量: {len(features_answer_list)}")
            
            del outputs, hidden_states
            torch.cuda.empty_cache()
            
            return {
                'features_answer_list': features_answer_list,
                'features_other_list': features_other_list,
                'aligned_tokens': cett_answer_decoded,
                'original_tokens': answer_text,
                'answer_indices': [int(idx) for idx in adjusted_answer_indices],
                'is_aligned': bool(is_match),
                'total_features': int(len(features_answer_list))
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

# ==================== 3. 弹性网络特征处理器 ====================
class ElasticNetFeatureProcessor:
    """弹性网络特征处理器"""
    
    def __init__(self, model_data):
        # 从模型数据中加载组件
        self.classifier = model_data['classifier']
        self.neuron_names = model_data.get('neuron_names', [])
        self.h_neuron_indices = model_data.get('h_neuron_indices', [])
        self.h_neuron_names = model_data.get('h_neuron_names', [])
        self.h_neuron_weights = model_data.get('h_neuron_weights', [])
        
        # 弹性网络特有的预处理组件
        self.var_selector = model_data.get('var_selector')
        self.percentile_selector = model_data.get('percentile_selector')
        self.scaler = model_data.get('scaler')
        
        self.logger = logging.getLogger(__name__)
        
        # 记录特征维度信息
        self.original_dim = len(self.neuron_names)
        
        if self.var_selector and hasattr(self.var_selector, 'get_support'):
            var_selected = np.sum(self.var_selector.get_support())
            self.logger.info(f"方差选择后特征: {var_selected}")
        
        if self.percentile_selector and hasattr(self.percentile_selector, 'get_support'):
            percentile_selected = np.sum(self.percentile_selector.get_support())
            self.logger.info(f"百分位数选择后特征: {percentile_selected}")
        
        self.logger.info(f"弹性网络特征处理信息:")
        self.logger.info(f"  原始特征维度: {self.original_dim}")
        self.logger.info(f"  H-Neurons数量: {len(self.h_neuron_indices)}")
        self.logger.info(f"  最佳L1比例: {model_data.get('best_params', {}).get('l1_ratio', 'N/A')}")
        self.logger.info(f"  最佳C值: {model_data.get('best_params', {}).get('C', 'N/A')}")
    
    def process_features(self, features_list: List[float]) -> np.ndarray:
        """处理特征，与训练时相同的pipeline"""
        if features_list is None:
            return None
        
        # 转换为numpy数组
        features = np.array(features_list, dtype=np.float32).reshape(1, -1)
        
        # 确保特征维度正确
        if features.shape[1] != self.original_dim:
            self.logger.error(f"特征维度不匹配! 预期: {self.original_dim}, 实际: {features.shape[1]}")
            return None
        
        # 应用与训练时相同的特征处理pipeline
        x_proc = features.copy()
        
        # 1. 方差选择
        if self.var_selector is not None:
            try:
                x_proc = self.var_selector.transform(x_proc)
                self.logger.debug("方差选择完成")
            except Exception as e:
                self.logger.warning(f"方差选择失败: {e}")
        
        # 2. 百分位数选择
        if self.percentile_selector is not None:
            try:
                x_proc = self.percentile_selector.transform(x_proc)
                self.logger.debug("百分位数选择完成")
            except Exception as e:
                self.logger.warning(f"百分位数选择失败: {e}")
        
        # 3. 标准化
        if self.scaler is not None:
            try:
                x_proc = self.scaler.transform(x_proc)
                self.logger.debug("标准化完成")
            except Exception as e:
                self.logger.warning(f"标准化失败: {e}")
        
        # 4. 检查并清理特征
        if np.any(np.isnan(x_proc)) or np.any(np.isinf(x_proc)):
            self.logger.warning("发现无效值，进行清理")
            x_proc = np.nan_to_num(x_proc, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.logger.debug(f"处理后特征形状: {x_proc.shape}")
        return x_proc
    
    def predict(self, features_list: List[float]) -> Tuple[float, bool]:
        """使用弹性网络模型预测"""
        x_proc = self.process_features(features_list)
        if x_proc is None:
            self.logger.error("特征处理失败")
            return 0.0, False
        
        try:
            # 弹性网络使用predict_proba
            prob = self.classifier.predict_proba(x_proc)[0][1]
            is_hallucination = prob > GlobalConfig.threshold
            
            self.logger.info(f"预测概率: {prob:.4f}, 阈值: {GlobalConfig.threshold}, 是否幻觉: {is_hallucination}")
            
            return float(prob), bool(is_hallucination)
        except Exception as e:
            self.logger.error(f"预测失败: {e}")
            return 0.0, False

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
    """递归地将numpy/pandas类型转换为Python原生类型"""
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
    elif hasattr(obj, 'item'):
        try:
            return obj.item()
        except:
            return obj
    else:
        return obj

# ==================== 6. 主程序逻辑 ====================
def main():
    config = GlobalConfig()
    
    # Step A: 加载弹性网络模型
    logger.info(f"正在加载弹性网络模型: {config.classifier_path}")
    try:
        with open(config.classifier_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info(f"弹性网络模型加载成功")
        
        # 检查关键组件
        required_components = ['classifier']
        missing = [c for c in required_components if c not in model_data]
        if missing:
            logger.error(f"缺少关键组件: {missing}")
            return
        
        # 初始化特征处理器
        feature_processor = ElasticNetFeatureProcessor(model_data)
        
    except Exception as e:
        logger.error(f"加载弹性网络模型失败: {e}")
        return

    # Step B: 加载LLM模型
    logger.info(f"正在加载LLM模型: {config.model_path}")
    try:
        model, tokenizer = ModelLoader.load_model_and_tokenizer(config)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # Step C: 初始化CETT特征提取器
    cett_extractor = ElasticNetCETTExtractor(model, tokenizer, config)

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
                    logger.warning(f"extracted_part不在model_answer中，使用完整回答")
                    answer_text = model_answer
                
                logger.info(f"模型回答: {model_answer[:100]}...")
                logger.info(f"提取部分: {answer_text[:100]}...")

                # 提取CETT特征
                logger.info("开始CETT特征提取...")
                features_result = cett_extractor.extract_features_for_sample(question, model_answer, answer_text)
                
                if features_result is None:
                    logger.error(f"样本 {idx} 特征提取失败，跳过")
                    failed_count += 1
                    continue
                
                if not features_result.get('is_aligned', False):
                    alignment_failures += 1
                    logger.warning(f"对齐失败，使用完整回答作为答案词元")
                
                # 处理特征并预测
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
                    "is_hallucination": is_hallu
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
                    "error": str(e)
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
            "alignment_success_rate": sum(1 for r in successful_results if r.get("cett_extraction_result", {}).get("is_aligned", False)) / len(successful_results) if successful_results else 0,
            "method": "elastic_net",
            "model_params": {
                "h_neurons_count": len(feature_processor.h_neuron_indices) if hasattr(feature_processor, 'h_neuron_indices') else 0,
                "l1_ratio": model_data.get('best_params', {}).get('l1_ratio', 'N/A'),
                "C_value": model_data.get('best_params', {}).get('C', 'N/A')
            },
            "threshold": config.threshold,
            "model_path": config.model_path,
            "classifier_path": config.classifier_path,
            "processing_time": datetime.now().isoformat()
        })
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("弹性网络检测统计报告")
        print(f"{'='*60}")
        print(f"总样本数: {summary['total_samples']}")
        print(f"成功处理: {summary['processed_samples']}")
        print(f"处理失败: {summary['failed_samples']}")
        print(f"对齐失败: {summary['alignment_failures']}")
        print(f"对齐成功率: {summary['alignment_success_rate']:.2%}")
        print(f"幻觉数量: {summary['hallucination_count']}")
        print(f"幻觉比例: {summary['hallucination_rate']:.2%}")
        print(f"平均概率: {summary['avg_hallucination_prob']:.4f}")
        print(f"\n模型参数:")
        print(f"  H-Neurons数量: {summary['model_params']['h_neurons_count']}")
        print(f"  L1比例: {summary['model_params']['l1_ratio']}")
        print(f"  C值: {summary['model_params']['C_value']}")
        print(f"\n详细结果: {config.output_path}")
        print(f"统计报告: {summary_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"生成统计报告失败: {e}")

if __name__ == "__main__":
    print("="*60)
    print("H-Neurons幻觉检测测试 - 弹性网络版本")
    print("="*60)
    
    # 运行主测试
    main()
    
    print("\n" + "="*60)
    print("批量测试完成!")
    print("="*60)