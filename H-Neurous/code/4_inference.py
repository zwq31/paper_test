# """=========================================================API提取答案词元================================
# H-Neurons测试推理管道 - 混合优化版本
# 修复JSON序列化问题
# """

# import os
# import json
# import pickle
# import torch
# import logging
# import time
# import re
# import numpy as np
# from datetime import datetime
# from typing import List, Dict, Tuple, Optional
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import bitsandbytes as bnb
# from tqdm import tqdm
# from openai import OpenAI
# from tenacity import retry, stop_after_attempt, wait_exponential
# import gc

# # ==================== 1. 全局配置 ====================
# class GlobalConfig:
#     # 路径配置
#     model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
#     classifier_path = "/mnt/project/H-Neurous/output/saga_chat/h_neurons_classifier.pkl" 
#     input_data_path = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train_data.json"
#     output_path = "/mnt/project/H-Neurous/data/test/train_inference.json"
    
#     # API配置 (DeepSeek)
#     api_key = "__SET_VIA_ENVIRONMENT__"
#     base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#     extract_model = "deepseek-v3"
    
#     # 推理参数
#     max_seq_length = 1024  # 与训练保持一致
#     max_new_tokens = 1024
#     threshold = 0.5  # 判定阈值
#     use_4bit = True  # 与训练保持一致
#     extract_all_layers = True  # 与训练保持一致

# # 日志监控
# logging.basicConfig(
#     level=logging.WARNING,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("/mnt/project/H-Neurous/logs/test_inference.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # ==================== 2. 答案提取器 ====================
# class AnswerExtractor:
#     def __init__(self, config):
#         self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
#         self.model_name = config.extract_model

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
#     def extract_span(self, question: str, model_answer: str) -> Optional[Dict]:
#         prompt = f"""请分析以下对话，并从模型回答中精确提取直接回答问题的部分。

# 问题：{question}

# 模型回答：{model_answer}


# 请执行以下任务：
# 1. 仔细分析问题在询问什么
# 2. 在模型回答中找到直接回答问题的具体内容
# 3. 确定这部分内容在回答中的精确位置（字符位置）
# 4. 提取这部分内容

# 输出要求：
# 1. 必须使用以下JSON格式：
# {{
#   "answer_text": "从模型回答中提取的答案文本（必须是原文，不能修改）",
#   "start_char": 答案在模型回答中的起始字符位置（从0开始）,
#   "end_char": 答案在模型回答中的结束字符位置
# }}

# 2. 字符位置必须精确，不能有偏差
# 4. 只输出JSON，不要有其他任何内容

# 示例：
# 问题："中国的首都是哪里？"
# 模型回答："中国的首都是北京，北京是一座历史悠久的城市。"
# 输出：
# {{
#   "answer_text": "北京",
#   "start_char": 6,
#   "end_char": 8,
# }}
# 注意：必须是北京，而不是中国的首都是北京"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.1
#             )
#             content = response.choices[0].message.content
#             match = re.search(r'\{.*\}', content, re.DOTALL)
#             return json.loads(match.group()) if match else None
#         except Exception as e:
#             logger.error(f"API提取失败: {e}")
#             return None

# # ==================== 3. CETT特征提取器（混合优化版本） ====================
# class HybridCETTExtractor:
#     """
#     CETT特征提取器 - 混合优化版本
#     保持字符映射对齐逻辑，但使用训练代码的向量化计算和列表格式
#     """
    
#     def __init__(self, model, tokenizer, config):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # 获取模型信息
#         self.num_layers = len(model.model.layers)
        
#         # 确定要提取的层 - 按照论文，提取所有层
#         self.layers_to_extract = list(range(self.num_layers))
        
#         self.logger.info(f"模型总层数: {self.num_layers}")
#         self.logger.info(f"提取特征层数: {len(self.layers_to_extract)}")
        
#         # 预计算所有层的权重范数（与训练代码一致，用于向量化计算）
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

#     def _validate_with_unicode_tolerance(self, orig_text, cett_text, max_tolerate=3):
#         """终极乱码容忍校验逻辑（保持不变）"""
#         # 1. 清洗掉所有空白符（空格、换行等）
#         s1 = "".join(orig_text.split())
#         s2 = "".join(cett_text.split())

#         # 2. 长度校验：如果去掉空格后长度都不一样，说明 Token 映射逻辑肯定错了
#         if len(s1) != len(s2):
#             return False, f"长度不一 (预期{len(s1)} vs 实际{len(s2)})"

#         # 3. 逐字比对
#         mismatches = 0
#         for c1, c2 in zip(s1, s2):
#             if c1 != c2:
#                 # 如果不相等，检查 c2 是不是乱码符号 \ufffd ()
#                 if c2 == '' or c2 == '\ufffd':
#                     mismatches += 1
#                 else:
#                     # 发现真正的字符不匹配（不是乱码导致的），直接失败
#                     return False, f"字符冲突 ('{c1}' vs '{c2}')"

#         # 4. 容忍度校验：如果乱码位太多，可能也不是我们要的
#         if mismatches > max_tolerate:
#             return False, f"乱码过多 ({mismatches}处)"

#         return True, "对齐成功"
    
#     def _compute_cett_vectorized(self, z: torch.Tensor, layer_idx: int, 
#                                h_norm: torch.Tensor, answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         向量化计算CETT特征 - 完全使用训练代码的实现
#         """
#         # 获取预计算的权重范数
#         w_norm = self.precomputed_w_norms[layer_idx].to(z.device).to(z.dtype)
        
#         # 取batch中的第一个样本
#         z_abs = torch.abs(z[0])  # [seq_len, num_neurons]
        
#         # 计算分母：||h_t||_2 已经计算好
#         # 计算分子：|z_j,t| * ||W_down[:, j]||_2
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
        
#         # 转换为numpy并清理GPU内存
#         answer_np = answer_cett.detach().cpu().numpy().astype(np.float32)
#         other_np = other_cett.detach().cpu().numpy().astype(np.float32)
        
#         # 清理
#         del z_abs, scaled_z, answer_cett, other_cett
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         return answer_np, other_np

#     def extract_features_for_sample(self, question: str, model_answer: str, answer_text: str) -> Optional[Dict]:
#         """
#         为单个样本提取CETT特征 - 混合版本
#         保持字符映射对齐逻辑，但使用向量化计算和列表格式
#         """
#         try:
#             self.logger.info(f"开始提取特征 - 问题: {question[:50]}...")
            
#             # 步骤1: 先单独tokenize model_answer，获取精确的答案词元
#             if not answer_text or answer_text not in model_answer:
#                 self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 answer_text = model_answer
            
#             # 找到答案在model_answer中的精确字符位置
#             start_char = model_answer.find(answer_text)
#             end_char = start_char + len(answer_text)
            
#             if start_char == -1:
#                 self.logger.warning(f"无法找到答案文本: {answer_text[:50]}...")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 answer_text = model_answer
#                 start_char = 0
#                 end_char = len(model_answer)
            
#             # 对model_answer进行tokenization，获取offset mapping
#             self.logger.info(f"单独tokenize model_answer，长度: {len(model_answer)}")
#             model_answer_enc = self.tokenizer(
#                 model_answer,
#                 return_offsets_mapping=True,
#                 add_special_tokens=False,
#                 return_tensors='pt'
#             )
#             model_answer_offsets = model_answer_enc['offset_mapping'][0].cpu().numpy()
            
#             # 基于字符位置找到model_answer中的答案token索引
#             answer_token_indices_in_model = []
#             for token_idx, (char_start, char_end) in enumerate(model_answer_offsets):
#                 # token完全在答案区域内，或者部分重叠但主要部分在答案内
#                 if (char_start >= start_char and char_end <= end_char) or \
#                 (char_start <= start_char and char_end > start_char + (end_char - start_char) / 2):
#                     answer_token_indices_in_model.append(token_idx)
            
#             if not answer_token_indices_in_model:
#                 self.logger.warning(f"无法找到model_answer中的答案token: {answer_text[:50]}...")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 # 回退：使用所有token作为答案词元
#                 answer_token_indices_in_model = list(range(len(model_answer_offsets)))
            
#             # 获取原始答案词元文本
#             seq_len = model_answer_enc['input_ids'].size(1)
#             model_answer_answer_tokens = [
#                 model_answer_enc['input_ids'][0][i].item() 
#                 for i in answer_token_indices_in_model
#                 if i < seq_len
#             ]
#             orig_answer_text = self.tokenizer.decode(
#                 model_answer_answer_tokens,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False
#             )
            
#             # 步骤1: 构建与训练代码完全相同的Chat消息格式
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
            
#             # 步骤2: 应用Chat模板，与训练代码完全一致
#             full_prompt = self.tokenizer.apply_chat_template(
#                 messages, 
#                 tokenize=False, 
#                 add_generation_prompt=False
#             )
            
#             # 步骤3: 对完整文本进行分词，获取offset mapping
#             self.logger.info(f"使用Chat模板构建prompt，长度: {len(full_prompt)}")
#             inputs = self.tokenizer(
#                 full_prompt,
#                 return_tensors="pt",
#                 max_length=self.config.max_seq_length,
#                 truncation=True,
#                 padding="max_length",
#                 return_offsets_mapping=True
#             )
            
#             input_ids = inputs['input_ids'].to(self.model.device)
#             attention_mask = inputs['attention_mask'].to(self.model.device)
#             full_offsets = inputs['offset_mapping'][0].cpu().numpy()
            
#             # 步骤4: 精确对齐 - 找到答案文本在完整prompt中的位置
#             # 在完整prompt中找到model_answer的起始位置
#             ans_start_in_prompt = full_prompt.rfind(model_answer)
#             if ans_start_in_prompt == -1:
#                 self.logger.warning("无法在完整prompt中找到model_answer，使用启发式定位")
#                 # 回退：尝试查找"assistant"内容
#                 ans_start_in_prompt = full_prompt.find("assistant") + len("assistant") + 5
#                 if ans_start_in_prompt <= len("assistant") + 5:
#                     ans_start_in_prompt = len(full_prompt) - len(model_answer)
            
#             # 找到答案文本在model_answer中的相对位置
#             rel_start_char = model_answer.find(answer_text)
#             if rel_start_char == -1:
#                 self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
#                 # 回退：使用完整回答作为答案文本
#                 answer_text = model_answer
#                 rel_start_char = 0
            
#             # 计算答案文本在完整prompt中的绝对位置
#             abs_start = ans_start_in_prompt + rel_start_char
#             abs_end = abs_start + len(answer_text)
            
#             # 步骤5: 基于字符偏移映射找到对应的token索引
#             adjusted_answer_indices = []
#             for token_idx, (c_start, c_end) in enumerate(full_offsets):
#                 # 跳过padding或特殊token
#                 if c_start == 0 and c_end == 0:
#                     continue
                
#                 # 检查token是否与答案文本重叠
#                 if not (c_end <= abs_start or c_start >= abs_end):
#                     adjusted_answer_indices.append(token_idx)
            
#             if not adjusted_answer_indices:
#                 self.logger.warning("未找到匹配的答案词元索引，回退到完整回答")
#                 # 回退：将整个回答作为答案词元
#                 for token_idx, (c_start, c_end) in enumerate(full_offsets):
#                     if c_start == 0 and c_end == 0:
#                         continue
#                     if c_start >= ans_start_in_prompt:
#                         adjusted_answer_indices.append(token_idx)
            
#             # 获取CETT对齐后的答案词元
#             prompt_answer_tokens = [
#                 input_ids[0][i].item()
#                 for i in adjusted_answer_indices
#                 if i < input_ids.size(1)
#             ]
#             cett_answer_decoded = self.tokenizer.decode(
#                 prompt_answer_tokens,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False
#             )
            
#             # 验证对齐结果
#             orig_answer_text = answer_text
#             is_match, match_msg = self._validate_with_unicode_tolerance(
#                 orig_answer_text, cett_answer_decoded
#             )
            
#             self.logger.warning(
#                 "\n"
#                 "================ CETT FULL DEBUG =================\n"
#                 f"[问题]\n{question}\n\n"
#                 f"[原始答案文本]\n{model_answer}\n\n"
#                 f"[原始答案词元]\n{orig_answer_text}\n\n"
#                 f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
#                 f"[对齐结果]\n{match_msg}\n"
#                 "=================================================="
#             )
            
#             if not is_match:
#                 self.logger.warning(f"对齐校验不通过: {match_msg}")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 # 重新调整答案索引
#                 adjusted_answer_indices = []
#                 for token_idx, (c_start, c_end) in enumerate(full_offsets):
#                     if c_start == 0 and c_end == 0:
#                         continue
#                     if c_start >= ans_start_in_prompt:
#                         adjusted_answer_indices.append(token_idx)
                
#                 # 重新解码
#                 prompt_answer_tokens = [
#                     input_ids[0][i].item()
#                     for i in adjusted_answer_indices
#                     if i < input_ids.size(1)
#                 ]
#                 cett_answer_decoded = self.tokenizer.decode(
#                     prompt_answer_tokens,
#                     skip_special_tokens=True,
#                     clean_up_tokenization_spaces=False
#                 )
#                 self.logger.info(f"回退后答案词元: {cett_answer_decoded[:100]}...")
#             else:
#                 self.logger.info(f"✓ 答案词元成功对齐: {match_msg}")

#             # 一次性前向传播获取所有层的隐藏状态
#             self.logger.info("开始前向传播...")
#             with torch.no_grad():
#                 outputs = self.model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     output_hidden_states=True,
#                     return_dict=True
#                 )
            
#             hidden_states = outputs.hidden_states  # 所有层的隐藏状态
            
#             # 向量化提取所有层的特征（使用训练代码的方法）
#             all_answer_features = []
#             all_other_features = []
            
#             self.logger.info(f"向量化计算CETT特征，共{len(self.layers_to_extract)}层")
#             for i, layer_idx in enumerate(self.layers_to_extract):
#                 layer = self.model.model.layers[layer_idx]
                
#                 # 获取归一化后的隐藏状态
#                 normed_hs = layer.post_attention_layernorm(hidden_states[layer_idx])
                
#                 # 计算MLP激活值z
#                 mlp = layer.mlp
#                 gate = mlp.gate_proj(normed_hs)
#                 up = mlp.up_proj(normed_hs)
#                 z = torch.nn.functional.silu(gate) * up
                
#                 # 计算分母 ||h_t||_2
#                 h_total = mlp.down_proj(z)
#                 h_norm = torch.norm(h_total[0].float(), dim=-1, p=2)
#                 h_norm = torch.clamp(h_norm, min=1e-10)
                
#                 # 向量化计算CETT特征（使用训练代码的向量化函数）
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
            
#             # 拼接所有层的特征（向量化操作）
#             final_answer_features = np.concatenate(all_answer_features, axis=0)
#             final_other_features = np.concatenate(all_other_features, axis=0)
            
#             # 转换为列表格式（更高效的JSON序列化）
#             features_answer_list = final_answer_features.tolist()
#             features_other_list = final_other_features.tolist()
            
#             self.logger.info(f"特征提取成功，特征数量: {len(features_answer_list)}")
            
#             # 清理前向传播结果
#             del outputs, hidden_states
#             torch.cuda.empty_cache()
            
#             # 修复JSON序列化问题：确保所有数据类型都是Python原生类型
#             return {
#                 'features_answer_list': features_answer_list,  # 列表格式，与训练代码一致
#                 'features_other_list': features_other_list,    # 列表格式，与训练代码一致
#                 'aligned_tokens': cett_answer_decoded,
#                 'original_tokens': orig_answer_text,
#                 'answer_indices': [int(idx) for idx in adjusted_answer_indices],  # 转换为Python int
#                 'is_aligned': bool(is_match),  # 确保是Python bool
#                 'total_features': int(len(features_answer_list))  # 转换为Python int
#             }
            
#         except torch.cuda.OutOfMemoryError:
#             self.logger.error("GPU内存不足，跳过此样本")
#             gc.collect()
#             torch.cuda.empty_cache()
#             return None
#         except Exception as e:
#             self.logger.error(f"提取特征失败: {e}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#             return None

# # ==================== 4. 特征处理器（使用训练代码的逻辑） ====================
# class FeatureProcessor:
#     """特征处理器 - 确保与训练时相同的处理流程"""
    
#     def __init__(self, classifier_bundle):
#         self.classifier = classifier_bundle['classifier']
#         self.var_selector = classifier_bundle.get('var_selector')
#         self.percentile_selector = classifier_bundle.get('percentile_selector')
#         self.scaler = classifier_bundle.get('scaler')
#         self.neuron_names = classifier_bundle.get('neuron_names', [])
#         self.logger = logging.getLogger(__name__)
        
#         # 记录特征维度信息
#         self.original_dim = len(self.neuron_names)
#         if self.percentile_selector and hasattr(self.percentile_selector, 'get_support'):
#             support_mask = self.percentile_selector.get_support()
#             self.selected_dim = int(np.sum(support_mask))  # 转换为Python int
#         else:
#             self.selected_dim = int(self.original_dim)
        
#         self.logger.info(f"特征处理信息:")
#         self.logger.info(f"  原始特征维度: {self.original_dim}")
#         self.logger.info(f"  选择后维度: {self.selected_dim}")
    
#     def process_features(self, features_list: List[float]) -> np.ndarray:
#         """处理特征，与训练时完全相同的pipeline"""
#         if features_list is None:
#             return None
        
#         # 转换为numpy数组
#         features = np.array(features_list, dtype=np.float32).reshape(1, -1)
        
#         # 确保特征维度正确
#         if features.shape[1] != self.original_dim:
#             self.logger.error(f"特征维度不匹配! 预期: {self.original_dim}, 实际: {features.shape[1]}")
#             return None
        
#         # 应用与训练时相同的特征处理pipeline
#         x_proc = features.copy()
        
#         # 1. 方差选择
#         if self.var_selector is not None:
#             x_proc = self.var_selector.transform(x_proc)
        
#         # 2. 百分位数选择
#         if self.percentile_selector is not None:
#             x_proc = self.percentile_selector.transform(x_proc)
        
#         # 3. 标准化
#         if self.scaler is not None:
#             x_proc = self.scaler.transform(x_proc)
        
#         # 4. 检查并清理特征
#         if np.any(np.isnan(x_proc)) or np.any(np.isinf(x_proc)):
#             x_proc = np.nan_to_num(x_proc, nan=0.0, posinf=1e6, neginf=-1e6)
        
#         return x_proc
    
#     def predict(self, features_list: List[float]) -> Tuple[float, bool]:
#         """预测"""
#         x_proc = self.process_features(features_list)
#         if x_proc is None:
#             return 0.0, False
        
#         prob = self.classifier.predict_proba(x_proc)[0][1]
#         is_hallucination = prob > GlobalConfig.threshold
        
#         return float(prob), bool(is_hallucination)

# # ==================== 5. 模型加载器 ====================
# class ModelLoader:
#     """模型和分词器加载器"""
    
#     @staticmethod
#     def load_model_and_tokenizer(config):
#         """加载模型和分词器"""
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
        
#         return model, tokenizer

# # ==================== 6. 辅助函数：Python类型转换 ====================
# def convert_to_python_types(obj):
#     """
#     递归地将numpy/pandas类型转换为Python原生类型
#     解决JSON序列化问题
#     """
#     if isinstance(obj, dict):
#         return {k: convert_to_python_types(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_to_python_types(item) for item in obj]
#     elif isinstance(obj, tuple):
#         return tuple(convert_to_python_types(item) for item in obj)
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.bool_):
#         return bool(obj)
#     elif isinstance(obj, np.ndarray):
#         return convert_to_python_types(obj.tolist())
#     elif hasattr(obj, 'item'):  # 其他可能的numpy标量类型
#         try:
#             return obj.item()
#         except:
#             return obj
#     else:
#         return obj

# # ==================== 7. 主程序逻辑 ====================
# def main():
#     config = GlobalConfig()
    
#     # Step A: 加载分类器组件
#     logger.info(f"正在加载分类器包: {config.classifier_path}")
#     try:
#         with open(config.classifier_path, 'rb') as f:
#             bundle = pickle.load(f)
        
#         logger.info(f"分类器加载成功")
        
#         # 检查关键组件
#         required_components = ['classifier', 'scaler', 'neuron_names']
#         missing = [c for c in required_components if c not in bundle]
#         if missing:
#             logger.error(f"缺少关键组件: {missing}")
#             return
        
#         # 初始化特征处理器
#         feature_processor = FeatureProcessor(bundle)
        
#     except Exception as e:
#         logger.error(f"加载分类器失败: {e}")
#         return

#     # Step B: 加载模型
#     logger.info(f"正在加载模型: {config.model_path}")
#     try:
#         model, tokenizer = ModelLoader.load_model_and_tokenizer(config)
#     except Exception as e:
#         logger.error(f"加载模型失败: {e}")
#         return

#     # Step C: 初始化工具
#     api_extractor = AnswerExtractor(config)
#     cett_extractor = HybridCETTExtractor(model, tokenizer, config)

#     # Step D: 读取数据
#     logger.info(f"正在读取测试数据: {config.input_data_path}")
#     try:
#         with open(config.input_data_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         logger.info(f"成功加载 {len(data)} 条测试数据")
#     except Exception as e:
#         logger.error(f"加载测试数据失败: {e}")
#         return
    
#     os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
#     logger.info(f"开始逐条检测，结果将保存到: {config.output_path}")

#     # 流式写入JSON文件
#     with open(config.output_path, 'w', encoding='utf-8') as f_out:
#         f_out.write("[\n")  # JSON数组开始
        
#         processed_count = 0
#         failed_count = 0
#         alignment_failures = 0
        
#         for idx, sample in enumerate(tqdm(data, desc="处理测试样本")):
#             question = sample.get("question", "").strip()
#             source_id = sample.get("source_id", f"sample_{idx}")
#             reference_answer = sample.get("answer", "")
            
#             if not question: 
#                 logger.warning(f"样本 {idx} 问题为空，跳过")
#                 continue
            
#             try:
#                 # 1. 生成模型回答
#                 logger.info(f"\n=== 处理样本 {idx+1}/{len(data)}: {source_id} ===")
                
#                 chat_msgs = [
#                     {
#                         "role": "system", 
#                         "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
#                     },
#                     {
#                         "role": "user", 
#                         "content": f"请回答下面的问题：\n\n问题：{question}"
#                     }
#                 ]
                
#                 gen_inputs = tokenizer.apply_chat_template(
#                     chat_msgs, 
#                     tokenize=True, 
#                     add_generation_prompt=True, 
#                     return_tensors="pt"
#                 ).to(model.device)
                
#                 with torch.no_grad():
#                     gen_out = model.generate(
#                         gen_inputs, 
#                         max_new_tokens=config.max_new_tokens, 
#                         do_sample=False,
#                         temperature=0.1
#                     )
                
#                 model_answer = tokenizer.decode(
#                     gen_out[0][gen_inputs.shape[1]:], 
#                     skip_special_tokens=True
#                 ).strip()
                
#                 logger.info(f"生成的回答: {model_answer[:100]}...")

#                 # 2. API 提取答案位置
#                 logger.info("调用API提取答案片段...")
#                 api_span = api_extractor.extract_span(question, model_answer)
                
#                 if not api_span:
#                     logger.warning("API提取失败，使用启发式规则")
#                     # 使用启发式规则
#                     if reference_answer and reference_answer in model_answer:
#                         answer_text = reference_answer
#                     else:
#                         # 提取第一句话
#                         sentences = model_answer.split('。')
#                         if sentences and sentences[0]:
#                             answer_text = sentences[0] + '。'
#                         else:
#                             answer_text = model_answer[:100]
#                     logger.info(f"启发式提取的答案: {answer_text[:100]}...")
#                 else:
#                     answer_text = api_span.get("answer_text", "")
#                     logger.info(f"API提取的答案: {answer_text[:100]}...")

#                 # 3. 提取CETT特征（使用混合优化版本）
#                 logger.info("开始CETT特征提取...")
#                 features_result = cett_extractor.extract_features_for_sample(question, model_answer, answer_text)
                
#                 if features_result is None:
#                     logger.error(f"样本 {idx} 特征提取失败，跳过")
#                     failed_count += 1
#                     continue
                
#                 if not features_result.get('is_aligned', False):
#                     alignment_failures += 1
#                     logger.warning(f"对齐失败，使用完整回答作为答案词元")
                
#                 # 4. 处理特征并预测（直接使用列表格式）
#                 logger.info("处理特征并预测...")
#                 features_list = features_result['features_answer_list']
#                 prob, is_hallu = feature_processor.predict(features_list)
                
#                 logger.info(f"预测结果 - 概率: {prob:.4f}, 是否幻觉: {is_hallu}")

#                 # 5. 整合结果并确保Python原生类型
#                 res_item = convert_to_python_types({
#                     "source_id": source_id,
#                     "question": question,
#                     "reference_answer": reference_answer,
#                     "model_answer": model_answer,
#                     "extracted_part": answer_text,
#                     "hallucination_prob": round(float(prob), 4),
#                     "is_hallucination": is_hallu,
#                 })
                
#                 # 流式写入文件
#                 if processed_count > 0: 
#                     f_out.write(",\n")
#                 f_out.write(json.dumps(res_item, ensure_ascii=False, indent=2))
#                 f_out.flush()
                
#                 processed_count += 1

#                 # 定期清理内存
#                 if (idx + 1) % 20 == 0:
#                     logger.info(f"已处理 {idx+1} 个样本，清理内存...")
#                     gc.collect()
#                     torch.cuda.empty_cache()

#             except Exception as e:
#                 logger.error(f"处理样本 {idx} 失败: {e}")
#                 failed_count += 1
#                 # 写入错误结果
#                 if processed_count > 0: 
#                     f_out.write(",\n")
#                 error_item = convert_to_python_types({
#                     "source_id": source_id,
#                     "question": question,
#                     "reference_answer": reference_answer,
#                     "model_answer": "",
#                     "extracted_part": "",
#                     "hallucination_prob": -1.0,
#                     "is_hallucination": False,
#                 })
#                 f_out.write(json.dumps(error_item, ensure_ascii=False, indent=2))
#                 f_out.flush()
#                 continue
        
#         f_out.write("\n]")  # JSON数组结束
    
#     # 生成汇总报告
#     logger.info(f"\n✨ 检测完成！最终结果保存至: {config.output_path}")
    
#     # 读取结果生成统计报告
#     try:
#         with open(config.output_path, 'r', encoding='utf-8') as f:
#             results = json.load(f)
        
#         successful_results = [r for r in results if "error" not in r]
#         failed_results = [r for r in results if "error" in r]
        
#         summary_path = config.output_path.replace('.json', '_summary.json')
#         summary = convert_to_python_types({
#             "total_samples": len(data),
#             "processed_samples": len(successful_results),
#             "failed_samples": len(failed_results),
#             "alignment_failures": alignment_failures,
#             "hallucination_count": sum(1 for r in successful_results if r["is_hallucination"]),
#             "hallucination_rate": sum(1 for r in successful_results if r["is_hallucination"]) / len(successful_results) if successful_results else 0,
#             "avg_hallucination_prob": np.mean([r["hallucination_prob"] for r in successful_results]) if successful_results else 0,
#             "alignment_success_rate": sum(1 for r in successful_results if r.get("is_aligned", False)) / len(successful_results) if successful_results else 0,
#             "feature_selection_info": {
#                 "original_dimension": feature_processor.original_dim,
#                 "selected_dimension": feature_processor.selected_dim,
#                 "selection_ratio": feature_processor.selected_dim / feature_processor.original_dim if feature_processor.original_dim > 0 else 0
#             },
#             "threshold": config.threshold,
#             "model_path": config.model_path,
#             "classifier_path": config.classifier_path,
#             "processing_time": datetime.now().isoformat()
#         })
        
#         with open(summary_path, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, ensure_ascii=False, indent=2)
        
#         print(f"\n{'='*60}")
#         print("检测统计报告")
#         print(f"{'='*60}")
#         print(f"总样本数: {summary['total_samples']}")
#         print(f"成功处理: {summary['processed_samples']}")
#         print(f"处理失败: {summary['failed_samples']}")
#         print(f"对齐失败: {summary['alignment_failures']}")
#         print(f"对齐成功率: {summary['alignment_success_rate']:.2%}")
#         print(f"幻觉数量: {summary['hallucination_count']}")
#         print(f"幻觉比例: {summary['hallucination_rate']:.2%}")
#         print(f"平均概率: {summary['avg_hallucination_prob']:.4f}")
#         print(f"\n特征选择信息:")
#         print(f"  原始特征维度: {summary['feature_selection_info']['original_dimension']}")
#         print(f"  选择后维度: {summary['feature_selection_info']['selected_dimension']}")
#         print(f"  选择比例: {summary['feature_selection_info']['selection_ratio']*100:.1f}%")
#         print(f"\n详细结果: {config.output_path}")
#         print(f"统计报告: {summary_path}")
#         print(f"{'='*60}")
        
#     except Exception as e:
#         print(f"生成统计报告失败: {e}")

# if __name__ == "__main__":
#     print("="*60)
#     print("H-Neurons幻觉检测测试（混合优化版本）")
#     print("="*60)
    
#     # 运行主测试
#     main()
    
#     print("\n" + "="*60)
#     print("批量测试完成!")
#     print("="*60)


# ===========================================================直接使用推理结果=================================================
"""
H-Neurons测试推理管道 - 混合优化版本
修复JSON序列化问题
直接使用已有数据中的extracted_part字段
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

# ==================== 1. 全局配置 ====================
class GlobalConfig:
    # 路径配置
    model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    classifier_path = "/mnt/project/H-Neurous/output/h_neurons_chat_vectorized_final_20260112_152113/h_neurons_classifier.pkl" 
    input_data_path = "/mnt/project/H-Neurous/data/test/test_inference_chat.json"  # 改为已生成的数据文件
    output_path = "/mnt/project/H-Neurous/data/test/test_inference_chat_all.json"
    
    # 推理参数
    max_seq_length = 1024  # 与训练保持一致
    use_4bit = True  # 与训练保持一致
    extract_all_layers = True  # 与训练保持一致
    threshold = 0.5  # 判定阈值

# 日志监控
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/mnt/project/H-Neurous/logs/test_inference_direct.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 2. CETT特征提取器（混合优化版本） ====================
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

# ==================== 3. 特征处理器（使用训练代码的逻辑） ====================
class FeatureProcessor:
    """特征处理器 - 确保与训练时相同的处理流程"""
    
    def __init__(self, classifier_bundle):
        self.classifier = classifier_bundle['classifier']
        self.var_selector = classifier_bundle.get('var_selector')
        self.percentile_selector = classifier_bundle.get('percentile_selector')
        self.scaler = classifier_bundle.get('scaler')
        self.neuron_names = classifier_bundle.get('neuron_names', [])
        self.logger = logging.getLogger(__name__)
        
        # # 记录特征维度信息
        self.original_dim = len(self.neuron_names)
        if self.percentile_selector and hasattr(self.percentile_selector, 'get_support'):
            support_mask = self.percentile_selector.get_support()
            self.selected_dim = int(np.sum(support_mask))  # 转换为Python int
        else:
            self.selected_dim = int(self.original_dim)
        
        self.logger.info(f"特征处理信息:")
        self.logger.info(f"  原始特征维度: {self.original_dim}")
        self.logger.info(f"  选择后维度: {self.selected_dim}")
    
    def process_features(self, features_list: List[float]) -> np.ndarray:
        """处理特征，与训练时完全相同的pipeline"""
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
            x_proc = self.var_selector.transform(x_proc)
        
        # 2. 百分位数选择
        if self.percentile_selector is not None:
            x_proc = self.percentile_selector.transform(x_proc)
        
        # 3. 标准化
        if self.scaler is not None:
            x_proc = self.scaler.transform(x_proc)
        
        # 4. 检查并清理特征
        if np.any(np.isnan(x_proc)) or np.any(np.isinf(x_proc)):
            x_proc = np.nan_to_num(x_proc, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return x_proc
    
    def predict(self, features_list: List[float]) -> Tuple[float, bool]:
        """预测"""
        x_proc = self.process_features(features_list)
        if x_proc is None:
            return 0.0, False
        
        prob = self.classifier.predict_proba(x_proc)[0][1]
        is_hallucination = prob > GlobalConfig.threshold
        
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
    
    # Step A: 加载分类器组件
    logger.info(f"正在加载分类器包: {config.classifier_path}")
    try:
        with open(config.classifier_path, 'rb') as f:
            bundle = pickle.load(f)
        
        logger.info(f"分类器加载成功")
        
        # 检查关键组件
        required_components = ['classifier', 'scaler', 'neuron_names']
        missing = [c for c in required_components if c not in bundle]
        if missing:
            logger.error(f"缺少关键组件: {missing}")
            return
        
        # 初始化特征处理器
        feature_processor = FeatureProcessor(bundle)
        
    except Exception as e:
        logger.error(f"加载分类器失败: {e}")
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
                
                # 处理特征并预测（直接使用列表格式）
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
            "feature_selection_info": {
                "original_dimension": feature_processor.original_dim,
                "selected_dimension": feature_processor.selected_dim,
                "selection_ratio": feature_processor.selected_dim / feature_processor.original_dim if feature_processor.original_dim > 0 else 0
            },
            "threshold": config.threshold,
            "model_path": config.model_path,
            "classifier_path": config.classifier_path,
            "processing_time": datetime.now().isoformat()
        })
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("检测统计报告")
        print(f"{'='*60}")
        print(f"总样本数: {summary['total_samples']}")
        print(f"成功处理: {summary['processed_samples']}")
        print(f"处理失败: {summary['failed_samples']}")
        print(f"对齐失败: {summary['alignment_failures']}")
        print(f"对齐成功率: {summary['alignment_success_rate']:.2%}")
        print(f"幻觉数量: {summary['hallucination_count']}")
        print(f"幻觉比例: {summary['hallucination_rate']:.2%}")
        print(f"平均概率: {summary['avg_hallucination_prob']:.4f}")
        print(f"\n特征选择信息:")
        print(f"  原始特征维度: {summary['feature_selection_info']['original_dimension']}")
        print(f"  选择后维度: {summary['feature_selection_info']['selected_dimension']}")
        print(f"  选择比例: {summary['feature_selection_info']['selection_ratio']*100:.1f}%")
        print(f"\n详细结果: {config.output_path}")
        print(f"统计报告: {summary_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"生成统计报告失败: {e}")

if __name__ == "__main__":
    print("="*60)
    print("H-Neurons幻觉检测测试（直接使用已有数据）")
    print("="*60)
    
    # 运行主测试
    main()
    
    print("\n" + "="*60)
    print("批量测试完成!")
    print("="*60)

# =========================================直接使用推理结果-二次训练============================================
# """
# H-Neurons测试推理管道 - 混合优化版本
# 修复JSON序列化问题
# 直接使用已有数据中的extracted_part字段
# 修改：仅移除特征选择步骤，保持其他所有内容不变
# """

# import os
# import json
# import pickle
# import torch
# import logging
# import time
# import re
# import numpy as np
# from datetime import datetime
# from typing import List, Dict, Tuple, Optional
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import bitsandbytes as bnb
# from tqdm import tqdm
# import gc

# # ==================== 1. 全局配置 ====================
# class GlobalConfig:
#     # 路径配置
#     model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
#     classifier_path = "/mnt/project/H-Neurous/output/h_neurons_second_training_20260113_155408/h_neurons_classifier.pkl" 
#     input_data_path = "/mnt/project/H-Neurous/data/test/test_inference_chat.json"  # 改为已生成的数据文件
#     output_path = "/mnt/project/H-Neurous/data/test/test_inference_chat_second.json"
    
#     # 推理参数
#     max_seq_length = 1024  # 与训练保持一致
#     use_4bit = True  # 与训练保持一致
#     extract_all_layers = True  # 与训练保持一致
#     threshold = 0.5  # 判定阈值

# # 日志监控
# logging.basicConfig(
#     level=logging.WARNING,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("/mnt/project/H-Neurous/logs/test_inference_direct.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # ==================== 2. CETT特征提取器（混合优化版本） ====================
# class HybridCETTExtractor:
#     """
#     CETT特征提取器 - 混合优化版本
#     保持字符映射对齐逻辑，但使用训练代码的向量化计算和列表格式
#     """
    
#     def __init__(self, model, tokenizer, config):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # 获取模型信息
#         self.num_layers = len(model.model.layers)
        
#         # 确定要提取的层 - 按照论文，提取所有层
#         self.layers_to_extract = list(range(self.num_layers))
        
#         self.logger.info(f"模型总层数: {self.num_layers}")
#         self.logger.info(f"提取特征层数: {len(self.layers_to_extract)}")
        
#         # 预计算所有层的权重范数（与训练代码一致，用于向量化计算）
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

#     def _validate_with_unicode_tolerance(self, orig_text, cett_text, max_tolerate=3):
#         """终极乱码容忍校验逻辑（保持不变）"""
#         # 1. 清洗掉所有空白符（空格、换行等）
#         s1 = "".join(orig_text.split())
#         s2 = "".join(cett_text.split())

#         # 2. 长度校验：如果去掉空格后长度都不一样，说明 Token 映射逻辑肯定错了
#         if len(s1) != len(s2):
#             return False, f"长度不一 (预期{len(s1)} vs 实际{len(s2)})"

#         # 3. 逐字比对
#         mismatches = 0
#         for c1, c2 in zip(s1, s2):
#             if c1 != c2:
#                 # 如果不相等，检查 c2 是不是乱码符号 \ufffd ()
#                 if c2 == '' or c2 == '\ufffd':
#                     mismatches += 1
#                 else:
#                     # 发现真正的字符不匹配（不是乱码导致的），直接失败
#                     return False, f"字符冲突 ('{c1}' vs '{c2}')"

#         # 4. 容忍度校验：如果乱码位太多，可能也不是我们要的
#         if mismatches > max_tolerate:
#             return False, f"乱码过多 ({mismatches}处)"

#         return True, "对齐成功"
    
#     def _compute_cett_vectorized(self, z: torch.Tensor, layer_idx: int, 
#                                h_norm: torch.Tensor, answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         向量化计算CETT特征 - 完全使用训练代码的实现
#         """
#         # 获取预计算的权重范数
#         w_norm = self.precomputed_w_norms[layer_idx].to(z.device).to(z.dtype)
        
#         # 取batch中的第一个样本
#         z_abs = torch.abs(z[0])  # [seq_len, num_neurons]
        
#         # 计算分母：||h_t||_2 已经计算好
#         # 计算分子：|z_j,t| * ||W_down[:, j]||_2
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
        
#         # 转换为numpy并清理GPU内存
#         answer_np = answer_cett.detach().cpu().numpy().astype(np.float32)
#         other_np = other_cett.detach().cpu().numpy().astype(np.float32)
        
#         # 清理
#         del z_abs, scaled_z, answer_cett, other_cett
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
        
#         return answer_np, other_np

#     def extract_features_for_sample(self, question: str, model_answer: str, answer_text: str) -> Optional[Dict]:
#         """
#         为单个样本提取CETT特征 - 混合版本
#         保持字符映射对齐逻辑，但使用向量化计算和列表格式
#         """
#         try:
#             self.logger.info(f"开始提取特征 - 问题: {question[:50]}...")
            
#             # 步骤1: 先单独tokenize model_answer，获取精确的答案词元
#             if not answer_text or answer_text not in model_answer:
#                 self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 answer_text = model_answer
            
#             # 找到答案在model_answer中的精确字符位置
#             start_char = model_answer.find(answer_text)
#             end_char = start_char + len(answer_text)
            
#             if start_char == -1:
#                 self.logger.warning(f"无法找到答案文本: {answer_text[:50]}...")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 answer_text = model_answer
#                 start_char = 0
#                 end_char = len(model_answer)
            
#             # 对model_answer进行tokenization，获取offset mapping
#             self.logger.info(f"单独tokenize model_answer，长度: {len(model_answer)}")
#             model_answer_enc = self.tokenizer(
#                 model_answer,
#                 return_offsets_mapping=True,
#                 add_special_tokens=False,
#                 return_tensors='pt'
#             )
#             model_answer_offsets = model_answer_enc['offset_mapping'][0].cpu().numpy()
            
#             # 基于字符位置找到model_answer中的答案token索引
#             answer_token_indices_in_model = []
#             for token_idx, (char_start, char_end) in enumerate(model_answer_offsets):
#                 # token完全在答案区域内，或者部分重叠但主要部分在答案内
#                 if (char_start >= start_char and char_end <= end_char) or \
#                 (char_start <= start_char and char_end > start_char + (end_char - start_char) / 2):
#                     answer_token_indices_in_model.append(token_idx)
            
#             if not answer_token_indices_in_model:
#                 self.logger.warning(f"无法找到model_answer中的答案token: {answer_text[:50]}...")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 # 回退：使用所有token作为答案词元
#                 answer_token_indices_in_model = list(range(len(model_answer_offsets)))
            
#             # 获取原始答案词元文本
#             seq_len = model_answer_enc['input_ids'].size(1)
#             model_answer_answer_tokens = [
#                 model_answer_enc['input_ids'][0][i].item() 
#                 for i in answer_token_indices_in_model
#                 if i < seq_len
#             ]
#             orig_answer_text = self.tokenizer.decode(
#                 model_answer_answer_tokens,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False
#             )
            
#             # 步骤1: 构建与训练代码完全相同的Chat消息格式
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
            
#             # 步骤2: 应用Chat模板，与训练代码完全一致
#             full_prompt = self.tokenizer.apply_chat_template(
#                 messages, 
#                 tokenize=False, 
#                 add_generation_prompt=False
#             )
            
#             # 步骤3: 对完整文本进行分词，获取offset mapping
#             self.logger.info(f"使用Chat模板构建prompt，长度: {len(full_prompt)}")
#             inputs = self.tokenizer(
#                 full_prompt,
#                 return_tensors="pt",
#                 max_length=self.config.max_seq_length,
#                 truncation=True,
#                 padding="max_length",
#                 return_offsets_mapping=True
#             )
            
#             input_ids = inputs['input_ids'].to(self.model.device)
#             attention_mask = inputs['attention_mask'].to(self.model.device)
#             full_offsets = inputs['offset_mapping'][0].cpu().numpy()
            
#             # 步骤4: 精确对齐 - 找到答案文本在完整prompt中的位置
#             # 在完整prompt中找到model_answer的起始位置
#             ans_start_in_prompt = full_prompt.rfind(model_answer)
#             if ans_start_in_prompt == -1:
#                 self.logger.warning("无法在完整prompt中找到model_answer，使用启发式定位")
#                 # 回退：尝试查找"assistant"内容
#                 ans_start_in_prompt = full_prompt.find("assistant") + len("assistant") + 5
#                 if ans_start_in_prompt <= len("assistant") + 5:
#                     ans_start_in_prompt = len(full_prompt) - len(model_answer)
            
#             # 找到答案文本在model_answer中的相对位置
#             rel_start_char = model_answer.find(answer_text)
#             if rel_start_char == -1:
#                 self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
#                 # 回退：使用完整回答作为答案文本
#                 answer_text = model_answer
#                 rel_start_char = 0
            
#             # 计算答案文本在完整prompt中的绝对位置
#             abs_start = ans_start_in_prompt + rel_start_char
#             abs_end = abs_start + len(answer_text)
            
#             # 步骤5: 基于字符偏移映射找到对应的token索引
#             adjusted_answer_indices = []
#             for token_idx, (c_start, c_end) in enumerate(full_offsets):
#                 # 跳过padding或特殊token
#                 if c_start == 0 and c_end == 0:
#                     continue
                
#                 # 检查token是否与答案文本重叠
#                 if not (c_end <= abs_start or c_start >= abs_end):
#                     adjusted_answer_indices.append(token_idx)
            
#             if not adjusted_answer_indices:
#                 self.logger.warning("未找到匹配的答案词元索引，回退到完整回答")
#                 # 回退：将整个回答作为答案词元
#                 for token_idx, (c_start, c_end) in enumerate(full_offsets):
#                     if c_start == 0 and c_end == 0:
#                         continue
#                     if c_start >= ans_start_in_prompt:
#                         adjusted_answer_indices.append(token_idx)
            
#             # 获取CETT对齐后的答案词元
#             prompt_answer_tokens = [
#                 input_ids[0][i].item()
#                 for i in adjusted_answer_indices
#                 if i < input_ids.size(1)
#             ]
#             cett_answer_decoded = self.tokenizer.decode(
#                 prompt_answer_tokens,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False
#             )
            
#             # 验证对齐结果
#             orig_answer_text = answer_text
#             is_match, match_msg = self._validate_with_unicode_tolerance(
#                 orig_answer_text, cett_answer_decoded
#             )
            
#             self.logger.warning(
#                 "\n"
#                 "================ CETT FULL DEBUG =================\n"
#                 f"[问题]\n{question}\n\n"
#                 f"[原始答案文本]\n{model_answer}\n\n"
#                 f"[原始答案词元]\n{orig_answer_text}\n\n"
#                 f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
#                 f"[对齐结果]\n{match_msg}\n"
#                 "=================================================="
#             )
            
#             if not is_match:
#                 self.logger.warning(f"对齐校验不通过: {match_msg}")
#                 self.logger.info("回退: 使用完整回答作为答案词元")
#                 # 重新调整答案索引
#                 adjusted_answer_indices = []
#                 for token_idx, (c_start, c_end) in enumerate(full_offsets):
#                     if c_start == 0 and c_end == 0:
#                         continue
#                     if c_start >= ans_start_in_prompt:
#                         adjusted_answer_indices.append(token_idx)
                
#                 # 重新解码
#                 prompt_answer_tokens = [
#                     input_ids[0][i].item()
#                     for i in adjusted_answer_indices
#                     if i < input_ids.size(1)
#                 ]
#                 cett_answer_decoded = self.tokenizer.decode(
#                     prompt_answer_tokens,
#                     skip_special_tokens=True,
#                     clean_up_tokenization_spaces=False
#                 )
#                 self.logger.info(f"回退后答案词元: {cett_answer_decoded[:100]}...")
#             else:
#                 self.logger.info(f"✓ 答案词元成功对齐: {match_msg}")

#             # 一次性前向传播获取所有层的隐藏状态
#             self.logger.info("开始前向传播...")
#             with torch.no_grad():
#                 outputs = self.model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     output_hidden_states=True,
#                     return_dict=True
#                 )
            
#             hidden_states = outputs.hidden_states  # 所有层的隐藏状态
            
#             # 向量化提取所有层的特征（使用训练代码的方法）
#             all_answer_features = []
#             all_other_features = []
            
#             self.logger.info(f"向量化计算CETT特征，共{len(self.layers_to_extract)}层")
#             for i, layer_idx in enumerate(self.layers_to_extract):
#                 layer = self.model.model.layers[layer_idx]
                
#                 # 获取归一化后的隐藏状态
#                 normed_hs = layer.post_attention_layernorm(hidden_states[layer_idx])
                
#                 # 计算MLP激活值z
#                 mlp = layer.mlp
#                 gate = mlp.gate_proj(normed_hs)
#                 up = mlp.up_proj(normed_hs)
#                 z = torch.nn.functional.silu(gate) * up
                
#                 # 计算分母 ||h_t||_2
#                 h_total = mlp.down_proj(z)
#                 h_norm = torch.norm(h_total[0].float(), dim=-1, p=2)
#                 h_norm = torch.clamp(h_norm, min=1e-10)
                
#                 # 向量化计算CETT特征（使用训练代码的向量化函数）
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
            
#             # 拼接所有层的特征（向量化操作）
#             final_answer_features = np.concatenate(all_answer_features, axis=0)
#             final_other_features = np.concatenate(all_other_features, axis=0)
            
#             # 转换为列表格式（更高效的JSON序列化）
#             features_answer_list = final_answer_features.tolist()
#             features_other_list = final_other_features.tolist()
            
#             self.logger.info(f"特征提取成功，特征数量: {len(features_answer_list)}")
            
#             # 清理前向传播结果
#             del outputs, hidden_states
#             torch.cuda.empty_cache()
            
#             # 修复JSON序列化问题：确保所有数据类型都是Python原生类型
#             return {
#                 'features_answer_list': features_answer_list,  # 列表格式，与训练代码一致
#                 'features_other_list': features_other_list,    # 列表格式，与训练代码一致
#                 'aligned_tokens': cett_answer_decoded,
#                 'original_tokens': orig_answer_text,
#                 'answer_indices': [int(idx) for idx in adjusted_answer_indices],  # 转换为Python int
#                 'is_aligned': bool(is_match),  # 确保是Python bool
#                 'total_features': int(len(features_answer_list))  # 转换为Python int
#             }
            
#         except torch.cuda.OutOfMemoryError:
#             self.logger.error("GPU内存不足，跳过此样本")
#             gc.collect()
#             torch.cuda.empty_cache()
#             return None
#         except Exception as e:
#             self.logger.error(f"提取特征失败: {e}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#             return None

# # ==================== 3. 特征处理器（修改：仅移除特征选择步骤） ====================
# # ==================== 3. 特征处理器（修改：使用selected_indices筛选1050个神经元） ====================
# class FeatureProcessor:
#     """特征处理器 - 确保与训练时相同的处理流程"""
    
#     def __init__(self, classifier_bundle):
#         self.classifier = classifier_bundle['classifier']
#         self.scaler = classifier_bundle.get('scaler')
#         self.selected_indices = classifier_bundle.get('selected_indices', [])  # 1050个神经元的索引
#         self.logger = logging.getLogger(__name__)
        
#         if not self.selected_indices:
#             self.logger.error("未找到selected_indices，无法进行特征筛选")
#             raise ValueError("分类器包中缺少selected_indices")
        
#         self.expected_dim = len(self.selected_indices)
#         self.logger.info(f"特征处理信息:")
#         self.logger.info(f"  将提取 {self.expected_dim} 个神经元的特征")
#         self.logger.info(f"  跳过特征选择步骤")
        
#         # 检查标准化器
#         if self.scaler is not None:
#             if hasattr(self.scaler, 'mean_'):
#                 scaler_dim = self.scaler.mean_.shape[0]
#                 if scaler_dim != self.expected_dim:
#                     self.logger.warning(f"标准化器维度({scaler_dim})与期望维度({self.expected_dim})不匹配!")
#             else:
#                 self.logger.warning("StandardScaler未拟合，将跳过标准化步骤")
#                 self.scaler = None
    
#     def filter_features(self, features_list: List[float]) -> List[float]:
#         """筛选特征，只保留选中的1050个神经元"""
#         if not self.selected_indices:
#             return features_list
        
#         # 确保索引在范围内
#         selected_features = []
        
#         for idx in self.selected_indices:
#             if idx < len(features_list):
#                 selected_features.append(features_list[idx])
#             else:
#                 self.logger.warning(f"神经元索引 {idx} 超出特征范围 (0-{len(features_list)-1})")
#                 selected_features.append(0.0)  # 超出范围则填充0
        
#         self.logger.info(f"特征筛选完成: 从 {len(features_list)} 个特征中筛选出 {len(selected_features)} 个特征")
        
#         return selected_features
    
#     def process_features(self, features_list: List[float]) -> np.ndarray:
#         """处理特征，先筛选，然后标准化"""
#         if features_list is None:
#             return None
        
#         # 第一步：筛选特征（只保留选中的1050个神经元）
#         filtered_features = self.filter_features(features_list)
        
#         # 转换为numpy数组
#         features = np.array(filtered_features, dtype=np.float32).reshape(1, -1)
        
#         # 检查维度
#         actual_dim = features.shape[1]
#         if actual_dim != self.expected_dim:
#             self.logger.error(f"特征维度不匹配! 期望: {self.expected_dim}, 实际: {actual_dim}")
#             return None
        
#         # 应用标准化
#         x_proc = features.copy()
        
#         # 标准化（如果标准化器可用且已拟合）
#         if self.scaler is not None:
#             try:
#                 x_proc = self.scaler.transform(x_proc)
#                 self.logger.debug(f"特征标准化完成，形状: {x_proc.shape}")
#             except Exception as e:
#                 self.logger.warning(f"标准化失败: {e}")
#                 # 如果标准化失败，我们仍然可以使用原始特征
        
#         # 检查并清理特征
#         if np.any(np.isnan(x_proc)) or np.any(np.isinf(x_proc)):
#             self.logger.warning("特征包含NaN或Inf，进行清理")
#             x_proc = np.nan_to_num(x_proc, nan=0.0, posinf=1e6, neginf=-1e6)
        
#         return x_proc
    
#     def predict(self, features_list: List[float]) -> Tuple[float, bool]:
#         """预测"""
#         x_proc = self.process_features(features_list)
#         if x_proc is None:
#             return 0.0, False
        
#         # 检查特征维度是否与分类器匹配
#         expected_features = self.classifier.n_features_in_
#         actual_features = x_proc.shape[1]
        
#         if expected_features != actual_features:
#             self.logger.error(f"特征维度不匹配! 分类器期望: {expected_features}, 实际: {actual_features}")
#             return 0.0, False
        
#         try:
#             prob = self.classifier.predict_proba(x_proc)[0][1]
#             is_hallucination = prob > GlobalConfig.threshold
            
#             self.logger.info(f"预测成功 - 概率: {prob:.4f}, 是否幻觉: {is_hallucination}")
            
#             return float(prob), bool(is_hallucination)
#         except Exception as e:
#             self.logger.error(f"预测失败: {e}")
#             return 0.0, False

# # ==================== 4. 模型加载器 ====================
# class ModelLoader:
#     """模型和分词器加载器"""
    
#     @staticmethod
#     def load_model_and_tokenizer(config):
#         """加载模型和分词器"""
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
        
#         return model, tokenizer

# # ==================== 5. 辅助函数：Python类型转换 ====================
# def convert_to_python_types(obj):
#     """
#     递归地将numpy/pandas类型转换为Python原生类型
#     解决JSON序列化问题
#     """
#     if isinstance(obj, dict):
#         return {k: convert_to_python_types(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_to_python_types(item) for item in obj]
#     elif isinstance(obj, tuple):
#         return tuple(convert_to_python_types(item) for item in obj)
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.bool_):
#         return bool(obj)
#     elif isinstance(obj, np.ndarray):
#         return convert_to_python_types(obj.tolist())
#     elif hasattr(obj, 'item'):  # 其他可能的numpy标量类型
#         try:
#             return obj.item()
#         except:
#             return obj
#     else:
#         return obj

# # ==================== 6. 主程序逻辑 ====================
# # ==================== 6. 主程序逻辑 ====================
# def main():
#     config = GlobalConfig()
    
#     # Step A: 加载分类器组件
#     logger.info(f"正在加载分类器包: {config.classifier_path}")
#     try:
#         with open(config.classifier_path, 'rb') as f:
#             bundle = pickle.load(f)
        
#         logger.info(f"分类器加载成功")
        
#         # 检查关键组件 - 对于第二次训练，需要classifier、scaler和selected_indices
#         required_components = ['classifier', 'selected_indices']
#         missing = [c for c in required_components if c not in bundle]
#         if missing:
#             logger.error(f"缺少关键组件: {missing}")
            
#             # 尝试查看bundle中有哪些键
#             logger.info(f"bundle中的键: {list(bundle.keys())}")
            
#             # 如果缺少selected_indices，尝试使用h_neuron_indices
#             if 'selected_indices' not in bundle and 'h_neuron_indices' in bundle:
#                 logger.info("使用h_neuron_indices作为selected_indices")
#                 bundle['selected_indices'] = bundle['h_neuron_indices']
#             else:
#                 return
        
#         selected_indices = bundle['selected_indices']
#         logger.info(f"发现 {len(selected_indices)} 个选中的神经元（第二次训练的输入）")
        
#         # 检查分类器期望的维度
#         classifier = bundle['classifier']
#         if hasattr(classifier, 'n_features_in_'):
#             logger.info(f"分类器期望的特征维度: {classifier.n_features_in_}")
        
#         # 初始化特征处理器
#         feature_processor = FeatureProcessor(bundle)
        
#     except Exception as e:
#         logger.error(f"加载分类器失败: {e}")
#         return

#     # Step B: 加载模型
#     logger.info(f"正在加载模型: {config.model_path}")
#     try:
#         model, tokenizer = ModelLoader.load_model_and_tokenizer(config)
#     except Exception as e:
#         logger.error(f"加载模型失败: {e}")
#         return

#     # Step C: 初始化CETT特征提取器
#     cett_extractor = HybridCETTExtractor(model, tokenizer, config)

#     # Step D: 读取数据
#     logger.info(f"正在读取测试数据: {config.input_data_path}")
#     try:
#         with open(config.input_data_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         logger.info(f"成功加载 {len(data)} 条测试数据")
#     except Exception as e:
#         logger.error(f"加载测试数据失败: {e}")
#         return
    
#     os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
#     logger.info(f"开始逐条检测，结果将保存到: {config.output_path}")

#     # 流式写入JSON文件
#     with open(config.output_path, 'w', encoding='utf-8') as f_out:
#         f_out.write("[\n")  # JSON数组开始
        
#         processed_count = 0
#         failed_count = 0
#         alignment_failures = 0
        
#         for idx, sample in enumerate(tqdm(data, desc="处理测试样本")):
#             question = sample.get("question", "").strip()
#             source_id = sample.get("source_id", f"sample_{idx}")
#             reference_answer = sample.get("reference_answer", "")
#             model_answer = sample.get("model_answer", "")
#             extracted_part = sample.get("extracted_part", "")
            
#             if not question or not model_answer: 
#                 logger.warning(f"样本 {idx} 问题或回答为空，跳过")
#                 continue
            
#             try:
#                 logger.info(f"\n=== 处理样本 {idx+1}/{len(data)}: {source_id} ===")
                
#                 # 使用已有的extracted_part字段
#                 answer_text = extracted_part
#                 if not answer_text or answer_text not in model_answer:
#                     logger.warning(f"extracted_part不在model_answer中，尝试使用完整回答: {answer_text[:50]}...")
#                     # 回退：使用完整回答作为答案词元
#                     answer_text = model_answer
                
#                 logger.info(f"模型回答: {model_answer[:100]}...")
#                 logger.info(f"提取部分: {answer_text[:100]}...")

#                 # 提取CETT特征（使用混合优化版本）
#                 logger.info("开始CETT特征提取...")
#                 features_result = cett_extractor.extract_features_for_sample(question, model_answer, answer_text)
                
#                 if features_result is None:
#                     logger.error(f"样本 {idx} 特征提取失败，跳过")
#                     failed_count += 1
#                     continue
                
#                 if not features_result.get('is_aligned', False):
#                     alignment_failures += 1
#                     logger.warning(f"对齐失败，使用完整回答作为答案词元")
                
#                 # 处理特征并预测（直接使用列表格式）
#                 logger.info("处理特征并预测...")
#                 features_list = features_result['features_answer_list']
#                 prob, is_hallu = feature_processor.predict(features_list)
                
#                 logger.info(f"预测结果 - 概率: {prob:.4f}, 是否幻觉: {is_hallu}")

#                 # 整合结果并确保Python原生类型
#                 res_item = convert_to_python_types({
#                     "source_id": source_id,
#                     "question": question,
#                     "reference_answer": reference_answer,
#                     "model_answer": model_answer,
#                     "extracted_part": answer_text,
#                     "hallucination_prob": round(float(prob), 4),
#                     "is_hallucination": is_hallu
#                 })
                
#                 # 流式写入文件
#                 if processed_count > 0: 
#                     f_out.write(",\n")
#                 f_out.write(json.dumps(res_item, ensure_ascii=False, indent=2))
#                 f_out.flush()
                
#                 processed_count += 1

#                 # 定期清理内存
#                 if (idx + 1) % 20 == 0:
#                     logger.info(f"已处理 {idx+1} 个样本，清理内存...")
#                     gc.collect()
#                     torch.cuda.empty_cache()

#             except Exception as e:
#                 logger.error(f"处理样本 {idx} 失败: {e}")
#                 failed_count += 1
#                 # 写入错误结果
#                 if processed_count > 0: 
#                     f_out.write(",\n")
#                 error_item = convert_to_python_types({
#                     "source_id": source_id,
#                     "question": question,
#                     "reference_answer": reference_answer,
#                     "model_answer": model_answer,
#                     "extracted_part": extracted_part,
#                     "hallucination_prob": -1.0,
#                     "is_hallucination": False,
#                     "error": str(e)
#                 })
#                 f_out.write(json.dumps(error_item, ensure_ascii=False, indent=2))
#                 f_out.flush()
#                 continue
        
#         f_out.write("\n]")  # JSON数组结束
    
#     # 生成汇总报告
#     logger.info(f"\n✨ 检测完成！最终结果保存至: {config.output_path}")
    
#     # 读取结果生成统计报告
#     try:
#         with open(config.output_path, 'r', encoding='utf-8') as f:
#             results = json.load(f)
        
#         successful_results = [r for r in results if "error" not in r]
#         failed_results = [r for r in results if "error" in r]
        
#         summary_path = config.output_path.replace('.json', '_summary.json')
#         summary = convert_to_python_types({
#             "total_samples": len(data),
#             "processed_samples": len(successful_results),
#             "failed_samples": len(failed_results),
#             "alignment_failures": alignment_failures,
#             "hallucination_count": sum(1 for r in successful_results if r["is_hallucination"]),
#             "hallucination_rate": sum(1 for r in successful_results if r["is_hallucination"]) / len(successful_results) if successful_results else 0,
#             "avg_hallucination_prob": np.mean([r["hallucination_prob"] for r in successful_results]) if successful_results else 0,
#             "alignment_success_rate": (len(successful_results) - alignment_failures) / len(successful_results) if successful_results else 0,
#             "feature_selection_info": {
#                 "original_dimension": 530432,  # Qwen2.5-7B的全维度神经元数
#                 "selected_neurons": len(selected_indices) if 'selected_indices' in bundle else 0,
#                 "feature_selection_applied": True,
#                 "standardization_applied": feature_processor.scaler is not None
#             },
#             "threshold": config.threshold,
#             "model_path": config.model_path,
#             "classifier_path": config.classifier_path,
#             "processing_time": datetime.now().isoformat()
#         })
        
#         with open(summary_path, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, ensure_ascii=False, indent=2)
        
#         print(f"\n{'='*60}")
#         print("检测统计报告")
#         print(f"{'='*60}")
#         print(f"总样本数: {summary['total_samples']}")
#         print(f"成功处理: {summary['processed_samples']}")
#         print(f"处理失败: {summary['failed_samples']}")
#         print(f"对齐失败: {summary['alignment_failures']}")
#         print(f"对齐成功率: {summary['alignment_success_rate']:.2%}")
#         print(f"幻觉数量: {summary['hallucination_count']}")
#         print(f"幻觉比例: {summary['hallucination_rate']:.2%}")
#         print(f"平均概率: {summary['avg_hallucination_prob']:.4f}")
#         print(f"\n特征处理信息:")
#         print(f"  原始特征维度: {summary['feature_selection_info']['original_dimension']}")
#         print(f"  选中的神经元数: {summary['feature_selection_info']['selected_neurons']}")
#         print(f"  特征选择应用: {summary['feature_selection_info']['feature_selection_applied']}")
#         print(f"  标准化应用: {summary['feature_selection_info']['standardization_applied']}")
#         print(f"\n详细结果: {config.output_path}")
#         print(f"统计报告: {summary_path}")
#         print(f"{'='*60}")
        
#     except Exception as e:
#         print(f"生成统计报告失败: {e}")

# if __name__ == "__main__":
#     print("="*60)
#     print("H-Neurons幻觉检测测试（直接使用已有数据）")
#     print("修改：仅移除特征选择步骤，其他保持不变")
#     print("="*60)
    
#     # 运行主测试
#     main()
    
#     print("\n" + "="*60)
#     print("批量测试完成!")
#     print("="*60)
# # ================================================本地模型提取答案词元===================================
# # """
# # H-Neurons测试推理管道 - 混合优化版本 + 本地答案提取
# # 修复JSON序列化问题，使用两个独立本地模型
# # """

# # import os
# # import json
# # import pickle
# # import torch
# # import logging
# # import time
# # import re
# # import numpy as np
# # from datetime import datetime
# # from typing import List, Dict, Tuple, Optional
# # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# # import bitsandbytes as bnb
# # from tqdm import tqdm
# # from tenacity import retry, stop_after_attempt, wait_exponential
# # import gc

# # # ==================== 1. 全局配置 ====================
# # class GlobalConfig:
# #     # 路径配置
# #     model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
# #     classifier_path = "/mnt/project/H-Neurous/output/train_trivia/h_neurons_classifier.pkl" 
# #     input_data_path = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train_data.json"
# #     output_path = "/mnt/project/H-Neurous/data/test/train_inference.json"
    
# #     # 本地答案提取配置（独立模型）
# #     extract_model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
# #     extract_max_new_tokens = 128  # 增加token数以获取更完整输出
    
# #     # 推理参数
# #     max_seq_length = 1024  # 与训练保持一致
# #     max_new_tokens = 1024
# #     threshold = 0.5  # 判定阈值
# #     use_4bit = True  # 与训练保持一致
# #     extract_all_layers = True  # 与训练保持一致

# # # 日志监控
# # logging.basicConfig(
# #     level=logging.WARNING,
# #     format='%(asctime)s - %(levelname)s - %(message)s',
# #     handlers=[
# #         logging.FileHandler("/mnt/project/H-Neurous/logs/test_inference_local.log"),
# #         logging.StreamHandler()
# #     ]
# # )
# # logger = logging.getLogger(__name__)

# # # ==================== 2. 本地答案提取器（独立模型） ====================
# # class LocalAnswerExtractor:
# #     def __init__(self, config):
# #         self.config = config
# #         self.extract_model = None
# #         self.extract_tokenizer = None
# #         self.device = None
# #         self.logger = logging.getLogger(__name__)
# #         self._model_loaded = False
        
# #     def load_extraction_model(self):
# #         """加载独立的本地模型用于答案提取"""
# #         if self._model_loaded:
# #             return True
            
# #         self.logger.info("加载独立的本地答案提取模型...")
        
# #         try:
# #             bnb_config = None
# #             if self.config.use_4bit:
# #                 bnb_config = BitsAndBytesConfig(
# #                     load_in_4bit=True,
# #                     bnb_4bit_compute_dtype=torch.float16,
# #                     bnb_4bit_use_double_quant=True,
# #                     bnb_4bit_quant_type="nf4"
# #                 )
            
# #             # 加载分词器
# #             self.extract_tokenizer = AutoTokenizer.from_pretrained(
# #                 self.config.extract_model_path,
# #                 trust_remote_code=True,
# #                 use_fast=True
# #             )
            
# #             if self.extract_tokenizer.pad_token is None:
# #                 self.extract_tokenizer.pad_token = self.extract_tokenizer.eos_token
            
# #             # 加载模型
# #             self.extract_model = AutoModelForCausalLM.from_pretrained(
# #                 self.config.extract_model_path,
# #                 quantization_config=bnb_config,
# #                 device_map="auto",
# #                 torch_dtype=torch.float16,
# #                 trust_remote_code=True
# #             )
            
# #             self.extract_model.eval()
# #             self.device = self.extract_model.device
# #             self._model_loaded = True
# #             self.logger.info(f"独立答案提取模型加载完成，设备: {self.device}")
# #             return True
            
# #         except Exception as e:
# #             self.logger.error(f"加载独立提取模型失败: {e}")
# #             return False
    
# #     def build_extraction_prompt(self, question: str, model_answer: str) -> str:
# #         """构建答案提取的提示词"""
# #         prompt = f"""请从以下回答中提取直接回答问题的部分：

# # 问题：{question}

# # 回答：{model_answer}

# # 请严格按照以下JSON格式输出，不要有任何额外内容：
# # {{
# #   "answer_text": "提取的答案文本",
# #   "start_char": 0,
# #   "end_char": 0
# # }}

# # 注意：answer_text必须是回答中的原文，不能修改。"""
# #         return prompt
    
# #     def parse_extraction_output(self, output: str) -> Optional[Dict]:
# #         """解析模型提取的输出"""
# #         try:
# #             # 清理输出
# #             output = output.strip()
# #             self.logger.debug(f"解析模型输出: {output[:200]}...")
            
# #             # 尝试直接解析整个输出为JSON
# #             try:
# #                 result = json.loads(output)
# #                 if isinstance(result, dict) and "answer_text" in result:
# #                     return result
# #             except json.JSONDecodeError:
# #                 pass
            
# #             # 查找JSON对象
# #             json_pattern = r'\{[^{}]*"answer_text"[^{}]*\}'
# #             matches = re.findall(json_pattern, output, re.DOTALL)
# #             if matches:
# #                 try:
# #                     result = json.loads(matches[0])
# #                     if "answer_text" in result:
# #                         return result
# #                 except json.JSONDecodeError:
# #                     pass
            
# #             # 提取文本和数字
# #             text_match = re.search(r'"answer_text"\s*:\s*"([^"]+)"', output)
# #             start_match = re.search(r'"start_char"\s*:\s*(\d+)', output)
# #             end_match = re.search(r'"end_char"\s*:\s*(\d+)', output)
            
# #             if text_match:
# #                 answer_text = text_match.group(1).strip()
# #                 start_char = int(start_match.group(1)) if start_match else 0
# #                 end_char = int(end_match.group(1)) if end_match else len(answer_text)
                
# #                 return {
# #                     'answer_text': answer_text,
# #                     'start_char': start_char,
# #                     'end_char': end_char
# #                 }
            
# #             # 简单的关键词提取
# #             if "answer_text" in output or "答案" in output:
# #                 # 尝试提取引号内的内容
# #                 quote_match = re.search(r'["\']([^"\']+)["\']', output)
# #                 if quote_match:
# #                     answer_text = quote_match.group(1).strip()
# #                     return {
# #                         'answer_text': answer_text,
# #                         'start_char': 0,
# #                         'end_char': len(answer_text)
# #                     }
            
# #             return None
            
# #         except Exception as e:
# #             self.logger.error(f"解析提取输出失败: {e}")
# #             return None
    
# #     def extract_span(self, question: str, model_answer: str) -> Optional[Dict]:
# #         """使用独立本地模型提取答案片段"""
# #         try:
# #             # 确保模型已加载
# #             if not self._model_loaded:
# #                 if not self.load_extraction_model():
# #                     # 如果模型加载失败，直接返回整个答案
# #                     self.logger.warning("提取模型加载失败，返回整个答案")
# #                     return {
# #                         'answer_text': model_answer,
# #                         'start_char': 0,
# #                         'end_char': len(model_answer)
# #                     }
            
# #             # 构建提取提示词
# #             prompt = self.build_extraction_prompt(question, model_answer)
            
# #             # 构造聊天消息
# #             chat_msgs = [
# #                 {
# #                     "role": "system",
# #                     "content": "你是一个精确的答案提取助手。请从回答中提取直接回答问题的部分，并以JSON格式输出。"
# #                 },
# #                 {
# #                     "role": "user",
# #                     "content": prompt
# #                 }
# #             ]
            
# #             # 应用聊天模板
# #             inputs = self.extract_tokenizer.apply_chat_template(
# #                 chat_msgs,
# #                 tokenize=True,
# #                 add_generation_prompt=True,
# #                 return_tensors="pt"
# #             ).to(self.device)
            
# #             # 生成提取结果
# #             with torch.no_grad():
# #                 outputs = self.extract_model.generate(
# #                     inputs,
# #                     max_new_tokens=self.config.extract_max_new_tokens,
# #                     do_sample=False,
# #                     temperature=0.1,
# #                     top_p=1.0,
# #                     pad_token_id=self.extract_tokenizer.pad_token_id,
# #                     eos_token_id=self.extract_tokenizer.eos_token_id
# #                 )
            
# #             # 解码输出
# #             generated_tokens = outputs[0][inputs.shape[1]:]
# #             extract_text = self.extract_tokenizer.decode(
# #                 generated_tokens,
# #                 skip_special_tokens=True
# #             ).strip()
            
# #             self.logger.info(f"提取模型输出: {extract_text[:200]}...")
            
# #             # 解析输出
# #             extraction_result = self.parse_extraction_output(extract_text)
            
# #             if extraction_result:
# #                 answer_text = extraction_result['answer_text']
                
# #                 # 验证答案文本是否在模型回答中
# #                 if answer_text not in model_answer:
# #                     self.logger.warning(f"提取的答案不在原回答中，返回整个答案")
# #                     # 返回整个答案
# #                     return {
# #                         'answer_text': model_answer,
# #                         'start_char': 0,
# #                         'end_char': len(model_answer)
# #                     }
                
# #                 # 重新计算字符位置
# #                 start_char = model_answer.find(answer_text)
# #                 if start_char != -1:
# #                     extraction_result['start_char'] = start_char
# #                     extraction_result['end_char'] = start_char + len(answer_text)
# #                     self.logger.info(f"成功提取答案: '{answer_text}' (位置: {start_char}-{extraction_result['end_char']})")
# #                     return extraction_result
# #                 else:
# #                     # 如果找不到位置，返回整个答案
# #                     self.logger.warning("无法找到答案在回答中的位置，返回整个答案")
# #                     return {
# #                         'answer_text': model_answer,
# #                         'start_char': 0,
# #                         'end_char': len(model_answer)
# #                     }
# #             else:
# #                 self.logger.warning("模型未返回有效JSON，返回整个答案")
# #                 return {
# #                     'answer_text': model_answer,
# #                     'start_char': 0,
# #                     'end_char': len(model_answer)
# #                 }
            
# #         except Exception as e:
# #             self.logger.error(f"本地提取失败: {e}")
# #             import traceback
# #             self.logger.error(traceback.format_exc())
# #             # 返回整个答案
# #             return {
# #                 'answer_text': model_answer,
# #                 'start_char': 0,
# #                 'end_char': len(model_answer)
# #             }

# # # ==================== 3. CETT特征提取器（混合优化版本） ====================
# # class HybridCETTExtractor:
# #     """
# #     CETT特征提取器 - 混合优化版本
# #     保持字符映射对齐逻辑，但使用训练代码的向量化计算和列表格式
# #     """
    
# #     def __init__(self, model, tokenizer, config):
# #         self.model = model
# #         self.tokenizer = tokenizer
# #         self.config = config
# #         self.logger = logging.getLogger(__name__)
        
# #         # 获取模型信息
# #         self.num_layers = len(model.model.layers)
        
# #         # 确定要提取的层 - 按照论文，提取所有层
# #         self.layers_to_extract = list(range(self.num_layers))
        
# #         self.logger.info(f"模型总层数: {self.num_layers}")
# #         self.logger.info(f"提取特征层数: {len(self.layers_to_extract)}")
        
# #         # 预计算所有层的权重范数（与训练代码一致，用于向量化计算）
# #         self.logger.info("预计算权重范数...")
# #         self.precomputed_w_norms = []
# #         for layer_idx in self.layers_to_extract:
# #             mlp = self.model.model.layers[layer_idx].mlp
# #             w_down = mlp.down_proj.weight
            
# #             # 反量化处理
# #             if hasattr(w_down, 'quant_state'):
# #                 w_down_dequant = bnb.functional.dequantize_4bit(w_down.data, w_down.quant_state)
# #             else:
# #                 w_down_dequant = w_down
            
# #             # 计算范数 ||W_down[:, j]||_2
# #             w_norm = torch.norm(w_down_dequant.float(), dim=0, p=2).cpu()
# #             self.precomputed_w_norms.append(w_norm)
            
# #             # 清理临时变量
# #             del w_down_dequant
        
# #         torch.cuda.empty_cache()
# #         self.logger.info("权重范数预计算完成")
        
# #         # 记录神经元数量信息
# #         if self.layers_to_extract:
# #             first_layer = self.model.model.layers[0].mlp
# #             self.neurons_per_layer = first_layer.up_proj.weight.shape[1]  # intermediate_size
# #             self.total_neurons = len(self.layers_to_extract) * self.neurons_per_layer
# #             self.logger.info(f"每层神经元数: {self.neurons_per_layer}")
# #             self.logger.info(f"总神经元数: {self.total_neurons}")

# #     def _validate_with_unicode_tolerance(self, orig_text, cett_text, max_tolerate=3):
# #         """终极乱码容忍校验逻辑（保持不变）"""
# #         # 1. 清洗掉所有空白符（空格、换行等）
# #         s1 = "".join(orig_text.split())
# #         s2 = "".join(cett_text.split())

# #         # 2. 长度校验：如果去掉空格后长度都不一样，说明 Token 映射逻辑肯定错了
# #         if len(s1) != len(s2):
# #             return False, f"长度不一 (预期{len(s1)} vs 实际{len(s2)})"

# #         # 3. 逐字比对
# #         mismatches = 0
# #         for c1, c2 in zip(s1, s2):
# #             if c1 != c2:
# #                 # 如果不相等，检查 c2 是不是乱码符号 \ufffd ()
# #                 if c2 == '' or c2 == '\ufffd':
# #                     mismatches += 1
# #                 else:
# #                     # 发现真正的字符不匹配（不是乱码导致的），直接失败
# #                     return False, f"字符冲突 ('{c1}' vs '{c2}')"

# #         # 4. 容忍度校验：如果乱码位太多，可能也不是我们要的
# #         if mismatches > max_tolerate:
# #             return False, f"乱码过多 ({mismatches}处)"

# #         return True, "对齐成功"
    
# #     def _compute_cett_vectorized(self, z: torch.Tensor, layer_idx: int, 
# #                                h_norm: torch.Tensor, answer_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
# #         """
# #         向量化计算CETT特征 - 完全使用训练代码的实现
# #         """
# #         # 获取预计算的权重范数
# #         w_norm = self.precomputed_w_norms[layer_idx].to(z.device).to(z.dtype)
        
# #         # 取batch中的第一个样本
# #         z_abs = torch.abs(z[0])  # [seq_len, num_neurons]
        
# #         # 计算分母：||h_t||_2 已经计算好
# #         # 计算分子：|z_j,t| * ||W_down[:, j]||_2
# #         scaled_z = z_abs / h_norm.unsqueeze(-1)
        
# #         # 创建所有索引的集合
# #         seq_len = z_abs.shape[0]
# #         all_indices = set(range(seq_len))
# #         valid_answer_indices = [idx for idx in answer_indices if idx < seq_len]
# #         other_indices = list(all_indices - set(valid_answer_indices))
        
# #         # 向量化聚合 - 直接计算均值
# #         answer_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
# #         other_cett = torch.zeros(self.neurons_per_layer, dtype=z.dtype, device=z.device)
        
# #         if valid_answer_indices:
# #             answer_cett = torch.mean(scaled_z[valid_answer_indices], dim=0) * w_norm
        
# #         if other_indices:
# #             other_cett = torch.mean(scaled_z[other_indices], dim=0) * w_norm
        
# #         # 转换为numpy并清理GPU内存
# #         answer_np = answer_cett.detach().cpu().numpy().astype(np.float32)
# #         other_np = other_cett.detach().cpu().numpy().astype(np.float32)
        
# #         # 清理
# #         del z_abs, scaled_z, answer_cett, other_cett
# #         if torch.cuda.is_available():
# #             torch.cuda.empty_cache()
        
# #         return answer_np, other_np

# #     def extract_features_for_sample(self, question: str, model_answer: str, answer_text: str) -> Optional[Dict]:
# #         """
# #         为单个样本提取CETT特征 - 混合版本
# #         保持字符映射对齐逻辑，但使用向量化计算和列表格式
# #         """
# #         try:
# #             self.logger.info(f"开始提取特征 - 问题: {question[:50]}...")
            
# #             # 步骤1: 先单独tokenize model_answer，获取精确的答案词元
# #             if not answer_text or answer_text not in model_answer:
# #                 self.logger.warning(f"答案文本不在model_answer中: {answer_text[:50]}...")
# #                 self.logger.info("使用完整回答作为答案词元")
# #                 answer_text = model_answer
            
# #             # 找到答案在model_answer中的精确字符位置
# #             start_char = model_answer.find(answer_text)
# #             end_char = start_char + len(answer_text)
            
# #             if start_char == -1:
# #                 self.logger.warning(f"无法找到答案文本: {answer_text[:50]}...")
# #                 self.logger.info("使用完整回答作为答案词元")
# #                 answer_text = model_answer
# #                 start_char = 0
# #                 end_char = len(model_answer)
            
# #             # 对model_answer进行tokenization，获取offset mapping
# #             self.logger.info(f"单独tokenize model_answer，长度: {len(model_answer)}")
# #             model_answer_enc = self.tokenizer(
# #                 model_answer,
# #                 return_offsets_mapping=True,
# #                 add_special_tokens=False,
# #                 return_tensors='pt'
# #             )
# #             model_answer_offsets = model_answer_enc['offset_mapping'][0].cpu().numpy()
            
# #             # 基于字符位置找到model_answer中的答案token索引
# #             answer_token_indices_in_model = []
# #             for token_idx, (char_start, char_end) in enumerate(model_answer_offsets):
# #                 # token完全在答案区域内，或者部分重叠但主要部分在答案内
# #                 if (char_start >= start_char and char_end <= end_char) or \
# #                 (char_start <= start_char and char_end > start_char + (end_char - start_char) / 2):
# #                     answer_token_indices_in_model.append(token_idx)
            
# #             if not answer_token_indices_in_model:
# #                 self.logger.warning(f"无法找到model_answer中的答案token: {answer_text[:50]}...")
# #                 self.logger.info("使用完整回答作为答案词元")
# #                 # 使用所有token作为答案词元
# #                 answer_token_indices_in_model = list(range(len(model_answer_offsets)))
            
# #             # 获取原始答案词元文本
# #             seq_len = model_answer_enc['input_ids'].size(1)
# #             model_answer_answer_tokens = [
# #                 model_answer_enc['input_ids'][0][i].item() 
# #                 for i in answer_token_indices_in_model
# #                 if i < seq_len
# #             ]
# #             orig_answer_text = self.tokenizer.decode(
# #                 model_answer_answer_tokens,
# #                 skip_special_tokens=True,
# #                 clean_up_tokenization_spaces=False
# #             )
            
# #             # 步骤2: 构建完整prompt并确保tokenization的一致性
# #             prompt = f"问题：{question}\n回答：{model_answer}"
# #             self.logger.info(f"构建完整prompt，长度: {len(prompt)}")
            
# #             # 关键：使用相同的tokenization参数，并且记录token到原始字符的映射
# #             inputs = self.tokenizer(
# #                 prompt,
# #                 return_tensors="pt",
# #                 max_length=self.config.max_seq_length,
# #                 truncation=True,
# #                 padding="max_length",
# #                 return_offsets_mapping=True
# #             )
            
# #             input_ids = inputs['input_ids'].to(self.model.device)
# #             attention_mask = inputs['attention_mask'].to(self.model.device)
# #             full_offsets = inputs['offset_mapping'][0].cpu().numpy()
            
# #             # 步骤3: 在完整prompt中找到"回答："前缀的位置
# #             answer_prefix = "回答："
# #             answer_prefix_pos = prompt.find(answer_prefix)
# #             if answer_prefix_pos == -1:
# #                 # 尝试其他可能的格式
# #                 answer_prefix = "回答:"
# #                 answer_prefix_pos = prompt.find(answer_prefix)
# #                 if answer_prefix_pos == -1:
# #                     self.logger.warning(f"找不到回答前缀: {question[:50]}...")
# #                     self.logger.info("假设回答从prompt末尾开始")
# #                     # 假设回答从prompt末尾开始
# #                     answer_prefix_pos = prompt.find(model_answer)
# #                     if answer_prefix_pos == -1:
# #                         # 如果还找不到，使用整个prompt
# #                         answer_prefix_pos = len(prompt) - len(model_answer)
            
# #             # 步骤4: 构建映射表 - 将model_answer的token映射到完整prompt的token
# #             model_answer_start_in_prompt = answer_prefix_pos + len(answer_prefix)
# #             model_answer_end_in_prompt = model_answer_start_in_prompt + len(model_answer)
            
# #             # 建立映射字典：model_answer token index -> prompt token index
# #             token_mapping = {}
            
# #             for prompt_token_idx, (char_start, char_end) in enumerate(full_offsets):
# #                 if char_start == 0 and char_end == 0:  # 跳过padding
# #                     continue
                
# #                 # 检查这个token是否在model_answer的字符范围内
# #                 if char_start >= model_answer_start_in_prompt and char_end <= model_answer_end_in_prompt:
# #                     # 计算在model_answer中的相对字符位置
# #                     relative_char_start = char_start - model_answer_start_in_prompt
# #                     relative_char_end = char_end - model_answer_start_in_prompt
                    
# #                     # 在model_answer的token中查找匹配的token
# #                     for model_token_idx, (model_char_start, model_char_end) in enumerate(model_answer_offsets):
# #                         if model_char_start == relative_char_start and model_char_end == relative_char_end:
# #                             token_mapping[model_token_idx] = prompt_token_idx
# #                             break
            
# #             # 步骤5: 使用映射找到完整prompt中的答案token索引
# #             adjusted_answer_indices = []
# #             for model_token_idx in answer_token_indices_in_model:
# #                 if model_token_idx in token_mapping:
# #                     adjusted_answer_indices.append(token_mapping[model_token_idx])
            
# #             if not adjusted_answer_indices:
# #                 self.logger.warning(f"无法映射答案token到完整prompt: {question[:50]}...")
# #                 self.logger.info("使用完整回答作为答案词元")
# #                 # 将整个回答作为答案词元
# #                 for prompt_token_idx, (char_start, char_end) in enumerate(full_offsets):
# #                     if char_start == 0 and char_end == 0:  # 跳过padding
# #                         continue
# #                     if char_start >= model_answer_start_in_prompt:
# #                         adjusted_answer_indices.append(prompt_token_idx)
            
# #             # 获取CETT对齐后的答案词元
# #             prompt_answer_tokens = [
# #                 input_ids[0][i].item()
# #                 for i in adjusted_answer_indices
# #                 if i < input_ids.size(1)
# #             ]
# #             cett_answer_decoded = self.tokenizer.decode(
# #                 prompt_answer_tokens,
# #                 skip_special_tokens=True,
# #                 clean_up_tokenization_spaces=False
# #             )
            
# #             # 终极乱码容忍校验逻辑
# #             is_match, match_msg = self._validate_with_unicode_tolerance(orig_answer_text, cett_answer_decoded)

# #             # 打印调试信息
# #             self.logger.warning(
# #                 "\n"
# #                 "================ CETT FULL DEBUG =================\n"
# #                 f"[问题]\n{question}\n\n"
# #                 f"[原始答案文本]\n{model_answer}\n\n"
# #                 f"[原始答案词元]\n{orig_answer_text}\n\n"
# #                 f"[CETT 对齐后答案词元]\n{cett_answer_decoded}\n\n"
# #                 f"[对齐结果]\n{match_msg}\n"
# #                 "=================================================="
# #             )

# #             if not is_match:
# #                 self.logger.warning(f"对齐校验不通过: {match_msg}")
# #                 self.logger.info("使用完整回答作为答案词元")
# #                 # 将整个回答作为答案词元
# #                 adjusted_answer_indices = []
# #                 for prompt_token_idx, (char_start, char_end) in enumerate(full_offsets):
# #                     if char_start == 0 and char_end == 0:  # 跳过padding
# #                         continue
# #                     if char_start >= model_answer_start_in_prompt:
# #                         adjusted_answer_indices.append(prompt_token_idx)
                
# #                 # 重新解码
# #                 prompt_answer_tokens = [
# #                     input_ids[0][i].item()
# #                     for i in adjusted_answer_indices
# #                     if i < input_ids.size(1)
# #                 ]
# #                 cett_answer_decoded = self.tokenizer.decode(
# #                     prompt_answer_tokens,
# #                     skip_special_tokens=True,
# #                     clean_up_tokenization_spaces=False
# #                 )
# #                 self.logger.info(f"回退后答案词元: {cett_answer_decoded[:100]}...")
# #             else:
# #                 self.logger.info(f"✓ 答案词元成功对齐: {match_msg}")

# #             # 一次性前向传播获取所有层的隐藏状态
# #             self.logger.info("开始前向传播...")
# #             with torch.no_grad():
# #                 outputs = self.model(
# #                     input_ids=input_ids,
# #                     attention_mask=attention_mask,
# #                     output_hidden_states=True,
# #                     return_dict=True
# #                 )
            
# #             hidden_states = outputs.hidden_states  # 所有层的隐藏状态
            
# #             # 向量化提取所有层的特征（使用训练代码的方法）
# #             all_answer_features = []
# #             all_other_features = []
            
# #             self.logger.info(f"向量化计算CETT特征，共{len(self.layers_to_extract)}层")
# #             for i, layer_idx in enumerate(self.layers_to_extract):
# #                 layer = self.model.model.layers[layer_idx]
                
# #                 # 获取归一化后的隐藏状态
# #                 normed_hs = layer.post_attention_layernorm(hidden_states[layer_idx])
                
# #                 # 计算MLP激活值z
# #                 mlp = layer.mlp
# #                 gate = mlp.gate_proj(normed_hs)
# #                 up = mlp.up_proj(normed_hs)
# #                 z = torch.nn.functional.silu(gate) * up
                
# #                 # 计算分母 ||h_t||_2
# #                 h_total = mlp.down_proj(z)
# #                 h_norm = torch.norm(h_total[0].float(), dim=-1, p=2)
# #                 h_norm = torch.clamp(h_norm, min=1e-10)
                
# #                 # 向量化计算CETT特征（使用训练代码的向量化函数）
# #                 answer_vec, other_vec = self._compute_cett_vectorized(
# #                     z, i, h_norm, adjusted_answer_indices
# #                 )
                
# #                 # 收集该层的特征
# #                 all_answer_features.append(answer_vec)
# #                 all_other_features.append(other_vec)
                
# #                 # 清理中间变量
# #                 del gate, up, z, h_total, h_norm, normed_hs
# #                 if torch.cuda.is_available():
# #                     torch.cuda.empty_cache()
            
# #             # 拼接所有层的特征（向量化操作）
# #             final_answer_features = np.concatenate(all_answer_features, axis=0)
# #             final_other_features = np.concatenate(all_other_features, axis=0)
            
# #             # 转换为列表格式（更高效的JSON序列化）
# #             features_answer_list = final_answer_features.tolist()
# #             features_other_list = final_other_features.tolist()
            
# #             self.logger.info(f"特征提取成功，特征数量: {len(features_answer_list)}")
            
# #             # 清理前向传播结果
# #             del outputs, hidden_states
# #             torch.cuda.empty_cache()
            
# #             # 修复JSON序列化问题：确保所有数据类型都是Python原生类型
# #             return {
# #                 'features_answer_list': features_answer_list,  # 列表格式，与训练代码一致
# #                 'features_other_list': features_other_list,    # 列表格式，与训练代码一致
# #                 'aligned_tokens': cett_answer_decoded,
# #                 'original_tokens': orig_answer_text,
# #                 'answer_indices': [int(idx) for idx in adjusted_answer_indices],  # 转换为Python int
# #                 'is_aligned': bool(is_match),  # 确保是Python bool
# #                 'total_features': int(len(features_answer_list))  # 转换为Python int
# #             }
            
# #         except torch.cuda.OutOfMemoryError:
# #             self.logger.error("GPU内存不足，跳过此样本")
# #             gc.collect()
# #             torch.cuda.empty_cache()
# #             return None
# #         except Exception as e:
# #             self.logger.error(f"提取特征失败: {e}")
# #             import traceback
# #             self.logger.error(traceback.format_exc())
# #             return None

# # # ==================== 4. 特征处理器（使用训练代码的逻辑） ====================
# # class FeatureProcessor:
# #     """特征处理器 - 确保与训练时相同的处理流程"""
    
# #     def __init__(self, classifier_bundle):
# #         self.classifier = classifier_bundle['classifier']
# #         self.var_selector = classifier_bundle.get('var_selector')
# #         self.percentile_selector = classifier_bundle.get('percentile_selector')
# #         self.scaler = classifier_bundle.get('scaler')
# #         self.neuron_names = classifier_bundle.get('neuron_names', [])
# #         self.logger = logging.getLogger(__name__)
        
# #         # 记录特征维度信息
# #         self.original_dim = len(self.neuron_names)
# #         if self.percentile_selector and hasattr(self.percentile_selector, 'get_support'):
# #             support_mask = self.percentile_selector.get_support()
# #             self.selected_dim = int(np.sum(support_mask))  # 转换为Python int
# #         else:
# #             self.selected_dim = int(self.original_dim)
        
# #         self.logger.info(f"特征处理信息:")
# #         self.logger.info(f"  原始特征维度: {self.original_dim}")
# #         self.logger.info(f"  选择后维度: {self.selected_dim}")
    
# #     def process_features(self, features_list: List[float]) -> np.ndarray:
# #         """处理特征，与训练时完全相同的pipeline"""
# #         if features_list is None:
# #             return None
        
# #         # 转换为numpy数组
# #         features = np.array(features_list, dtype=np.float32).reshape(1, -1)
        
# #         # 确保特征维度正确
# #         if features.shape[1] != self.original_dim:
# #             self.logger.error(f"特征维度不匹配! 预期: {self.original_dim}, 实际: {features.shape[1]}")
# #             return None
        
# #         # 应用与训练时相同的特征处理pipeline
# #         x_proc = features.copy()
        
# #         # 1. 方差选择
# #         if self.var_selector is not None:
# #             x_proc = self.var_selector.transform(x_proc)
        
# #         # 2. 百分位数选择
# #         if self.percentile_selector is not None:
# #             x_proc = self.percentile_selector.transform(x_proc)
        
# #         # 3. 标准化
# #         if self.scaler is not None:
# #             x_proc = self.scaler.transform(x_proc)
        
# #         # 4. 检查并清理特征
# #         if np.any(np.isnan(x_proc)) or np.any(np.isinf(x_proc)):
# #             x_proc = np.nan_to_num(x_proc, nan=0.0, posinf=1e6, neginf=-1e6)
        
# #         return x_proc
    
# #     def predict(self, features_list: List[float]) -> Tuple[float, bool]:
# #         """预测"""
# #         x_proc = self.process_features(features_list)
# #         if x_proc is None:
# #             return 0.0, False
        
# #         prob = self.classifier.predict_proba(x_proc)[0][1]
# #         is_hallucination = prob > GlobalConfig.threshold
        
# #         return float(prob), bool(is_hallucination)

# # # ==================== 5. 模型加载器 ====================
# # class ModelLoader:
# #     """模型和分词器加载器"""
    
# #     @staticmethod
# #     def load_model_and_tokenizer(config):
# #         """加载模型和分词器"""
# #         logger.info(f"加载模型: {config.model_path}")
        
# #         bnb_config = None
# #         if config.use_4bit:
# #             bnb_config = BitsAndBytesConfig(
# #                 load_in_4bit=True,
# #                 bnb_4bit_compute_dtype=torch.float16,
# #                 bnb_4bit_use_double_quant=True,
# #                 bnb_4bit_quant_type="nf4"
# #             )
        
# #         logger.info("加载分词器...")
# #         tokenizer = AutoTokenizer.from_pretrained(
# #             config.model_path,
# #             trust_remote_code=True,
# #             use_fast=True
# #         )
        
# #         if tokenizer.pad_token is None:
# #             tokenizer.pad_token = tokenizer.eos_token
        
# #         logger.info("加载模型...")
# #         model = AutoModelForCausalLM.from_pretrained(
# #             config.model_path,
# #             quantization_config=bnb_config,
# #             device_map="auto",
# #             torch_dtype=torch.float16,
# #             trust_remote_code=True
# #         )
        
# #         model.eval()
# #         logger.info(f"模型加载完成")
        
# #         return model, tokenizer

# # # ==================== 6. 辅助函数：Python类型转换 ====================
# # def convert_to_python_types(obj):
# #     """
# #     递归地将numpy/pandas类型转换为Python原生类型
# #     解决JSON序列化问题
# #     """
# #     if isinstance(obj, dict):
# #         return {k: convert_to_python_types(v) for k, v in obj.items()}
# #     elif isinstance(obj, list):
# #         return [convert_to_python_types(item) for item in obj]
# #     elif isinstance(obj, tuple):
# #         return tuple(convert_to_python_types(item) for item in obj)
# #     elif isinstance(obj, np.integer):
# #         return int(obj)
# #     elif isinstance(obj, np.floating):
# #         return float(obj)
# #     elif isinstance(obj, np.bool_):
# #         return bool(obj)
# #     elif isinstance(obj, np.ndarray):
# #         return convert_to_python_types(obj.tolist())
# #     elif hasattr(obj, 'item'):  # 其他可能的numpy标量类型
# #         try:
# #             return obj.item()
# #         except:
# #             return obj
# #     else:
# #         return obj

# # # ==================== 7. 主程序逻辑 ====================
# # def main():
# #     config = GlobalConfig()
    
# #     # Step A: 加载分类器组件
# #     logger.info(f"正在加载分类器包: {config.classifier_path}")
# #     try:
# #         with open(config.classifier_path, 'rb') as f:
# #             bundle = pickle.load(f)
        
# #         logger.info(f"分类器加载成功")
        
# #         # 检查关键组件
# #         required_components = ['classifier', 'scaler', 'neuron_names']
# #         missing = [c for c in required_components if c not in bundle]
# #         if missing:
# #             logger.error(f"缺少关键组件: {missing}")
# #             return
        
# #         # 初始化特征处理器
# #         feature_processor = FeatureProcessor(bundle)
        
# #     except Exception as e:
# #         logger.error(f"加载分类器失败: {e}")
# #         return

# #     # Step B: 加载主模型（用于生成回答）
# #     logger.info(f"正在加载主模型: {config.model_path}")
# #     try:
# #         model, tokenizer = ModelLoader.load_model_and_tokenizer(config)
# #     except Exception as e:
# #         logger.error(f"加载主模型失败: {e}")
# #         return

# #     # Step C: 初始化本地答案提取器（独立模型）
# #     logger.info(f"初始化本地答案提取器，使用模型: {config.extract_model_path}")
# #     api_extractor = LocalAnswerExtractor(config)
    
# #     # 预加载提取模型
# #     api_extractor.load_extraction_model()

# #     # Step D: 初始化CETT特征提取器
# #     cett_extractor = HybridCETTExtractor(model, tokenizer, config)

# #     # Step E: 读取数据
# #     logger.info(f"正在读取测试数据: {config.input_data_path}")
# #     try:
# #         with open(config.input_data_path, 'r', encoding='utf-8') as f:
# #             data = json.load(f)
# #         logger.info(f"成功加载 {len(data)} 条测试数据")
# #     except Exception as e:
# #         logger.error(f"加载测试数据失败: {e}")
# #         return
    
# #     os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
# #     logger.info(f"开始逐条检测，结果将保存到: {config.output_path}")

# #     # 流式写入JSON文件
# #     with open(config.output_path, 'w', encoding='utf-8') as f_out:
# #         f_out.write("[\n")  # JSON数组开始
        
# #         processed_count = 0
# #         failed_count = 0
# #         alignment_failures = 0
        
# #         for idx, sample in enumerate(tqdm(data, desc="处理测试样本")):
# #             question = sample.get("question", "").strip()
# #             source_id = sample.get("source_id", f"sample_{idx}")
# #             reference_answer = sample.get("answer", "")
            
# #             if not question: 
# #                 logger.warning(f"样本 {idx} 问题为空，跳过")
# #                 continue
            
# #             try:
# #                 # 1. 生成模型回答（使用主模型）
# #                 logger.info(f"\n=== 处理样本 {idx+1}/{len(data)}: {source_id} ===")
                
# #                 chat_msgs = [
# #                     {
# #                         "role": "system", 
# #                         "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
# #                     },
# #                     {
# #                         "role": "user", 
# #                         "content": f"请回答下面的问题：\n\n问题：{question}"
# #                     }
# #                 ]
                
# #                 gen_inputs = tokenizer.apply_chat_template(
# #                     chat_msgs, 
# #                     tokenize=True, 
# #                     add_generation_prompt=True, 
# #                     return_tensors="pt"
# #                 ).to(model.device)
                
# #                 with torch.no_grad():
# #                     gen_out = model.generate(
# #                         gen_inputs, 
# #                         max_new_tokens=config.max_new_tokens, 
# #                         do_sample=False,
# #                         temperature=0.1
# #                     )
                
# #                 model_answer = tokenizer.decode(
# #                     gen_out[0][gen_inputs.shape[1]:], 
# #                     skip_special_tokens=True
# #                 ).strip()
                
# #                 logger.info(f"生成的回答: {model_answer[:100]}...")

# #                 # 2. 使用独立本地模型提取答案位置
# #                 logger.info("调用本地模型提取答案片段...")
# #                 api_span = api_extractor.extract_span(question, model_answer)
                
# #                 if not api_span:
# #                     logger.warning("本地模型提取失败，使用整个答案")
# #                     answer_text = model_answer
# #                 else:
# #                     answer_text = api_span.get("answer_text", "")
# #                     if not answer_text:
# #                         logger.warning("提取的答案为空，使用整个答案")
# #                         answer_text = model_answer
                
# #                 logger.info(f"使用的答案: {answer_text[:100]}...")

# #                 # 3. 提取CETT特征（使用混合优化版本）
# #                 logger.info("开始CETT特征提取...")
# #                 features_result = cett_extractor.extract_features_for_sample(question, model_answer, answer_text)
                
# #                 if features_result is None:
# #                     logger.error(f"样本 {idx} 特征提取失败，跳过")
# #                     failed_count += 1
# #                     continue
                
# #                 if not features_result.get('is_aligned', False):
# #                     alignment_failures += 1
# #                     logger.warning(f"对齐失败，使用完整回答作为答案词元")
                
# #                 # 4. 处理特征并预测（直接使用列表格式）
# #                 logger.info("处理特征并预测...")
# #                 features_list = features_result['features_answer_list']
# #                 prob, is_hallu = feature_processor.predict(features_list)
                
# #                 logger.info(f"预测结果 - 概率: {prob:.4f}, 是否幻觉: {is_hallu}")

# #                 # 5. 整合结果并确保Python原生类型
# #                 res_item = convert_to_python_types({
# #                     "source_id": source_id,
# #                     "question": question,
# #                     "reference_answer": reference_answer,
# #                     "model_answer": model_answer,
# #                     "extracted_part": answer_text,
# #                     "hallucination_prob": round(float(prob), 4),
# #                     "is_hallucination": is_hallu,
# #                     "extraction_method": "local_model" if api_span and answer_text != model_answer else "full_answer",
# #                     "is_aligned": features_result.get('is_aligned', False)
# #                 })
                
# #                 # 流式写入文件
# #                 if processed_count > 0: 
# #                     f_out.write(",\n")
# #                 f_out.write(json.dumps(res_item, ensure_ascii=False, indent=2))
# #                 f_out.flush()
                
# #                 processed_count += 1

# #                 # 定期清理内存
# #                 if (idx + 1) % 10 == 0:
# #                     logger.info(f"已处理 {idx+1} 个样本，清理内存...")
# #                     gc.collect()
# #                     torch.cuda.empty_cache()

# #             except Exception as e:
# #                 logger.error(f"处理样本 {idx} 失败: {e}")
# #                 failed_count += 1
# #                 # 写入错误结果
# #                 if processed_count > 0: 
# #                     f_out.write(",\n")
# #                 error_item = convert_to_python_types({
# #                     "source_id": source_id,
# #                     "question": question,
# #                     "reference_answer": reference_answer,
# #                     "model_answer": "",
# #                     "extracted_part": "",
# #                     "hallucination_prob": -1.0,
# #                     "is_hallucination": False,
# #                     "error": str(e)[:200]
# #                 })
# #                 f_out.write(json.dumps(error_item, ensure_ascii=False, indent=2))
# #                 f_out.flush()
# #                 continue
        
# #         f_out.write("\n]")  # JSON数组结束
    
# #     # 生成汇总报告
# #     logger.info(f"\n✨ 检测完成！最终结果保存至: {config.output_path}")
    
# #     # 读取结果生成统计报告
# #     try:
# #         with open(config.output_path, 'r', encoding='utf-8') as f:
# #             results = json.load(f)
        
# #         successful_results = [r for r in results if r.get("model_answer", "") and r.get("hallucination_prob", -1) >= 0]
# #         failed_results = [r for r in results if r.get("model_answer", "") == "" or r.get("hallucination_prob", -1) < 0]
        
# #         summary_path = config.output_path.replace('.json', '_summary.json')
        
# #         # 统计提取方法
# #         local_extraction_count = sum(1 for r in successful_results if r.get("extraction_method") == "local_model")
# #         full_answer_count = sum(1 for r in successful_results if r.get("extraction_method") == "full_answer")
        
# #         summary = convert_to_python_types({
# #             "total_samples": len(data),
# #             "processed_samples": len(successful_results),
# #             "failed_samples": len(failed_results),
# #             "alignment_failures": alignment_failures,
# #             "hallucination_count": sum(1 for r in successful_results if r["is_hallucination"]),
# #             "hallucination_rate": sum(1 for r in successful_results if r["is_hallucination"]) / len(successful_results) if successful_results else 0,
# #             "avg_hallucination_prob": np.mean([r["hallucination_prob"] for r in successful_results]) if successful_results else 0,
# #             "alignment_success_rate": sum(1 for r in successful_results if r.get("is_aligned", False)) / len(successful_results) if successful_results else 0,
# #             "extraction_statistics": {
# #                 "local_model_extraction": local_extraction_count,
# #                 "full_answer_extraction": full_answer_count,
# #                 "local_model_success_rate": local_extraction_count / len(successful_results) if successful_results else 0
# #             },
# #             "feature_selection_info": {
# #                 "original_dimension": feature_processor.original_dim,
# #                 "selected_dimension": feature_processor.selected_dim,
# #                 "selection_ratio": feature_processor.selected_dim / feature_processor.original_dim if feature_processor.original_dim > 0 else 0
# #             },
# #             "model_info": {
# #                 "main_model": config.model_path,
# #                 "extraction_model": config.extract_model_path
# #             },
# #             "threshold": config.threshold,
# #             "classifier_path": config.classifier_path,
# #             "processing_time": datetime.now().isoformat()
# #         })
        
# #         with open(summary_path, 'w', encoding='utf-8') as f:
# #             json.dump(summary, f, ensure_ascii=False, indent=2)
        
# #         print(f"\n{'='*60}")
# #         print("检测统计报告")
# #         print(f"{'='*60}")
# #         print(f"总样本数: {summary['total_samples']}")
# #         print(f"成功处理: {summary['processed_samples']}")
# #         print(f"处理失败: {summary['failed_samples']}")
# #         print(f"对齐失败: {summary['alignment_failures']}")
# #         print(f"对齐成功率: {summary['alignment_success_rate']:.2%}")
# #         print(f"\n答案提取统计:")
# #         print(f"  本地模型提取: {summary['extraction_statistics']['local_model_extraction']}")
# #         print(f"  完整答案: {summary['extraction_statistics']['full_answer_extraction']}")
# #         print(f"  本地模型成功率: {summary['extraction_statistics']['local_model_success_rate']:.2%}")
# #         print(f"\n幻觉检测结果:")
# #         print(f"  幻觉数量: {summary['hallucination_count']}")
# #         print(f"  幻觉比例: {summary['hallucination_rate']:.2%}")
# #         print(f"  平均概率: {summary['avg_hallucination_prob']:.4f}")
# #         print(f"\n特征选择信息:")
# #         print(f"  原始特征维度: {summary['feature_selection_info']['original_dimension']}")
# #         print(f"  选择后维度: {summary['feature_selection_info']['selected_dimension']}")
# #         print(f"  选择比例: {summary['feature_selection_info']['selection_ratio']*100:.1f}%")
# #         print(f"\n模型信息:")
# #         print(f"  主模型: {summary['model_info']['main_model']}")
# #         print(f"  提取模型: {summary['model_info']['extraction_model']}")
# #         print(f"\n详细结果: {config.output_path}")
# #         print(f"统计报告: {summary_path}")
# #         print(f"{'='*60}")
        
# #     except Exception as e:
# #         print(f"生成统计报告失败: {e}")
# #         import traceback
# #         print(traceback.format_exc())

# # if __name__ == "__main__":
# #     print("="*60)
# #     print("H-Neurons幻觉检测测试（混合优化版本 + 本地答案提取）")
# #     print("="*60)
    
# #     # 运行主测试
# #     main()
    
# #     print("\n" + "="*60)
# #     print("批量测试完成!")
# #     print("="*60)