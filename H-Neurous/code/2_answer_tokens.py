# """
# 阿里云百炼API答案词元批量提取器（简化版）
# 直接使用批量API进行高效提取
# """

# import json
# import time
# import hashlib
# import logging
# import os
# import re
# import sys
# from typing import Dict, List, Optional, Any
# from datetime import datetime
# import numpy as np

# from openai import OpenAI

# # 添加项目根目录到Python路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # 直接导入配置
# from configs.config import model_config, api_config, data_config
# from modules.model_loader import ModelLoader

# # 配置日志
# logger = logging.getLogger(__name__)

# class DashscopeBatchExtractor:
#     """阿里云百炼API批量答案词元提取器"""
    
#     def __init__(self):
#         """初始化API客户端"""
#         logger.info("初始化阿里云百炼API批量提取器...")
        
#         self.client = OpenAI(
#             api_key=model_config.dashscope_api_key,
#             base_url=model_config.dashscope_base_url
#         )
        
#         self.model_name = model_config.dashscope_model
        
#         # 批量处理配置
#         self.batch_size = 1000  # 每批处理的样本数
#         self.max_wait_hours = 72  # 最大等待时间
        
#         # 确保缓存目录存在
#         os.makedirs(api_config.location_cache_dir, exist_ok=True)
        
#         logger.info(f"API客户端初始化完成，模型: {self.model_name}")
#         logger.info(f"批量配置: 批次大小={self.batch_size}, 最大等待时间={self.max_wait_hours}小时")
    
#     def _generate_cache_key(self, question: str, model_answer: str, standard_answer: str = None) -> str:
#         """生成缓存键"""
#         content = f"dashscope::{question}|||{model_answer}|||{standard_answer}"
#         return hashlib.md5(content.encode('utf-8')).hexdigest()
    
#     def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
#         """获取缓存的提取结果"""
#         cache_file = os.path.join(api_config.location_cache_dir, f"{cache_key}.json")
        
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, 'r', encoding='utf-8') as f:
#                     cached_data = json.load(f)
                
#                 # 检查缓存是否过期（7天）
#                 if time.time() - cached_data.get('timestamp', 0) < 7 * 24 * 3600:
#                     logger.debug(f"使用缓存提取结果: {cache_key}")
#                     return cached_data
#             except Exception as e:
#                 logger.warning(f"读取缓存失败: {e}")
        
#         return None
    
#     def _save_cached_result(self, cache_key: str, result: Dict):
#         """保存提取结果到缓存"""
#         cache_file = os.path.join(api_config.location_cache_dir, f"{cache_key}.json")
        
#         try:
#             result['timestamp'] = time.time()
            
#             with open(cache_file, 'w', encoding='utf-8') as f:
#                 json.dump(result, f, ensure_ascii=False, indent=2)
            
#             logger.debug(f"保存提取结果到缓存: {cache_file}")
#         except Exception as e:
#             logger.warning(f"保存缓存失败: {e}")
    
#     def _build_extraction_prompt(self, question: str, model_answer: str, standard_answer: Optional[str] = None) -> str:
#         """构建答案提取提示词"""
#         prompt = f"""请分析以下对话，并从模型回答中精确提取直接回答问题的部分。

# 问题：{question}

# 模型回答：{model_answer}

# {f"参考答案（用于参考，但不是必须完全匹配）：{standard_answer}" if standard_answer else ""}

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
#   "end_char": 答案在模型回答中的结束字符位置,
#   "confidence": 提取置信度（0-1之间的小数，表示你有多确定这是正确答案）,
#   "reason": "简要说明为什么选择这部分作为答案"
# }}

# 2. 字符位置必须精确，不能有偏差
# 3. 如果模型回答中没有明确的答案，将answer_text设为空字符串，confidence设为0
# 4. 只输出JSON，不要有其他任何内容"""

#         return prompt
    
#     def _parse_extraction_response(self, response_text: str) -> Optional[Dict]:
#         """解析提取响应"""
#         try:
#             # 提取JSON部分
#             json_pattern = r'\{.*\}'
#             match = re.search(json_pattern, response_text, re.DOTALL)
            
#             if not match:
#                 logger.warning(f"响应中没有找到JSON: {response_text[:200]}...")
#                 return None
            
#             json_str = match.group()
#             result = json.loads(json_str)
            
#             # 验证必需字段
#             required_fields = ['answer_text', 'start_char', 'end_char', 'confidence']
#             for field in required_fields:
#                 if field not in result:
#                     logger.warning(f"响应缺少必需字段: {field}")
#                     return None
            
#             # 验证数据类型
#             try:
#                 result['start_char'] = int(result['start_char'])
#                 result['end_char'] = int(result['end_char'])
#                 result['confidence'] = float(result['confidence'])
#             except (ValueError, TypeError) as e:
#                 logger.warning(f"响应字段类型错误: {e}")
#                 return None
            
#             # 验证位置有效性
#             if result['start_char'] < 0 or result['end_char'] <= result['start_char']:
#                 logger.warning(f"响应位置无效: start={result['start_char']}, end={result['end_char']}")
#                 return None
            
#             return result
            
#         except json.JSONDecodeError as e:
#             logger.error(f"解析响应JSON失败: {e}, 响应: {response_text[:200]}...")
#             return None
#         except Exception as e:
#             logger.error(f"解析响应失败: {e}")
#             return None
    
#     def _convert_char_to_token_indices(self, text: str, start_char: int, end_char: int, 
#                                      tokenizer) -> List[int]:
#         """将字符位置转换为token索引"""
#         try:
#             # 对整个文本进行tokenization
#             encoding = tokenizer.encode_plus(
#                 text,
#                 return_offsets_mapping=True,
#                 add_special_tokens=False
#             )
            
#             tokens = encoding['input_ids']
#             offset_mapping = encoding['offset_mapping']
            
#             # 找到答案区域内的token索引
#             answer_token_indices = []
            
#             for idx, (char_start, char_end) in enumerate(offset_mapping):
#                 # 检查token是否与答案区域有重叠
#                 if not (char_end <= start_char or char_start >= end_char):
#                     answer_token_indices.append(idx)
            
#             # 如果找不到匹配的token，返回整个回答的token
#             if not answer_token_indices:
#                 logger.warning(f"无法找到匹配的token，返回所有token")
#                 return list(range(len(tokens)))
            
#             return answer_token_indices
            
#         except Exception as e:
#             logger.error(f"字符位置转token索引失败: {e}")
#             # 返回整个回答的token作为后备
#             try:
#                 full_tokens = tokenizer.encode(text, add_special_tokens=False)
#                 return list(range(len(full_tokens)))
#             except:
#                 return []
    
#     def _create_batch_input_file(self, samples_batch: List[Dict[str, Any]], batch_num: int, output_dir: str) -> str:
#         """创建批量提取请求的JSONL文件"""
#         input_file_path = os.path.join(output_dir, f"batch_{batch_num}_input.jsonl")
        
#         with open(input_file_path, 'w', encoding='utf-8') as f:
#             for i, sample in enumerate(samples_batch):
#                 custom_id = f"batch_{batch_num}_sample_{i}"
                
#                 question = sample.get('question', '')
#                 model_answer = sample.get('model_answer', '')
#                 standard_answer = sample.get('standard_answer', '')
                
#                 # 构建提示词
#                 prompt = self._build_extraction_prompt(question, model_answer, standard_answer)
                
#                 batch_request = {
#                     "custom_id": custom_id,
#                     "method": "POST",
#                     "url": "/v1/chat/completions",
#                     "body": {
#                         "model": self.model_name,
#                         "messages": [
#                             {
#                                 "role": "system", 
#                                 "content": "你是一个专业的文本分析助手，负责从回答中精确定位答案部分。严格按照要求的JSON格式返回结果。"
#                             },
#                             {
#                                 "role": "user", 
#                                 "content": prompt
#                             }
#                         ],
#                         "max_tokens": 1000,
#                         "temperature": 0.1
#                     }
#                 }
                
#                 f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
        
#         logger.info(f"创建批次 {batch_num} 输入文件: {len(samples_batch)} 个请求")
#         return input_file_path
    
#     def _submit_batch_job(self, input_file_path: str, batch_num: int) -> Optional[str]:
#         """提交批量任务"""
#         try:
#             with open(input_file_path, 'rb') as file:
#                 uploaded_file = self.client.files.create(file=file, purpose="batch")
            
#             batch_job = self.client.batches.create(
#                 input_file_id=uploaded_file.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window=f"{self.max_wait_hours}h"
#             )
            
#             logger.info(f"批次 {batch_num} 提交成功，任务ID: {batch_job.id}")
#             return batch_job.id
            
#         except Exception as e:
#             logger.error(f"批次 {batch_num} 提交失败: {e}")
#             return None
    
#     def _monitor_batch_progress(self, batch_id: str, batch_num: int) -> Optional[str]:
#         """监控批量任务进度"""
#         logger.info(f"监控批次 {batch_num} 进度: {batch_id}")
        
#         start_time = time.time()
#         last_status = None
        
#         while True:
#             try:
#                 batch_status = self.client.batches.retrieve(batch_id)
#                 current_status = batch_status.status
                
#                 if current_status != last_status:
#                     elapsed_hours = (time.time() - start_time) / 3600
#                     logger.info(f"批次 {batch_num} 状态: {current_status} (已等待 {elapsed_hours:.1f} 小时)")
#                     last_status = current_status
                
#                 if current_status == "completed":
#                     logger.info(f"批次 {batch_num} 处理完成!")
#                     return batch_status.output_file_id
                    
#                 elif current_status in ["failed", "expired", "cancelled"]:
#                     logger.error(f"批次 {batch_num} 失败: {current_status}")
#                     return None
                    
#                 else:
#                     # 还在处理中，等待5分钟后再次检查
#                     time.sleep(300)
                    
#             except Exception as e:
#                 logger.warning(f"检查批次 {batch_num} 状态失败: {e}")
#                 time.sleep(300)
    
#     def _process_batch_results(self, output_file_id: str, original_batch: List[Dict[str, Any]], 
#                               batch_num: int, tokenizer) -> List[Dict[str, Any]]:
#         """处理批量结果"""
#         try:
#             # 下载结果
#             result_content = self.client.files.content(output_file_id)
#             results = result_content.text.split('\n')
            
#             processed_samples = []
#             success_count = 0
#             error_count = 0
            
#             logger.info(f"下载批次 {batch_num} 结果: {len(results)} 行，开始解析...")
            
#             # 构建映射表
#             sample_mapping = {}
#             for i, sample in enumerate(original_batch):
#                 custom_id = f"batch_{batch_num}_sample_{i}"
#                 sample_mapping[custom_id] = sample
            
#             # 解析每一行结果
#             for line in results:
#                 if not line.strip():
#                     continue
                    
#                 try:
#                     result_obj = json.loads(line)
#                     custom_id = result_obj.get("custom_id")
#                     response = result_obj.get("response", {})
                    
#                     if response.get("status_code") == 200:
#                         body = response.get("body", {})
#                         choices = body.get("choices", [])
                        
#                         if choices:
#                             content = choices[0].get("message", {}).get("content", "")
                            
#                             # 提取JSON
#                             json_match = re.search(r'\{.*\}', content, re.DOTALL)
#                             if json_match:
#                                 try:
#                                     extraction_result = json.loads(json_match.group(0))
                                    
#                                     if custom_id in sample_mapping:
#                                         original_sample = sample_mapping[custom_id]
                                        
#                                         # 处理提取结果
#                                         question = original_sample.get('question', '')
#                                         model_answer = original_sample.get('model_answer', '')
#                                         answer_text = extraction_result.get('answer_text', '')
#                                         start_char = extraction_result.get('start_char', 0)
#                                         end_char = extraction_result.get('end_char', 0)
#                                         confidence = extraction_result.get('confidence', 0.0)
                                        
#                                         # 验证并修正位置
#                                         if answer_text and answer_text in model_answer:
#                                             actual_start = model_answer.find(answer_text)
#                                             if actual_start != -1 and actual_start != start_char:
#                                                 start_char = actual_start
#                                                 end_char = actual_start + len(answer_text)
                                        
#                                         # 转换为token索引
#                                         answer_token_indices = []
#                                         if tokenizer and answer_text:
#                                             answer_token_indices = self._convert_char_to_token_indices(
#                                                 model_answer, start_char, end_char, tokenizer
#                                             )
                                        
#                                         # 构建处理后的样本
#                                         processed_sample = original_sample.copy()
#                                         processed_sample.update({
#                                             'answer_text': answer_text,
#                                             'answer_token_indices': answer_token_indices,
#                                             'answer_token_count': len(answer_token_indices),
#                                             'start_char': start_char,
#                                             'end_char': end_char,
#                                             'confidence': confidence,
#                                             'method': 'dashscope_batch_api',
#                                             'extraction_time': time.time(),
#                                             'batch_id': batch_num,
#                                             'custom_id': custom_id
#                                         })
                                        
#                                         # 保存到缓存
#                                         cache_key = self._generate_cache_key(question, model_answer, 
#                                                                            original_sample.get('standard_answer'))
#                                         self._save_cached_result(cache_key, {
#                                             'answer_text': answer_text,
#                                             'start_char': start_char,
#                                             'end_char': end_char,
#                                             'confidence': confidence,
#                                             'method': 'dashscope_batch_api'
#                                         })
                                        
#                                         processed_samples.append(processed_sample)
#                                         success_count += 1
                                    
#                                 except json.JSONDecodeError:
#                                     error_count += 1
#                             else:
#                                 error_count += 1
#                         else:
#                             error_count += 1
#                     else:
#                         error_count += 1
                        
#                 except Exception:
#                     error_count += 1
            
#             logger.info(f"批次 {batch_num} 解析完成: 成功 {success_count}, 失败 {error_count}")
#             return processed_samples
            
#         except Exception as e:
#             logger.error(f"下载或解析批次 {batch_num} 结果失败: {e}")
#             return []
    
#     def _load_data(self) -> List[Dict[str, Any]]:
#         """加载数据"""
#         logger.info("加载数据...")
        
#         # 直接指定数据路径
#         faithful_path = data_config.faithful_data_path
#         hallucination_path = data_config.hallucination_data_path
        
#         logger.info(f"忠实数据路径: {faithful_path}")
#         logger.info(f"幻觉数据路径: {hallucination_path}")
        
#         # 加载忠实数据
#         with open(faithful_path, 'r', encoding='utf-8') as f:
#             faithful_data = json.load(f)
        
#         # 加载幻觉数据
#         with open(hallucination_path, 'r', encoding='utf-8') as f:
#             hallucination_data = json.load(f)
        
#         logger.info(f"忠实样本数: {len(faithful_data)}")
#         logger.info(f"幻觉样本数: {len(hallucination_data)}")
        
#         # 添加标签
#         for item in faithful_data:
#             item['label'] = 0  # 0表示忠实
#             if 'standard_answer' not in item:
#                 item['standard_answer'] = item.get('answer', '')
        
#         for item in hallucination_data:
#             item['label'] = 1  # 1表示幻觉
#             if 'standard_answer' not in item:
#                 item['standard_answer'] = item.get('answer', '')
        
#         # 合并数据
#         all_data = faithful_data + hallucination_data
        
#         logger.info(f"总样本数: {len(all_data)}")
        
#         return all_data
    
#     def batch_extract(self):
#         """批量提取主函数"""
#         logger.info("="*60)
#         logger.info("开始阿里云百炼API答案词元批量提取")
#         logger.info("="*60)
        
#         start_time = time.time()
        
#         try:
#             # 1. 加载数据
#             logger.info("步骤1: 加载数据")
#             all_data = self._load_data()
            
#             # 限制样本数量（如果需要）
#             max_samples = None  # 设为None表示不限制
#             if max_samples and max_samples < len(all_data):
#                 all_data = all_data[:max_samples]
#                 logger.info(f"限制样本数量为: {max_samples}")
            
#             # 2. 加载tokenizer
#             logger.info("步骤2: 加载tokenizer")
#             _, tokenizer = ModelLoader.load_model_and_tokenizer()
            
#             # 3. 创建输出目录
#             logger.info("步骤3: 创建输出目录")
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             output_dir = f"{data_config.output_dir}/batch_extraction_{timestamp}"
#             os.makedirs(output_dir, exist_ok=True)
            
#             # 4. 分批处理
#             logger.info("步骤4: 分批处理")
#             total_batches = (len(all_data) + self.batch_size - 1) // self.batch_size
#             all_processed_samples = []
            
#             for batch_num in range(1, total_batches + 1):
#                 batch_start = (batch_num - 1) * self.batch_size
#                 batch_end = batch_num * self.batch_size
#                 batch_data = all_data[batch_start:batch_end]
                
#                 logger.info(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_data)} 个样本)")
                
#                 # 4.1 创建批量文件
#                 input_file = self._create_batch_input_file(batch_data, batch_num, output_dir)
                
#                 # 4.2 提交批量任务
#                 batch_id = self._submit_batch_job(input_file, batch_num)
#                 if not batch_id:
#                     logger.error(f"批次 {batch_num} 提交失败，跳过")
#                     continue
                
#                 # 4.3 监控进度
#                 output_file_id = self._monitor_batch_progress(batch_id, batch_num)
#                 if not output_file_id:
#                     logger.error(f"批次 {batch_num} 处理失败，跳过")
#                     continue
                
#                 # 4.4 处理结果
#                 processed_batch = self._process_batch_results(output_file_id, batch_data, batch_num, tokenizer)
                
#                 if processed_batch:
#                     # 4.5 保存批次结果
#                     batch_output_file = os.path.join(output_dir, f"batch_{batch_num}_results.json")
#                     with open(batch_output_file, 'w', encoding='utf-8') as f:
#                         json.dump(processed_batch, f, ensure_ascii=False, indent=2)
                    
#                     all_processed_samples.extend(processed_batch)
                    
#                     # 4.6 清理临时文件
#                     if os.path.exists(input_file):
#                         os.remove(input_file)
                    
#                     logger.info(f"批次 {batch_num} 完成! 累计完成 {len(all_processed_samples)} 个样本")
#                 else:
#                     logger.error(f"批次 {batch_num} 未生成有效结果")
            
#             # 5. 保存所有结果
#             logger.info("步骤5: 保存所有结果")
#             final_output_file = os.path.join(output_dir, f"all_extracted_samples.json")
#             with open(final_output_file, 'w', encoding='utf-8') as f:
#                 json.dump(all_processed_samples, f, ensure_ascii=False, indent=2)
            
#             # 6. 分析提取结果
#             logger.info("步骤6: 分析提取结果")
#             self._analyze_results(all_processed_samples, output_dir)
            
#             # 7. 打印总结
#             total_time = time.time() - start_time
#             hours = int(total_time // 3600)
#             minutes = int((total_time % 3600) // 60)
            
#             logger.info("\n" + "="*60)
#             logger.info(f"批量提取完成!")
#             logger.info(f"总样本数: {len(all_data)}")
#             logger.info(f"成功提取: {len(all_processed_samples)}")
#             logger.info(f"总处理时间: {hours}小时{minutes}分钟")
#             logger.info(f"输出文件: {final_output_file}")
#             logger.info("="*60)
            
#             return all_processed_samples
            
#         except Exception as e:
#             logger.error(f"批量提取失败: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#             raise
    
#     def _analyze_results(self, processed_samples: List[Dict[str, Any]], output_dir: str):
#         """分析提取结果"""
#         logger.info("分析提取结果...")
        
#         total_samples = len(processed_samples)
        
#         # 统计提取成功率
#         successful_extractions = []
#         failed_extractions = []
        
#         for sample in processed_samples:
#             token_indices = sample.get('answer_token_indices', [])
#             confidence = sample.get('confidence', 0)
            
#             if token_indices and confidence > 0.5:
#                 successful_extractions.append(sample)
#             else:
#                 failed_extractions.append(sample)
        
#         # 统计token数量分布
#         token_counts = [len(s.get('answer_token_indices', [])) for s in successful_extractions]
        
#         stats = {
#             'total_samples': total_samples,
#             'successful_extractions': len(successful_extractions),
#             'failed_extractions': len(failed_extractions),
#             'success_rate': len(successful_extractions) / total_samples if total_samples > 0 else 0,
#             'avg_token_count': np.mean(token_counts) if token_counts else 0,
#             'min_token_count': min(token_counts) if token_counts else 0,
#             'max_token_count': max(token_counts) if token_counts else 0,
#             'avg_confidence': np.mean([s.get('confidence', 0) for s in successful_extractions]) if successful_extractions else 0,
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         # 保存统计信息
#         stats_file = os.path.join(output_dir, f"extraction_stats.json")
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             json.dump(stats, f, ensure_ascii=False, indent=2)
        
#         # 打印统计信息
#         logger.info(f"\n提取结果统计:")
#         logger.info(f"  总样本数: {stats['total_samples']}")
#         logger.info(f"  成功提取: {stats['successful_extractions']} ({stats['success_rate']:.2%})")
#         logger.info(f"  失败提取: {stats['failed_extractions']}")
#         logger.info(f"  平均答案词元数: {stats['avg_token_count']:.2f}")
#         logger.info(f"  平均置信度: {stats['avg_confidence']:.3f}")
        
#         # 保存失败样本（如果有）
#         if failed_extractions:
#             failed_file = os.path.join(output_dir, f"failed_samples.json")
#             with open(failed_file, 'w', encoding='utf-8') as f:
#                 json.dump(failed_extractions, f, ensure_ascii=False, indent=2)
#             logger.info(f"失败样本保存到: {failed_file}")


# def main():
#     """主入口函数"""
#     # 配置日志
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(f"/mnt/project/H-Neurous/logs/dashscope_batch_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
#             logging.StreamHandler()
#         ]
#     )
    
#     # 创建提取器并执行批量提取
#     extractor = DashscopeBatchExtractor()
#     extractor.batch_extract()


# if __name__ == "__main__":
#     main()



# """
# 阿里云百炼API答案词元批量提取器（独立完整版）
# 直接使用批量API进行高效提取 - 所有配置内嵌
# """

# import json
# import time
# import hashlib
# import logging
# import os
# import re
# import sys
# from typing import Dict, List, Optional, Any
# from datetime import datetime
# import numpy as np
# from dataclasses import dataclass, field
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch

# from openai import OpenAI

# # =============== 配置部分 ===============

# @dataclass
# class ModelConfig:
#     """模型配置"""
#     # 阿里云百炼API配置
#     dashscope_api_key: str = "__SET_VIA_ENVIRONMENT__"
#     dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#     dashscope_model: str = "deepseek-v3"
    
#     # 本地模型路径（用于tokenization）
#     local_model_path: str = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    
#     # 4bit量化配置
#     use_4bit: bool = True
#     device_map: str = "auto"
#     torch_dtype: str = "bfloat16"
#     max_seq_length: int = 1024

# @dataclass
# class DataConfig:
#     """数据配置"""
#     # 数据文件路径
#     faithful_data_path: str = "/mnt/project/H-Neurous/data/faithful.json"
#     hallucination_data_path: str = "/mnt/project/H-Neurous/data/hallucination.json"
    
#     # 输出目录
#     output_dir: str = "/mnt/project/H-Neurous/output"
    
#     def __post_init__(self):
#         """创建目录"""
#         os.makedirs(self.output_dir, exist_ok=True)

# @dataclass
# class APIConfig:
#     """API配置"""
#     # 缓存目录
#     location_cache_dir: str = "/mnt/project/H-Neurous/cache/locations"
    
#     # API调用参数
#     max_retries: int = 3
#     timeout: int = 30
#     temperature: float = 0.1
    
#     def __post_init__(self):
#         """创建目录"""
#         os.makedirs(self.location_cache_dir, exist_ok=True)

# # 创建配置实例
# model_config = ModelConfig()
# data_config = DataConfig()
# api_config = APIConfig()

# # =============== 模型加载器 ===============

# class ModelLoader:
#     """模型加载器"""
    
#     @staticmethod
#     def load_model_and_tokenizer():
#         """加载模型和分词器"""
#         logger.info(f"加载模型: {model_config.local_model_path}")
        
#         try:
#             # 配置量化
#             bnb_config = None
#             if model_config.use_4bit:
#                 bnb_config = BitsAndBytesConfig(
#                     load_in_4bit=True,
#                     bnb_4bit_compute_dtype=torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16,
#                     bnb_4bit_use_double_quant=True,
#                     bnb_4bit_quant_type="nf4"
#                 )
            
#             # 加载分词器
#             logger.info("加载分词器...")
#             tokenizer = AutoTokenizer.from_pretrained(
#                 model_config.local_model_path,
#                 trust_remote_code=True,
#                 use_fast=True
#             )
            
#             # 设置padding token
#             if tokenizer.pad_token is None:
#                 tokenizer.pad_token = tokenizer.eos_token
            
#             # 加载模型
#             logger.info("加载模型...")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_config.local_model_path,
#                 quantization_config=bnb_config,
#                 device_map=model_config.device_map,
#                 torch_dtype=torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16,
#                 trust_remote_code=True
#             )
            
#             # 设置为评估模式
#             model.eval()
            
#             logger.info(f"模型加载完成，设备: {model.device}")
            
#             return model, tokenizer
            
#         except Exception as e:
#             logger.error(f"加载模型失败: {e}")
#             # 创建简单分词器作为后备
#             logger.warning("使用简单分词器作为后备方案")
            
#             class SimpleTokenizer:
#                 def encode(self, text, add_special_tokens=False):
#                     return list(text.encode('utf-8'))
                
#                 def encode_plus(self, text, return_offsets_mapping=False, add_special_tokens=False):
#                     tokens = self.encode(text, add_special_tokens)
#                     if return_offsets_mapping:
#                         # 简单实现：每个字符一个token
#                         offset_mapping = [(i, i+1) for i in range(len(text))]
#                         return {'input_ids': tokens, 'offset_mapping': offset_mapping}
#                     return {'input_ids': tokens}
            
#             return None, SimpleTokenizer()

# # =============== 主提取器类 ===============

# logger = logging.getLogger(__name__)

# class DashscopeBatchExtractor:
#     """阿里云百炼API批量答案词元提取器"""
    
#     def __init__(self):
#         """初始化API客户端"""
#         logger.info("初始化阿里云百炼API批量提取器...")
        
#         self.client = OpenAI(
#             api_key=model_config.dashscope_api_key,
#             base_url=model_config.dashscope_base_url
#         )
        
#         self.model_name = model_config.dashscope_model
        
#         # 批量处理配置
#         self.batch_size = 1000  # 每批处理的样本数
#         self.max_wait_hours = 72  # 最大等待时间
        
#         # 确保缓存目录存在
#         os.makedirs(api_config.location_cache_dir, exist_ok=True)
        
#         logger.info(f"API客户端初始化完成，模型: {self.model_name}")
#         logger.info(f"批量配置: 批次大小={self.batch_size}, 最大等待时间={self.max_wait_hours}小时")
    
#     def _generate_cache_key(self, question: str, model_answer: str, standard_answer: str = None) -> str:
#         """生成缓存键"""
#         content = f"dashscope::{question}|||{model_answer}|||{standard_answer}"
#         return hashlib.md5(content.encode('utf-8')).hexdigest()
    
#     def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
#         """获取缓存的提取结果"""
#         cache_file = os.path.join(api_config.location_cache_dir, f"{cache_key}.json")
        
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, 'r', encoding='utf-8') as f:
#                     cached_data = json.load(f)
                
#                 # 检查缓存是否过期（7天）
#                 if time.time() - cached_data.get('timestamp', 0) < 7 * 24 * 3600:
#                     logger.debug(f"使用缓存提取结果: {cache_key}")
#                     return cached_data
#             except Exception as e:
#                 logger.warning(f"读取缓存失败: {e}")
        
#         return None
    
#     def _save_cached_result(self, cache_key: str, result: Dict):
#         """保存提取结果到缓存"""
#         cache_file = os.path.join(api_config.location_cache_dir, f"{cache_key}.json")
        
#         try:
#             result['timestamp'] = time.time()
            
#             with open(cache_file, 'w', encoding='utf-8') as f:
#                 json.dump(result, f, ensure_ascii=False, indent=2)
            
#             logger.debug(f"保存提取结果到缓存: {cache_file}")
#         except Exception as e:
#             logger.warning(f"保存缓存失败: {e}")
    
#     def _build_extraction_prompt(self, question: str, model_answer: str, standard_answer: Optional[str] = None) -> str:
#         """构建答案提取提示词"""
#         prompt = f"""请分析以下对话，并从模型回答中精确提取直接回答问题的部分。

# 问题：{question}

# 模型回答：{model_answer}

# {f"参考答案（用于参考，但不是必须完全匹配）：{standard_answer}" if standard_answer else ""}

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
#   "end_char": 答案在模型回答中的结束字符位置,
#   "confidence": 提取置信度（0-1之间的小数，表示你有多确定这是正确答案）,
#   "reason": "简要说明为什么选择这部分作为答案"
# }}

# 2. 字符位置必须精确，不能有偏差
# 3. 如果模型回答中没有明确的答案，将answer_text设为空字符串，confidence设为0
# 4. 只输出JSON，不要有其他任何内容"""

#         return prompt
    
#     def _parse_extraction_response(self, response_text: str) -> Optional[Dict]:
#         """解析提取响应"""
#         try:
#             # 提取JSON部分
#             json_pattern = r'\{.*\}'
#             match = re.search(json_pattern, response_text, re.DOTALL)
            
#             if not match:
#                 logger.warning(f"响应中没有找到JSON: {response_text[:200]}...")
#                 return None
            
#             json_str = match.group()
#             result = json.loads(json_str)
            
#             # 验证必需字段
#             required_fields = ['answer_text', 'start_char', 'end_char', 'confidence']
#             for field in required_fields:
#                 if field not in result:
#                     logger.warning(f"响应缺少必需字段: {field}")
#                     return None
            
#             # 验证数据类型
#             try:
#                 result['start_char'] = int(result['start_char'])
#                 result['end_char'] = int(result['end_char'])
#                 result['confidence'] = float(result['confidence'])
#             except (ValueError, TypeError) as e:
#                 logger.warning(f"响应字段类型错误: {e}")
#                 return None
            
#             # 验证位置有效性
#             if result['start_char'] < 0 or result['end_char'] <= result['start_char']:
#                 logger.warning(f"响应位置无效: start={result['start_char']}, end={result['end_char']}")
#                 return None
            
#             return result
            
#         except json.JSONDecodeError as e:
#             logger.error(f"解析响应JSON失败: {e}, 响应: {response_text[:200]}...")
#             return None
#         except Exception as e:
#             logger.error(f"解析响应失败: {e}")
#             return None
    
#     def _convert_char_to_token_indices(self, text: str, start_char: int, end_char: int, 
#                                      tokenizer) -> List[int]:
#         """将字符位置转换为token索引"""
#         try:
#             # 对整个文本进行tokenization
#             encoding = tokenizer.encode_plus(
#                 text,
#                 return_offsets_mapping=True,
#                 add_special_tokens=False
#             )
            
#             tokens = encoding['input_ids']
#             offset_mapping = encoding['offset_mapping']
            
#             # 找到答案区域内的token索引
#             answer_token_indices = []
            
#             for idx, (char_start, char_end) in enumerate(offset_mapping):
#                 # 检查token是否与答案区域有重叠
#                 if not (char_end <= start_char or char_start >= end_char):
#                     answer_token_indices.append(idx)
            
#             # 如果找不到匹配的token，返回整个回答的token
#             if not answer_token_indices:
#                 logger.warning(f"无法找到匹配的token，返回所有token")
#                 return list(range(len(tokens)))
            
#             return answer_token_indices
            
#         except Exception as e:
#             logger.error(f"字符位置转token索引失败: {e}")
#             # 返回整个回答的token作为后备
#             try:
#                 full_tokens = tokenizer.encode(text, add_special_tokens=False)
#                 return list(range(len(full_tokens)))
#             except:
#                 return []
    
#     def _create_batch_input_file(self, samples_batch: List[Dict[str, Any]], batch_num: int, output_dir: str) -> str:
#         """创建批量提取请求的JSONL文件"""
#         input_file_path = os.path.join(output_dir, f"batch_{batch_num}_input.jsonl")
        
#         with open(input_file_path, 'w', encoding='utf-8') as f:
#             for i, sample in enumerate(samples_batch):
#                 custom_id = f"batch_{batch_num}_sample_{i}"
                
#                 question = sample.get('question', '')
#                 model_answer = sample.get('model_answer', '')
#                 standard_answer = sample.get('standard_answer', '')
                
#                 # 构建提示词
#                 prompt = self._build_extraction_prompt(question, model_answer, standard_answer)
                
#                 batch_request = {
#                     "custom_id": custom_id,
#                     "method": "POST",
#                     "url": "/v1/chat/completions",
#                     "body": {
#                         "model": self.model_name,
#                         "messages": [
#                             {
#                                 "role": "system", 
#                                 "content": "你是一个专业的文本分析助手，负责从回答中精确定位答案部分。严格按照要求的JSON格式返回结果。"
#                             },
#                             {
#                                 "role": "user", 
#                                 "content": prompt
#                             }
#                         ],
#                         "max_tokens": 1000,
#                         "temperature": 0.1
#                     }
#                 }
                
#                 f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
        
#         logger.info(f"创建批次 {batch_num} 输入文件: {len(samples_batch)} 个请求")
#         return input_file_path
    
#     def _submit_batch_job(self, input_file_path: str, batch_num: int) -> Optional[str]:
#         """提交批量任务"""
#         try:
#             with open(input_file_path, 'rb') as file:
#                 uploaded_file = self.client.files.create(file=file, purpose="batch")
            
#             batch_job = self.client.batches.create(
#                 input_file_id=uploaded_file.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window=f"{self.max_wait_hours}h"
#             )
            
#             logger.info(f"批次 {batch_num} 提交成功，任务ID: {batch_job.id}")
#             return batch_job.id
            
#         except Exception as e:
#             logger.error(f"批次 {batch_num} 提交失败: {e}")
#             return None
    
#     def _monitor_batch_progress(self, batch_id: str, batch_num: int) -> Optional[str]:
#         """监控批量任务进度"""
#         logger.info(f"监控批次 {batch_num} 进度: {batch_id}")
        
#         start_time = time.time()
#         last_status = None
        
#         while True:
#             try:
#                 batch_status = self.client.batches.retrieve(batch_id)
#                 current_status = batch_status.status
                
#                 if current_status != last_status:
#                     elapsed_hours = (time.time() - start_time) / 3600
#                     logger.info(f"批次 {batch_num} 状态: {current_status} (已等待 {elapsed_hours:.1f} 小时)")
#                     last_status = current_status
                
#                 if current_status == "completed":
#                     logger.info(f"批次 {batch_num} 处理完成!")
#                     return batch_status.output_file_id
                    
#                 elif current_status in ["failed", "expired", "cancelled"]:
#                     logger.error(f"批次 {batch_num} 失败: {current_status}")
#                     return None
                    
#                 else:
#                     # 还在处理中，等待5分钟后再次检查
#                     time.sleep(300)
                    
#             except Exception as e:
#                 logger.warning(f"检查批次 {batch_num} 状态失败: {e}")
#                 time.sleep(300)
    
#     def _process_batch_results(self, output_file_id: str, original_batch: List[Dict[str, Any]], 
#                               batch_num: int, tokenizer) -> List[Dict[str, Any]]:
#         """处理批量结果"""
#         try:
#             # 下载结果
#             result_content = self.client.files.content(output_file_id)
#             results = result_content.text.split('\n')
            
#             processed_samples = []
#             success_count = 0
#             error_count = 0
            
#             logger.info(f"下载批次 {batch_num} 结果: {len(results)} 行，开始解析...")
            
#             # 构建映射表
#             sample_mapping = {}
#             for i, sample in enumerate(original_batch):
#                 custom_id = f"batch_{batch_num}_sample_{i}"
#                 sample_mapping[custom_id] = sample
            
#             # 解析每一行结果
#             for line in results:
#                 if not line.strip():
#                     continue
                    
#                 try:
#                     result_obj = json.loads(line)
#                     custom_id = result_obj.get("custom_id")
#                     response = result_obj.get("response", {})
                    
#                     if response.get("status_code") == 200:
#                         body = response.get("body", {})
#                         choices = body.get("choices", [])
                        
#                         if choices:
#                             content = choices[0].get("message", {}).get("content", "")
                            
#                             # 提取JSON
#                             json_match = re.search(r'\{.*\}', content, re.DOTALL)
#                             if json_match:
#                                 try:
#                                     extraction_result = json.loads(json_match.group(0))
                                    
#                                     if custom_id in sample_mapping:
#                                         original_sample = sample_mapping[custom_id]
                                        
#                                         # 处理提取结果
#                                         question = original_sample.get('question', '')
#                                         model_answer = original_sample.get('model_answer', '')
#                                         answer_text = extraction_result.get('answer_text', '')
#                                         start_char = extraction_result.get('start_char', 0)
#                                         end_char = extraction_result.get('end_char', 0)
#                                         confidence = extraction_result.get('confidence', 0.0)
                                        
#                                         # 验证并修正位置
#                                         if answer_text and answer_text in model_answer:
#                                             actual_start = model_answer.find(answer_text)
#                                             if actual_start != -1 and actual_start != start_char:
#                                                 start_char = actual_start
#                                                 end_char = actual_start + len(answer_text)
                                        
#                                         # 转换为token索引
#                                         answer_token_indices = []
#                                         if tokenizer and answer_text:
#                                             answer_token_indices = self._convert_char_to_token_indices(
#                                                 model_answer, start_char, end_char, tokenizer
#                                             )
                                        
#                                         # 构建处理后的样本
#                                         processed_sample = original_sample.copy()
#                                         processed_sample.update({
#                                             'answer_text': answer_text,
#                                             'answer_token_indices': answer_token_indices,
#                                             'answer_token_count': len(answer_token_indices),
#                                             'start_char': start_char,
#                                             'end_char': end_char,
#                                             'confidence': confidence,
#                                             'method': 'dashscope_batch_api',
#                                             'extraction_time': time.time(),
#                                             'batch_id': batch_num,
#                                             'custom_id': custom_id
#                                         })
                                        
#                                         # 保存到缓存
#                                         cache_key = self._generate_cache_key(question, model_answer, 
#                                                                            original_sample.get('standard_answer'))
#                                         self._save_cached_result(cache_key, {
#                                             'answer_text': answer_text,
#                                             'start_char': start_char,
#                                             'end_char': end_char,
#                                             'confidence': confidence,
#                                             'method': 'dashscope_batch_api'
#                                         })
                                        
#                                         processed_samples.append(processed_sample)
#                                         success_count += 1
                                    
#                                 except json.JSONDecodeError:
#                                     error_count += 1
#                             else:
#                                 error_count += 1
#                         else:
#                             error_count += 1
#                     else:
#                         error_count += 1
                        
#                 except Exception:
#                     error_count += 1
            
#             logger.info(f"批次 {batch_num} 解析完成: 成功 {success_count}, 失败 {error_count}")
#             return processed_samples
            
#         except Exception as e:
#             logger.error(f"下载或解析批次 {batch_num} 结果失败: {e}")
#             return []
    
#     def _load_data(self) -> List[Dict[str, Any]]:
#         """加载数据"""
#         logger.info("加载数据...")
        
#         # 直接指定数据路径
#         faithful_path = data_config.faithful_data_path
#         hallucination_path = data_config.hallucination_data_path
        
#         logger.info(f"忠实数据路径: {faithful_path}")
#         logger.info(f"幻觉数据路径: {hallucination_path}")
        
#         # 检查文件是否存在
#         if not os.path.exists(faithful_path):
#             logger.error(f"忠实数据文件不存在: {faithful_path}")
#             raise FileNotFoundError(f"忠实数据文件不存在: {faithful_path}")
        
#         if not os.path.exists(hallucination_path):
#             logger.error(f"幻觉数据文件不存在: {hallucination_path}")
#             raise FileNotFoundError(f"幻觉数据文件不存在: {hallucination_path}")
        
#         # 加载忠实数据
#         with open(faithful_path, 'r', encoding='utf-8') as f:
#             faithful_data = json.load(f)
        
#         # 加载幻觉数据
#         with open(hallucination_path, 'r', encoding='utf-8') as f:
#             hallucination_data = json.load(f)
        
#         logger.info(f"忠实样本数: {len(faithful_data)}")
#         logger.info(f"幻觉样本数: {len(hallucination_data)}")
        
#         # 添加标签
#         for item in faithful_data:
#             item['label'] = 0  # 0表示忠实
#             if 'standard_answer' not in item:
#                 item['standard_answer'] = item.get('answer', '')
        
#         for item in hallucination_data:
#             item['label'] = 1  # 1表示幻觉
#             if 'standard_answer' not in item:
#                 item['standard_answer'] = item.get('answer', '')
        
#         # 合并数据
#         all_data = faithful_data + hallucination_data
        
#         logger.info(f"总样本数: {len(all_data)}")
        
#         return all_data
    
#     def batch_extract(self):
#         """批量提取主函数"""
#         logger.info("="*60)
#         logger.info("开始阿里云百炼API答案词元批量提取")
#         logger.info("="*60)
        
#         start_time = time.time()
        
#         try:
#             # 1. 加载数据
#             logger.info("步骤1: 加载数据")
#             all_data = self._load_data()
            
#             # 限制样本数量（如果需要）
#             max_samples = None  # 设为None表示不限制
#             if max_samples and max_samples < len(all_data):
#                 all_data = all_data[:max_samples]
#                 logger.info(f"限制样本数量为: {max_samples}")
            
#             # 2. 加载tokenizer
#             logger.info("步骤2: 加载tokenizer")
#             _, tokenizer = ModelLoader.load_model_and_tokenizer()
            
#             # 3. 创建输出目录
#             logger.info("步骤3: 创建输出目录")
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             output_dir = f"{data_config.output_dir}/batch_extraction_{timestamp}"
#             os.makedirs(output_dir, exist_ok=True)
            
#             # 4. 分批处理
#             logger.info("步骤4: 分批处理")
#             total_batches = (len(all_data) + self.batch_size - 1) // self.batch_size
#             all_processed_samples = []
            
#             for batch_num in range(1, total_batches + 1):
#                 batch_start = (batch_num - 1) * self.batch_size
#                 batch_end = batch_num * self.batch_size
#                 batch_data = all_data[batch_start:batch_end]
                
#                 logger.info(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_data)} 个样本)")
                
#                 # 4.1 创建批量文件
#                 input_file = self._create_batch_input_file(batch_data, batch_num, output_dir)
                
#                 # 4.2 提交批量任务
#                 batch_id = self._submit_batch_job(input_file, batch_num)
#                 if not batch_id:
#                     logger.error(f"批次 {batch_num} 提交失败，跳过")
#                     continue
                
#                 # 4.3 监控进度
#                 output_file_id = self._monitor_batch_progress(batch_id, batch_num)
#                 if not output_file_id:
#                     logger.error(f"批次 {batch_num} 处理失败，跳过")
#                     continue
                
#                 # 4.4 处理结果
#                 processed_batch = self._process_batch_results(output_file_id, batch_data, batch_num, tokenizer)
                
#                 if processed_batch:
#                     # 4.5 保存批次结果
#                     batch_output_file = os.path.join(output_dir, f"batch_{batch_num}_results.json")
#                     with open(batch_output_file, 'w', encoding='utf-8') as f:
#                         json.dump(processed_batch, f, ensure_ascii=False, indent=2)
                    
#                     all_processed_samples.extend(processed_batch)
                    
#                     # 4.6 清理临时文件
#                     if os.path.exists(input_file):
#                         os.remove(input_file)
                    
#                     logger.info(f"批次 {batch_num} 完成! 累计完成 {len(all_processed_samples)} 个样本")
#                 else:
#                     logger.error(f"批次 {batch_num} 未生成有效结果")
            
#             # 5. 保存所有结果
#             logger.info("步骤5: 保存所有结果")
#             final_output_file = os.path.join(output_dir, f"all_extracted_samples.json")
#             with open(final_output_file, 'w', encoding='utf-8') as f:
#                 json.dump(all_processed_samples, f, ensure_ascii=False, indent=2)
            
#             # 6. 分析提取结果
#             logger.info("步骤6: 分析提取结果")
#             self._analyze_results(all_processed_samples, output_dir)
            
#             # 7. 打印总结
#             total_time = time.time() - start_time
#             hours = int(total_time // 3600)
#             minutes = int((total_time % 3600) // 60)
            
#             logger.info("\n" + "="*60)
#             logger.info(f"批量提取完成!")
#             logger.info(f"总样本数: {len(all_data)}")
#             logger.info(f"成功提取: {len(all_processed_samples)}")
#             logger.info(f"总处理时间: {hours}小时{minutes}分钟")
#             logger.info(f"输出文件: {final_output_file}")
#             logger.info("="*60)
            
#             return all_processed_samples
            
#         except Exception as e:
#             logger.error(f"批量提取失败: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#             raise
    
#     def _analyze_results(self, processed_samples: List[Dict[str, Any]], output_dir: str):
#         """分析提取结果"""
#         logger.info("分析提取结果...")
        
#         total_samples = len(processed_samples)
        
#         if total_samples == 0:
#             logger.warning("没有处理成功的样本")
#             return
        
#         # 统计提取成功率
#         successful_extractions = []
#         failed_extractions = []
        
#         for sample in processed_samples:
#             token_indices = sample.get('answer_token_indices', [])
#             confidence = sample.get('confidence', 0)
            
#             if token_indices and confidence > 0.5:
#                 successful_extractions.append(sample)
#             else:
#                 failed_extractions.append(sample)
        
#         # 统计token数量分布
#         token_counts = []
#         for s in successful_extractions:
#             token_count = len(s.get('answer_token_indices', []))
#             if token_count > 0:
#                 token_counts.append(token_count)
        
#         # 计算置信度
#         confidences = []
#         for s in successful_extractions:
#             conf = s.get('confidence', 0)
#             if conf > 0:
#                 confidences.append(conf)
        
#         stats = {
#             'total_samples': total_samples,
#             'successful_extractions': len(successful_extractions),
#             'failed_extractions': len(failed_extractions),
#             'success_rate': len(successful_extractions) / total_samples if total_samples > 0 else 0,
#             'avg_token_count': np.mean(token_counts) if token_counts else 0,
#             'min_token_count': min(token_counts) if token_counts else 0,
#             'max_token_count': max(token_counts) if token_counts else 0,
#             'avg_confidence': np.mean(confidences) if confidences else 0,
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         # 保存统计信息
#         stats_file = os.path.join(output_dir, f"extraction_stats.json")
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             json.dump(stats, f, ensure_ascii=False, indent=2)
        
#         # 打印统计信息
#         logger.info(f"\n提取结果统计:")
#         logger.info(f"  总样本数: {stats['total_samples']}")
#         logger.info(f"  成功提取: {stats['successful_extractions']} ({stats['success_rate']:.2%})")
#         logger.info(f"  失败提取: {stats['failed_extractions']}")
#         logger.info(f"  平均答案词元数: {stats['avg_token_count']:.2f}")
#         logger.info(f"  平均置信度: {stats['avg_confidence']:.3f}")
        
#         # 保存失败样本（如果有）
#         if failed_extractions:
#             failed_file = os.path.join(output_dir, f"failed_samples.json")
#             with open(failed_file, 'w', encoding='utf-8') as f:
#                 json.dump(failed_extractions, f, ensure_ascii=False, indent=2)
#             logger.info(f"失败样本保存到: {failed_file}")


# def main():
#     """主入口函数"""
#     # 创建日志目录
#     log_dir = "/mnt/project/H-Neurous/logs"
#     os.makedirs(log_dir, exist_ok=True)
    
#     # 配置日志
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(f"{log_dir}/dashscope_batch_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
#             logging.StreamHandler()
#         ]
#     )
    
#     # 创建提取器并执行批量提取
#     extractor = DashscopeBatchExtractor()
#     extractor.batch_extract()


# if __name__ == "__main__":
#     main()

    
"""
阿里云百炼API答案词元实时提取器
实时调用API进行答案词元提取
"""

import json
import time
import hashlib
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logger = logging.getLogger(__name__)

class DashscopeRealtimeExtractor:
    """阿里云百炼API实时答案词元提取器"""
    
    def __init__(self):
        """初始化API客户端"""
        logger.info("初始化阿里云百炼API实时提取器...")
        
        # 直接指定API配置（根据您的实际配置修改）
        self.api_key = "__SET_VIA_ENVIRONMENT__"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = "deepseek-v3"
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 处理配置
        self.batch_size = 1  # 实时处理，每次1个
        self.max_retries = 3
        self.timeout = 30
        self.temperature = 0.1
        
        # 确保缓存目录存在
        self.cache_dir = "/mnt/project/H-Neurous/cache/locations"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"API客户端初始化完成，模型: {self.model_name}")
    
    def _generate_cache_key(self, question: str, model_answer: str, standard_answer: str = None) -> str:
        """生成缓存键"""
        content = f"dashscope::{question}|||{model_answer}|||{standard_answer}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """获取缓存的提取结果"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 检查缓存是否过期（7天）
                if time.time() - cached_data.get('timestamp', 0) < 7 * 24 * 3600:
                    logger.debug(f"使用缓存提取结果: {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        
        return None
    
    def _save_cached_result(self, cache_key: str, result: Dict):
        """保存提取结果到缓存"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            result['timestamp'] = time.time()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"保存提取结果到缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_dashscope_api(self, prompt: str) -> str:
        """实时调用阿里云百炼API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本分析助手，负责从回答中精确定位答案部分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000,
                timeout=self.timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"阿里云百炼API调用失败: {e}")
            raise
    
    def _build_extraction_prompt(self, question: str, model_answer: str, standard_answer: Optional[str] = None) -> str:
        """构建答案提取提示词"""
        prompt = f"""请分析以下对话，并从模型回答中精确提取直接回答问题的部分。

问题：{question}

模型回答：{model_answer}


请执行以下任务：
1. 仔细分析问题在询问什么
2. 在模型回答中找到直接回答问题的具体内容
3. 确定这部分内容在回答中的精确位置（字符位置）
4. 提取这部分内容

输出要求：
1. 必须使用以下JSON格式：
{{
  "answer_text": "从模型回答中提取的答案文本（必须是原文，不能修改）",
  "start_char": 答案在模型回答中的起始字符位置（从0开始）,
  "end_char": 答案在模型回答中的结束字符位置,
  "confidence": 提取置信度（0-1之间的小数，表示你有多确定这是正确答案）,
  "reason": "简要说明为什么选择这部分作为答案"
}}

2. 字符位置必须精确，不能有偏差
3. 如果模型回答中没有明确的答案，将answer_text设为空字符串，confidence设为0
4. 只输出JSON，不要有其他任何内容

示例：
问题："中国的首都是哪里？"
模型回答："中国的首都是北京，北京是一座历史悠久的城市。"
输出：
{{
  "answer_text": "北京",
  "start_char": 6,
  "end_char": 8,
  "confidence": 0.95,
  "reason": "这是直接回答首都问题的城市名称"
}}
注意：必须是北京，而不是中国的首都是北京"""

        return prompt
    
    def _parse_extraction_response(self, response_text: str) -> Optional[Dict]:
        """解析提取响应"""
        try:
            # 提取JSON部分
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if not match:
                logger.warning(f"响应中没有找到JSON: {response_text[:200]}...")
                return None
            
            json_str = match.group()
            result = json.loads(json_str)
            
            # 验证必需字段
            required_fields = ['answer_text', 'start_char', 'end_char', 'confidence']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"响应缺少必需字段: {field}")
                    return None
            
            # 验证数据类型
            try:
                result['start_char'] = int(result['start_char'])
                result['end_char'] = int(result['end_char'])
                result['confidence'] = float(result['confidence'])
            except (ValueError, TypeError) as e:
                logger.warning(f"响应字段类型错误: {e}")
                return None
            
            # 验证位置有效性
            if result['start_char'] < 0 or result['end_char'] <= result['start_char']:
                logger.warning(f"响应位置无效: start={result['start_char']}, end={result['end_char']}")
                return None
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"解析响应JSON失败: {e}, 响应: {response_text[:200]}...")
            return None
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return None
    
    def _find_best_match(self, text: str, target: str) -> Optional[Dict]:
        """在文本中找到与目标最相似的子串"""
        if not target or not text:
            return None
        
        target_len = len(target)
        best_match = None
        best_score = 0
        
        # 滑动窗口搜索
        for i in range(len(text) - target_len + 1):
            substring = text[i:i+target_len]
            
            # 简单的相似度计算
            if substring == target:
                return {
                    'text': substring,
                    'start': i,
                    'end': i + target_len,
                    'score': 1.0
                }
            
            # 计算字符重叠度
            overlap = sum(1 for a, b in zip(substring, target) if a == b)
            score = overlap / max(len(substring), len(target))
            
            if score > best_score:
                best_score = score
                best_match = {
                    'text': substring,
                    'start': i,
                    'end': i + target_len,
                    'score': score
                }
        
        if best_score > 0.7:  # 相似度阈值
            return best_match
        return None
    
    def extract_answer_position(self, question: str, model_answer: str, 
                              standard_answer: Optional[str] = None) -> Dict:
        """实时提取答案位置"""
        logger.debug(f"提取答案位置: 问题长度={len(question)}, 回答长度={len(model_answer)}")
        
        # 首先检查缓存
        cache_key = self._generate_cache_key(question, model_answer, standard_answer)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result is not None:
            logger.debug(f"使用缓存结果: {cache_key}")
            return cached_result
        
        # 构建提示词
        prompt = self._build_extraction_prompt(question, model_answer, standard_answer)
        
        try:
            # 调用API
            response = self._call_dashscope_api(prompt)
            
            # 解析响应
            result = self._parse_extraction_response(response)
            
            if result:
                # 验证提取的文本是否确实在模型回答中
                answer_text = result['answer_text']
                if answer_text and answer_text in model_answer:
                    # 验证位置是否匹配
                    actual_start = model_answer.find(answer_text)
                    if actual_start != -1 and actual_start != result['start_char']:
                        logger.warning(f"API返回的位置不准确，修正: {result['start_char']} -> {actual_start}")
                        result['start_char'] = actual_start
                        result['end_char'] = actual_start + len(answer_text)
                else:
                    # 如果文本不在回答中，尝试查找最相似的子串
                    if answer_text:
                        best_match = self._find_best_match(model_answer, answer_text)
                        if best_match:
                            result['answer_text'] = best_match['text']
                            result['start_char'] = best_match['start']
                            result['end_char'] = best_match['end']
                            result['confidence'] *= 0.8  # 降低置信度
                            logger.info(f"修正答案文本位置: {best_match['text'][:50]}...")
                        else:
                            logger.warning(f"API提取的答案文本不在原回答中: {answer_text[:50]}...")
                            # 清空无效结果
                            result['answer_text'] = ""
                            result['confidence'] = 0.0
                
                result['method'] = 'dashscope_realtime_api'
                logger.debug(f"API提取成功: {result['answer_text'][:50]}..., 置信度: {result['confidence']:.3f}")
                
                # 保存到缓存
                self._save_cached_result(cache_key, result)
                return result
            else:
                logger.warning("API解析失败，返回空结果")
                return {
                    'answer_text': '',
                    'start_char': 0,
                    'end_char': 0,
                    'confidence': 0.0,
                    'method': 'failed'
                }
                
        except Exception as e:
            logger.error(f"API提取失败: {e}")
            return {
                'answer_text': '',
                'start_char': 0,
                'end_char': 0,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def _convert_char_to_token_indices(self, text: str, start_char: int, end_char: int, 
                                     tokenizer) -> List[int]:
        """将字符位置转换为token索引"""
        try:
            # 对整个文本进行tokenization
            encoding = tokenizer.encode_plus(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            
            tokens = encoding['input_ids']
            offset_mapping = encoding['offset_mapping']
            
            # 找到答案区域内的token索引
            answer_token_indices = []
            
            for idx, (char_start, char_end) in enumerate(offset_mapping):
                # 检查token是否与答案区域有重叠
                if not (char_end <= start_char or char_start >= end_char):
                    answer_token_indices.append(idx)
            
            # 如果找不到匹配的token，返回整个回答的token
            if not answer_token_indices:
                logger.warning(f"无法找到匹配的token，返回所有token")
                return list(range(len(tokens)))
            
            return answer_token_indices
            
        except Exception as e:
            logger.error(f"字符位置转token索引失败: {e}")
            # 返回整个回答的token作为后备
            try:
                full_tokens = tokenizer.encode(text, add_special_tokens=False)
                return list(range(len(full_tokens)))
            except:
                return []
    
    def extract_answer_tokens(self, sample: Dict, tokenizer) -> Dict:
        """提取单个样本的答案词元"""
        question = sample.get('question', '')
        model_answer = sample.get('model_answer', '')
        standard_answer = sample.get('standard_answer', '')
        label = sample.get('label', 0)
        
        # 提取答案位置
        position_result = self.extract_answer_position(question, model_answer, standard_answer)
        
        answer_text = position_result['answer_text']
        start_char = position_result['start_char']
        end_char = position_result['end_char']
        confidence = position_result['confidence']
        
        # 将字符位置转换为token索引
        answer_token_indices = []
        if answer_text:
            answer_token_indices = self._convert_char_to_token_indices(
                model_answer, start_char, end_char, tokenizer
            )
        
        # 构建结果
        result = sample.copy()
        result.update({
            'answer_text': answer_text,
            'answer_token_indices': answer_token_indices,
            'answer_token_count': len(answer_token_indices),
            'start_char': start_char,
            'end_char': end_char,
            'confidence': confidence,
            'method': position_result.get('method', 'dashscope_realtime_api'),
            'extraction_time': time.time()
        })
        
        return result
    
    def load_data(self, input_file: str) -> List[Dict]:
        """加载输入数据"""
        logger.info(f"加载输入数据: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"加载 {len(data)} 个样本")
        
        # 验证数据格式
        required_fields = ['question', 'model_answer', 'label']
        for i, sample in enumerate(data[:5]):
            for field in required_fields:
                if field not in sample:
                    logger.warning(f"样本 {i} 缺少字段: {field}")
        
        # 统计类别分布
        label_counts = {}
        for sample in data:
            label = sample.get('label', 0)
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label, count in label_counts.items():
            logger.info(f"标签 {label} (0=忠实,1=幻觉): {count} 个样本")
        
        return data
    
    def realtime_extract(self, input_file: str, output_file: str):
        """实时提取主函数"""
        logger.info("="*60)
        logger.info("开始阿里云百炼API答案词元实时提取")
        logger.info("="*60)
        logger.info(f"输入文件: {input_file}")
        logger.info(f"输出文件: {output_file}")
        
        start_time = time.time()
        
        try:
            # 1. 加载数据
            logger.info("步骤1: 加载数据")
            all_data = self.load_data(input_file)
            
            # 2. 加载tokenizer
            logger.info("步骤2: 加载tokenizer")
            try:
                from transformers import AutoTokenizer
                # 使用您项目中已有的模型路径
                model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
                
                # 设置padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                logger.info("tokenizer加载成功")
            except Exception as e:
                logger.error(f"加载tokenizer失败: {e}")
                # 创建一个简单的tokenizer备用
                class SimpleTokenizer:
                    def encode(self, text, add_special_tokens=False):
                        return list(text.encode('utf-8'))
                    
                    def encode_plus(self, text, return_offsets_mapping=False, add_special_tokens=False):
                        tokens = self.encode(text, add_special_tokens)
                        if return_offsets_mapping:
                            # 简单实现：每个字符一个token
                            offset_mapping = [(i, i+1) for i in range(len(text))]
                            return {'input_ids': tokens, 'offset_mapping': offset_mapping}
                        return {'input_ids': tokens}
                
                tokenizer = SimpleTokenizer()
                logger.warning("使用简单tokenizer，字符位置转换可能不准确")
            
            # 3. 实时处理每个样本
            logger.info("步骤3: 实时处理样本")
            processed_samples = []
            success_count = 0
            failed_count = 0
            
            # 使用tqdm显示进度
            for i, sample in enumerate(tqdm(all_data, desc="提取答案词元")):
                try:
                    # 提取答案词元
                    processed_sample = self.extract_answer_tokens(sample, tokenizer)
                    processed_samples.append(processed_sample)
                    success_count += 1
                    
                    # 定期显示进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i+1}/{len(all_data)} 个样本")
                    
                    # 避免API调用过于频繁
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"处理样本 {i} 失败: {e}")
                    # 保存原始样本作为失败记录
                    sample['extraction_error'] = str(e)
                    processed_samples.append(sample)
                    failed_count += 1
            
            # 4. 保存结果
            logger.info("步骤4: 保存结果")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_samples, f, ensure_ascii=False, indent=2)
            
            # 5. 分析提取结果
            logger.info("步骤5: 分析提取结果")
            self._analyze_results(processed_samples, output_file)
            
            # 6. 打印总结
            total_time = time.time() - start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            logger.info("\n" + "="*60)
            logger.info(f"实时提取完成!")
            logger.info(f"总样本数: {len(all_data)}")
            logger.info(f"成功提取: {success_count}")
            logger.info(f"失败提取: {failed_count}")
            logger.info(f"总处理时间: {hours}小时{minutes}分钟{seconds}秒")
            logger.info(f"输出文件: {output_file}")
            logger.info("="*60)
            
            return processed_samples
            
        except Exception as e:
            logger.error(f"实时提取失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _analyze_results(self, processed_samples: List[Dict], output_file: str):
        """分析提取结果"""
        logger.info("分析提取结果...")
        
        total_samples = len(processed_samples)
        
        # 统计提取成功率
        successful_extractions = []
        failed_extractions = []
        
        for sample in processed_samples:
            token_indices = sample.get('answer_token_indices', [])
            confidence = sample.get('confidence', 0)
            
            if token_indices and confidence > 0.5:
                successful_extractions.append(sample)
            else:
                failed_extractions.append(sample)
        
        # 统计token数量分布
        token_counts = [len(s.get('answer_token_indices', [])) for s in successful_extractions]
        
        stats = {
            'total_samples': total_samples,
            'successful_extractions': len(successful_extractions),
            'failed_extractions': len(failed_extractions),
            'success_rate': len(successful_extractions) / total_samples if total_samples > 0 else 0,
            'avg_token_count': np.mean(token_counts) if token_counts else 0,
            'min_token_count': min(token_counts) if token_counts else 0,
            'max_token_count': max(token_counts) if token_counts else 0,
            'avg_confidence': np.mean([s.get('confidence', 0) for s in successful_extractions]) if successful_extractions else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'output_file': output_file
        }
        
        # 保存统计信息
        stats_dir = os.path.dirname(output_file)
        stats_file = os.path.join(stats_dir, f"extraction_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        logger.info(f"\n提取结果统计:")
        logger.info(f"  总样本数: {stats['total_samples']}")
        logger.info(f"  成功提取: {stats['successful_extractions']} ({stats['success_rate']:.2%})")
        logger.info(f"  失败提取: {stats['failed_extractions']}")
        logger.info(f"  平均答案词元数: {stats['avg_token_count']:.2f}")
        logger.info(f"  平均置信度: {stats['avg_confidence']:.3f}")
        
        # 保存失败样本（如果有）
        if failed_extractions:
            failed_file = os.path.join(stats_dir, f"failed_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_extractions, f, ensure_ascii=False, indent=2)
            logger.info(f"失败样本保存到: {failed_file}")


def main():
    """主入口函数"""
    # 配置日志
    log_dir = "/mnt/project/H-Neurous/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/dashscope_realtime_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # 指定输入和输出文件路径
    # 请根据您的实际情况修改这些路径
    input_file = "/mnt/project/H-Neurous/data/add.json"  # 替换为您的输入文件路径
    output_file = "/mnt/project/H-Neurous/data/failed_retry_fixed.json"  # 替换为您的输出文件路径
    
    print("\n" + "="*60)
    print("阿里云百炼API答案词元实时提取器")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("="*60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 创建提取器并执行实时提取
    try:
        extractor = DashscopeRealtimeExtractor()
        result = extractor.realtime_extract(input_file, output_file)
        
        print("\n" + "="*60)
        print("提取完成!")
        print(f"结果保存到: {output_file}")
        print("="*60)
        
        # 打印一些示例
        if result and len(result) > 0:
            print(f"\n前3个样本的提取结果:")
            for i, sample in enumerate(result[:3]):
                question = sample.get('question', '')[:50]
                answer_text = sample.get('answer_text', '')[:50]
                token_count = sample.get('answer_token_count', 0)
                confidence = sample.get('confidence', 0)
                method = sample.get('method', 'unknown')
                print(f"  样本 {i+1}: {question}...")
                print(f"    答案: {answer_text}...")
                print(f"    词元数: {token_count}, 置信度: {confidence:.3f}, 方法: {method}")
                print()
        
    except Exception as e:
        print(f"提取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
