"""
阿里云百炼API答案词元提取器
直接调用DeepSeek-V3提取答案词元位置
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
import pickle
import os
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from configs.config import model_config, api_config

logger = logging.getLogger(__name__)

class DashscopeAnswerExtractor:
    """阿里云百炼API答案词元提取器"""
    
    def __init__(self):
        """初始化API客户端"""
        logger.info("初始化阿里云百炼API客户端...")
        
        self.client = OpenAI(
            api_key=model_config.dashscope_api_key,
            base_url=model_config.dashscope_base_url
        )
        
        self.model_name = model_config.dashscope_model
        
        # 确保缓存目录存在
        os.makedirs(api_config.location_cache_dir, exist_ok=True)
        
        logger.info(f"API客户端初始化完成，使用模型: {self.model_name}")
    
    def _generate_cache_key(self, question: str, model_answer: str, standard_answer: str = None) -> str:
        """生成缓存键"""
        content = f"dashscope::{question}|||{model_answer}|||{standard_answer}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """获取缓存的提取结果"""
        if not api_config.cache_locations:
            return None
        
        cache_file = os.path.join(api_config.location_cache_dir, f"{cache_key}.json")
        
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
        if not api_config.cache_locations:
            return
        
        cache_file = os.path.join(api_config.location_cache_dir, f"{cache_key}.json")
        
        try:
            result['timestamp'] = time.time()
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"保存提取结果到缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    @retry(
        stop=stop_after_attempt(api_config.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_dashscope_api(self, prompt: str) -> str:
        """调用阿里云百炼API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本分析助手，负责从回答中精确定位答案部分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=api_config.temperature,
                max_tokens=1000,
                timeout=api_config.timeout
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

{f"参考答案（用于参考，但不是必须完全匹配）：{standard_answer}" if standard_answer else ""}

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
}}"""

        return prompt
    
    def _parse_api_response(self, response_text: str) -> Optional[Dict]:
        """解析API响应"""
        try:
            # 提取JSON部分
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if not match:
                logger.warning(f"API响应中没有找到JSON: {response_text[:200]}...")
                return None
            
            json_str = match.group()
            result = json.loads(json_str)
            
            # 验证必需字段
            required_fields = ['answer_text', 'start_char', 'end_char', 'confidence']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"API响应缺少必需字段: {field}")
                    return None
            
            # 验证数据类型
            try:
                result['start_char'] = int(result['start_char'])
                result['end_char'] = int(result['end_char'])
                result['confidence'] = float(result['confidence'])
            except (ValueError, TypeError) as e:
                logger.warning(f"API响应字段类型错误: {e}")
                return None
            
            # 验证位置有效性
            if result['start_char'] < 0 or result['end_char'] <= result['start_char']:
                logger.warning(f"API响应位置无效: start={result['start_char']}, end={result['end_char']}")
                return None
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"解析API响应JSON失败: {e}, 响应: {response_text[:200]}...")
            return None
        except Exception as e:
            logger.error(f"解析API响应失败: {e}")
            return None
    
    def extract_answer_position(self, question: str, model_answer: str, 
                              standard_answer: Optional[str] = None) -> Dict:
        """
        使用阿里云百炼API提取答案位置
        
        返回:
            Dict: 包含答案文本、位置和置信度的字典
        """
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
            result = self._parse_api_response(response)
            
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
                result['method'] = 'dashscope_api'
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
            score = self._calculate_similarity(substring, target)
            
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
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度"""
        if not s1 or not s2:
            return 0.0
        
        # 简单的Jaccard相似度
        set1 = set(s1)
        set2 = set(s2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_answer_tokens(self, question: str, model_answer: str, tokenizer,
                            standard_answer: Optional[str] = None) -> Dict:
        """
        提取答案词元
        
        返回:
            Dict: 包含答案词元索引和其他信息的字典
        """
        # 提取答案位置
        position_result = self.extract_answer_position(question, model_answer, standard_answer)
        
        if not position_result or not position_result.get('answer_text'):
            logger.warning(f"无法提取答案位置，返回空结果")
            return {
                'answer_token_indices': [],
                'answer_text': '',
                'confidence': 0.0,
                'method': position_result.get('method', 'unknown'),
                'error': '无法提取答案位置'
            }
        
        answer_text = position_result['answer_text']
        start_char = position_result['start_char']
        end_char = position_result['end_char']
        confidence = position_result['confidence']
        
        # 将字符位置转换为token索引
        answer_token_indices = self._convert_char_to_token_indices(
            model_answer, start_char, end_char, tokenizer
        )
        
        result = {
            'answer_token_indices': answer_token_indices,
            'answer_token_count': len(answer_token_indices),
            'answer_text': answer_text,
            'start_char': start_char,
            'end_char': end_char,
            'confidence': confidence,
            'method': position_result.get('method', 'dashscope_api'),
            'extraction_time': time.time()
        }
        
        logger.debug(f"提取到 {len(answer_token_indices)} 个答案词元，置信度: {confidence:.3f}")
        
        return result
    
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
                # 使用宽松条件：只要有重叠就包含
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
    
    def batch_extract_answer_tokens(self, samples: List[Dict], tokenizer, 
                                   batch_size: int = 10) -> List[Dict]:
        """
        批量提取答案词元
        
        返回:
            List[Dict]: 包含答案词元索引的样本列表
        """
        logger.info(f"批量提取答案词元，共 {len(samples)} 个样本")
        
        processed_samples = []
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"已处理 {i}/{len(samples)} 个样本")
            
            question = sample.get('question', '')
            model_answer = sample.get('model_answer', '')
            standard_answer = sample.get('standard_answer', '')
            label = sample.get('label', 0)
            
            if not question or not model_answer:
                logger.warning(f"样本 {i} 缺少问题或回答")
                continue
            
            # 提取答案词元
            extraction_result = self.extract_answer_tokens(
                question, model_answer, tokenizer, standard_answer
            )
            
            # 创建新的样本
            new_sample = sample.copy()
            new_sample.update(extraction_result)
            new_sample['label'] = label
            
            processed_samples.append(new_sample)
        
        logger.info(f"批量提取完成: {len(processed_samples)}/{len(samples)} 个样本成功")
        
        return processed_samples