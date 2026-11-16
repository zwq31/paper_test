"""
本地模型答案词元批量提取器（独立完整版）
使用本地模型进行高效提取 - 所有配置内嵌
"""

import json
import time
import os
import re
import sys
import torch
import gc
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LocalModelBatchExtractor:
    """本地模型批量答案词元提取器"""
    
    def __init__(self):
        """初始化本地模型"""
        print("初始化本地模型批量提取器...")
        
        # 配置
        self.local_model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
        
        # 模型配置
        self.use_4bit = True
        self.max_new_tokens = 1024
        self.temperature = 0.1
        self.top_p = 0.9
        self.timeout = 120
        
        # 数据配置
        self.faithful_data_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/faithful.json"
        self.hallucination_data_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/hallucination.json"
        self.output_dir = "/mnt/project/H-Neurous/data/trivia_qa/inference"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 批量处理配置
        self.batch_size = 100  # 每批处理的样本数（减少批量以避免显存不足）
        
        # 加载模型和tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        print(f"本地模型初始化完成，模型路径: {self.local_model_path}")
        print(f"批量配置: 批次大小={self.batch_size}")
    
    def _load_model_and_tokenizer(self):
        """加载本地模型和分词器"""
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("加载模型...")
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.local_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # 设置为评估模式
        model.eval()
        
        print(f"模型加载完成，设备: {model.device}")
        
        return model, tokenizer
    
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
    
    def _call_local_model(self, prompt: str) -> str:
        """调用本地模型生成"""
        try:
            # 构建聊天消息
            chat_msgs = [
                {
                    "role": "system",
                    "content": "你是一个专业的文本分析助手，负责从回答中精确定位答案部分。严格按照要求的JSON格式返回结果。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Tokenize输入
            inputs = self.tokenizer.apply_chat_template(
                chat_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码生成的部分
            generated_tokens = outputs[0][inputs.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # 清理显存
            torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            print(f"本地模型调用失败: {e}")
            # 清理显存
            torch.cuda.empty_cache()
            raise
    
    def _parse_extraction_response(self, response_text: str) -> Optional[Dict]:
        """解析提取响应"""
        try:
            # 提取JSON部分
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if not match:
                print(f"响应中没有找到JSON: {response_text[:200]}...")
                return None
            
            json_str = match.group()
            result = json.loads(json_str)
            
            # 验证必需字段
            required_fields = ['answer_text', 'start_char', 'end_char', 'confidence']
            for field in required_fields:
                if field not in result:
                    print(f"响应缺少必需字段: {field}")
                    return None
            
            # 验证数据类型
            try:
                result['start_char'] = int(result['start_char'])
                result['end_char'] = int(result['end_char'])
                result['confidence'] = float(result['confidence'])
            except (ValueError, TypeError) as e:
                print(f"响应字段类型错误: {e}")
                return None
            
            # 验证位置有效性
            if result['start_char'] < 0 or result['end_char'] <= result['start_char']:
                print(f"响应位置无效: start={result['start_char']}, end={result['end_char']}")
                return None
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"解析响应JSON失败: {e}, 响应: {response_text[:200]}...")
            return None
        except Exception as e:
            print(f"解析响应失败: {e}")
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
    
    def _convert_char_to_token_indices(self, text: str, start_char: int, end_char: int) -> List[int]:
        """将字符位置转换为token索引"""
        try:
            # 对整个文本进行tokenization
            encoding = self.tokenizer.encode_plus(
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
                print(f"无法找到匹配的token，返回所有token")
                full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return list(range(len(full_tokens)))
            
            return answer_token_indices
            
        except Exception as e:
            print(f"字符位置转token索引失败: {e}")
            # 返回整个回答的token作为后备
            try:
                full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return list(range(len(full_tokens)))
            except:
                return []
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        print("加载数据...")
        
        print(f"忠实数据路径: {self.faithful_data_path}")
        print(f"幻觉数据路径: {self.hallucination_data_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.faithful_data_path):
            raise FileNotFoundError(f"忠实数据文件不存在: {self.faithful_data_path}")
        
        if not os.path.exists(self.hallucination_data_path):
            raise FileNotFoundError(f"幻觉数据文件不存在: {self.hallucination_data_path}")
        
        # 加载忠实数据
        with open(self.faithful_data_path, 'r', encoding='utf-8') as f:
            faithful_data = json.load(f)
        
        # 加载幻觉数据
        with open(self.hallucination_data_path, 'r', encoding='utf-8') as f:
            hallucination_data = json.load(f)
        
        print(f"忠实样本数: {len(faithful_data)}")
        print(f"幻觉样本数: {len(hallucination_data)}")
        
        # 添加标签
        for item in faithful_data:
            item['label'] = 0  # 0表示忠实
            if 'standard_answer' not in item:
                item['standard_answer'] = item.get('answer', '')
        
        for item in hallucination_data:
            item['label'] = 1  # 1表示幻觉
            if 'standard_answer' not in item:
                item['standard_answer'] = item.get('answer', '')
        
        # 合并数据
        all_data = faithful_data + hallucination_data
        
        print(f"总样本数: {len(all_data)}")
        
        return all_data
    
    def batch_extract(self):
        """批量提取主函数"""
        print("="*60)
        print("开始本地模型答案词元批量提取")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # 1. 加载数据
            print("步骤1: 加载数据")
            all_data = self._load_data()
            
            # 限制样本数量（如果需要）
            max_samples = None  # 设为None表示不限制
            if max_samples and max_samples < len(all_data):
                all_data = all_data[:max_samples]
                print(f"限制样本数量为: {max_samples}")
            
            # 2. 创建输出目录
            print("步骤2: 创建输出目录")
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_dir = f"{self.output_dir}/batch_extraction_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. 分批处理
            print("步骤3: 分批处理")
            total_batches = (len(all_data) + self.batch_size - 1) // self.batch_size
            all_processed_samples = []
            
            # 流式写入主文件
            final_output_file = os.path.join(output_dir, f"answer_tokens.json")
            
            with open(final_output_file, 'w', encoding='utf-8') as final_f:
                final_f.write("[\n")
                
                for batch_num in range(1, total_batches + 1):
                    batch_start = (batch_num - 1) * self.batch_size
                    batch_end = batch_num * self.batch_size
                    batch_data = all_data[batch_start:batch_end]
                    
                    print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_data)} 个样本)")
                    
                    # 处理批次中的每个样本
                    for i, sample in enumerate(tqdm(batch_data, desc=f"批次 {batch_num}")):
                        try:
                            question = sample.get('question', '')
                            model_answer = sample.get('model_answer', '')
                            standard_answer = sample.get('standard_answer', '')
                            
                            # 构建提示词
                            prompt = self._build_extraction_prompt(question, model_answer, standard_answer)
                            
                            # 调用本地模型
                            response = self._call_local_model(prompt)
                            
                            # 解析响应
                            extraction_result = self._parse_extraction_response(response)
                            
                            if extraction_result:
                                answer_text = extraction_result.get('answer_text', '')
                                start_char = extraction_result.get('start_char', 0)
                                end_char = extraction_result.get('end_char', 0)
                                confidence = extraction_result.get('confidence', 0.0)
                                
                                # 验证并修正位置
                                if answer_text and answer_text in model_answer:
                                    actual_start = model_answer.find(answer_text)
                                    if actual_start != -1 and actual_start != start_char:
                                        print(f"模型返回的位置不准确，修正: {start_char} -> {actual_start}")
                                        start_char = actual_start
                                        end_char = actual_start + len(answer_text)
                                
                                # 转换为token索引
                                answer_token_indices = []
                                if answer_text:
                                    answer_token_indices = self._convert_char_to_token_indices(
                                        model_answer, start_char, end_char
                                    )
                                
                                # 构建处理后的样本
                                processed_sample = sample.copy()
                                processed_sample.update({
                                    'answer_text': answer_text,
                                    'answer_token_indices': answer_token_indices,
                                    'answer_token_count': len(answer_token_indices),
                                    'start_char': start_char,
                                    'end_char': end_char,
                                    'confidence': confidence,
                                    'method': 'local_model_batch',
                                    'extraction_time': time.time(),
                                    'batch_id': batch_num
                                })
                                
                            else:
                                # 模型解析失败
                                processed_sample = sample.copy()
                                processed_sample.update({
                                    'answer_text': '',
                                    'answer_token_indices': [],
                                    'answer_token_count': 0,
                                    'start_char': 0,
                                    'end_char': 0,
                                    'confidence': 0.0,
                                    'method': 'failed',
                                    'extraction_time': time.time(),
                                    'batch_id': batch_num
                                })
                            
                            # 流式写入到主文件
                            if batch_num > 1 or i > 0:
                                final_f.write(",\n")
                            json.dump(processed_sample, final_f, ensure_ascii=False, indent=2)
                            final_f.flush()
                            
                            all_processed_samples.append(processed_sample)
                            
                            # 定期清理显存
                            if (i + 1) % 10 == 0:
                                gc.collect()
                                torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"处理样本失败: {e}")
                            # 保存原始样本作为失败记录
                            sample['extraction_error'] = str(e)
                            all_processed_samples.append(sample)
                            
                            # 流式写入失败样本
                            if batch_num > 1 or i > 0:
                                final_f.write(",\n")
                            json.dump(sample, final_f, ensure_ascii=False, indent=2)
                            final_f.flush()
                    
                    # 每个批次完成后清理显存
                    gc.collect()
                    torch.cuda.empty_cache()
                
                final_f.write("\n]\n")
            
            # 4. 分析提取结果
            print("步骤4: 分析提取结果")
            self._analyze_results(all_processed_samples, output_dir)
            
            # 5. 打印总结
            total_time = time.time() - start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            
            print("\n" + "="*60)
            print(f"批量提取完成!")
            print(f"总样本数: {len(all_data)}")
            print(f"成功处理: {len(all_processed_samples)}")
            print(f"总处理时间: {hours}小时{minutes}分钟")
            print(f"输出文件: {final_output_file}")
            print("="*60)
            
            return all_processed_samples
            
        except Exception as e:
            print(f"批量提取失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _analyze_results(self, processed_samples: List[Dict[str, Any]], output_dir: str):
        """分析提取结果"""
        print("分析提取结果...")
        
        total_samples = len(processed_samples)
        
        if total_samples == 0:
            print("没有处理成功的样本")
            return
        
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
        token_counts = []
        for s in successful_extractions:
            token_count = len(s.get('answer_token_indices', []))
            if token_count > 0:
                token_counts.append(token_count)
        
        # 计算置信度
        confidences = []
        for s in successful_extractions:
            conf = s.get('confidence', 0)
            if conf > 0:
                confidences.append(conf)
        
        stats = {
            'total_samples': total_samples,
            'successful_extractions': len(successful_extractions),
            'failed_extractions': len(failed_extractions),
            'success_rate': len(successful_extractions) / total_samples if total_samples > 0 else 0,
            'avg_token_count': np.mean(token_counts) if token_counts else 0,
            'min_token_count': min(token_counts) if token_counts else 0,
            'max_token_count': max(token_counts) if token_counts else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, f"extraction_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        print(f"\n提取结果统计:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  成功提取: {stats['successful_extractions']} ({stats['success_rate']:.2%})")
        print(f"  失败提取: {stats['failed_extractions']}")
        print(f"  平均答案词元数: {stats['avg_token_count']:.2f}")
        print(f"  平均置信度: {stats['avg_confidence']:.3f}")
        
        # 保存失败样本（如果有）
        if failed_extractions:
            failed_file = os.path.join(output_dir, f"failed_samples.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_extractions, f, ensure_ascii=False, indent=2)
            print(f"失败样本保存到: {failed_file}")


def main():
    """主入口函数"""
    print("\n" + "="*60)
    print("本地模型答案词元批量提取器")
    print("="*60)
    
    # 创建提取器并执行批量提取
    extractor = LocalModelBatchExtractor()
    result = extractor.batch_extract()
    
    print("\n" + "="*60)
    print("提取完成!")
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


if __name__ == "__main__":
    main()