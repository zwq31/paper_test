# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import json
# import logging
# from typing import List, Dict, Any
# from tqdm import tqdm
# from openai import OpenAI
# import numpy as np
# import evaluate
# import bert_score

# # 设置日志配置
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='evaluation1.log',
#     filemode='a',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# # 配置参数
# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_yuanshi/test_v2/lou.json"
# OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_yuanshi/test_v2/test_claude1111111111111"  # 目录
# EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "evaluation_results.json")
# EVALUATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "evaluation_summary.json")

# # 创建输出目录
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 初始化评估指标
# rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")

# # 初始化OpenAI客户端（阿里云百炼）
# client = OpenAI(
#     api_key="__SET_VIA_ENVIRONMENT__",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )

# def load_test_results() -> List[Dict[str, Any]]:
#     """加载测试结果数据"""
#     logger.info(f"加载测试结果: {INPUT_FILE}")
#     with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#         test_results = json.load(f)
#     logger.info(f"测试结果数量: {len(test_results)}")
#     return test_results

# def calculate_basic_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
#     """计算基础评估指标（ROUGE和BERTScore）"""
#     metrics = {}
#     try:
#         # ROUGE
#         rouge_results = rouge.compute(
#             predictions=predictions,
#             references=references,
#             use_stemmer=True
#         )
#         metrics["rouge_1"] = rouge_results["rouge1"] * 100
#         metrics["rouge_2"] = rouge_results["rouge2"] * 100
#         metrics["rouge_l"] = rouge_results["rougeL"] * 100

#         # BERTScore
#         bert_score.utils.model2layers["/mnt/project/bert-base-chinese"] = 12

#         bertscore_results = bertscore.compute(
#             predictions=predictions,
#             references=references,
#             lang="zh",
#             model_type="/mnt/project/bert-base-chinese",
#             batch_size=16,
#             rescale_with_baseline=False
#         )

#         metrics["bertscore_precision"] = np.mean(bertscore_results["precision"]) * 100
#         metrics["bertscore_recall"] = np.mean(bertscore_results["recall"]) * 100
#         metrics["bertscore_f1"] = np.mean(bertscore_results["f1"]) * 100

#     except Exception as e:
#         logger.error(f"计算基础指标时出错: {e}")
#         metrics = {k: 0.0 for k in ["rouge_1", "rouge_2", "rouge_l",
#                                     "bertscore_precision", "bertscore_recall", "bertscore_f1"]}
#     return metrics

# def evaluate_correctness( question: str, model_answer: str, reference_answer: str) -> Dict[str, Any]:
#     """
#     正确性评估：使用大模型判断模型回答是否与参考答案一致
#     回答不一致即视为错误（可视为广义幻觉）
#     """
#     prompt = f"""请判断模型回答是否与参考答案在事实上一致。


# 问题:
# {question}

# 模型回答:
# {model_answer}

# 参考答案:
# {reference_answer}

# 评估标准：
# 1. 模型回答是否因为信息不全拒绝回答
# 2. 模型回答是否与参考答案在关键事实上一致？
# 3. 是否遗漏重要信息？
# 4. 是否包含错误内容？只需要看是否有错误，不需要看是否有虚构，因为关键事实如果正确，其他的内容一般正确。

# 请使用以下格式回答：
# {{
#   "refusal": true,                   //是否拒绝回答（true/false）
#   "correct": true,                    // 是否正确（true/false）
#   "error_types": ["事实错误", "遗漏"], // 错误类型（可选），拒答为空
#   "score": 5,                         // 准确性评分（1-5分，越高越好），拒答为1分
#   "reason": "判断理由，要求尽量简短"
# }}"""

#     try:
#         response = client.chat.completions.create(
#             model="deepseek-v3",
#             messages=[
#                 {"role": "system", "content": "你是一个专业的评估专家，判断模型回答是否拒答或者是否正确，严格按照json格式返回。"},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=800,
#             temperature=0.3
#         )
#         result = json.loads(response.choices[0].message.content)
#         # 补全字段
#         result.setdefault("error_types", [])
#         result.setdefault("reason", "无说明")
#         return result
#     except Exception as e:
#         logger.error(f"正确性评估出错: {e}")
#         return {
#             "refusal": False,
#             "correct": False,
#             "error_types": ["评估失败"],
#             "score": 1,
#             "reason": f"评估失败: {str(e)}"
#         }

# def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
#     """处理单个样本的评估"""
#     try:
#         eval_sample = {
#             "source_id": sample.get("source_id", ""),
#             "question": sample["question"],
#             "answer": sample["standard_answer"],
#             "model_answer": sample["model_answer"],
#             "evaluation_results": {
#                 "basic_metrics": {},
#                 "correctness": {}  # 原为 hallucination，改为 correctness
#             },
#             "refusal_decision": sample.get("retrieval_decision", {})
            
#         }

#         # 计算基础指标
#         basic_metrics = calculate_basic_metrics(
#             [eval_sample["model_answer"]],
#             [eval_sample["answer"]]
#         )
#         eval_sample["evaluation_results"]["basic_metrics"] = basic_metrics

#         # # 构造增强 context（加入 pub_date）
#         # enhanced_context = f"【发布日期】{eval_sample['pub_date']}\n\n【内容】{eval_sample['context']}"

#         # 正确性评估
#         correctness_result = evaluate_correctness(
#             question=eval_sample["question"],
#             model_answer=eval_sample["model_answer"],
#             reference_answer=eval_sample["answer"]
#         )
#         eval_sample["evaluation_results"]["correctness"] = correctness_result

#         logger.info(f"成功评估样本 {eval_sample['source_id']}")
#         return eval_sample

#     except Exception as e:
#         logger.error(f"评估样本出错: {e}")
#         return None

# def analyze_evaluation_results(evaluated_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """分析评估结果"""
#     logger.info("分析评估结果...")
#     total_samples = len(evaluated_samples)

#     # 基础指标平均值
#     basic_metrics_avg = {
#         "rouge_1": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_1"] for s in evaluated_samples]),
#         "rouge_2": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_2"] for s in evaluated_samples]),
#         "rouge_l": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_l"] for s in evaluated_samples]),
#         "bertscore_precision": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_precision"] for s in evaluated_samples]),
#         "bertscore_recall": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_recall"] for s in evaluated_samples]),
#         "bertscore_f1": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_f1"] for s in evaluated_samples])
#     }

#     # ✅ 计算拒答率
#     refusal_count = sum(
#         1 for s in evaluated_samples
#         if s["evaluation_results"]["correctness"].get("refusal", False)
#     )
#     refudal_rate  = refusal_count/total_samples if total_samples > 0 else 0

#     # ✅ 计算正确率
#     correct_count = sum(
#         1 for s in evaluated_samples
#         if s["evaluation_results"]["correctness"].get("correct", False)
#     )
#     accuracy = correct_count / total_samples if total_samples > 0 else 0
#     # correct_count = sum(
#     # 1 for s in evaluated_samples
#     # if not s["evaluation_results"]["correctness"].get("refusal", False)  # 没有拒答
#     # and s["evaluation_results"]["correctness"].get("correct", False)      # 并且回答正确
#     # )
#     refusal_false_count = sum(
#         1 for s in evaluated_samples
#         if not s["evaluation_results"]["correctness"].get("refusal", False)  # 没有拒答

#     )
#     accuracy = correct_count / refusal_false_count if refusal_false_count > 0 else 0
#     # 汇总结果
#     summary = {
#         "total_samples": total_samples,
#         # "refusal_count":refusal_count,
#         # "refusal_false_count":refusal_false_count,
#         "correct_count": correct_count,
#         # "refudal_rate":f"{refudal_rate:.2%}",
#         "accuracy_rate": f"{accuracy:.2%}",
#         # "basic_metrics_average": basic_metrics_avg
#     }

#     return summary

# def main():
#     """主函数"""
#     logger.info("开始评估流程...")

#     # 加载测试结果
#     try:
#         test_results = load_test_results()
#     except Exception as e:
#         logger.error(f"加载测试结果失败: {e}")
#         return

#     evaluated_samples = []

#     # 逐条评估
#     for idx, sample in enumerate(tqdm(test_results, total=len(test_results), desc="评估中")):
#     # for idx, sample in enumerate(tqdm(test_results[1380:], total=len(test_results) - 1380, desc="评估中")): 
#         result = process_single_sample(sample)
#         if result:
#             evaluated_samples.append(result)
#             # 实时保存，防止中断丢失
#             with open(EVALUATION_RESULTS_PATH, 'w', encoding='utf-8') as f:
#                 json.dump(evaluated_samples, f, ensure_ascii=False, indent=2)

#     # 分析结果
#     try:
#         summary = analyze_evaluation_results(evaluated_samples)
#         logger.info("评估分析完成")
#     except Exception as e:
#         logger.error(f"分析失败: {e}")
#         return

#     # 保存总结
#     try:
#         with open(EVALUATION_SUMMARY_PATH, 'w', encoding='utf-8') as f:
#             json.dump(summary, f, ensure_ascii=False, indent=2)
#         logger.info(f"评估总结已保存: {EVALUATION_SUMMARY_PATH}")
#     except Exception as e:
#         logger.error(f"保存总结失败: {e}")
#         return

#     # # 打印关键指标
#     # logger.info("\n=== 评估总结 ===")
#     # logger.info(f"总样本数: {summary['total_samples']}")
#     # logger.info(f"正确样本数: {summary['correct_count']}")
#     # logger.info(f"准确率: {summary['accuracy_rate']}")
#     # logger.info(f"ROUGE-L: {summary['basic_metrics_average']['rouge_l']:.2f}")
#     # logger.info(f"BERTScore-F1: {summary['basic_metrics_average']['bertscore_f1']:.2f}")
#     # logger.info("评估完成")

# if __name__ == "__main__":
#     main()



import os
import json
import logging
import time
import sys
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import evaluate
import bert_score

# ============================================================================
# 配置参数 - 批量处理
# ============================================================================
INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_m3/test_qwen2.5.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_m3/test_qwen2.5"  # 目录
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "evaluation_results.json")
EVALUATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "evaluation_summary.json")

# 批量处理配置
MODEL_NAME = "deepseek-v3"
BATCH_SIZE = 4000  # 每批处理的样本数
MAX_WAIT_HOURS = 72  # 最大等待时间

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 设置日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(OUTPUT_DIR, 'evaluation_batch.log'),
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 添加控制台输出
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def log_progress(message):
    """记录进度信息"""
    logger.info(message)
    print(message)

# 初始化评估指标
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# 初始化OpenAI客户端（阿里云百炼）
client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def load_test_results() -> List[Dict[str, Any]]:
    """加载测试结果数据"""
    log_progress(f"加载测试结果: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        test_results = json.load(f)
    log_progress(f"测试结果数量: {len(test_results)}")
    return test_results

def calculate_basic_metrics_batch(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """批量计算基础评估指标（ROUGE和BERTScore）"""
    metrics = {}
    try:
        # ROUGE
        rouge_results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        metrics["rouge_1"] = rouge_results["rouge1"] * 100
        metrics["rouge_2"] = rouge_results["rouge2"] * 100
        metrics["rouge_l"] = rouge_results["rougeL"] * 100

        # BERTScore
        bert_score.utils.model2layers["/mnt/project/bert-base-chinese"] = 12

        bertscore_results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="zh",
            model_type="/mnt/project/bert-base-chinese",
            batch_size=16,
            rescale_with_baseline=False
        )

        metrics["bertscore_precision"] = np.mean(bertscore_results["precision"]) * 100
        metrics["bertscore_recall"] = np.mean(bertscore_results["recall"]) * 100
        metrics["bertscore_f1"] = np.mean(bertscore_results["f1"]) * 100

    except Exception as e:
        logger.error(f"计算基础指标时出错: {e}")
        metrics = {k: 0.0 for k in ["rouge_1", "rouge_2", "rouge_l",
                                    "bertscore_precision", "bertscore_recall", "bertscore_f1"]}
    return metrics

def create_batch_file(samples_batch: List[Dict[str, Any]], batch_num: int) -> str:
    """创建批量评估输入文件"""
    input_file_path = os.path.join(OUTPUT_DIR, f"batch_input_{batch_num}.jsonl")
    
    with open(input_file_path, 'w', encoding='utf-8') as f:
        for sample in samples_batch:
            custom_id = f"{sample.get('source_id', 'unknown')}_{samples_batch.index(sample)}"
            
            prompt = f"""请判断模型回答是否与参考答案在事实上一致。

问题:
{sample['question']}

模型回答:
{sample['model_answer']}

参考答案:
{sample['standard_answer']}

评估标准：
1. 模型回答是否拒绝回答,比如答案中包含“根据现有资料，我无法回答这个问题。”之类的表述
2. 主要事实是否一致？（核心数字、关键实体、主要结论等）
3. 是否遗漏大量重要信息？
4. 是否包含严重错误？

请使用以下格式回答：
{{
  "refusal": true,                   //是否拒绝回答（true/false）
  "correct": true,                    // 是否正确（true/false）
  "error_types": ["事实错误", "遗漏"], // 错误类型（可选），拒答为空
  "score": 5,                         // 准确性评分（1-5分，越高越好），拒答为1分,3分以上认为可接受
  "reason": "判断理由，要求尽量简短"
}}"""

            batch_request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "你是一个专业的评估专家，判断模型回答是否拒答或者是否正确，严格按照json格式返回。"
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 800,
                    "temperature": 0.3
                }
            }
            
            f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
    
    log_progress(f"📄 创建批次 {batch_num} 输入文件: {len(samples_batch)} 个请求")
    return input_file_path

def submit_batch_job(input_file_path: str, batch_num: int) -> str:
    """提交批量任务"""
    try:
        with open(input_file_path, 'rb') as file:
            uploaded_file = client.files.create(file=file, purpose="batch")
        
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window=f"{MAX_WAIT_HOURS}h"
        )
        
        log_progress(f"✅ 批次 {batch_num} 提交成功")
        log_progress(f"🆔 任务ID: {batch_job.id}")
        return batch_job.id
        
    except Exception as e:
        log_progress(f"❌ 批次 {batch_num} 提交失败: {e}")
        return None

def monitor_batch_progress(batch_id: str, batch_num: int) -> str:
    """监控批量任务进度"""
    log_progress(f"🔍 开始监控批次 {batch_num} 进度: {batch_id}")
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            batch_status = client.batches.retrieve(batch_id)
            current_status = batch_status.status
            
            if current_status != last_status:
                elapsed_hours = (time.time() - start_time) / 3600
                log_progress(f"🔄 批次 {batch_num} 状态: {current_status} (已等待 {elapsed_hours:.1f} 小时)")
                last_status = current_status
            
            if current_status == "completed":
                log_progress(f"✅ 批次 {batch_num} 处理完成!")
                return batch_status.output_file_id
                
            elif current_status in ["failed", "expired", "cancelled"]:
                log_progress(f"❌ 批次 {batch_num} 失败: {current_status}")
                if hasattr(batch_status, 'errors'):
                    log_progress(f"错误详情: {batch_status.errors}")
                return None
                
            else:
                # 还在处理中，等待5分钟后再次检查
                time.sleep(300)
                
        except Exception as e:
            log_progress(f"⚠️ 检查批次 {batch_num} 状态失败: {e}")
            time.sleep(300)

def download_and_process_batch_results(output_file_id: str, original_batch: List[Dict[str, Any]], batch_num: int) -> List[Dict[str, Any]]:
    """下载并处理批次结果"""
    try:
        # 下载结果
        result_content = client.files.content(output_file_id)
        results = result_content.text.split('\n')
        
        evaluated_samples = []
        success_count = 0
        error_count = 0
        
        log_progress(f"📥 下载批次 {batch_num} 结果: {len(results)} 行，开始解析...")
        
        # 构建映射表
        sample_mapping = {}
        for sample in original_batch:
            custom_id = f"{sample.get('source_id', 'unknown')}_{original_batch.index(sample)}"
            sample_mapping[custom_id] = sample
        
        # 解析每一行结果
        for line_num, line in enumerate(results, 1):
            if not line.strip():
                continue
                
            try:
                result_obj = json.loads(line)
                custom_id = result_obj.get("custom_id")
                response = result_obj.get("response", {})
                
                if response.get("status_code") == 200:
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        
                        # 提取JSON
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                correctness_result = json.loads(json_match.group(0))
                                
                                if custom_id in sample_mapping:
                                    original_sample = sample_mapping[custom_id]
                                    
                                    # 计算基础指标
                                    basic_metrics = calculate_basic_metrics_batch(
                                        [original_sample["model_answer"]],
                                        [original_sample["standard_answer"]]
                                    )
                                    
                                    # 构建评估结果
                                    eval_sample = {
                                        "source_id": original_sample.get("source_id", ""),
                                        "question": original_sample["question"],
                                        "answer": original_sample["standard_answer"],
                                        "model_answer": original_sample["model_answer"],
                                        "evaluation_results": {
                                            "basic_metrics": basic_metrics,
                                            "correctness": correctness_result
                                        }
                                    }
                                    
                                    evaluated_samples.append(eval_sample)
                                    success_count += 1
                                    
                                    if success_count % 20 == 0:  # 每20个显示一次进度
                                        log_progress(f"✅ 批次 {batch_num}: 已成功处理 {success_count} 个样本")
                                else:
                                    error_count += 1
                                    
                            except json.JSONDecodeError as e:
                                error_count += 1
                                log_progress(f"❌ 批次 {batch_num} JSON解析失败: {e}")
                        else:
                            error_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
        
        log_progress(f"📊 批次 {batch_num} 解析完成: 成功 {success_count}, 失败 {error_count}")
        return evaluated_samples
        
    except Exception as e:
        log_progress(f"❌ 下载或解析批次 {batch_num} 结果失败: {e}")
        return []

def save_incremental_results(evaluated_samples: List[Dict[str, Any]], output_path: str):
    """增量保存结果"""
    # 如果文件已存在，读取现有数据
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    # 合并数据
    existing_data.extend(evaluated_samples)
    
    # 保存到临时文件
    temp_path = output_path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    # 替换原文件
    os.replace(temp_path, output_path)
    
    log_progress(f"💾 增量保存完成: 总数据量 {len(existing_data)} 条")

def get_processed_batches() -> set:
    """获取已处理的批次信息"""
    processed_file = os.path.join(OUTPUT_DIR, "processed_batches.json")
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_processed_batch(batch_num: int):
    """保存已处理的批次信息"""
    processed_file = os.path.join(OUTPUT_DIR, "processed_batches.json")
    processed_batches = get_processed_batches()
    processed_batches.add(batch_num)
    
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(list(processed_batches), f, indent=2)
    
    log_progress(f"📝 记录批次 {batch_num} 为已处理")

def cleanup_batch_files(batch_num: int):
    """清理批次临时文件"""
    input_file = os.path.join(OUTPUT_DIR, f"batch_input_{batch_num}.jsonl")
    
    if os.path.exists(input_file):
        try:
            os.remove(input_file)
            log_progress(f"🧹 已清理批次 {batch_num} 临时文件")
        except:
            log_progress(f"⚠️ 清理批次 {batch_num} 临时文件失败")

def analyze_evaluation_results(evaluated_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析评估结果"""
    logger.info("分析评估结果...")
    total_samples = len(evaluated_samples)

    # 基础指标平均值
    basic_metrics_avg = {
        "rouge_1": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_1"] for s in evaluated_samples]),
        "rouge_2": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_2"] for s in evaluated_samples]),
        "rouge_l": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_l"] for s in evaluated_samples]),
        "bertscore_precision": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_precision"] for s in evaluated_samples]),
        "bertscore_recall": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_recall"] for s in evaluated_samples]),
        "bertscore_f1": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_f1"] for s in evaluated_samples])
    }

    # 计算拒答率
    refusal_count = sum(
        1 for s in evaluated_samples
        if s["evaluation_results"]["correctness"].get("refusal", False)
    )
    refudal_rate = refusal_count / total_samples if total_samples > 0 else 0

    # 计算正确率
    correct_count = sum(
        1 for s in evaluated_samples
        if s["evaluation_results"]["correctness"].get("correct", False)
    )
    
    refusal_false_count = sum(
        1 for s in evaluated_samples
        if not s["evaluation_results"]["correctness"].get("refusal", False)
    )
    
    accuracy = correct_count / refusal_false_count if refusal_false_count > 0 else 0

    # 汇总结果
    summary = {
        "total_samples": total_samples,
        "refusal_count": refusal_count,
        "refusal_false_count": refusal_false_count,
        "correct_count": correct_count,
        "refusal_rate": f"{refudal_rate:.2%}",
        "accuracy_rate": f"{accuracy:.2%}",
        "basic_metrics_average": basic_metrics_avg
    }

    return summary

def main():
    """主函数 - 批量评估"""
    log_progress("🚀 开始批量评估流程...")
    
    print("\n" + "="*70)
    print("📊 批量评估配置")
    print("="*70)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"模型: {MODEL_NAME}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"最大等待时间: {MAX_WAIT_HOURS} 小时")
    print("="*70)

    start_time = time.time()

    try:
        # 1. 加载测试结果
        log_progress("📥 步骤1: 加载测试结果...")
        test_results = load_test_results()
        if not test_results:
            log_progress("❌ 没有加载到测试结果，退出")
            return

        # 2. 分批处理
        total_batches = (len(test_results) + BATCH_SIZE - 1) // BATCH_SIZE
        processed_batches = get_processed_batches()
        
        log_progress(f"📦 数据分 {total_batches} 批处理")
        log_progress(f"📋 已处理批次: {processed_batches}")
        
        completed_batches = 0
        total_evaluated = 0
        
        for batch_num in range(1, total_batches + 1):
            if batch_num in processed_batches:
                log_progress(f"⏭️  跳过已处理的批次 {batch_num}")
                continue
            
            batch_start = (batch_num - 1) * BATCH_SIZE
            batch_end = batch_num * BATCH_SIZE
            batch_data = test_results[batch_start:batch_end]
            
            log_progress(f"\n🔄 开始处理批次 {batch_num}/{total_batches}")
            log_progress(f"📊 批次数据: {len(batch_data)} 条")
            
            # 2.1 创建批量文件
            input_file = create_batch_file(batch_data, batch_num)
            
            # 2.2 提交批量任务
            batch_id = submit_batch_job(input_file, batch_num)
            if not batch_id:
                log_progress(f"❌ 批次 {batch_num} 提交失败，跳过")
                continue
            
            # 2.3 监控进度
            output_file_id = monitor_batch_progress(batch_id, batch_num)
            if not output_file_id:
                log_progress(f"❌ 批次 {batch_num} 处理失败，跳过")
                continue
            
            # 2.4 下载和处理结果
            evaluated_batch = download_and_process_batch_results(output_file_id, batch_data, batch_num)
            
            if evaluated_batch:
                # 2.5 增量保存结果
                save_incremental_results(evaluated_batch, EVALUATION_RESULTS_PATH)
                
                # 2.6 记录为已处理
                save_processed_batch(batch_num)
                
                # 2.7 清理临时文件
                cleanup_batch_files(batch_num)
                
                completed_batches += 1
                total_evaluated += len(evaluated_batch)
                
                log_progress(f"✅ 批次 {batch_num} 完成!")
                log_progress(f"📈 当前进度: {completed_batches}/{total_batches} 批次")
                log_progress(f"📊 累计评估: {total_evaluated} 条数据")
                
                # 计算预计剩余时间
                elapsed_time = time.time() - start_time
                if completed_batches > 0:
                    avg_time_per_batch = elapsed_time / completed_batches
                    remaining_batches = total_batches - completed_batches
                    estimated_remaining = avg_time_per_batch * remaining_batches
                    
                    hours = int(estimated_remaining // 3600)
                    minutes = int((estimated_remaining % 3600) // 60)
                    
                    log_progress(f"⏳ 预计剩余时间: {hours}小时{minutes}分钟")
            else:
                log_progress(f"❌ 批次 {batch_num} 未生成有效评估结果")

        # 3. 加载所有评估结果并分析
        log_progress("\n📊 步骤3: 分析最终评估结果...")
        if os.path.exists(EVALUATION_RESULTS_PATH):
            with open(EVALUATION_RESULTS_PATH, 'r', encoding='utf-8') as f:
                all_evaluated_samples = json.load(f)
            
            summary = analyze_evaluation_results(all_evaluated_samples)
            
            # 保存总结
            with open(EVALUATION_SUMMARY_PATH, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # 打印关键指标
            log_progress("\n" + "="*70)
            log_progress("🎉 评估完成!")
            log_progress("="*70)
            log_progress(f"总样本数: {summary['total_samples']}")
            log_progress(f"拒答样本数: {summary['refusal_count']}")
            log_progress(f"非拒答样本数: {summary['refusal_false_count']}")
            log_progress(f"正确样本数: {summary['correct_count']}")
            log_progress(f"拒答率: {summary['refusal_rate']}")
            log_progress(f"准确率: {summary['accuracy_rate']}")
            log_progress(f"ROUGE-L: {summary['basic_metrics_average']['rouge_l']:.2f}")
            log_progress(f"BERTScore-F1: {summary['basic_metrics_average']['bertscore_f1']:.2f}")
            log_progress("="*70)
        
        # 计算总时间
        total_time = time.time() - start_time
        days = int(total_time // 86400)
        hours = int((total_time % 86400) // 3600)
        minutes = int((total_time % 3600) // 60)
        
        log_progress(f"\n⏱️  总处理时间: {days}天{hours}小时{minutes}分钟")
        log_progress(f"📈 完成批次: {completed_batches}/{total_batches}")
        log_progress(f"📊 总评估数据量: {total_evaluated} 条")
        log_progress(f"💾 输出文件: {EVALUATION_RESULTS_PATH}")
        log_progress(f"📋 总结文件: {EVALUATION_SUMMARY_PATH}")
        
    except Exception as e:
        log_progress(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()