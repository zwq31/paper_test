#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import evaluate
import bert_score

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='evaluation.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


# 配置参数
INPUT_FILE = "../data/test_results_merge.json"
OUTPUT_DIR = "../data/evaluation_results"
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_DIR, "evaluation_results.json")
EVALUATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "evaluation_summary.json")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化评估指标
# 初始化评估指标
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# 初始化OpenAI客户端
client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def load_test_results():
    """加载测试结果数据"""
    logger.info(f"加载测试结果: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        test_results = json.load(f)
    logger.info(f"测试结果数量: {len(test_results)}")
    return test_results

def calculate_basic_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算基础评估指标（ROUGE和BERTScore）"""
    metrics = {}
    
    try:
        # 计算ROUGE得分
        rouge_results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        metrics["rouge_1"] = rouge_results["rouge1"] * 100
        metrics["rouge_2"] = rouge_results["rouge2"] * 100
        metrics["rouge_l"] = rouge_results["rougeL"] * 100
        
        # 计算BERTScore
        # 手动添加模型路径到 model2layers 映射
        bert_score.utils.model2layers["/mnt/project/bert-base-chinese"] = 12  # 替换为实际层数
        
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
        metrics = {
            "rouge_1": 0,
            "rouge_2": 0,
            "rouge_l": 0,
            "bertscore_precision": 0,
            "bertscore_recall": 0,
            "bertscore_f1": 0,
        }
    
    return metrics

def evaluate_situational(context: str, question: str, model_answer: str, reference_answer: str) -> Dict[str, Any]:
    """情境评估"""
    prompt = f"""请你作为一个严格的情境理解评估员，判断模型回答是否完整理解了给定上下文语境并据此准确作答。

上下文信息：
{context}

问题：
{question}

模型的回答：
{model_answer}

参考答案：
{reference_answer}

你需要评估模型是否：
1. 明确基于上下文回答，而非泛泛其谈；
2. 是否遗漏或扭曲了关键信息；
3. 是否能体现对背景的理解。

请用以下格式回答：
{{
  "contextual_alignment": true/false,   // 是否基于上下文回答
  "missing_key_info": true/false,       // 是否遗漏关键信息
  "score": 1-5,                         // 总体得分（1-5分）
  "reason": "模型是否依据情境作答的分析原因"
}}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "你是一个专业的情境理解评估专家，负责评估模型回答是否准确理解并基于给定上下文作答。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        logger.error(f"情境评估出错: {e}")
        return {
            "contextual_alignment": False,
            "missing_key_info": True,
            "score": 1,
            "reason": f"评估过程出错: {str(e)}"
        }

def evaluate_conceptual(context: str, question: str, model_answer: str, reference_answer: str) -> Dict[str, Any]:
    """概念评估"""
    prompt = f"""请作为一个专业评估员判断以下回答是否准确传达了问题中关键概念的内涵及其在特定语境下的体现。

上下文信息：
{context}

问题：
{question}

模型的回答：
{model_answer}

参考答案：
{reference_answer}

请评估以下方面：
1. 模型是否准确说明了问题中关键概念的定义或核心含义；
2. 是否结合了情境案例说明概念应用；
3. 是否存在概念混淆或错误归因。

回答格式如下：
{{
  "concept_understanding": true/false,     // 是否理解核心概念
  "application_correctness": true/false,   // 是否正确应用概念
  "score": 1-5,                            // 总体得分（1-5分）
  "reason": "评估结论解释"
}}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "你是一个专业的概念理解评估专家，负责评估模型对关键概念的理解和应用是否准确。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        logger.error(f"概念评估出错: {e}")
        return {
            "concept_understanding": False,
            "application_correctness": False,
            "score": 1,
            "reason": f"评估过程出错: {str(e)}"
        }

def evaluate_hallucination(context: str, question: str, model_answer: str, reference_answer: str) -> Dict[str, Any]:
    """幻觉评估"""
    prompt = f"""请判断模型是否产生了与提供上下文无关的虚构信息（幻觉）。

上下文：
{context}

问题：
{question}

模型回答：
{model_answer}

参考答案：
{reference_answer}

评估标准：
1. 是否出现上下文中未提及、与问题无关的虚构实体/事件；    
2. 是否有与事实不符的内容；
3. 是否有内容完全不在语料中出现过。

请使用以下格式回答：
{{
  "hallucination_detected": true/false,          // 是否检测到幻觉
  "hallucinated_content": ["内容1", "内容2"],    // 幻觉内容列表
  "score": 1-5,                                 // 总体得分（1-5分）
  "reason": "为什么判断是幻觉或不是"
}}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "你是一个专业的幻觉检测专家，负责评估模型回答中是否存在与上下文无关的虚构信息。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        logger.error(f"幻觉评估出错: {e}")
        return {
            "hallucination_detected": True,
            "hallucinated_content": ["评估过程出错"],
            "score": 1,
            "reason": f"评估过程出错: {str(e)}"
        }

def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个样本的评估
    
    Args:
        sample: 包含问题、答案、模型回答等信息的样本字典
    
    Returns:
        评估后的样本字典，包含所有评估结果
    """
    try:
        # 构建评估样本
        eval_sample = {
            "question": sample["question"],
            "answer": sample["answer"],
            "model_answer": sample["model_answer"],
            "base_model_answer": sample.get("base_model_answer", ""),
            "source_id": sample.get("source_id", ""),
            "context": sample.get("context", ""),
            "type": sample.get("type", "unknown"),
            "evaluation_results": {
                "basic_metrics": {},
                "situational": {},
                "conceptual": {},
                "hallucination": {}
            }
        }
        
        # 计算基础指标
        basic_metrics = calculate_basic_metrics(
            [eval_sample["model_answer"]], 
            [eval_sample["answer"]]
        )
        eval_sample["evaluation_results"]["basic_metrics"] = basic_metrics
        
        # 对所有样本进行幻觉评估
        eval_sample["evaluation_results"]["hallucination"] = evaluate_hallucination(
            eval_sample["context"],
            eval_sample["question"],
            eval_sample["model_answer"],
            eval_sample["answer"]
        )
        
        # 根据问题类型进行情境或概念评估
        if eval_sample["type"] == "situational":
            eval_sample["evaluation_results"]["situational"] = evaluate_situational(
                eval_sample["context"],
                eval_sample["question"],
                eval_sample["model_answer"],
                eval_sample["answer"]
            )
        elif eval_sample["type"] == "conceptual":
            eval_sample["evaluation_results"]["conceptual"] = evaluate_conceptual(
                eval_sample["context"],
                eval_sample["question"],
                eval_sample["model_answer"],
                eval_sample["answer"]
            )
        
        logger.info(f"成功评估样本 {eval_sample['source_id']}")
        return eval_sample
        
    except Exception as e:
        logger.error(f"评估样本 {sample.get('source_id', 'unknown')} 时出错: {e}")
        return None

def analyze_evaluation_results(evaluated_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析评估结果"""
    logger.info("分析评估结果...")
    
    # 初始化统计变量
    total_samples = len(evaluated_samples)
    situational_samples = [s for s in evaluated_samples if s["type"] == "situational"]
    conceptual_samples = [s for s in evaluated_samples if s["type"] == "conceptual"]
    
    # 计算基础指标平均值
    basic_metrics_avg = {
        "rouge_1": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_1"] for s in evaluated_samples]),
        "rouge_2": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_2"] for s in evaluated_samples]),
        "rouge_l": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_l"] for s in evaluated_samples]),
        "bertscore_precision": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_precision"] for s in evaluated_samples]),
        "bertscore_recall": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_recall"] for s in evaluated_samples]),
        "bertscore_f1": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_f1"] for s in evaluated_samples])
    }
    
    # 计算情境评估平均分
    situational_scores = [s["evaluation_results"]["situational"]["score"] for s in situational_samples]
    situational_avg = np.mean(situational_scores) if situational_scores else 0
    
    # 计算概念评估平均分
    conceptual_scores = [s["evaluation_results"]["conceptual"]["score"] for s in conceptual_samples]
    conceptual_avg = np.mean(conceptual_scores) if conceptual_scores else 0
    
    # 计算幻觉检出率
    hallucination_detected = sum(1 for s in evaluated_samples 
                               if s["evaluation_results"]["hallucination"]["score"]<3)
    hallucination_rate = hallucination_detected / total_samples if total_samples > 0 else 0
    
    # 汇总结果
    summary = {
        "total_samples": total_samples,
        "situational_samples": len(situational_samples),
        "conceptual_samples": len(conceptual_samples),
        "basic_metrics_average": basic_metrics_avg,
        "situational_average_score": float(situational_avg),
        "conceptual_average_score": float(conceptual_avg),
        "hallucination_detection_rate": float(hallucination_rate),
        # "low_score_samples": {
        #     "situational": [s["source_id"] for s in situational_samples 
        #                   if s["evaluation_results"]["situational"]["score"] < 3],
        #     "conceptual": [s["source_id"] for s in conceptual_samples 
        #                  if s["evaluation_results"]["conceptual"]["score"] < 3],
        #     "hallucination": [s["source_id"] for s in evaluated_samples 
        #                     if s["evaluation_results"]["hallucination"]["score"] < 3]
        # }
    }
    
    return summary

def main():
    """主函数"""
    logger.info("开始评估流程...")
    
    # 加载测试结果
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            test_results = json.load(f)
        logger.info(f"成功加载 {len(test_results)} 条测试结果")
    except Exception as e:
        logger.error(f"加载测试结果时出错: {e}")
        return

    
    # 存储评估结果
    evaluated_samples = []
    
    # 处理指定范围的样本
    for idx, sample in tqdm(enumerate(test_results), total=len(test_results), desc="评估所有样本"):
        result = process_single_sample(sample)
        if result:
            evaluated_samples.append(result)
            # 每评估一条就保存一次
            with open(EVALUATION_RESULTS_PATH, 'w', encoding='utf-8') as f:
                json.dump(evaluated_samples, f, ensure_ascii=False, indent=2)
    
    # 分析评估结果
    try:
        summary = analyze_evaluation_results(evaluated_samples)
        logger.info("评估结果分析完成")
    except Exception as e:
        logger.error(f"分析评估结果时出错: {e}")
        return
    
    # 保存评估总结
    try:
        with open(EVALUATION_SUMMARY_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"评估总结已保存到: {EVALUATION_SUMMARY_PATH}")
    except Exception as e:
        logger.error(f"保存评估总结时出错: {e}")
        return
    
    # 打印关键指标
    logger.info("\n评估总结:")
    logger.info(f"总样本数: {summary['total_samples']}")
    logger.info(f"情境问题平均分: {summary['situational_average_score']:.2f}")
    logger.info(f"概念问题平均分: {summary['conceptual_average_score']:.2f}")
    logger.info(f"幻觉检出率: {summary['hallucination_detection_rate']:.2%}")
    logger.info(f"ROUGE-L平均分: {summary['basic_metrics_average']['rouge_l']:.2f}")
    logger.info(f"BERTScore-F1平均分: {summary['basic_metrics_average']['bertscore_f1']:.2f}")
    
    logger.info("评估流程完成")

if __name__ == "__main__":
    main() 