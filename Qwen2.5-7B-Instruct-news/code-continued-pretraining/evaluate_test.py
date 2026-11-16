import re
from typing import List, Dict
import evaluate
import bert_score
import numpy as np
import logging

# 初始化评估器
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def extract_news_contents(filepath: str, limit: int = None) -> List[str]:
    """从新闻文件中提取所有新闻内容字段"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 分割每条新闻（<|endoftext|> 作为结束标记）
    articles = raw_text.split("<|endoftext|>")

    contents = []
    for article in articles:
        # 提取“新闻内容”字段
        match = re.search(r"新闻内容：(.*?)$", article.strip(), re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content:
                contents.append(content)
                if limit is not None and len(contents) >= limit:
                    break  # 达到限制数量后停止处理
    return contents


def calculate_basic_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算基础评估指标（ROUGE 和 BERTScore）"""
    metrics = {}

    try:
        # 计算 ROUGE
        rouge_results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        metrics["rouge_1"] = rouge_results["rouge1"] * 100
        metrics["rouge_2"] = rouge_results["rouge2"] * 100
        metrics["rouge_l"] = rouge_results["rougeL"] * 100

        # 注册中文模型的层数
        bert_score.utils.model2layers["/mnt/project/bert-base-chinese"] = 12

        # 计算 BERTScore
        bertscore_results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="zh",
            model_type="/mnt/project/bert-base-chinese",
            batch_size=16,
            rescale_with_baseline=False
        )
        metrics["bertscore_precision"] = float(np.mean(bertscore_results["precision"])) * 100
        metrics["bertscore_recall"] = float(np.mean(bertscore_results["recall"])) * 100
        metrics["bertscore_f1"] = float(np.mean(bertscore_results["f1"])) * 100

    except Exception as e:

        metrics = {
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }

    return metrics


if __name__ == "__main__":
    # 设置文件路径
    reference_file = "../data-continued-pretraining/news_3year.txt"
    prediction_file = "../data-continued-pretraining/generated_news_3year.txt"

    # 读取新闻内容
    generated_contents = extract_news_contents(prediction_file)
    print(len(generated_contents))
    reference_contents = extract_news_contents(reference_file,limit=len(generated_contents))
 

    # 确保数量一致
    assert len(reference_contents) == len(generated_contents), \
        f"参考内容数量 ({len(reference_contents)}) 与生成内容数量 ({len(generated_contents)}) 不一致！"

    # 评估
    results = calculate_basic_metrics(generated_contents, reference_contents)

    # 打印结果
    print("评估结果：")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")
