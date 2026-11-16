import json
import numpy as np

INPUT_FILE = "../data-2016news-finetune/evaluation_results/evaluation_results.json"

def load_evaluation_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_evaluation_results(evaluated_samples):
    print("分析评估结果...")

    # 总样本数
    total_samples = len(evaluated_samples)

    # 类型划分
    situational_samples = [s for s in evaluated_samples if s.get("type") == "situational"]
    conceptual_samples = [s for s in evaluated_samples if s.get("type") == "conceptual"]

    # 平均基本指标（ROUGE 和 BERTScore）
    def avg(metric_name):
        values = [s["evaluation_results"]["basic_metrics"].get(metric_name, 0) for s in evaluated_samples]
        return np.mean(values) if values else 0

    basic_metrics_avg = {
        "rouge_1": avg("rouge_1"),
        "rouge_2": avg("rouge_2"),
        "rouge_l": avg("rouge_l"),
        "bertscore_precision": avg("bertscore_precision"),
        "bertscore_recall": avg("bertscore_recall"),
        "bertscore_f1": avg("bertscore_f1")
    }

    # 平均情境评估分数
    situational_scores = [
        s["evaluation_results"]["situational"].get("score", 0)
        for s in situational_samples if "situational" in s["evaluation_results"]
    ]
    situational_avg = np.mean(situational_scores) if situational_scores else 0

    # 平均概念评估分数
    conceptual_scores = [
        s["evaluation_results"]["conceptual"].get("score", 0)
        for s in conceptual_samples if "conceptual" in s["evaluation_results"]
    ]
    conceptual_avg = np.mean(conceptual_scores) if conceptual_scores else 0

    # 幻觉检出率（score < 3 视为存在幻觉）
    hallucination_detected = sum(
    1 for s in evaluated_samples
    if s["evaluation_results"].get("hallucination", {}).get("hallucination_detected", False) is True
)
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

if __name__ == "__main__":
    results = load_evaluation_results(INPUT_FILE)
    summary = analyze_evaluation_results(results)
    with open("../data-2016news-finetune/evaluation_results/evaluation_summary_true.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
