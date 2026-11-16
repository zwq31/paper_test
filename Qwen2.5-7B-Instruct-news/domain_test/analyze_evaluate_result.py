import json
import numpy as np

INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_train_test/output/liangmobamn/test_old_data/evaluation_results.json"

def load_evaluation_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_evaluation_results(evaluated_samples):
    """分析评估结果"""
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

    # ✅ 计算正确率
    # correct_count = sum(
    #     1 for s in evaluated_samples
    #     if s["evaluation_results"]["correctness"].get("correct", False)
    # )

    correct_count = sum(
    1 for s in evaluated_samples
    if s["evaluation_results"]["correctness"].get("score", 0) >= 4
    )
    accuracy = correct_count / total_samples if total_samples > 0 else 0

    # 汇总结果
    summary = {
        "total_samples": total_samples,
        "correct_count": correct_count,
        "accuracy": float(accuracy),
        "accuracy_rate": f"{accuracy:.2%}",
        "basic_metrics_average": basic_metrics_avg
    }
    
    return summary
    # # ✅ 计算拒答率
    # refusal_count = sum(
    #     1 for s in evaluated_samples
    #     if s["evaluation_results"]["correctness"].get("refusal", False)
    # )
    # refudal_rate  = refusal_count/total_samples if total_samples > 0 else 0

    # # ✅ 计算正确率
    # correct_count = sum(
    #     1 for s in evaluated_samples
    #     if s["evaluation_results"]["correctness"].get("correct", False)
    # )
    # accuracy = correct_count / total_samples if total_samples > 0 else 0
    # correct_count = sum(
    # 1 for s in evaluated_samples
    # if not s["evaluation_results"]["correctness"].get("refusal", False)  # 没有拒答
    # and s["evaluation_results"]["correctness"].get("correct", False)      # 并且回答正确
    # )
    # refusal_false_count = sum(
    #     1 for s in evaluated_samples
    #     if not s["evaluation_results"]["correctness"].get("refusal", False)  # 没有拒答

    # )
    # accuracy = correct_count / refusal_false_count if refusal_false_count > 0 else 0
    # # 汇总结果
    # summary = {
    #     "total_samples": total_samples,
    #     "refusal_count":refusal_count,
    #     "refusal_false_count":refusal_false_count,
    #     "correct_count": correct_count,
    #     "refudal_rate":f"{refudal_rate:.2%}",
    #     "accuracy_rate": f"{accuracy:.2%}",
    #     "basic_metrics_average": basic_metrics_avg
    # }

    # return summary



if __name__ == "__main__":
    results = load_evaluation_results(INPUT_FILE)
    summary = analyze_evaluation_results(results)
    with open("/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_train_test/output/liangmobamn/test_old_data/evaluation_summary_4.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
