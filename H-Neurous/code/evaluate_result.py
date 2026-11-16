import json
from pathlib import Path
import numpy as np
from collections import defaultdict

# ===== 配置路径（请务必修改 FILE_B）=====
FILE_A = "/mnt/project/H-Neurous/data/test/test_inference_chat/evaluation_results.json"
FILE_B = "/mnt/project/H-Neurous/data/test/test_inference_elastic_net.json"  # ← 替换为实际路径！

def load_by_question(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            data = [data]
    q_dict = {}
    for item in data:
        q = item.get("question", "").strip()
        if q and q not in q_dict:
            q_dict[q] = item
    return q_dict

def analyze_evaluation_results_with_hallucination(file_a, file_b):
    """使用 is_hallucination 判断模型是否回答，score 判断是否应该回答"""
    dict_a = load_by_question(file_a)
    dict_b = load_by_question(file_b)
    
    common_questions = set(dict_a.keys()) & set(dict_b.keys())
    total_samples = len(common_questions)
    
    counters = {
        'total': total_samples,
        'model_answered': 0,
        'model_refused': 0,
        'score_ge_4': 0,
        'score_lt_4': 0,
        'answered_and_score_ge_4': 0,
        'answered_and_score_lt_4': 0,
        'refused_and_score_lt_4': 0,
        'refused_and_score_ge_4': 0,
    }
    
    error_types_count = defaultdict(int)
    all_scores = []

    for q in common_questions:
        item_a = dict_a[q]
        item_b = dict_b[q]

        score = item_a.get("evaluation_results", {}).get("correctness", {}).get("score", 0)
        is_hallucination = item_b.get("is_hallucination", True)
        error_types = item_a.get("evaluation_results", {}).get("correctness", {}).get("error_types", [])
        
        all_scores.append(score)

        if score >= 4:
            counters['score_ge_4'] += 1
        else:
            counters['score_lt_4'] += 1

        # 关键修正：is_hallucination=True 表示模型未有效回答（视为拒答）
        model_answered = not is_hallucination

        if model_answered:
            counters['model_answered'] += 1
            if score > 3:
                counters['answered_and_score_ge_4'] += 1
            else:
                counters['answered_and_score_lt_4'] += 1
                error_types_count["Boundary violation (answered when should refuse)"] += 1
        else:
            counters['model_refused'] += 1
            if score > 3:
                counters['refused_and_score_ge_4'] += 1
            else:
                counters['refused_and_score_lt_4'] += 1

        for et in error_types:
            error_types_count[et] += 1

    # --- 以下指标计算逻辑保持不变 ---
    refusal_rate = counters['model_refused'] / total_samples if total_samples > 0 else 0
    refusal_precision = counters['refused_and_score_lt_4'] / counters['model_refused'] if counters['model_refused'] > 0 else 0
    refusal_recall = counters['refused_and_score_lt_4'] / counters['score_lt_4'] if counters['score_lt_4'] > 0 else 0
    refusal_f1 = 2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall) if (refusal_precision + refusal_recall) > 0 else 0

    answer_accuracy = counters['answered_and_score_ge_4'] / counters['model_answered'] if counters['model_answered'] > 0 else 0
    answer_recall = counters['answered_and_score_ge_4'] / counters['score_ge_4'] if counters['score_ge_4'] > 0 else 0
    answer_f1 = 2 * answer_accuracy * answer_recall / (answer_accuracy + answer_recall) if (answer_accuracy + answer_recall) > 0 else 0

    overall_accuracy = (counters['answered_and_score_ge_4'] + counters['refused_and_score_lt_4']) / total_samples if total_samples > 0 else 0
    global_hallucination_rate = counters['answered_and_score_lt_4'] / total_samples if total_samples > 0 else 0

    basic_metrics_avg = {"rouge_1": 0, "rouge_2": 0, "rouge_l": 0, "bertscore_precision": 0, "bertscore_recall": 0, "bertscore_f1": 0}
    try:
        basic_metrics_avg = {
            "rouge_1": float(np.mean([dict_a[q]["evaluation_results"]["basic_metrics"]["rouge_1"] for q in common_questions])),
            "rouge_2": float(np.mean([dict_a[q]["evaluation_results"]["basic_metrics"]["rouge_2"] for q in common_questions])),
            "rouge_l": float(np.mean([dict_a[q]["evaluation_results"]["basic_metrics"]["rouge_l"] for q in common_questions])),
            "bertscore_precision": float(np.mean([dict_a[q]["evaluation_results"]["basic_metrics"]["bertscore_precision"] for q in common_questions])),
            "bertscore_recall": float(np.mean([dict_a[q]["evaluation_results"]["basic_metrics"]["bertscore_recall"] for q in common_questions])),
            "bertscore_f1": float(np.mean([dict_a[q]["evaluation_results"]["basic_metrics"]["bertscore_f1"] for q in common_questions]))
        }
    except Exception as e:
        pass

    score_distribution = defaultdict(int)
    for s in all_scores:
        score_distribution[s] += 1

    summary = {
        "total_samples": counters['total'],
        "score_ge_4_count": counters['score_ge_4'],
        "score_lt_4_count": counters['score_lt_4'],
        "model_answered_count": counters['model_answered'],
        "model_refused_count": counters['model_refused'],
        "answered_and_score_ge_4": counters['answered_and_score_ge_4'],
        "answered_and_score_lt_4": counters['answered_and_score_lt_4'],
        "refused_and_score_lt_4": counters['refused_and_score_lt_4'],
        "refused_and_score_ge_4": counters['refused_and_score_ge_4'],

        "knowledge_boundary_metrics": {
            "refusal_rate": float(refusal_rate),
            "refusal_rate_percentage": f"{refusal_rate:.2%}",
            "refusal_precision": float(refusal_precision),
            "refusal_precision_percentage": f"{refusal_precision:.2%}",
            "refusal_recall": float(refusal_recall),
            "refusal_recall_percentage": f"{refusal_recall:.2%}",
            "refusal_f1": float(refusal_f1),
            "refusal_f1_percentage": f"{refusal_f1:.2%}"
        },

        "answer_accuracy_metrics": {
            "answer_accuracy": float(answer_accuracy),
            "answer_accuracy_percentage": f"{answer_accuracy:.2%}",
            "answer_recall": float(answer_recall),
            "answer_recall_percentage": f"{answer_recall:.2%}",
            "answer_f1": float(answer_f1),
            "answer_f1_percentage": f"{answer_f1:.2%}"
        },

        "hallucination_metrics": {
            "hallucination_count": counters['answered_and_score_lt_4'],
            "global_hallucination_rate": float(global_hallucination_rate),
            "global_hallucination_rate_percentage": f"{global_hallucination_rate:.2%}"
        },

        "overall_metrics": {
            "overall_accuracy": float(overall_accuracy),
            "overall_accuracy_percentage": f"{overall_accuracy:.2%}"
        },

        "basic_metrics_average": basic_metrics_avg,
        "error_types_distribution": dict(error_types_count),
        "score_distribution": dict(score_distribution),

        "confusion_matrix_counts": {
            "true_positive": counters['answered_and_score_ge_4'],
            "false_positive": counters['answered_and_score_lt_4'],
            "true_negative": counters['refused_and_score_lt_4'],
            "false_negative": counters['refused_and_score_ge_4']
        }
    }

    return summary

# ========== 保留你原有的打印函数 ==========
def print_detailed_report(summary):
    print("=" * 60)
    print("幻觉分类器性能评估报告")
    print("=" * 60)
    
    print(f"\n📊 基本统计:")
    print(f"   总样本数: {summary['total_samples']}")
    print(f"   正确样本数: {summary['score_ge_4_count']}")
    print(f"   幻觉样本数: {summary['score_lt_4_count']}")
    print(f"   初始准确率: {((summary['score_ge_4_count'] / summary['total_samples']) * 100):.2f}%")
    print(f"   判断为正确样本数: {summary['model_answered_count']}")
    print(f"   判断为幻觉样本数: {summary['model_refused_count']}")
    
    
    print(f"\n🎯 幻觉检测性能:")
    kb_metrics = summary['knowledge_boundary_metrics']
    print(f"   精确率: {kb_metrics['refusal_precision_percentage']} (预测为幻觉中，实际是幻觉的比例)")
    print(f"   召回率: {kb_metrics['refusal_recall_percentage']} (实际是幻觉中，被正确预测的比例)")
    print(f"   F1分数: {kb_metrics['refusal_f1_percentage']}")
    
    print(f"\n✅  正确回答检测性能:")
    acc_metrics = summary['answer_accuracy_metrics']
    print(f"   准确率: {acc_metrics['answer_accuracy_percentage']} (预测为正确中，实际正确的比例)")
    print(f"   召回率: {acc_metrics['answer_recall_percentage']} (实际正确中，被正确预测的比例)")
    print(f"   F1分数: {acc_metrics['answer_f1_percentage']}")
    
    print(f"\n⚠️ 幻觉分析（全局错误回答率）:")
    hallu = summary['hallucination_metrics']
    print(f"   幻觉回答数量: {hallu['hallucination_count']}")
    print(f"   全局幻觉率: {hallu['global_hallucination_rate_percentage']} "
          f"(错误回答数 / 总样本数，反映系统整体幻觉频率)")

    print(f"\n🏆 总体表现:")
    overall = summary['overall_metrics']
    print(f"   总体准确率: {overall['overall_accuracy_percentage']} (正确样本判为正确+幻觉样本判为幻觉)/总数")
    
    print(f"\n🔍 详细统计:")
    cm = summary['confusion_matrix_counts']
    print(f"   真正例(TP): {cm['true_positive']} (正确样本判为正确)")
    print(f"   假正例(FN): {cm['false_positive']} (幻觉样本判为正确)")
    print(f"   真负例(TN): {cm['true_negative']} (幻觉样本判为幻觉)")
    print(f"   假负例(FN): {cm['false_negative']} (正确样本判为幻觉)")
    
    if summary['error_types_distribution']:
        print(f"\n❌ 错误类型分布:")
        for error_type, count in summary['error_types_distribution'].items():
            print(f"   {error_type}: {count}")
    
    if summary['score_distribution']:
        print(f"\n📈 分数分布:")
        for score, count in sorted(summary['score_distribution'].items()):
            percentage = count / summary['total_samples'] * 100
            print(f"   分数 {score}: {count} ({percentage:.1f}%)")

# ========== 主程序 ==========
if __name__ == "__main__":
    summary = analyze_evaluation_results_with_hallucination(FILE_A, FILE_B)
    print_detailed_report(summary)