import json
from pathlib import Path
import numpy as np
from collections import defaultdict

INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/test_decision_tree_2400/merged_evaluation.json"

def load_evaluation_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_evaluation_results(evaluated_samples):
    """分析评估结果 - 使用全局幻觉率"""
    total_samples = len(evaluated_samples)
    
    counters = {
        'total': total_samples,
        'model_answered': 0,
        'model_refused': 0,
        'score_ge_4': 0,
        'score_lt_4': 0,
        'answered_and_score_ge_4': 0,
        'answered_and_score_lt_4': 0,  # ← 这就是幻觉回答
        'refused_and_score_lt_4': 0,
        'refused_and_score_ge_4': 0,
    }
    
    error_types_count = defaultdict(int)
    
    for sample in evaluated_samples:
        should_answer = sample.get("refusal_decision", {}).get("should_answer", True)
        score = sample.get("evaluation_results", {}).get("correctness", {}).get("score", 0)
        error_types = sample.get("evaluation_results", {}).get("correctness", {}).get("error_types", [])
        
        is_refusal = not should_answer
        
        if is_refusal:
            counters['model_refused'] += 1
        else:
            counters['model_answered'] += 1
        
        if score >= 4:
            counters['score_ge_4'] += 1
            if not is_refusal:
                counters['answered_and_score_ge_4'] += 1
            else:
                counters['refused_and_score_ge_4'] += 1
        else:
            counters['score_lt_4'] += 1
            if not is_refusal:
                counters['answered_and_score_lt_4'] += 1  # 幻觉
            else:
                counters['refused_and_score_lt_4'] += 1
                
        for error_type in error_types:
            error_types_count[error_type] += 1
    
    # 知识边界指标
    refusal_rate = counters['model_refused'] / total_samples if total_samples > 0 else 0
    refusal_precision = counters['refused_and_score_lt_4'] / counters['model_refused'] if counters['model_refused'] > 0 else 0
    refusal_recall = counters['refused_and_score_lt_4'] / counters['score_lt_4'] if counters['score_lt_4'] > 0 else 0
    refusal_f1 = 2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall) if (refusal_precision + refusal_recall) > 0 else 0
    
    # 回答准确性
    answer_accuracy = counters['answered_and_score_ge_4'] / counters['model_answered'] if counters['model_answered'] > 0 else 0
    answer_recall = counters['answered_and_score_ge_4'] / counters['score_ge_4'] if counters['score_ge_4'] > 0 else 0
    answer_f1 = 2 * answer_accuracy * answer_recall / (answer_accuracy + answer_recall) if (answer_accuracy + answer_recall) > 0 else 0
    
    # 总体准确率
    overall_accuracy = (counters['answered_and_score_ge_4'] + counters['refused_and_score_lt_4']) / total_samples if total_samples > 0 else 0
    
    # ✅ 全局幻觉率：错误回答 / 总样本
    global_hallucination_rate = counters['answered_and_score_lt_4'] / total_samples if total_samples > 0 else 0
    
    # 基础指标
    basic_metrics_avg = {
        "rouge_1": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_1"] for s in evaluated_samples]),
        "rouge_2": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_2"] for s in evaluated_samples]),
        "rouge_l": np.mean([s["evaluation_results"]["basic_metrics"]["rouge_l"] for s in evaluated_samples]),
        "bertscore_precision": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_precision"] for s in evaluated_samples]),
        "bertscore_recall": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_recall"] for s in evaluated_samples]),
        "bertscore_f1": np.mean([s["evaluation_results"]["basic_metrics"]["bertscore_f1"] for s in evaluated_samples])
    }
    
    score_distribution = defaultdict(int)
    for sample in evaluated_samples:
        score = sample.get("evaluation_results", {}).get("correctness", {}).get("score", 0)
        score_distribution[score] += 1
    
    summary = {
        "total_samples": counters['total'],
        "score_ge_4_count": counters['score_ge_4'],
        "score_lt_4_count": counters['score_lt_4'],
        "model_answered_count": counters['model_answered'],
        "model_refused_count": counters['model_refused'],
        "answered_and_score_ge_4": counters['answered_and_score_ge_4'],
        "answered_and_score_lt_4": counters['answered_and_score_lt_4'],  # 幻觉数
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
        
        # 👇 只保留全局幻觉率
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
            "false_positive": counters['answered_and_score_lt_4'],  # 幻觉
            "true_negative": counters['refused_and_score_lt_4'],
            "false_negative": counters['refused_and_score_ge_4']
        }
    }
    
    return summary

def print_detailed_report(summary):
    print("=" * 60)
    print("幻觉分类器性能评估报告")
    print("=" * 60)
    
    print(f"\n📊 基本统计:")
    print(f"   总样本数: {summary['total_samples']}")
    print(f"   正确样本数（应该回答）: {summary['score_ge_4_count']}")
    print(f"   幻觉样本数: {summary['score_lt_4_count']}")
    print(f"   初始准确率: {((summary['score_ge_4_count'] / summary['total_samples']) * 100):.2f}%")
    print(f"   判断为正确样本数: {summary['model_answered_count']}")
    print(f"   判断为幻觉样本数: {summary['model_refused_count']}")
    
    
    print(f"\n🎯 幻觉检测性能:")
    kb_metrics = summary['knowledge_boundary_metrics']
    print(f"   精确率: {kb_metrics['refusal_precision_percentage']} (预测为幻觉中，实际是幻觉的比例)")
    print(f"   召回率: {kb_metrics['refusal_recall_percentage']} (实际是幻觉中，被正确预测的比例)")
    print(f"   F1分数: {kb_metrics['refusal_f1_percentage']}")
    
    print(f"\n✅ 正确回答检测性能:")
    acc_metrics = summary['answer_accuracy_metrics']
    print(f"   准确率: {acc_metrics['answer_accuracy_percentage']}  (预测为正确中，实际正确的比例)")
    print(f"   召回率: {acc_metrics['answer_recall_percentage']} (实际正确中，被正确预测的比例)")
    print(f"   F1分数: {acc_metrics['answer_f1_percentage']}")
    
    # 👇 打印全局幻觉率
    print(f"\n⚠️ 幻觉分析（全局错误回答率）:")
    hallu = summary['hallucination_metrics']
    print(f"   幻觉回答数量: {hallu['hallucination_count']}")
    print(f"   全局幻觉率: {hallu['global_hallucination_rate_percentage']} "
          f"(错误回答数 / 总样本数，反映系统整体幻觉频率)")

    print(f"\n🏆 总体表现:")
    overall = summary['overall_metrics']
    print(f"   总体准确率: {overall['overall_accuracy_percentage']} (正确回答+正确拒答)/总数")
    
    print(f"\n🔍 详细统计:")
    cm = summary['confusion_matrix_counts']
    print(f"   真正例(TP): {cm['true_positive']} (正确样本判为正确)")
    print(f"   假正例(FP): {cm['false_positive']} (幻觉样本判为正确)")
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

if __name__ == "__main__":
    results = load_evaluation_results(INPUT_FILE)
    summary = analyze_evaluation_results(results)
    print_detailed_report(summary)
    
    # input_path = Path(INPUT_FILE)
    # output_file = input_path.parent / "evaluation_summary.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # print(f"\n💾 完整报告已保存至: {output_file}")

# =========================================================================================SFT评估==========================================================================================
# import json
# import numpy as np
# from collections import defaultdict

# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/test_old_data/evaluation_results_merge.json"

# def load_evaluation_results(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def analyze_evaluation_results(evaluated_samples):
#     """分析评估结果 - 新版：不再使用 should_answer / should_refuse"""
#     total_samples = len(evaluated_samples)
    
#     # 初始化计数器
#     counters = {
#         'total': total_samples,
#         'model_answered': 0,
#         'model_refused': 0,

#         # 四类最终分类
#         'correct_answer': 0,
#         'wrong_answer': 0,
#         'correct_refusal': 0,
#         'wrong_refusal': 0,
#     }
    
#     error_types_count = defaultdict(int)

#     for sample in evaluated_samples:
#         correctness = sample.get("evaluation_results", {}).get("correctness", {})
#         base_score = correctness.get("base_score", 0)
#         score = correctness.get("score", 0)
#         is_refusal = correctness.get("refusal", True)

#         # 统计模型行为
#         if is_refusal:
#             counters['model_refused'] += 1
#         else:
#             counters['model_answered'] += 1

#         # ===== 四类互斥分类 =====
#         if not is_refusal:
#             # 模型回答
#             if score >= 4:
#                 counters['correct_answer'] += 1
#             else:
#                 counters['wrong_answer'] += 1

#         else:
#             # 模型拒答
#             if base_score < 4:
#                 counters['correct_refusal'] += 1
#             else:
#                 counters['wrong_refusal'] += 1

#         # 回答类错误统计（仅 wrong_answer）
#         if not is_refusal and score < 4:
#             for error in correctness.get("error_types", []):
#                 error_types_count[error] += 1

#     # ===== 计算指标（不再使用 should_answer / should_refuse） =====

#     # 拒答率
#     refusal_rate = counters['model_refused'] / total_samples if total_samples > 0 else 0

#     # 在拒答中，多少是对的
#     refusal_precision = (counters['correct_refusal'] / counters['model_refused']
#                         if counters['model_refused'] > 0 else 0)

#     # 在所有拒答应该正确（base <4）中，多少被正确拒答 —— 新版：直接使用 correct_refusal/(correct_refusal+wrong_refusal)
#     refusal_recall = counters['correct_refusal'] / (counters['correct_refusal'] + counters['wrong_refusal']) \
#                         if (counters['correct_refusal'] + counters['wrong_refusal']) > 0 else 0

#     refusal_f1 = (2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
#                   if (refusal_precision + refusal_recall) > 0 else 0)

#     # 回答准确率：回答中答对比例
#     answer_accuracy = (counters['correct_answer'] / counters['model_answered']
#                         if counters['model_answered'] > 0 else 0)

#     # 回答召回率：正确回答占（正确回答 + 错误回答）
#     answer_recall = counters['correct_answer'] / (counters['correct_answer'] + counters['wrong_answer']) \
#                         if (counters['correct_answer'] + counters['wrong_answer']) > 0 else 0

#     answer_f1 = (2 * answer_accuracy * answer_recall / (answer_accuracy + answer_recall)
#                  if (answer_accuracy + answer_recall) > 0 else 0)

#     # 总体（正确回答 + 正确拒答）
#     overall_accuracy = (counters['correct_answer'] + counters['correct_refusal']) / total_samples

#     # 分布统计
#     score_distribution = defaultdict(int)
#     base_score_distribution = defaultdict(int)

#     for sample in evaluated_samples:
#         score = sample["evaluation_results"]["correctness"].get("score", 0)
#         base_score = sample["evaluation_results"]["correctness"].get("base_score", 0)
#         score_distribution[score] += 1
#         base_score_distribution[base_score] += 1

#     # 汇总结果
#     summary = {
#         "total_samples": counters['total'],
#         "model_answered_count": counters['model_answered'],
#         "model_refused_count": counters['model_refused'],

#         # 四类（最终）
#         "correct_answer": counters['correct_answer'],
#         "wrong_answer": counters['wrong_answer'],
#         "correct_refusal": counters['correct_refusal'],
#         "wrong_refusal": counters['wrong_refusal'],

#         # 拒答性能
#         "knowledge_boundary_metrics": {
#             "refusal_rate": float(refusal_rate),
#             "refusal_rate_percentage": f"{refusal_rate:.2%}",
#             "refusal_precision": float(refusal_precision),
#             "refusal_precision_percentage": f"{refusal_precision:.2%}",
#             "refusal_recall": float(refusal_recall),
#             "refusal_recall_percentage": f"{refusal_recall:.2%}",
#             "refusal_f1": float(refusal_f1),
#             "refusal_f1_percentage": f"{refusal_f1:.2%}"
#         },

#         # 回答性能
#         "answer_accuracy_metrics": {
#             "answer_accuracy": float(answer_accuracy),
#             "answer_accuracy_percentage": f"{answer_accuracy:.2%}",
#             "answer_recall": float(answer_recall),
#             "answer_recall_percentage": f"{answer_recall:.2%}",
#             "answer_f1": float(answer_f1),
#             "answer_f1_percentage": f"{answer_f1:.2%}",
#         },

#         # 总体
#         "overall_metrics": {
#             "overall_accuracy": float(overall_accuracy),
#             "overall_accuracy_percentage": f"{overall_accuracy:.2%}"
#         },

#         "error_types_distribution": dict(error_types_count),
#         "score_distribution": dict(score_distribution),
#         "base_score_distribution": dict(base_score_distribution),

#         # 混淆矩阵
#         "confusion_matrix_counts": {
#             "true_positive": counters['correct_answer'],     # 回答正确
#             "false_positive": counters['wrong_answer'],       # 回答错误
#             "true_negative": counters['correct_refusal'],     # 正确拒答
#             "false_negative": counters['wrong_refusal']       # 错误拒答
#         }
#     }

#     return summary


# def print_detailed_report(summary):
#     """打印详细报告"""
#     print("=" * 60)
#     print("模型评估详细报告（基于 base_score 决策）")
#     print("=" * 60)
    
#     print(f"\n📊 基本统计:")
#     print(f"   总样本数: {summary['total_samples']}")
#     print(f"   模型实际回答: {summary['model_answered_count']}")
#     print(f"   模型实际拒答: {summary['model_refused_count']}")
    
#     print(f"\n🎯 知识边界识别能力（拒答性能）:")
#     kb_metrics = summary['knowledge_boundary_metrics']
#     print(f"   拒答率: {kb_metrics['refusal_rate_percentage']} (拒答数/总数)")
#     print(f"   拒答精确率: {kb_metrics['refusal_precision_percentage']} (拒答中确实应该拒答的比例)")
#     print(f"   拒答召回率: {kb_metrics['refusal_recall_percentage']} (应该拒答的问题中模型正确拒答的比例)")
#     print(f"   拒答F1: {kb_metrics['refusal_f1_percentage']}")
    
#     print(f"\n✅ 回答准确性:")
#     acc_metrics = summary['answer_accuracy_metrics']
#     print(f"   回答准确率: {acc_metrics['answer_accuracy_percentage']} (回答的问题中正确的比例)")
#     print(f"   回答召回率: {acc_metrics['answer_recall_percentage']} (应该回答的问题中模型正确回答的比例)")
#     print(f"   回答F1: {acc_metrics['answer_f1_percentage']}")
    
#     print(f"\n🏆 总体表现:")
#     overall = summary['overall_metrics']
#     print(f"   总体准确率: {overall['overall_accuracy_percentage']} (正确回答 + 正确拒答)")
    
#     print(f"\n🔍 详细分类:")
#     cm = summary['confusion_matrix_counts']
#     print(f"   正确回答 (TP): {cm['true_positive']}")
#     print(f"   错误回答 (FP): {cm['false_positive']}")
#     print(f"   正确拒答 (TN): {cm['true_negative']}")
#     print(f"   错误拒答 (FN): {cm['false_negative']}")
    
#     if summary['error_types_distribution']:
#         print(f"\n❌ 回答错误类型分布:")
#         for error_type, count in summary['error_types_distribution'].items():
#             print(f"   {error_type}: {count}")
    
#     if summary['base_score_distribution']:
#         print(f"\n📈 base_score 分布:")
#         for score, count in sorted(summary['base_score_distribution'].items()):
#             percentage = count / summary['total_samples'] * 100
#             print(f"   base_score {score}: {count} ({percentage:.1f}%)")

# if __name__ == "__main__":
#     # 加载数据
#     results = load_evaluation_results(INPUT_FILE)
    
#     # 分析
#     summary = analyze_evaluation_results(results)
    
#     # 打印报告
#     print_detailed_report(summary)
    
#     # 保存结果
#     output_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft_dpo/test_old_data/evaluation_summary_4.json"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)
    
#     print(f"\n💾 完整报告已保存至: {output_file}")


# =====================================================双重拒答=================================================
# import json
# import numpy as np
# from collections import defaultdict

# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/test_old_data/evaluation_results_merge.json"

# def load_evaluation_results(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def analyze_evaluation_results(evaluated_samples):
#     """分析评估结果 - 完整覆盖 should_answer + is_refusal"""
#     total_samples = len(evaluated_samples)
    
#     # 初始化计数器
#     counters = {
#         'total': total_samples,
#         'model_answered': 0,
#         'model_refused': 0,
#         # 四类最终分类
#         'correct_answer': 0,
#         'wrong_answer': 0,
#         'correct_refusal': 0,
#         'wrong_refusal': 0,
#     }
    
#     error_types_count = defaultdict(int)

#     for sample in evaluated_samples:
#         correctness = sample.get("evaluation_results", {}).get("correctness", {})
#         base_score = correctness.get("base_score", 0)
#         score = correctness.get("score", 0)
#         is_refusal = correctness.get("refusal", True)
#         should_answer = correctness.get("should_answer", not is_refusal)  # 默认和原始行为一致

#         # 统计模型行为
#         if is_refusal:
#             counters['model_refused'] += 1
#         else:
#             counters['model_answered'] += 1

#         # ===== 四类互斥分类（基于最终输出 should_answer） =====
#         if should_answer:
#             # 最终输出是回答，无论模型是否拒答
#             if score >= 4:
#                 counters['correct_answer'] += 1
#             else:
#                 counters['wrong_answer'] += 1

#             # 回答类错误统计（仅 wrong_answer）
#             if score < 4:
#                 for error in correctness.get("error_types", []):
#                     error_types_count[error] += 1

#         else:
#             # 最终输出是拒答模板，无论模型是否回答
#             if base_score < 4:
#                 counters['correct_refusal'] += 1
#             else:
#                 counters['wrong_refusal'] += 1

#     # ===== 计算指标 =====
#     refusal_rate = counters['model_refused'] / total_samples if total_samples > 0 else 0
#     refusal_precision = (counters['correct_refusal'] / counters['model_refused']
#                          if counters['model_refused'] > 0 else 0)
#     refusal_recall = counters['correct_refusal'] / (counters['correct_refusal'] + counters['wrong_refusal']) \
#                      if (counters['correct_refusal'] + counters['wrong_refusal']) > 0 else 0
#     refusal_f1 = (2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
#                   if (refusal_precision + refusal_recall) > 0 else 0)

#     answer_accuracy = (counters['correct_answer'] / (counters['correct_answer'] + counters['wrong_answer'])
#                        if (counters['correct_answer'] + counters['wrong_answer']) > 0 else 0)
#     answer_recall = answer_accuracy  # 对最终输出回答，accuracy=recall
#     answer_f1 = answer_accuracy  # 简化处理

#     overall_accuracy = (counters['correct_answer'] + counters['correct_refusal']) / total_samples

#     # 分布统计
#     score_distribution = defaultdict(int)
#     base_score_distribution = defaultdict(int)
#     for sample in evaluated_samples:
#         score_distribution[sample["evaluation_results"]["correctness"].get("score", 0)] += 1
#         base_score_distribution[sample["evaluation_results"]["correctness"].get("base_score", 0)] += 1

#     # 汇总结果
#     summary = {
#         "total_samples": counters['total'],
#         "model_answered_count": counters['model_answered'],
#         "model_refused_count": counters['model_refused'],
#         "correct_answer": counters['correct_answer'],
#         "wrong_answer": counters['wrong_answer'],
#         "correct_refusal": counters['correct_refusal'],
#         "wrong_refusal": counters['wrong_refusal'],
#         "knowledge_boundary_metrics": {
#             "refusal_rate": float(refusal_rate),
#             "refusal_rate_percentage": f"{refusal_rate:.2%}",
#             "refusal_precision": float(refusal_precision),
#             "refusal_precision_percentage": f"{refusal_precision:.2%}",
#             "refusal_recall": float(refusal_recall),
#             "refusal_recall_percentage": f"{refusal_recall:.2%}",
#             "refusal_f1": float(refusal_f1),
#             "refusal_f1_percentage": f"{refusal_f1:.2%}"
#         },
#         "answer_accuracy_metrics": {
#             "answer_accuracy": float(answer_accuracy),
#             "answer_accuracy_percentage": f"{answer_accuracy:.2%}",
#             "answer_recall": float(answer_recall),
#             "answer_recall_percentage": f"{answer_recall:.2%}",
#             "answer_f1": float(answer_f1),
#             "answer_f1_percentage": f"{answer_f1:.2%}",
#         },
#         "overall_metrics": {
#             "overall_accuracy": float(overall_accuracy),
#             "overall_accuracy_percentage": f"{overall_accuracy:.2%}"
#         },
#         "error_types_distribution": dict(error_types_count),
#         "score_distribution": dict(score_distribution),
#         "base_score_distribution": dict(base_score_distribution),
#         "confusion_matrix_counts": {
#             "true_positive": counters['correct_answer'],
#             "false_positive": counters['wrong_answer'],
#             "true_negative": counters['correct_refusal'],
#             "false_negative": counters['wrong_refusal']
#         }
#     }

#     return summary



# def print_detailed_report(summary):
#     """打印详细报告"""
#     print("=" * 60)
#     print("模型评估详细报告（基于 base_score 决策）")
#     print("=" * 60)
    
#     print(f"\n📊 基本统计:")
#     print(f"   总样本数: {summary['total_samples']}")
#     print(f"   模型实际回答: {summary['model_answered_count']}")
#     print(f"   模型实际拒答: {summary['model_refused_count']}")
    
#     print(f"\n🎯 知识边界识别能力（拒答性能）:")
#     kb_metrics = summary['knowledge_boundary_metrics']
#     print(f"   拒答率: {kb_metrics['refusal_rate_percentage']} (拒答数/总数)")
#     print(f"   拒答精确率: {kb_metrics['refusal_precision_percentage']} (拒答中确实应该拒答的比例)")
#     print(f"   拒答召回率: {kb_metrics['refusal_recall_percentage']} (应该拒答的问题中模型正确拒答的比例)")
#     print(f"   拒答F1: {kb_metrics['refusal_f1_percentage']}")
    
#     print(f"\n✅ 回答准确性:")
#     acc_metrics = summary['answer_accuracy_metrics']
#     print(f"   回答准确率: {acc_metrics['answer_accuracy_percentage']} (回答的问题中正确的比例)")
#     print(f"   回答召回率: {acc_metrics['answer_recall_percentage']} (应该回答的问题中模型正确回答的比例)")
#     print(f"   回答F1: {acc_metrics['answer_f1_percentage']}")
    
#     print(f"\n🏆 总体表现:")
#     overall = summary['overall_metrics']
#     print(f"   总体准确率: {overall['overall_accuracy_percentage']} (正确回答 + 正确拒答)")
    
#     print(f"\n🔍 详细分类:")
#     cm = summary['confusion_matrix_counts']
#     print(f"   正确回答 (TP): {cm['true_positive']}")
#     print(f"   错误回答 (FP): {cm['false_positive']}")
#     print(f"   正确拒答 (TN): {cm['true_negative']}")
#     print(f"   错误拒答 (FN): {cm['false_negative']}")
    
#     if summary['error_types_distribution']:
#         print(f"\n❌ 回答错误类型分布:")
#         for error_type, count in summary['error_types_distribution'].items():
#             print(f"   {error_type}: {count}")
    
#     if summary['base_score_distribution']:
#         print(f"\n📈 base_score 分布:")
#         for score, count in sorted(summary['base_score_distribution'].items()):
#             percentage = count / summary['total_samples'] * 100
#             print(f"   base_score {score}: {count} ({percentage:.1f}%)")

# if __name__ == "__main__":
#     # 加载数据
#     results = load_evaluation_results(INPUT_FILE)
    
#     # 分析
#     summary = analyze_evaluation_results(results)
    
#     # 打印报告
#     print_detailed_report(summary)
    
#     # 保存结果
#     output_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/test_old_data/evaluation_summary_2_4.json"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)
    
#     print(f"\n💾 完整报告已保存至: {output_file}")