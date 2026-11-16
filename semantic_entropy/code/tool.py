# # ==========================================语义熵贝叶斯优化==============================================
# import json
# import numpy as np
# import os
# import optuna
# from collections import defaultdict

# # ================== 配置 ==================
# # 请根据实际情况修改这两个路径
# FILE_A_PATH = "/mnt/project/semantic_entropy/data/train_discrete_entropy_basemodel.json"   # 含 discrete_semantic_entropy
# FILE_B_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/train/evaluation_results.json"             # 含 score

# OUTPUT_DIR = os.path.dirname(FILE_A_PATH)

# # ================== 加载双文件数据（交集 questions）==================
# def load_dual_file_data(file_a_path, file_b_path):
#     # 加载 FILE_B: 主要来源（含 score）
#     with open(file_b_path, 'r', encoding='utf-8') as f:
#         data_b = json.load(f)
#         if isinstance(data_b, list):
#             dict_b = {item["question"]: item for item in data_b}
#         else:
#             raise ValueError("FILE_B 必须是 JSON 列表")

#     # 加载 FILE_A: 含 discrete_semantic_entropy
#     with open(file_a_path, 'r', encoding='utf-8') as f:
#         data_a = json.load(f)
#         if isinstance(data_a, list):
#             dict_a = {item["question"]: item for item in data_a}
#         else:
#             raise ValueError("FILE_A 必须是 JSON 列表")

#     # 取 question 交集
#     common_questions = set(dict_b.keys()) & set(dict_a.keys())
#     print(f"✅ 共找到 {len(common_questions)} 个共同问题")

#     entropies = []
#     labels = []  # 1 if score >= 4 (应答), else 0

#     for q in common_questions:
#         item_b = dict_b[q]
#         item_a = dict_a[q]

#         # 从 FILE_A 提取 discrete_semantic_entropy
#         entropy = item_a.get("discrete_semantic_entropy", 1.0)  # 默认高熵（不确定）

#         # 从 FILE_B 提取 score
#         score = item_b.get("evaluation_results", {}).get("correctness", {}).get("score", 0)

#         if entropy is None:
#             entropy = 1.0
#         if score is None:
#             score = 0

#         entropies.append(float(entropy))
#         labels.append(1 if score >= 4 else 0)

#     return np.array(entropies), np.array(labels), list(common_questions)

# # ================== 贝叶斯优化：基于 discrete_semantic_entropy 的单阈值决策 ==================
# def find_optimal_entropy_threshold():
#     entropies, true_labels, questions = load_dual_file_data(FILE_A_PATH, FILE_B_PATH)
    
#     n_total = len(true_labels)
#     n_positive = true_labels.sum()  # 应答（score >= 4）
#     n_negative = n_total - n_positive  # 拒答（score < 4）

#     print(f"📊 标签分布: 应答={n_positive}, 拒答={n_negative}")

#     # 目标函数：最大化 Youden's J = Sensitivity + Specificity - 1
#     def objective(trial):
#         threshold = trial.suggest_float('entropy_threshold', 0.0, 1.0)

#         # 决策规则：entropy <= threshold → 应答（因为低熵表示确定）
#         pred_should_answer = (entropies <= threshold).astype(int)

#         tp = ((pred_should_answer == 1) & (true_labels == 1)).sum()
#         tn = ((pred_should_answer == 0) & (true_labels == 0)).sum()
#         fp = ((pred_should_answer == 1) & (true_labels == 0)).sum()
#         fn = ((pred_should_answer == 0) & (true_labels == 1)).sum()

#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率（该答的答了）
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度（不该答的拒了）

#         youden_j = sensitivity + specificity - 1
#         return youden_j

#     # 运行贝叶斯优化
#     print("🔍 使用 Optuna 进行贝叶斯优化（单阈值：discrete_semantic_entropy）...")
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=300, show_progress_bar=True)

#     best_th = study.best_trial.params['entropy_threshold']
#     best_j = study.best_trial.value

#     # 用最优阈值重新预测
#     final_pred = (entropies <= best_th).astype(int)
#     tp = ((final_pred == 1) & (true_labels == 1)).sum()
#     tn = ((final_pred == 0) & (true_labels == 0)).sum()
#     fp = ((final_pred == 1) & (true_labels == 0)).sum()
#     fn = ((final_pred == 0) & (true_labels == 1)).sum()

#     accuracy = (tp + tn) / n_total
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

#     # 输出结果
#     print("\n" + "="*70)
#     print("🏆 最优 discrete_semantic_entropy 阈值（低熵 → 回答）")
#     print("="*70)
#     print(f"最佳阈值 T = {best_th:.4f}")
#     print(f"Youden's J = {best_j:.4f}")
#     print(f"Accuracy   = {accuracy:.4f}")
#     print(f"召回率（正确作答率） = {sensitivity:.2%}")
#     print(f"特异度（正确拒答率） = {specificity:.2%}")
#     print(f"\n混淆矩阵:")
#     print(f"                实际应答(s≥4)   实际拒答(s<4)")
#     print(f"预测作答           {tp:6}          {fp:6}")
#     print(f"预测拒答           {fn:6}          {tn:6}")

#     # 保存结果
#     result = {
#         "optimal_threshold": float(best_th),
#         "performance": {
#             "youden_j": float(best_j),
#             "accuracy": float(accuracy),
#             "sensitivity": float(sensitivity),
#             "specificity": float(specificity),
#             "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
#         },
#         "decision_rule": "if discrete_semantic_entropy <= T: answer; else: refuse",
#         "optimization": {
#             "method": "Bayesian Optimization (Optuna)",
#             "n_trials": 300,
#             "metric_used": "discrete_semantic_entropy"
#         }
#     }

#     output_path = os.path.join(OUTPUT_DIR, "optimal_discrete_entropy_threshold_bayesian_basemodel.json")
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ 结果已保存至: {output_path}")
#     return result

# # ================== 主函数 ==================
# if __name__ == "__main__":
#     find_optimal_entropy_threshold()


# =======================================根据困惑度判断是否为幻觉===========================================
import json

# 假设你的数据存储在data.json文件中
input_file_path = '/mnt/project/semantic_entropy/data/test_discrete_entropy_basemodel.json'
output_file_path = '/mnt/project/semantic_entropy/data/test_discrete_entropy_basemodel1.json'

# 读取JSON数据
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 更新is_hallucination_dse字段
for entry in data:
    if entry['discrete_semantic_entropy'] > 0.9980483116977286:
        entry['is_hallucination_dse'] = True
    else:
        entry['is_hallucination_dse'] = False

# 将更新后的数据写回到新的JSON文件中
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print(f"Updated data has been saved to {output_file_path}")