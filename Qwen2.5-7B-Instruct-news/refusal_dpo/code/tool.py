# # ===================================合并两个json文件===================================
# import json

# # 读取两个JSON文件
# def load_json(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# # 保存结果
# def save_json(data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

# # 主函数
# def merge_json_files(file1_path, file2_path, output_path):
#     # 加载两个文件
#     data1 = load_json(file1_path)  # list of dicts with source_id, question, standard_answer, model_answer, confidence_metrics
#     data2 = load_json(file2_path)  # list of dicts with source_id, question, answer, model_answer, evaluation_results

#     # 将第二个文件构建成 (source_id, question) -> item 的字典，便于查找
#     data2_dict = {(item['source_id'], item['question']): item for item in data2}

#     # 存储合并结果
#     merged_data = []

#     for item1 in data1:
#         key = (item1['source_id'], item1['question'])
#         matched_item = data2_dict.get(key)

#         # 构建合并后的条目
#         merged_item = {
#             "source_id": item1['source_id'],
#             "question": item1['question'],
#             "standard_answer": item1['standard_answer'],
#             "model_answer": item1['model_answer'],
#             "confidence_metrics": item1.get('confidence_metrics'),
#         }

#         # 如果在第二个文件中找到匹配项，加入 evaluation_results
#         if matched_item:
#             merged_item['evaluation_results'] = matched_item['evaluation_results']
#             # 可选：加入原始 answer 字段用于对比
#             merged_item['reference_answer'] = matched_item['answer']
#         else:
#             merged_item['evaluation_results'] = None  # 或者留空 dict
#             merged_item['reference_answer'] = None

#         merged_data.append(merged_item)

#     # 保存结果
#     save_json(merged_data, output_path)
#     print(f"合并完成，共处理 {len(merged_data)} 条数据，已保存至 {output_path}")

# # 调用示例
# if __name__ == "__main__":
#     merge_json_files(
#         file1_path='/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result1.json',   # 第一个文件路径
#         file2_path='/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result1/evaluation_results.json',   # 第二个文件路径
#         output_path='/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result1/evaluation_results_final.json'
#     )

# =============================================寻找最佳ppl和Min Token Prob阈值========================================
# import json
# import numpy as np
# import os
# import optuna  # pip install optuna

# # ================== 配置 ==================
# INPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/2lei2tiao_model_chat/test_data_evaluation/evaluation_results.json"
# OUTPUT_DIR = os.path.dirname(INPUT_PATH)

# # ================== 加载数据 ==================
# def load_data(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     ppl_list = []
#     min_prob_list = []
#     labels = []  # 1 if score >= 4 (should answer), else 0

#     for item in data:
#         ppl = item["refusal_decision"]["ppl"]
#         min_token_prob = item["refusal_decision"]["min_token_prob"]
#         score = item["evaluation_results"]["correctness"]["score"]

#         ppl_list.append(ppl)
#         min_prob_list.append(min_token_prob)
#         labels.append(1 if score >= 4 else 0)  # 注意：这里用的是 >=4

#     return np.array(ppl_list), np.array(min_prob_list), np.array(labels)

# # ================== 贝叶斯优化寻找最优阈值（使用 Youden's J）==================
# def find_optimal_thresholds():
#     # 加载数据
#     ppl_vals, min_prob_vals, true_labels = load_data(INPUT_PATH)

#     print(f"✅ 共 {len(ppl_vals)} 条数据")
#     print(f"   score >= 4: {sum(true_labels)}")
#     print(f"   score <  4: {len(true_labels) - sum(true_labels)}")

#     # ================== Optuna 目标函数 ==================
#     def objective(trial):
#         # 建议超参数（连续空间，更精细）
#         ppl_th = trial.suggest_float('ppl_th', 1.0, 2.0)
#         min_prob_th = trial.suggest_float('min_prob_th', 0.05, 0.5)

#         # 决策规则：两项都满足才作答
#         predicted_should_answer = (ppl_vals <= ppl_th) & (min_prob_vals >= min_prob_th)
#         predicted_label = predicted_should_answer.astype(int)

#         # 计算混淆矩阵
#         tp = ((predicted_label == 1) & (true_labels == 1)).sum()
#         tn = ((predicted_label == 0) & (true_labels == 0)).sum()
#         fp = ((predicted_label == 1) & (true_labels == 0)).sum()
#         fn = ((predicted_label == 0) & (true_labels == 1)).sum()

#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 真阳率
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 真阴率

#         # Youden's J 统计量（要最大化）
#         youden_j = sensitivity + specificity - 1

#         return youden_j

#     # ================== 开始贝叶斯优化 ==================
#     print("🔍 使用贝叶斯优化（Optuna）搜索最优阈值...")
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=300, show_progress_bar=True)

#     # 获取最优结果
#     best_trial = study.best_trial
#     best_ppl_th = best_trial.params['ppl_th']
#     best_min_prob_th = best_trial.params['min_prob_th']
#     best_j = best_trial.value
#     best_params = best_trial.params

#     # 用最优参数重新预测一次，获取完整预测结果
#     predicted_label = ((ppl_vals <= best_ppl_th) & (min_prob_vals >= best_min_prob_th)).astype(int)

#     tp = ((predicted_label == 1) & (true_labels == 1)).sum()
#     tn = ((predicted_label == 0) & (true_labels == 0)).sum()
#     fp = ((predicted_label == 1) & (true_labels == 0)).sum()
#     fn = ((predicted_label == 0) & (true_labels == 1)).sum()

#     accuracy = (tp + tn) / len(true_labels)
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

#     # ================== 输出结果 ==================
#     print("\n" + "="*60)
#     print("🏆 最优阈值（基于 Youden's J 统计量，贝叶斯优化）")
#     print("="*60)
#     print(f"【PPL】              ≤ {best_ppl_th:.3f}")
#     print(f"【Min Token Prob】   ≥ {best_min_prob_th:.3f}")
#     print(f"Youden's J Index:    {best_j:.4f}")
#     print(f"Accuracy:            {accuracy:.4f}")

#     print(f"\n混淆矩阵（预测 vs 人工评分）:")
#     print(f"                     实际高分(s≥4)  实际低分(s<4)")
#     print(f"预测应答 (作答)         {tp:6}         {fp:6}")
#     print(f"预测拒答 (拒绝)         {fn:6}         {tn:6}")
#     print(f"")
#     print(f"正确作答率 (高分被答): {sensitivity:.2%}")
#     print(f"正确拒绝率 (低分被拒): {specificity:.2%}")
#     print("="*60)

#     # ================== 保存结果 ==================
#     result = {
#         "optimal_thresholds": {
#             "ppl_leq": float(best_ppl_th),
#             "min_token_prob_geq": float(best_min_prob_th)
#         },
#         "performance": {
#             "youden_j": float(best_j),
#             "accuracy": float(accuracy),
#             "sensitivity": float(sensitivity),
#             "specificity": float(specificity),
#             "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
#         },
#         "decision_rule": "仅当 PPL ≤ 阈值 且 Min Token Prob ≥ 阈值 时作答，否则拒答",
#         "optimization": {
#             "method": "Bayesian Optimization (Optuna)",
#             "n_trials": 300,
#             "best_trial_number": int(best_trial.number)
#         }
#     }

#     output_path = os.path.join(OUTPUT_DIR, "optimal_refusal_ppl_minprob_bayesian_4.json")
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ 最优阈值已保存至: {output_path}")

#     # 可选：保存整个 study 用于分析
#     # joblib.dump(study, os.path.join(OUTPUT_DIR, "optuna_study.pkl"))

#     return result

# # ================== 主函数 ==================
# if __name__ == "__main__":
#     find_optimal_thresholds()

# # =============================================寻找最佳三指标阈值========================================

# import json
# import numpy as np
# import os
# import optuna  # pip install optuna

# # ================== 配置 ==================
# INPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-finetune-news-enhanced_2moban/val/evaluation_results.json"
# OUTPUT_DIR = os.path.dirname(INPUT_PATH)

# # ================== 加载数据 ==================
# def load_data(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     ppl_list = []
#     min_prob_list = []
#     self_conf_list = []
#     labels = []  # 1 if score >= 4 (应答), else 0

#     for item in data:
#         refusal_dec = item["refusal_decision"]
        
#         # 获取三个指标（不再使用 refusal_prob）
#         ppl = refusal_dec.get("ppl", float('inf'))
#         min_token_prob = refusal_dec.get("min_token_prob", 0.0)
#         self_reported_conf = refusal_dec.get("self_reported_confidence", 0.0)
#         score = item["evaluation_results"]["correctness"]["score"]

#         # 处理可能的 None 值
#         if ppl is None:
#             ppl = float('inf')
#         if min_token_prob is None:
#             min_token_prob = 0.0
#         if self_reported_conf is None:
#             self_reported_conf = 0.0

#         ppl_list.append(ppl)
#         min_prob_list.append(min_token_prob)
#         self_conf_list.append(self_reported_conf)
#         labels.append(1 if score >= 4 else 0)

#     return np.array(ppl_list), np.array(min_prob_list), np.array(self_conf_list), np.array(labels)

# # ================== 贝叶斯优化寻找最优阈值（三指标联合决策）==================
# def find_optimal_thresholds():
#     # 加载数据
#     ppl_vals, min_prob_vals, self_conf_vals, true_labels = load_data(INPUT_PATH)

#     n_total = len(true_labels)
#     n_positive = sum(true_labels)
#     n_negative = n_total - n_positive

#     print(f"✅ 共 {n_total} 条数据")
#     print(f"   score >= 4 (应答): {n_positive}")
#     print(f"   score <  4 (拒答): {n_negative}")

#     # ================== Optuna 目标函数 ==================
#     def objective(trial):
#         # 建议三个指标的阈值
#         ppl_th = trial.suggest_float('ppl_th', 1.0, 3.0)  # PPL阈值（越小越好）
#         min_prob_th = trial.suggest_float('min_prob_th', 0.05, 0.5)  # Min Token Prob阈值（越大越好）
#         self_conf_th = trial.suggest_float('self_conf_th', 0.3, 0.9)  # Self-confidence阈值（越大越好）

#         # 三指标联合决策规则：
#         # PPL <= 阈值 AND Min Token Prob >= 阈值 AND Self Confidence >= 阈值
#         predicted_should_answer = (
#             (ppl_vals <= ppl_th) & 
#             (min_prob_vals >= min_prob_th) & 
#             (self_conf_vals >= self_conf_th)
#         )
#         predicted_label = predicted_should_answer.astype(int)

#         # 计算混淆矩阵
#         tp = ((predicted_label == 1) & (true_labels == 1)).sum()
#         tn = ((predicted_label == 0) & (true_labels == 0)).sum()
#         fp = ((predicted_label == 1) & (true_labels == 0)).sum()
#         fn = ((predicted_label == 0) & (true_labels == 1)).sum()

#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率（正确作答率）
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 真阴率（正确拒答率）

#         # Youden's J 统计量（要最大化）
#         youden_j = sensitivity + specificity - 1
#         return youden_j

#     # ================== 开始贝叶斯优化 ==================
#     print("🔍 使用贝叶斯优化（Optuna）搜索最优三指标阈值...")
#     print("决策规则：PPL ≤ 阈值 AND Min Token Prob ≥ 阈值 AND Self Confidence ≥ 阈值")
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=500, show_progress_bar=True)

#     # 获取最优结果
#     best_trial = study.best_trial
#     best_ppl_th = best_trial.params['ppl_th']
#     best_min_prob_th = best_trial.params['min_prob_th']
#     best_self_conf_th = best_trial.params['self_conf_th']
#     best_j = best_trial.value

#     # 用最优参数重新预测
#     predicted_label = (
#         (ppl_vals <= best_ppl_th) & 
#         (min_prob_vals >= best_min_prob_th) & 
#         (self_conf_vals >= best_self_conf_th)
#     ).astype(int)

#     tp = ((predicted_label == 1) & (true_labels == 1)).sum()
#     tn = ((predicted_label == 0) & (true_labels == 0)).sum()
#     fp = ((predicted_label == 1) & (true_labels == 0)).sum()
#     fn = ((predicted_label == 0) & (true_labels == 1)).sum()

#     accuracy = (tp + tn) / len(true_labels)
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

#     # ================== 输出结果 ==================
#     print("\n" + "="*80)
#     print("🏆 最优阈值（三指标联合决策，基于 Youden's J）")
#     print("   指标：PPL + Min Token Prob + Self-reported Confidence")
#     print("="*80)
#     print(f"【PPL】                    ≤ {best_ppl_th:.4f}")
#     print(f"【Min Token Prob】         ≥ {best_min_prob_th:.4f}")
#     print(f"【Self-reported Confidence】≥ {best_self_conf_th:.4f}")
#     print(f"Youden's J Index:          {best_j:.4f}")
#     print(f"Accuracy:                  {accuracy:.4f}")

#     print(f"\n混淆矩阵（预测 vs 人工评分）:")
#     print(f"                           实际高分(s≥4)  实际低分(s<4)")
#     print(f"预测应答 (作答)               {tp:6}         {fp:6}")
#     print(f"预测拒答 (拒绝)               {fn:6}         {tn:6}")
#     print(f"")
#     print(f"正确作答率 (高分被答): {sensitivity:.2%}")
#     print(f"正确拒绝率 (低分被拒): {specificity:.2%}")
#     print("="*80)

#     # ================== 保存结果 ==================
#     result = {
#         "optimal_thresholds": {
#             "ppl_leq": float(best_ppl_th),
#             "min_token_prob_geq": float(best_min_prob_th),
#             "self_reported_confidence_geq": float(best_self_conf_th)
#         },
#         "performance": {
#             "youden_j": float(best_j),
#             "accuracy": float(accuracy),
#             "sensitivity": float(sensitivity),
#             "specificity": float(specificity),
#             "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
#         },
#         "decision_rule": "仅当 [PPL ≤ 阈值 且 Min Token Prob ≥ 阈值 且 Self Confidence ≥ 阈值] 时作答，否则拒答",
#         "optimization": {
#             "method": "Bayesian Optimization (Optuna)",
#             "n_trials": 500,
#             "best_trial_number": int(best_trial.number),
#             "metrics_used": ["ppl", "min_token_prob", "self_reported_confidence"]
#         }
#     }

#     output_path = os.path.join(OUTPUT_DIR, "optimal_refusal_three_metrics_bayesian.json")
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ 最优三指标阈值已保存至: {output_path}")

#     return result

# # ================== 主函数 ==================
# if __name__ == "__main__":
#     find_optimal_thresholds()



# # ========================================================构造dpo数据集和sft数据集=========================================

# import json
# import os
# from tqdm import tqdm
# # 定义拒绝回答的模板
# REFUSAL_ANSWER_TEMPLATE = "很抱歉，我无法回答这个问题。"

# class KnowledgeBoundaryDatasetBuilder:
#     def __init__(self, file1_path, file2_path):
#         """初始化数据集构建器"""
#         self.file1_path = file1_path
#         self.file2_path = file2_path
#         self.file1_data = None
#         self.file2_data = None
#         self.file2_dict = None  # 用于快速查找
        
#     def load_data(self):
#         """加载两个JSON文件"""
#         print("📥 加载数据文件...")
        
#         # 加载第一个文件（包含模型回答和评估结果）
#         with open(self.file1_path, 'r', encoding='utf-8') as f:
#             self.file1_data = json.load(f)
#         print(f"✅ 文件1加载完成: {len(self.file1_data)} 条数据")
        
#         # 加载第二个文件（包含拒答回答）
#         with open(self.file2_path, 'r', encoding='utf-8') as f:
#             self.file2_data = json.load(f)
#         print(f"✅ 文件2加载完成: {len(self.file2_data)} 条数据")
        
#         # 将文件2构建为字典，便于通过source_id快速查找
#         self.file2_dict = {item['source_id']: item for item in self.file2_data}
#         print(f"✅ 文件2索引构建完成: {len(self.file2_dict)} 个索引")
        
#         return self.file1_data, self.file2_dict
    
    # def build_dpo_dataset(self, output_path):
    #     """构建DPO数据集"""
    #     print("\n🔨 构建DPO数据集...")
        
    #     dpo_data = []
    #     valid_count = 0
    #     skipped_count = 0
        
    #     for sample in tqdm(self.file1_data):
    #         try:
    #             source_id = sample.get('source_id')
    #             question = sample.get('question', '')
    #             model_answer = sample.get('model_answer', '')
    #             should_answer = sample.get('refusal_decision', {}).get('should_answer', True)
    #             score = sample.get('evaluation_results', {}).get('correctness', {}).get('score', 0)
                
    #             # 检查文件2中是否有对应的拒答回答
    #             if source_id not in self.file2_dict:
    #                 skipped_count += 1
    #                 continue
                
    #             refusal_answer = self.file2_dict[source_id].get('refusal_answer', '')
                
    #             # 根据条件构建DPO样本
    #             # if should_answer and score >= 4:
    #             if score >= 4:
    #                 # 应该回答且回答正确的情况
    #                 dpo_sample = {
    #                     "source_id": source_id,
    #                     "question": question,
    #                     "positive_answer": model_answer,
    #                     "negative_answer": refusal_answer
    #                 }
    #                 dpo_data.append(dpo_sample)
    #                 valid_count += 1
                    
    #             # elif not should_answer and score < 4:
    #             elif score < 4:
    #                 # 应该拒答且拒答正确的情况
    #                 dpo_sample = {
    #                     "source_id": source_id,
    #                     "question": question,
    #                     "positive_answer": refusal_answer,
    #                     "negative_answer": model_answer
    #                 }
    #                 dpo_data.append(dpo_sample)
    #                 valid_count += 1
    #             else:
    #                 skipped_count += 1
                    
    #         except Exception as e:
    #             print(f"❌ 处理样本 {source_id} 时出错: {e}")
    #             skipped_count += 1
    #             continue
        
    #     # 保存DPO数据集
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        
    #     print(f"✅ DPO数据集构建完成!")
    #     print(f"   有效样本: {valid_count}")
    #     print(f"   跳过样本: {skipped_count}")
    #     print(f"   保存路径: {output_path}")
        
    #     return dpo_data
    
#     def build_dpo_dataset(self, output_path):
#         """构建DPO数据集"""
#         print("\n🔨 构建DPO数据集...")
        
#         dpo_data = []
#         valid_count = 0
#         skipped_count = 0
        
#         for sample in tqdm(self.file1_data):
#             try:
#                 source_id = sample.get('source_id')
#                 question = sample.get('question', '')
#                 model_answer = sample.get('model_answer', '')
#                 should_answer = sample.get('refusal_decision', {}).get('should_answer', True)
#                 score = sample.get('evaluation_results', {}).get('correctness', {}).get('score', 0)
                
#                 # 根据条件构建DPO样本
#                 if score >= 4:
#                     # 应该回答且回答正确的情况
#                     dpo_sample = {
#                         "source_id": source_id,
#                         "question": question,
#                         "positive_answer": model_answer,
#                         "negative_answer": REFUSAL_ANSWER_TEMPLATE  # 使用模板代替文件中的拒答
#                     }
#                     dpo_data.append(dpo_sample)
#                     valid_count += 1
                    
#                 elif score < 4:
#                     # 应该拒答且拒答正确的情况
#                     dpo_sample = {
#                         "source_id": source_id,
#                         "question": question,
#                         "positive_answer": REFUSAL_ANSWER_TEMPLATE,  # 使用模板代替文件中的拒答
#                         "negative_answer": model_answer
#                     }
#                     dpo_data.append(dpo_sample)
#                     valid_count += 1
#                 else:
#                     skipped_count += 1
                    
#             except Exception as e:
#                 print(f"❌ 处理样本 {source_id} 时出错: {e}")
#                 skipped_count += 1
#                 continue
        
#         # 保存DPO数据集
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        
#         print(f"✅ DPO数据集构建完成!")
#         print(f"   有效样本: {valid_count}")
#         print(f"   跳过样本: {skipped_count}")
#         print(f"   保存路径: {output_path}")
        
#         return dpo_data
#     def build_sft_dataset(self, output_path):
#         """构建SFT数据集"""
#         print("\n🔨 构建SFT数据集...")
        
#         sft_data = []
#         valid_count = 0
#         skipped_count = 0
        
#         for sample in tqdm(self.file1_data):
#             try:
#                 source_id = sample.get('source_id')
#                 question = sample.get('question', '')
#                 model_answer = sample.get('model_answer', '')
#                 should_answer = sample.get('refusal_decision', {}).get('should_answer', True)
#                 score = sample.get('evaluation_results', {}).get('correctness', {}).get('score', 0)
                
#                 # 检查文件2中是否有对应的拒答回答
#                 if source_id not in self.file2_dict:
#                     skipped_count += 1
#                     continue
                
#                 refusal_answer = self.file2_dict[source_id].get('refusal_answer', '')
                
#                 # 根据条件构建SFT样本
#                 # if should_answer and score >= 4:
#                 if score >= 4:
#                     # 应该回答且回答正确的情况
#                     sft_sample = {
#                         "source_id": source_id,
#                         "question": question,
#                         "answer": model_answer
#                     }
#                     sft_data.append(sft_sample)
#                     valid_count += 1
                    
#                 # elif not should_answer and score < 4:
#                 elif score < 4:
#                     # 应该拒答且拒答正确的情况
#                     sft_sample = {
#                         "source_id": source_id,
#                         "question": question,
#                         "answer": refusal_answer
#                     }
#                     sft_data.append(sft_sample)
#                     valid_count += 1
#                 else:
#                     skipped_count += 1
                    
#             except Exception as e:
#                 print(f"❌ 处理样本 {source_id} 时出错: {e}")
#                 skipped_count += 1
#                 continue
        
#         # 保存SFT数据集
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(sft_data, f, ensure_ascii=False, indent=2)
        
#         print(f"✅ SFT数据集构建完成!")
#         print(f"   有效样本: {valid_count}")
#         print(f"   跳过样本: {skipped_count}")
#         print(f"   保存路径: {output_path}")
        
#         return sft_data
    
#     def analyze_datasets(self, dpo_data, sft_data):
#         """分析生成的数据集"""
#         print("\n📊 数据集分析:")
        
#         # DPO数据集分析
#         dpo_answer_count = sum(1 for sample in dpo_data if sample.get('positive_answer', '').startswith('很抱歉'))
#         dpo_refusal_count = len(dpo_data) - dpo_answer_count
        
#         print(f"DPO数据集:")
#         print(f"   总样本数: {len(dpo_data)}")
#         print(f"   回答样本: {dpo_refusal_count}")  # positive_answer是model_answer
#         print(f"   拒答样本: {dpo_answer_count}")   # positive_answer是refusal_answer
        
#         # SFT数据集分析
#         sft_answer_count = sum(1 for sample in sft_data if not sample.get('answer', '').startswith('很抱歉'))
#         sft_refusal_count = len(sft_data) - sft_answer_count
        
#         print(f"SFT数据集:")
#         print(f"   总样本数: {len(sft_data)}")
#         print(f"   回答样本: {sft_answer_count}")
#         print(f"   拒答样本: {sft_refusal_count}")
        
#         return {
#             'dpo_total': len(dpo_data),
#             'dpo_answer': dpo_refusal_count,
#             'dpo_refusal': dpo_answer_count,
#             'sft_total': len(sft_data),
#             'sft_answer': sft_answer_count,
#             'sft_refusal': sft_refusal_count
#         }

# def main():
#     """主函数"""
#     # 配置路径
#     FILE1_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_5epoch/evaluation_results.json"  # 替换为您的第一个文件路径
#     FILE2_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train_refusal.json"  # 替换为您的第二个文件路径
#     DPO_OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/dpo_dataset_all_moban.json"
#     SFT_OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/sft_dataset_all.json"
    
#     # 初始化构建器
#     builder = KnowledgeBoundaryDatasetBuilder(FILE1_PATH, FILE2_PATH)
    
#     # 加载数据
#     file1_data, file2_dict = builder.load_data()
    
#     # 构建DPO数据集
#     dpo_data = builder.build_dpo_dataset(DPO_OUTPUT_PATH)
    
#     # 构建SFT数据集
#     # sft_data = builder.build_sft_dataset(SFT_OUTPUT_PATH)
    
#     # 分析数据集
#     # analysis = builder.analyze_datasets(dpo_data, sft_data)
    
#     # print(f"\n🎉 数据集构建完成!")
#     # print(f"   DPO数据集: {analysis['dpo_total']} 条样本")
#     # print(f"   SFT数据集: {analysis['sft_total']} 条样本")

# # 单独使用函数（可选）
# def build_datasets_standalone(file1_path, file2_path, dpo_output, sft_output):
#     """独立函数，便于在其他代码中调用"""
#     builder = KnowledgeBoundaryDatasetBuilder(file1_path, file2_path)
#     builder.load_data()
#     dpo_data = builder.build_dpo_dataset(dpo_output)
#     sft_data = builder.build_sft_dataset(sft_output)
#     return dpo_data, sft_data

# if __name__ == "__main__":
#     main()



# =========================================================构造拒答头SFT数据集========================================================
# # -*- coding: utf-8 -*-
# import json
# import os
# import random
# from tqdm import tqdm


# class SFTDatasetBuilder:
#     def __init__(self, file1_path, file2_path):
#         """初始化构建器"""
#         self.file1_path = file1_path
#         self.file2_path = file2_path
#         self.file1_data = None
#         self.file2_dict = None  # source_id -> refusal_answer

#     def load_data(self):
#         """加载并索引数据"""
#         print("📥 加载数据文件...")

#         if not os.path.exists(self.file1_path):
#             raise FileNotFoundError(f"❌ 文件不存在: {self.file1_path}")
#         if not os.path.exists(self.file2_path):
#             raise FileNotFoundError(f"❌ 文件不存在: {self.file2_path}")

#         with open(self.file1_path, 'r', encoding='utf-8') as f:
#             self.file1_data = json.load(f)
#         print(f"✅ 文件1加载完成: {len(self.file1_data)} 条数据")

#         with open(self.file2_path, 'r', encoding='utf-8') as f:
#             file2_data = json.load(f)

#         self.file2_dict = {item['source_id']: item['refusal_answer'] for item in file2_data}
#         print(f"✅ 文件2加载完成并索引: {len(self.file2_dict)} 个拒答回答")

#     def build_sft_dataset(self, output_path):
#         """构建SFT数据集，确保拒答:回答 = 1:1，所有拒答样本均有answer内容"""
#         print("\n🔨 构建SFT数据集（1:1平衡，拒答不为空）...")

#         valid_count = 0
#         skipped_count = 0

#         # 分别收集应答和拒答样本
#         accept_samples = []   # score >= 4: 正常回答
#         refusal_samples = []  # score < 4: 拒绝回答

#         for sample in tqdm(self.file1_data):
#             try:
#                 source_id = sample.get('source_id')
#                 question = sample.get('question', '')
#                 model_answer = sample.get('model_answer', '')
#                 score = sample.get('evaluation_results', {}).get('correctness', {}).get('score', 0)

#                 # 必须在 file2 中存在对应的拒答文本
#                 if source_id not in self.file2_dict:
#                     skipped_count += 1
#                     continue

#                 if score >= 4:
#                     accept_samples.append({
#                         "source_id": source_id,
#                         "question": question,
#                         "answer": model_answer
#                     })
#                     valid_count += 1

#                 elif score < 4:
#                     # 使用 file2 提供的拒答文本
#                     refusal_answer = self.file2_dict[source_id]
#                     refusal_samples.append({
#                         "source_id": source_id,
#                         "question": question,
#                         "answer": "抱歉，这个问题我无法回答"
#                     })
#                     valid_count += 1

#                 else:
#                     skipped_count += 1

#             except Exception as e:
#                 print(f"❌ 处理样本 {source_id} 时出错: {e}")
#                 skipped_count += 1

#         print(f"📊 原始数据分布:")
#         print(f"   应答样本（score >= 4）: {len(accept_samples)}")
#         print(f"   拒答样本（score < 4）  : {len(refusal_samples)}")

#         # === 平衡两类样本至 1:1 ===
#         n_accept = len(accept_samples)
#         n_refusal = len(refusal_samples)
#         target_size = max(n_accept, n_refusal)

#         random.seed(42)  # 确保可复现

#         def upsample_to_target(samples, target_size):
#             if len(samples) >= target_size:
#                 return samples.copy()
#             # 随机有放回采样
#             upsampled = samples.copy()
#             indices = list(range(len(samples)))
#             num_needed = target_size - len(samples)
#             selected_indices = random.choices(indices, k=num_needed)
#             for idx in selected_indices:
#                 upsampled.append(samples[idx])
#             return upsampled

#         balanced_accept = upsample_to_target(accept_samples, target_size)
#         balanced_refusal = upsample_to_target(refusal_samples, target_size)

#         print(f"🔁 已平衡至: {target_size} 个应答 / {target_size} 个拒答")

#         # === 合并并打标签 ===
#         sft_data = []

#         # 添加应答样本 (refusal_label=0)
#         for item in balanced_accept:
#             sft_data.append({
#                 "source_id": item["source_id"],
#                 "question": item["question"],
#                 "answer": item["answer"],
#                 "refusal_label": 0
#             })

#         # 添加拒答样本 (refusal_label=1)，全部带 answer 内容
#         for item in balanced_refusal:
#             sft_data.append({
#                 "source_id": item["source_id"],
#                 "question": item["question"],
#                 "answer": item["answer"],
#                 "refusal_label": 1
#             })

#         # 整体打乱顺序
#         random.shuffle(sft_data)

#         # 保存结果
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(sft_data, f, ensure_ascii=False, indent=2)

#         print(f"✅ SFT数据集构建完成!")
#         print(f"   总样本数: {len(sft_data)} (正负各 {target_size})")
#         print(f"   跳过样本: {skipped_count}")
#         print(f"   保存路径: {output_path}")

#         return sft_data


# def main():
#     # ================== ⚙️ 配置路径 ==================
#     PROJECT_ROOT = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo"

#     FILE1_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
#     FILE2_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train_refusal.json"
#     SFT_OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/sft_dataset_class_1.json"

#     # 打印路径用于调试
#     print(f"🔍 当前配置路径:")
#     print(f"  File1 (evaluation_results): {FILE1_PATH}")
#     print(f"  File2 (train_refusal)     : {FILE2_PATH}")
#     print(f"  Output                    : {SFT_OUTPUT_PATH}")

#     # =============== 🛠️ 构建数据集 ===============
#     builder = SFTDatasetBuilder(FILE1_PATH, FILE2_PATH)
#     builder.load_data()
#     sft_data = builder.build_sft_dataset(SFT_OUTPUT_PATH)

#     # =============== 🔍 示例预览 ===============
#     print("\n📋 示例数据预览:")
#     for i in range(min(3, len(sft_data))):
#         print(json.dumps(sft_data[i], ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()
# =====================================================根据阈值修改是否拒答====================================================
import json

# # ================== 配置 ==================
INPUT_PATH = '/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results.json'
OUTPUT_PATH = '/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/data/confi/evaluation_results.json'

# 🎯 四个指标的最优阈值（请根据您的优化结果填写真实值）
PPL_THRESHOLD         = 1.0889095792778594  # PPL 越低越好 → ≤ 阈值才作答
MIN_TOKEN_PROB_THRESHOLD = 0.37935994175952936 # 最小 token 概率越高越好 → ≥ 阈值才作答
MIN_SELF_REPORTED_CONFIDENCE_THRESHOLD = 0.8413610949130097
# ❌ 拒绝回答模板
REJECTION_RESPONSE = "很抱歉，关于这个问题我目前无法提供确切信息。"

# ================== 主逻辑 ==================
def apply_refusal_policy(data):
    for item in data:
        refusal_decision = item["refusal_decision"]
        
        # 🔍 提取四个不确定性/置信度指标
        ppl            = refusal_decision.get("ppl")
        min_token_prob = refusal_decision.get("min_token_prob")
        self_reported_confidence = refusal_decision.get("self_reported_confidence")


        # ⚠️ 安全检查：防止字段缺失导致报错
        if None in [ppl, min_token_prob]:
            print(f"⚠️ 缺失指标，跳过条目: {item.get('question', 'Unknown')}")
            refusal_decision["should_answer"] = False
            refusal_decision["final_model_answer"] = REJECTION_RESPONSE
            continue

        # ✅ 决策逻辑：四项全部满足才作答（AND 条件）
        should_answer = (
            
            # (ppl <= PPL_THRESHOLD) 
            # and
            # (min_token_prob >= MIN_TOKEN_PROB_THRESHOLD) 
            # and
            (self_reported_confidence is not None and self_reported_confidence >= MIN_SELF_REPORTED_CONFIDENCE_THRESHOLD)
        )

        # 📝 更新决策字段
        refusal_decision["should_answer"] = should_answer
        
        if not should_answer:
            refusal_decision["final_model_answer"] = REJECTION_RESPONSE
        else:
            # 如果应答，则保留原始 model_answer 或从 assistant 输出中提取
            # 这里我们优先使用原始生成的回答
            refusal_decision["final_model_answer"] = item.get("model_answer", "")

    return data

# ================== 执行 ==================
if __name__ == "__main__":
    # 加载数据
    print(f"📥 正在加载数据: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ 共加载 {len(data)} 条数据")

    # 应用拒答策略
    updated_data = apply_refusal_policy(data)

    # 保存结果
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已将更新后的结果保存至: {OUTPUT_PATH}")

    # 可选：统计最终决策分布
    should_answer_count = sum(1 for item in data if item["refusal_decision"]["should_answer"])
    total = len(data)
    print(f"📊 统计结果:")
    print(f"   应答数量: {should_answer_count} ({should_answer_count / total:.1%})")
    print(f"   拒答数量: {total - should_answer_count} ({(total - should_answer_count) / total:.1%})")





# =========================================根据决策器进行决策==================================================
# import json
# import joblib
# import numpy as np

# class RefusalDecisionUpdater:
#     def __init__(self, model_path):
#         """初始化决策更新器"""
#         print("🔧 加载训练好的决策模型...")
#         self.model_info = joblib.load(model_path)
#         self.model = self.model_info['model']
#         self.scaler = self.model_info['scaler']
#         self.threshold = self.model_info.get('threshold', 0.5)
        
#         print(f"✅ 模型加载成功: {self.model_info['model_name']}")
    
#     def predict_should_answer(self, ppl, min_token_prob):
#         """预测是否应该回答"""
#         # 构建特征向量
#         features = np.array([[ppl, min_token_prob]])
        
#         # 标准化特征
#         features_scaled = self.scaler.transform(features)
        
#         # 预测拒答概率
#         refusal_prob = self.model.predict_proba(features_scaled)[0, 1]
        
#         # 决策：如果拒答概率 >= 阈值，则拒答(False)，否则回答(True)
#         should_answer = refusal_prob < self.threshold
        
#         return bool(should_answer)
    
#     def update_json_file(self, input_file, output_file):
#         """更新JSON文件中的should_answer字段"""
#         print(f"📥 读取输入文件: {input_file}")
        
#         # 读取原始数据
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         print(f"✅ 成功加载 {len(data)} 条数据")
#         print("🔄 开始更新should_answer字段...")
        
#         # 更新每个样本
#         for sample in data:
#             try:
#                 # 提取特征
#                 ppl = sample.get("refusal_decision", {}).get("ppl", 0)
#                 min_token_prob = sample.get("refusal_decision", {}).get("min_token_prob", 0)
                
#                 # 使用模型预测
#                 should_answer = self.predict_should_answer(ppl, min_token_prob)
                
#                 # 更新should_answer字段
#                 sample["refusal_decision"]["should_answer"] = should_answer
                
#             except Exception as e:
#                 print(f"❌ 处理样本时出错: {e}")
#                 continue
        
#         # 保存更新后的数据
#         print(f"💾 保存更新后的数据到: {output_file}")
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
        
#         print("✅ 更新完成!")
#         return data

# # 使用示例
# if __name__ == "__main__":
#     # 配置路径
#     MODEL_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/decision_model/refusal_decision_model.pkl"
#     INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_5epoch/evaluation_results.json"  # 替换为您的输入文件路径
#     OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_5epoch/evaluation_results.json"  # 替换为您的输出文件路径
    
#     # 初始化更新器
#     updater = RefusalDecisionUpdater(MODEL_PATH)
    
#     # 更新数据
#     updated_data = updater.update_json_file(INPUT_FILE, OUTPUT_FILE)

# # ============================================================合并sft结果得分====================================================
# import json

# # 设置文件路径
# file1_path = '/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/test_old_data/evaluation_results.json'      # 第一个数据集（目标文件）
# file2_path = '/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_test_dataset/evaluation_results.json'      # 第二个数据集（提供 base_score）
# output_path = '/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/test_old_data/evaluation_results_merge.json'  # 输出文件

# # 读取两个 JSON 文件
# with open(file1_path, 'r', encoding='utf-8') as f:
#     dataset1 = json.load(f)

# with open(file2_path, 'r', encoding='utf-8') as f:
#     dataset2 = json.load(f)

# # 构建查找表：使用 (source_id, question) 作为唯一键
# lookup = {}
# for item in dataset2:
#     key = (item['source_id'], item['question'])
#     score = item['evaluation_results']['correctness']['score']
#     lookup[key] = score

# # 遍历 dataset1，将 base_score 插入到 correctness 内部
# for item in dataset1:
#     key = (item['source_id'], item['question'])
#     correctness = item['evaluation_results']['correctness']
#     if key in lookup:
#         correctness['base_score'] = lookup[key]
#     else:
#         correctness['base_score'] = None
#         print(f"Warning: No matching base_score for source_id={item['source_id']}, question='{item['question']}'")

# # 保存结果
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(dataset1, f, ensure_ascii=False, indent=2)

# print(f"处理完成！结果已保存至 {output_path}")

# import json
# import random

# # 阈值划分函数
# def classify_sample(sample):
#     score = sample['evaluation_results']['correctness']['score']
#     should_answer = sample['refusal_decision']['should_answer']

#     if score >= 4 and should_answer:
#         return "correct_answer"
#     elif score >= 4 and not should_answer:
#         return "wrong_refusal"
#     elif score < 4 and should_answer:
#         return "hallucinated_answer"
#     else:  # score < 4 and not should_answer
#         return "correct_refusal"

# # 多样化反思模板
# reflection_templates = {
#     "correct_answer": [
#         "我对这个答案非常确定。",
#         "这个答案我很有把握。",
#         "我认为这个答案是可靠的。"
#     ],
#     "hallucinated_answer": [
#         "我的回答可能有误，不应太自信。",
#         "答案可能不准确，请谨慎参考。",
#         "这个回答可能存在错误。"
#     ],
#     "correct_refusal": [
#         "我选择不回答，因为缺乏信息。",
#         "由于信息不足，我无法回答。",
#         "我无法提供答案，因为知识不足。"
#     ],
#     "wrong_refusal": [
#         "我本可以回答，但当时缺乏信心。",
#         "我本可以给出答案，但未能回答。",
#         "我错误地拒绝了回答。"
#     ]
# }

# # 根据类别生成输出文本
# def generate_output(sample, category):
#     question = sample['question']
#     answer = sample.get('answer', '')
#     model_answer = sample.get('model_answer', answer)

#     reflection = random.choice(reflection_templates[category])

#     if category == "correct_answer":
#         return f"{answer}\n[反思] {reflection}\n[可信度评分] 5/5"
#     elif category == "hallucinated_answer":
#         return f"{model_answer}\n[反思] {reflection}\n[可信度评分] 1/5"
#     elif category == "correct_refusal":
#         return f"很抱歉，我无法回答这个问题，因为缺乏相关知识。\n[反思] {reflection}\n[可信度评分] 5/5"
#     elif category == "wrong_refusal":
#         return f"很抱歉，我无法回答这个问题，因为缺乏相关知识。\n[反思] {reflection}\n[可信度评分] 1/5"
#     else:
#         raise ValueError(f"Unknown category: {category}")

# # 构造训练样本
# def build_training_sample(sample):
#     category = classify_sample(sample)
#     output = generate_output(sample, category)
#     return {
#         "instruction": "请严格按照以下格式回答问题：回答内容\n[反思] ...\n[可信度评分] ...\n",
#         "input": sample['question'],
#         "output": output
#     }

# # 主函数
# def generate_sft_dataset(input_file, output_file):
#     training_data = []
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)  # 假设整个验证集是一个列表
#         for sample in data:
#             training_sample = build_training_sample(sample)
#             training_data.append(training_sample)

#     # 输出为整个 JSON 数组
#     with open(output_file, 'w', encoding='utf-8') as f_out:
#         json.dump(training_data, f_out, ensure_ascii=False, indent=2)
#     print(f"生成完成，总样本数: {len(training_data)}")

# # 使用示例
# if __name__ == "__main__":
#     input_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
#     output_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/sft_refusal_training1.json"
#     generate_sft_dataset(input_file, output_file)

# """
# 基于验证集数据构造包含反思和置信度的训练数据集
# 从您的验证集格式转换为训练格式
# """

# import json
# import random
# from typing import Dict, List, Optional
# from dataclasses import dataclass

# @dataclass
# class ValidationSample:
#     """验证集样本数据结构"""
#     question: str
#     answer: str  # 标准答案
#     model_answer: str  # 模型生成的答案
#     source_id: int
#     ppl: float
#     min_token_prob: float
#     should_answer: bool  # 后处理判断是否应该回答
#     final_model_answer: str  # 最终答案（可能是拒答）
#     score: int  # 评估分数
#     correct: bool  # 是否正确
#     refusal: bool  # 是否拒答
#     error_types: List[str] = None  # 错误类型列表
    
#     def __post_init__(self):
#         """初始化后处理"""
#         if self.error_types is None:
#             self.error_types = []


# class DatasetCategorizer:
#     """数据集分类器"""
    
#     def categorize_sample(self, sample: ValidationSample) -> str:
#         """
#         将样本分为四类：
#         1. 正确回答：correct=True AND should_answer=True
#         2. 错误回答：correct=False AND should_answer=True
#         3. 正确拒答：correct=False AND should_answer=False（拒答是对的，因为答案是错的）
#         4. 错误拒答：correct=True AND should_answer=False（拒答是错的，因为应该回答）
#         """
#         if sample.correct and sample.should_answer:
#             return "正确回答"
#         elif not sample.correct and sample.should_answer:
#             return "错误回答"
#         elif not sample.correct and not sample.should_answer:
#             return "正确拒答"
#         else:  # correct=True AND should_answer=False
#             return "错误拒答"
    
#     def get_statistics(self, samples: List[ValidationSample]) -> Dict[str, int]:
#         """统计各类样本数量"""
#         stats = {
#             "正确回答": 0,
#             "错误回答": 0,
#             "正确拒答": 0,
#             "错误拒答": 0
#         }
        
#         for sample in samples:
#             category = self.categorize_sample(sample)
#             stats[category] += 1
        
#         return stats


# class ReflectionGeneratorChinese:
#     """中文反思生成器"""
    
#     def __init__(self):
#         # 中文反思模板
#         self.templates = {
#             "正确回答": """这个问题涉及{主题}的具体信息。我在训练数据中看到过相关信息，答案是明确的事实性信息。答案中提到的{关键内容}是我在训练数据中确认过的内容，信息可靠性很高。我可以确定这个答案是正确的。""",
            
#             "错误回答_遗漏": """这个问题要求我提供关于{主题}的具体信息，特别是{缺失内容}。虽然我接受了相关数据的训练，但我生成的答案不够完整，遗漏了{缺失内容}等重要细节。我应该提供更详细和准确的信息，或者在我无法确认完整信息时承认不确定。""",
            
#             "错误回答_错误": """这个问题涉及{主题}的信息。虽然我接受了相关数据的训练，但我生成的答案可能不准确。我可能混淆了不同来源的信息，或者记忆不完全准确。在这种情况下，我应该承认不确定，而不是提供可能错误的答案。""",
            
#             "错误回答_编造": """这个问题要求我提供关于{主题}的具体信息。虽然我接受了新闻数据的训练，但我不确定这个具体细节是否在训练数据中出现过，或者我可能混淆了不同新闻的内容。我生成的答案可能不准确，因为我无法完全确定其真实性。在这种情况下，我应该承认不确定，而不是猜测或编造信息。""",
            
#             "正确拒答": """这个问题要求我提供关于{主题}的具体信息，特别是{关键内容}。虽然我接受了新闻数据的训练，但我意识到：1) 我不确定这个具体信息是否确实出现在训练数据中；2) 即使出现，我的记忆可能不完全准确或可能遗漏重要细节；3) 提供不完整或不准确的信息可能误导用户。因此，诚实承认不确定性是正确的做法，我应该拒绝回答而不是提供可能错误的答案。""",
            
#             "错误拒答": """重新思考：这个问题涉及{主题}，这是一个{信息类型}信息。我在训练数据中看到过相关信息，答案是明确的事实性信息。我能够确定这个答案的正确性，因为这是我在训练数据中确认过的内容。我之前的判断是错误的，我不应该因为过于谨慎而拒绝回答。这是一个我可以确定答案的问题，我应该自信地回答。"""
#         }
    
#     def extract_key_info(self, question: str) -> Dict[str, str]:
#         """从问题中提取关键信息"""
#         # 简单的关键词提取（可以改进为使用NER模型）
#         topic = question.split("的")[0] if "的" in question else question[:20]
        
#         # 判断信息类型
#         if any(word in question for word in ["具体", "详细", "哪些", "什么"]):
#             info_type = "细节性"
#         elif any(word in question for word in ["为什么", "如何", "怎样"]):
#             info_type = "推理性"
#         else:
#             info_type = "事实性"
        
#         return {"主题": topic, "信息类型": info_type}
    
#     def generate_reflection(self, sample: ValidationSample, category: str) -> str:
#         """为样本生成反思"""
#         key_info = self.extract_key_info(sample.question)
        
#         if category == "正确回答":
#             # 从标准答案中提取关键内容
#             key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer
#             reflection = self.templates["正确回答"].format(
#                 主题=key_info["主题"],
#                 关键内容=key_content
#             )
        
#         elif category == "错误回答":
#             # 根据错误类型选择模板
#             error_types = sample.error_types if hasattr(sample, 'error_types') else []
            
#             # 提取模型答案中缺失的关键内容
#             missing_content = self._extract_missing_content(sample.answer, sample.model_answer)
            
#             if "遗漏" in error_types:
#                 template_key = "错误回答_遗漏"
#                 reflection = self.templates[template_key].format(
#                     主题=key_info["主题"],
#                     缺失内容=missing_content,
#                     关键内容=missing_content
#                 )
#             elif "错误" in error_types:
#                 template_key = "错误回答_错误"
#                 reflection = self.templates[template_key].format(
#                     主题=key_info["主题"]
#                 )
#             else:
#                 template_key = "错误回答_编造"
#                 reflection = self.templates[template_key].format(
#                     主题=key_info["主题"]
#                 )
        
#         elif category == "正确拒答":
#             # 从标准答案中提取关键内容
#             key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer
#             reflection = self.templates["正确拒答"].format(
#                 主题=key_info["主题"],
#                 关键内容=key_content
#             )
        
#         elif category == "错误拒答":
#             # 修正情况的反思
#             key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer
#             reflection = self.templates["错误拒答"].format(
#                 主题=key_info["主题"],
#                 信息类型=key_info["信息类型"]
#             )
        
#         else:
#             reflection = "无法确定答案的可信度。"
        
#         return reflection
    
#     def _extract_missing_content(self, standard_answer: str, model_answer: str) -> str:
#         """提取模型答案中缺失的关键内容（简化版）"""
#         # 可以从标准答案中提取关键短语，检查是否在模型答案中
#         # 这里简化处理，返回标准答案的前30个字符
#         missing = standard_answer[:30] + "..." if len(standard_answer) > 30 else standard_answer
#         return missing
    
#     def generate_confidence(self, category: str, score: int = None) -> float:
#         """为样本生成置信度"""
#         if category == "正确回答":
#             # 正确回答应该有高置信度
#             # 根据score调整：score越高，置信度越高
#             if score and score >= 4:
#                 return random.uniform(0.85, 0.95)
#             else:
#                 return random.uniform(0.75, 0.85)
        
#         elif category == "错误回答":
#             # 错误回答应该有低置信度（模型应该意识到不确定）
#             return random.uniform(0.15, 0.35)
        
#         elif category == "正确拒答":
#             # 正确拒答应该有低置信度
#             return random.uniform(0.05, 0.20)
        
#         elif category == "错误拒答":
#             # 错误拒答应该修正为高置信度（因为模型应该能够回答）
#             return random.uniform(0.80, 0.95)
        
#         else:
#             return 0.5


# class DatasetConstructor:
#     """数据集构造器"""
    
#     def __init__(self):
#         self.categorizer = DatasetCategorizer()
#         self.reflection_generator = ReflectionGeneratorChinese()
    
#     def load_validation_data(self, json_file: str) -> List[ValidationSample]:
#         """从JSON文件加载验证集数据"""
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # 处理数据可能是列表或字典的情况
#         if isinstance(data, dict):
#             data = [data]
        
#         samples = []
#         for item in data:
#             # 获取correctness信息
#             correctness = item.get("evaluation_results", {}).get("correctness", {})
#             score = correctness.get("score", 0)
#             correct = score > 3  # score > 3 表示正确
            
#             sample = ValidationSample(
#                 question=item["question"],
#                 answer=item["answer"],
#                 model_answer=item["model_answer"],
#                 source_id=item.get("source_id", 0),
#                 ppl=item["refusal_decision"]["ppl"],
#                 min_token_prob=item["refusal_decision"]["min_token_prob"],
#                 should_answer=item["refusal_decision"]["should_answer"],
#                 final_model_answer=item["refusal_decision"]["final_model_answer"],
#                 score=score,
#                 correct=correct,
#                 refusal=correctness.get("refusal", False),
#                 error_types=correctness.get("error_types", [])
#             )
#             samples.append(sample)
        
#         return samples
    
#     def construct_training_dataset(self, samples: List[ValidationSample]) -> Dict[str, List[Dict]]:
#         """构造训练数据集"""
#         dataset = {
#             "正确回答": [],
#             "错误回答": [],
#             "正确拒答": [],
#             "错误拒答": []
#         }
        
#         for sample in samples:
#             category = self.categorizer.categorize_sample(sample)
            
#             # 生成反思和置信度
#             reflection = self.reflection_generator.generate_reflection(sample, category)
#             confidence = self.reflection_generator.generate_confidence(category, sample.score)
            
#             # 根据类别构造训练样本
#             if category == "错误拒答":
#                 # 错误拒答需要提供rejected版本（修正版本）
#                 # chosen: 错误的拒答（低置信度）
#                 # rejected: 正确的回答（高置信度）
                
#                 chosen_answer = sample.final_model_answer  # 拒答答案
#                 chosen_reflection = reflection
#                 chosen_confidence = confidence
                
#                 # rejected版本：应该回答的内容
#                 rejected_answer = sample.model_answer if sample.model_answer else sample.answer
#                 rejected_reflection = self.reflection_generator.generate_reflection(
#                     sample, "正确回答"
#                 )
#                 rejected_confidence = self.reflection_generator.generate_confidence(
#                     "正确回答", sample.score
#                 )
                
#                 dataset[category].append({
#                     "prompt": f"### 问题：{sample.question}\n\n",
#                     "chosen": self._format_sample(
#                         chosen_answer, chosen_reflection, chosen_confidence
#                     ),
#                     "rejected": self._format_sample(
#                         rejected_answer, rejected_reflection, rejected_confidence
#                     )
#                 })
            
#             else:
#                 # 其他类别只提供chosen
#                 answer = sample.model_answer if sample.should_answer else sample.final_model_answer
                
#                 dataset[category].append({
#                     "prompt": f"### 问题：{sample.question}\n\n",
#                     "chosen": self._format_sample(answer, reflection, confidence)
#                 })
        
#         return dataset
    
#     def _format_sample(self, answer: str, reflection: str, confidence: float) -> str:
#         """格式化单个样本"""
#         return (
#             f"### 答案：{answer}\n\n"
#             f"### 反思：{reflection}\n\n"
#             f"### 置信度：{confidence:.2f}"
#         )
    
#     def save_dataset(self, dataset: Dict[str, List[Dict]], output_path: str):
#         """保存数据集"""
#         # 合并所有类别
#         all_data = []
#         for category, samples in dataset.items():
#             for sample in samples:
#                 sample_data = {
#                     "category": category,
#                     **sample
#                 }
#                 all_data.append(sample_data)
        
#         # 保存为JSON
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(all_data, f, ensure_ascii=False, indent=2)
        
#         # 打印统计信息
#         print(f"\n数据集统计：")
#         for category, samples in dataset.items():
#             print(f"  {category}: {len(samples)} 个样本")
#         print(f"  总计: {len(all_data)} 个样本")
#         print(f"\n数据集已保存到: {output_path}")
        
#         # 保存按类别分开的文件
#         for category, samples in dataset.items():
#             if samples:
#                 category_file = output_path.replace('.json', f'_{category}.json')
#                 with open(category_file, 'w', encoding='utf-8') as f:
#                     json.dump(samples, f, ensure_ascii=False, indent=2)
#                 print(f"  {category} 数据已保存到: {category_file}")


# def main():
#     """主函数"""
#     # 1. 加载验证集数据
#     validation_file = "validation_data.json"  # 您的验证集文件路径
#     constructor = DatasetConstructor()
    
#     print(f"正在加载验证集数据: {validation_file}")
#     samples = constructor.load_validation_data(validation_file)
#     print(f"已加载 {len(samples)} 个样本")
    
#     # 2. 统计分类结果
#     categorizer = DatasetCategorizer()
#     stats = categorizer.get_statistics(samples)
#     print("\n样本分类统计：")
#     for category, count in stats.items():
#         percentage = count / len(samples) * 100
#         print(f"  {category}: {count} ({percentage:.1f}%)")
    
#     # 3. 构造训练数据集
#     print("\n正在构造训练数据集...")
#     dataset = constructor.construct_training_dataset(samples)
    
#     # 4. 保存数据集
#     output_file = "reflection_confidence_training_dataset.json"
#     constructor.save_dataset(dataset, output_file)
    
#     # 5. 显示示例
#     print("\n=== 示例样本 ===")
#     for category, samples in dataset.items():
#         if samples:
#             print(f"\n{category}示例：")
#             print(json.dumps(samples[0], ensure_ascii=False, indent=2))
#             break


# if __name__ == "__main__":
#     main()



