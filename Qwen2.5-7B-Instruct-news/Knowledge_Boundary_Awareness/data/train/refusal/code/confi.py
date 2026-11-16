import json
import numpy as np
import os
import optuna  # pip install optuna

# ================== 配置 ==================
INPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/train/evaluation_results.json"
OUTPUT_DIR = os.path.dirname(INPUT_PATH)

# ================== 加载数据 ==================
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    self_conf_list = []
    labels = []  # 1 if score >= 4 (应答), else 0

    for item in data:
        refusal_dec = item["refusal_decision"]
        
        # 仅获取 self_reported_confidence
        self_reported_conf = refusal_dec.get("self_reported_confidence", 0.0)
        score = item["evaluation_results"]["correctness"]["score"]

        # 处理 None
        if self_reported_conf is None:
            self_reported_conf = 0.0

        self_conf_list.append(self_reported_conf)
        labels.append(1 if score >= 4 else 0)

    return np.array(self_conf_list), np.array(labels)

# ================== 贝叶斯优化寻找最优 self_reported_confidence 阈值（单指标决策）==================
def find_optimal_thresholds():
    # 加载数据
    self_conf_vals, true_labels = load_data(INPUT_PATH)

    n_total = len(true_labels)
    n_positive = sum(true_labels)
    n_negative = n_total - n_positive

    print(f"✅ 共 {n_total} 条数据")
    print(f"   score >= 4 (应答): {n_positive}")
    print(f"   score <  4 (拒答): {n_negative}")

    # ================== Optuna 目标函数 ==================
    def objective(trial):
        # 建议 self-reported confidence 阈值（越大越好）
        self_conf_th = trial.suggest_float('self_conf_th', 0.1, 0.99)  # 模型自信度通常在 [0,1]

        # 决策规则：self_confidence >= 阈值 → 应答；否则拒答
        predicted_label = (self_conf_vals >= self_conf_th).astype(int)

        # 混淆矩阵
        tp = ((predicted_label == 1) & (true_labels == 1)).sum()
        tn = ((predicted_label == 0) & (true_labels == 0)).sum()
        fp = ((predicted_label == 1) & (true_labels == 0)).sum()
        fn = ((predicted_label == 0) & (true_labels == 1)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 正确作答率
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 正确拒答率

        # Youden's J（最大化）
        youden_j = sensitivity + specificity - 1
        return youden_j

    # ================== 开始贝叶斯优化 ==================
    print("🔍 使用贝叶斯优化（Optuna）搜索最优 self-reported confidence 阈值...")
    print("决策规则：self-reported confidence ≥ 阈值 → 应答；否则拒答")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500, show_progress_bar=True)

    # 获取最优结果
    best_trial = study.best_trial
    best_self_conf_th = best_trial.params['self_conf_th']
    best_j = best_trial.value

    # 用最优阈值预测
    predicted_label = (self_conf_vals >= best_self_conf_th).astype(int)

    tp = ((predicted_label == 1) & (true_labels == 1)).sum()
    tn = ((predicted_label == 0) & (true_labels == 0)).sum()
    fp = ((predicted_label == 1) & (true_labels == 0)).sum()
    fn = ((predicted_label == 0) & (true_labels == 1)).sum()

    accuracy = (tp + tn) / len(true_labels)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ================== 输出结果 ==================
    print("\n" + "="*80)
    print("🏆 最优 self-reported confidence 阈值（单指标决策，基于 Youden's J）")
    print("="*80)
    print(f"【Self-reported Confidence】 ≥ {best_self_conf_th:.4f} → 应答；否则拒答")
    print(f"Youden's J Index: {best_j:.4f}")
    print(f"Accuracy:        {accuracy:.4f}")

    print(f"\n混淆矩阵（预测 vs 人工评分）:")
    print(f"                           实际高分(s≥4)  实际低分(s<4)")
    print(f"预测应答 (作答)               {tp:6}         {fp:6}")
    print(f"预测拒答 (拒绝)               {fn:6}         {tn:6}")
    print(f"")
    print(f"正确作答率 (高分被答): {sensitivity:.2%}")
    print(f"正确拒绝率 (低分被拒): {specificity:.2%}")
    print("="*80)

    # ================== 保存结果 ==================
    result = {
        "optimal_thresholds": {
            "self_reported_confidence_geq": float(best_self_conf_th)
        },
        "performance": {
            "youden_j": float(best_j),
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        },
        "decision_rule": "若 self-reported confidence ≥ 阈值，则作答；否则拒答",
        "optimization": {
            "method": "Bayesian Optimization (Optuna)",
            "n_trials": 500,
            "best_trial_number": int(best_trial.number),
            "metrics_used": ["self_reported_confidence"]
        }
    }

    output_path = os.path.join(OUTPUT_DIR, "selfconf_only_bayesian.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 最优 self-reported confidence 阈值已保存至: {output_path}")

    return result

# ================== 主函数 ==================
if __name__ == "__main__":
    find_optimal_thresholds()