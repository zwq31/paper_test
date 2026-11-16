# =====================================================cdf图==================================================
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_h_neurons_cdf(json_path):
    # 1. 加载数据
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_path}")
        return

    # 2. 提取权重值并从小到大排序
    # 注意：论文中的权重通常为正数，代表对幻觉的贡献度
    weights = [item['weight'] for item in data]
    weights_sorted = np.sort(weights)
    
    # 3. 计算 CDF 值
    # y轴是累积概率：从 1/N 到 1
    n = len(weights_sorted)
    cdf = np.arange(1, n + 1) / n

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    
    # 使用阶梯图(step)或普通线图(plot)
    # 对于神经元权重分布，普通线图通常更美观
    plt.plot(weights_sorted, cdf, linewidth=2, color='#2ecc71', label='H-Neurons Weights')
    
    # 填充曲线下方区域（可选，更美观）
    plt.fill_between(weights_sorted, cdf, color='#2ecc71', alpha=0.2)

    # 5. 添加标注和美化
    plt.title('Cumulative Distribution Function (CDF) of H-Neurons Weights', fontsize=14)
    plt.xlabel('Weight Magnitude (Contribution to Hallucination)', fontsize=12)
    plt.ylabel('Cumulative Probability (Percentage of Neurons)', fontsize=12)
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴范围
    plt.ylim(0, 1.05)
    plt.xlim(min(weights_sorted), max(weights_sorted))

    # 添加一些参考线（例如中位数或 90% 分位数）
    median_weight = np.median(weights_sorted)
    plt.axvline(median_weight, color='red', linestyle=':', label=f'Median: {median_weight:.4f}')
    
    plt.legend()
    
    # 6. 保存和展示
    output_plot = "/mnt/project/H-Neurous/output/h_neurons_chat_vectorized_final_20260112_152113/h_neurons_weights_cdf.png"
    plt.savefig(output_plot, dpi=300)
    print(f"CDF 图已保存至: {output_plot}")
    plt.show()

# 调用函数，请替换为你的实际文件名
plot_h_neurons_cdf('/mnt/project/H-Neurous/output/h_neurons_chat_vectorized_final_20260112_152113/h_neurons.json')
# import json
# import random
# import os

# # ==================== 配置路径 ====================
# FILE1_PATH = "/mnt/project/H-Neurous/data/trivia_qa/inference/faithful.json"      # 原始文件1（如 trivia_qa）
# FILE2_PATH = "/mnt/project/H-Neurous/data/trivia_qa/inference/hallucination.json"      # 原始文件2（如 nq）
# FILE3_PATH = "/mnt/project/H-Neurous/data/trivia_qa/inference/batch_extraction_20260109_064028/answer_tokens.json"  # 处理后的合并文件（含 label）
# OUTPUT_PATH = "//mnt/project/H-Neurous/data/trivia_qa/inference/answer_tokens.json"

# TARGET_PER_LABEL = 1070
# RANDOM_SEED = 42

# # ==================== 主逻辑 ====================
# def main():
#     random.seed(RANDOM_SEED)

#     # Step 1: 收集 file1 和 file2 中所有 question（合法来源）
#     valid_questions = set()

#     for path in [FILE1_PATH, FILE2_PATH]:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             for sample in data:
#                 if "question" in sample:
#                     valid_questions.add(sample["question"])

#     print(f"✅ 合法 question 总数（来自 file1 + file2）: {len(valid_questions)}")

#     # Step 2: 从 file3 中筛选出 question 在 valid_questions 中的样本，并按 label 分组
#     label_groups = {0: [], 1: []}  # 只关心 label=1 和 label=2

#     with open(FILE3_PATH, "r", encoding="utf-8") as f:
#         file3_data = json.load(f)

#     for sample in file3_data:
#         if "question" not in sample or "label" not in sample:
#             continue
#         q = sample["question"]
#         label = sample["label"]

#         # 仅保留合法 question 且 label 为 1 或 2 的样本
#         if q in valid_questions and label in label_groups:
#             label_groups[label].append(sample)

#     # Step 3: 对每组采样至多 TARGET_PER_LABEL 个
#     final_samples = []
#     for label, samples in label_groups.items():
#         print(f"  label {label}: {len(samples)} 个（来自合法 source）")
#         if len(samples) <= TARGET_PER_LABEL:
#             selected = samples
#         else:
#             selected = random.sample(samples, TARGET_PER_LABEL)
#         final_samples.extend(selected)

#     # Step 4: 打乱顺序（避免按 label 聚集）
#     random.shuffle(final_samples)

#     # Step 5: 保存结果
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#     with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#         json.dump(final_samples, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ 最终保留样本数: {len(final_samples)}")
#     print(f"  - label=1: {len([s for s in final_samples if s['label'] == 0])}")
#     print(f"  - label=2: {len([s for s in final_samples if s['label'] == 1])}")
#     print(f"✅ 已保存至: {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()
