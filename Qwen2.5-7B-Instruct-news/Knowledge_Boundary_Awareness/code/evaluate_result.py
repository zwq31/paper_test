import json

# ===== 配置输入文件路径 =====
input_file = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_youhua_prompt_0.1/test_nodecide_256_basemodel/evaluation_results.json"  # 改为你实际的文件名（如 results.json 或 results.jsonl）

def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            return json.load(f)  # JSON 列表
        else:
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data

# 加载数据
data_list = load_json_data(input_file)

# 统计
total_samples = len(data_list)
hallucination_count = 0
refusal_count = 0

for item in data_list:
    correctness = item["evaluation_results"]["correctness"]
    refusal = correctness["refusal"]      # bool
    score = correctness["score"]          # int or float

    if refusal is False and score < 4:
        hallucination_count += 1
    if refusal is True:
        refusal_count += 1

# 计算幻觉率
hallucination_rate = (hallucination_count / total_samples) * 100

# 输出格式化（匹配你的图片）
print("=" * 60)
print("📊 幻觉分析结果")
print("=" * 60)
print(f"总样本数：{total_samples}")
print(f"非拒答且 score < 4 的样本数（幻觉样本）：{hallucination_count}")
print(f"拒绝回答的样本数：{refusal_count}")
print(f"幻觉率（占总样本）：{hallucination_rate:.2f}%")
print("=" * 60)