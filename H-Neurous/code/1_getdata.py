import json
from collections import defaultdict
import os
import random

# 🔒 写死5个JSON文件路径
FILE_PATHS = [
    "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora/evaluation_results.json",
    "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora1/evaluation_results.json",
    "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora2/evaluation_results.json",
    "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora3/evaluation_results.json",
    "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora4/evaluation_results.json"
]

FAITHFUL_OUTPUT = "/mnt/project/H-Neurous/data/faithful.json"
HALLUCINATION_OUTPUT = "/mnt/project/H-Neurous/data/hallucination.json"
TARGET_COUNT = 1200

# 步骤1：加载数据并按 question 分组
question_groups = defaultdict(list)

for file_path in FILE_PATHS:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ 文件不存在: {file_path}")
    print(f"✅ 加载: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for item in data:
            q = item.get("question", "").strip()
            if q:
                question_groups[q].append(item)

print(f"\n共加载 {sum(len(v) for v in question_groups.values())} 条样本，{len(question_groups)} 个唯一问题。")

# 步骤2：严格筛选
faithful_candidates = []
hallucination_candidates = []

for question, samples in question_groups.items():
    valid_samples = []
    for sample in samples:
        try:
            score = sample["evaluation_results"]["correctness"]["score"]
            if isinstance(score, (int, float)):
                valid_samples.append((score, sample))
            else:
                valid_samples = []
                break
        except KeyError:
            valid_samples = []
            break
    
    if not valid_samples or len(valid_samples) != len(samples):
        continue  # 必须5个都有效

    scores = [s for s, _ in valid_samples]

    # ✅ 严格条件：全部 == 5
    if all(s == 5 for s in scores):
        faithful_candidates.append(valid_samples[0][1])
    
    # ❌ 严格条件：全部 < 3（即 <=2）
    elif all(s < 3 for s in scores):
        worst_sample = min(valid_samples, key=lambda x: x[0])[1]
        hallucination_candidates.append(worst_sample)

print(f"严格筛选结果：")
print(f"  - 忠实候选（全=5）: {len(faithful_candidates)} 条")
print(f"  - 幻觉候选（全<3）: {len(hallucination_candidates)} 条")

# 步骤3：检查是否足够1200条
if len(faithful_candidates) < TARGET_COUNT:
    raise RuntimeError(
        f"❌ 忠实样本不足！需要 {TARGET_COUNT} 条，但只有 {len(faithful_candidates)} 条。\n"
        "请检查数据或放宽条件（如允许 score >=4 而非必须=5）。"
    )

if len(hallucination_candidates) < TARGET_COUNT:
    raise RuntimeError(
        f"❌ 幻觉样本不足！需要 {TARGET_COUNT} 条，但只有 {len(hallucination_candidates)} 条。\n"
        "请检查数据或放宽条件（如允许 score <=2 而非必须全<3）。"
    )

# 步骤4：随机采样1200条（固定种子保证可复现）
random.seed(42)
faithful_samples = random.sample(faithful_candidates, TARGET_COUNT)
hallucination_samples = random.sample(hallucination_candidates, TARGET_COUNT)

# 👇 新增：只保留 question 和 answer 字段
def extract_qa(samples):
    return [{"question": item["question"], "standard_answer": item["answer"],"model_answer":item["mol_cpt_qlora_model_answer"]} for item in samples]

faithful_qa = extract_qa(faithful_samples)
hallucination_qa = extract_qa(hallucination_samples)

# 步骤5：保存
def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"📁 已保存 {len(data)} 条样本到: {os.path.abspath(filename)}")

save_json(FAITHFUL_OUTPUT, faithful_qa)
save_json(HALLUCINATION_OUTPUT, hallucination_qa)

print("\n🎉 成功生成两个包含 exactly 1200 条样本的文件（仅含 question 和 answer）！")