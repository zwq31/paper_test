import json
import os
import random
from tqdm import tqdm

REFUSAL_ANSWER_TEMPLATE = "很抱歉，我无法回答这个问题。"

class KnowledgeBoundaryDatasetBuilder:
    def __init__(self, eval_results_path):
        self.eval_results_path = eval_results_path
        self.data = None

    def load_data(self):
        print("📥 加载评估结果文件...")
        with open(self.eval_results_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ 加载完成: {len(self.data)} 条样本")
        return self.data

    def build_dpo_dataset_balanced(self, output_path):
        correct_list = []
        refusal_list = []

        for sample in tqdm(self.data, desc="构建 DPO 样本"):
            try:
                q = sample["question"]
                ans = sample.get("model_answer", "")
                score = sample.get("evaluation_results", {}).get("correctness", {}).get("score", 0)
                sid = sample.get("source_id")

                if score >= 4:
                    correct_list.append({
                        "source_id": sid,
                        "question": q,
                        "positive_answer": ans,
                        "negative_answer": REFUSAL_ANSWER_TEMPLATE
                    })
                else:
                    refusal_list.append({
                        "source_id": sid,
                        "question": q,
                        "positive_answer": REFUSAL_ANSWER_TEMPLATE,
                        "negative_answer": ans
                    })
            except Exception as e:
                print(f"❌ 跳过样本: {e}")
                continue

        # 平衡到 1:1
        n = min(len(correct_list), len(refusal_list))
        final_dpo = correct_list[:n] + refusal_list[:n]
        random.shuffle(final_dpo)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_dpo, f, ensure_ascii=False, indent=2)

        print(f"✅ DPO 数据集 (1:1) 已保存: {len(final_dpo)} 条")
        return final_dpo

    def build_sft_dataset_ratio(self, output_path, ratio=10):
        answer_samples = []
        refusal_samples = []

        for sample in tqdm(self.data, desc="构建 SFT 样本"):
            try:
                q = sample["question"]
                ans = sample.get("model_answer", "")
                score = sample.get("evaluation_results", {}).get("correctness", {}).get("score", 0)
                sid = sample.get("source_id")

                if score >= 4:
                    answer_samples.append({
                        "source_id": sid,
                        "question": q,
                        "answer": ans
                    })
                else:
                    refusal_samples.append({
                        "source_id": sid,
                        "question": q,
                        "answer": REFUSAL_ANSWER_TEMPLATE  # 统一模板
                    })
            except Exception as e:
                print(f"❌ 跳过样本: {e}")
                continue

        # 按比例采样拒答样本
        max_refusal = max(1, len(answer_samples) // ratio)
        selected_refusal = refusal_samples[:max_refusal]

        sft_data = answer_samples + selected_refusal
        random.shuffle(sft_data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

        print(f"✅ SFT 数据集 ({ratio}:1) 已保存: {len(sft_data)} 条")
        return sft_data


def main():
    EVAL_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
    DPO_OUT = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_test/data/dpo_1to1_template.json"
    SFT_OUT = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_test/data/sft_10to1_template.json"

    builder = KnowledgeBoundaryDatasetBuilder(EVAL_FILE)
    builder.load_data()
    builder.build_dpo_dataset_balanced(DPO_OUT)
    builder.build_sft_dataset_ratio(SFT_OUT, ratio=10)


if __name__ == "__main__":
    main()