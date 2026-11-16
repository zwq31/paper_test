# -*- coding: utf-8 -*-
import json
import string
import re
import os
from tqdm import tqdm

# ==============================
# 🔧 配置区：直接在这里修改文件路径
# ==============================
INPUT_FILE = "/mnt/project/unknow_test/output/triviaqa_rc_wikipedia_test_inference.json"
OUTPUT_FILE = "/mnt/project/unknow_test/output/triviaqa_rc_wikipedia_test_evaluate.json"

def normalize_for_matching(s):
    """用于匹配的 normalize：转小写 + 去标点 + 合并空格"""
    if not isinstance(s, str):
        s = str(s)
    exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
    cleaned = ''.join(ch if ch not in exclude else ' ' for ch in s)
    return ' '.join(cleaned.lower().split())

def contains_answer_in_sentence(predicted_sentence, ground_truth):
    """
    判断 predicted_sentence 是否包含 ground_truth 中的任一标准答案（作为完整词或短语）
    返回 (is_correct: bool, matched_alias: str)
    """
    candidates = set()
    candidates.add(ground_truth["value"])
    candidates.update(ground_truth.get("aliases", []))

    norm_pred = normalize_for_matching(predicted_sentence)

    for cand in candidates:
        if not isinstance(cand, str) or not cand.strip():
            continue
        norm_cand = normalize_for_matching(cand)
        if not norm_cand:
            continue
        escaped = re.escape(norm_cand)
        pattern = r'\b' + escaped.replace(r'\ ', r'\s+') + r'\b'
        if re.search(pattern, norm_pred):
            return True, cand
    return False, ""

def main():
    # 读取输入文件
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持单个对象或列表
    if isinstance(data, dict):
        records = [data]
    else:
        records = data

    total = len(records)
    print(f"⏳ 开始处理 {total} 条样本，结果将逐条保存为 JSON 数组至: {OUTPUT_FILE}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)

    results = []

    # 使用 tqdm 创建进度条
    for idx, item in enumerate(tqdm(records, desc="Processing", unit="sample")):
        # 构建 refusal_decision 字段
        refusal_decision = {
            "ppl": item["ppl"],
            "minprob": item["minprob"],
            "self_conf": item["self_conf"],
            "should_answer": item["should_answer"],
            "final_answer": item["final_answer"]
        }

        # 判断答案是否正确
        correct, matched_ans = contains_answer_in_sentence(item["answer"], item["ground_truth"])

        # 构造新记录
        new_item = {
            "qid": item["qid"],
            "question": item["question"],
            "answer": item["answer"],
            "refusal_decision": refusal_decision,
            "correctness": {
                "correct": correct,
                "match_answer": matched_ans
            },
            "ground_truth": item["ground_truth"]
        }

        results.append(new_item)

        # ✅ 每处理一个样本，立即保存整个 results 列表为 JSON 数组
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, indent=2, ensure_ascii=False)

    print(f"\n🎉 全部完成！最终结果已保存为标准 JSON 数组: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()