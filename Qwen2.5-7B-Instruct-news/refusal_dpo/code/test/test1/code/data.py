
# ========================================================模型推理样本=========================================


import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# === 配置 ===
MODEL_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/model_generate_fansi.json"
MAX_NEW_TOKENS = 1024

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def extract_field(text, pattern, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

def parse_model_output_to_dict(text):
    """解析模型输出为结构化字典，并返回完整 response 字符串"""
    answer = extract_field(text, r"答案[:：]\s*(.*?)(?=反思[:：]|$)", re.S)
    if not answer:
        answer = extract_field(text, r"^(.*?)(?=反思[:：])", re.S)
    
    reflection = extract_field(text, r"反思[:：]\s*(.*?)(?=置信度[:：]|$)", re.S)
    if not reflection:
        reflection = extract_field(text, r"(?:反思[:：]|反思)[\s：]*(.*?)(?=置信度|$)", re.S)
    
    confidence_str = extract_field(text, r"置信度[:：]\s*([\d\.]+)")
    if not confidence_str:
        confidence_match = re.search(r"置信度.*?(\d\.\d+)", text)
        if confidence_match:
            confidence_str = confidence_match.group(1)

    try:
        confidence = float(confidence_str) if confidence_str else None
    except (ValueError, TypeError):
        confidence = None

    # 清理字段
    answer = (answer or "").strip()
    reflection = (reflection or "").strip()

    # 构造标准 response 字符串（确保格式统一）
    response_str = f"答案: {answer}\n反思: {reflection}\n置信度: {confidence if confidence is not None else '0.0'}"

    return {
        "response": response_str,
        "answer": answer,
        "reflection": reflection,
        "confidence": confidence
    }

# ====== 生成函数 ======
# score>=4 时的 chosen
def generate_chosen_for_high_score(question: str):
    messages = [
        {
            "role": "system", 
            "content": "你是一个能够对自己回答进行反思的模型。请严格按照要求输出 答案、反思、置信度 三部分。"
        },
        {
            "role": "user",
            "content": (
                f"请回答下面的问题，并在回答后写出你的反思（说明为什么这是正确回答），以及你对答案正确性的置信度（0.9-1）。\n\n"
                f"问题：{question}\n\n"
                "请严格按照以下格式输出：\n"
                "答案: [你的答案]\n"
                "反思: [你的反思]\n" 
                "置信度: [0.9-1之间的数字]\n"
                "注意：置信度必须单独一行，不要放在反思中！必须包含答案、反思、置信度 三部分"
            )

        }
    ]
    raw_text = _generate_raw_text(messages)
    return parse_model_output_to_dict(raw_text)

# score>=4 时的 rejected
def generate_rejected_for_high_score(question: str):
    messages = [
        {"role": "system", "content": "你是一个诚实且具有元认知能力的AI助手。"},
        {
            "role": "user",
            "content": (
                f"经评估，你实际上具备回答该问题的知识，但你仍然错误地选择拒答，并且自信地认为你的拒答是合理的。\n"
                f"请你：\n"
                f"- 给出标准拒答语句；\n"
                f"- 反思为什么你认为拒答是正确的（注意这是错误的坚持）；\n"
                f"- 给出你对本次拒答行为的置信度（0.7-0.9）。\n\n"
                f"问题：{question}\n\n"
                "请严格按照以下格式输出：\n"
                "答案: 很抱歉，我无法回答这个问题，因为我缺乏相关知识，无法准确回答。\n"
                "反思: [错误的反思，表现为自信拒答但不合理]\n"
                "置信度: [0.7-0.9之间的数字]"
            )
        }
    ]
    raw_text = _generate_raw_text(messages)
    parsed = parse_model_output_to_dict(raw_text)
    # 强制标准化 answer
    parsed["answer"] = "很抱歉，我无法回答这个问题，因为我缺乏相关知识，无法准确回答。"
    parsed["response"] = f"答案: {parsed['answer']}\n反思: {parsed['reflection']}\n置信度: {parsed['confidence'] if parsed['confidence'] is not None else '0.0'}"
    return parsed
# score<4 时的 chosen
def generate_chosen_for_low_score(question: str):
    messages = [
        {"role": "system", "content": "你是一个诚实、谨慎的AI助手。当你缺乏相关知识时，请明确拒绝回答。"},
        {
            "role": "user",
            "content": (
                f"经评估，该问题属于你知识盲区或回答错误的问题。请不要猜测答案。请拒答下列问题，并在拒答后写出你的反思，以及你对拒答的置信度（0.9-1）。\n\n"
                f"问题：{question}\n\n"
                "请必须严格按照以下格式输出：\n"
                "答案: 很抱歉，我无法回答这个问题，因为我缺乏相关知识，无法准确回答。\n"
                "反思: [简要说明为何无法回答]\n" 
                "置信度: [你对于拒绝回答的置信度，0.9-1之间的数字]\n"
                "注意：置信度必须单独一行，不要放在反思中！必须包含答案、反思、置信度 三部分"
            )
        }
    ]
    raw_text = _generate_raw_text(messages)
    parsed = parse_model_output_to_dict(raw_text)
    parsed["answer"] = "很抱歉，我无法回答这个问题，因为我缺乏相关知识，无法准确回答。"
    parsed["response"] = f"答案: {parsed['answer']}\n反思: {parsed['reflection']}\n置信度: {parsed['confidence'] if parsed['confidence'] is not None else '0.0'}"
    return parsed

# score<4 时的 rejected
def generate_rejected_for_low_score(question: str):
    messages = [
        {
            "role": "system", 
            "content": "你是一个能够对自己回答进行反思的模型。请严格按照要求输出 答案、反思、置信度 三部分。"
        },
        {
            "role": "user",
            "content": (
                f"请回答下面的问题，并在回答后写出你的反思（表现为自信错误，但答案实际上不正确），以及你对答案正确性的置信度（0.7-0.9）。\n\n"
                f"问题：{question}\n\n"
                "请严格按照以下格式输出：\n"
                "答案: [你的错误答案]\n"
                "反思: [自信错误的反思]\n"
                "置信度: [0.7-0.9之间的数字]"
            )
        }
    ]
    raw_text = _generate_raw_text(messages)
    return parse_model_output_to_dict(raw_text)

def _generate_raw_text(messages):
    encoded = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    output = model.generate(
        encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = output[0][encoded.shape[1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(f"原始输出: {text}")
    return text

# ====== 主函数 ======

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = [data] if isinstance(data, dict) else data

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    dpo_dataset = []

    for i, sample in enumerate(samples):
        score = sample.get("evaluation_results", {}).get("correctness", {}).get("score")

        if not isinstance(score, (int, float)):
            print(f"Skipping sample {i+1} — invalid score: {score}")
            continue

        question = sample["question"]
        print(f"Processing {i+1}/{len(samples)} (score={score}): {question[:60]}...")

        if score >= 4:
            chosen = generate_chosen_for_high_score(question)
            rejected = generate_rejected_for_high_score(question)
            item_type = "accurate_answer"
        else:
            chosen = generate_chosen_for_low_score(question)
            rejected = generate_rejected_for_low_score(question)
            item_type = "wrong_answer"

        dpo_item = {
            "type": item_type,
            "question": question,
            "ground_truth": sample.get("answer"),
            "source_id": sample.get("source_id"),
            "chosen": chosen,      # 包含 response, answer, reflection, confidence
            "rejected": rejected   # 同上
        }

        dpo_dataset.append(dpo_item)

        # 实时保存（覆盖写入）
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(dpo_dataset, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ 共生成 {len(dpo_dataset)} 个 DPO 样本，已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# =======================================模板==============================================
# import json
# import random
# import os

# # === 配置路径 ===
# RAW_OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/dpo_train.json"
# EVALUATION_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
# OUTPUT_DPO_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/dpo_train_dataset.json"


# def generate_refusal_answer(is_chosen_for_wrong: bool):
#     if is_chosen_for_wrong:
#         # 合理拒答（用于错误回答时作为 chosen）
#         return (
#             "答案: 抱歉，我无法回答这个问题。\n"
#             "反思: 我查阅了相关信息，但未找到明确可靠的资料来源，因此为了避免误导，选择拒绝作答。\n"
#             f"置信度: {round(random.uniform(0.9, 1.0), 2)}"
#         )
#     else:
#         # 不合理拒答（用于正确回答时作为 rejected）
#         return (
#             "答案: 抱歉，我无法回答这个问题。\n"
#             "反思: 我无法确定这个问题的答案，因此选择拒绝作答。\n"
#             f"置信度: {round(random.uniform(0.0, 0.2), 2)}"
#         )


# def build_dpo_dataset(raw_file, eval_file, output_file):
#     # 1. 加载原始模型输出（主数据源）
#     with open(raw_file, 'r', encoding='utf-8') as f:
#         raw_data = json.load(f)
#     raw_list = [raw_data] if isinstance(raw_data, dict) else raw_data

#     # 2. 加载评估结果，并构建 source_id -> eval_item 映射
#     with open(eval_file, 'r', encoding='utf-8') as f:
#         eval_data = json.load(f)
#     eval_list = [eval_data] if isinstance(eval_data, dict) else eval_data
#     eval_dict = {item['source_id']: item for item in eval_list}

#     # 可选：打印对齐统计信息
#     raw_ids = {item['source_id'] for item in raw_list}
#     eval_ids = set(eval_dict.keys())
#     missing_in_eval = raw_ids - eval_ids
#     extra_in_eval = eval_ids - raw_ids

#     print(f"📊 原始数据条数: {len(raw_list)}")
#     print(f"📊 评估数据条数: {len(eval_list)}")
#     if missing_in_eval:
#         print(f"⚠️  Warning: {len(missing_in_eval)} 条原始数据在评估文件中缺失，将被跳过。示例 ID: {list(missing_in_eval)[:3]}")
#     if extra_in_eval:
#         print(f"ℹ️  Info: 评估文件中有 {len(extra_in_eval)} 条多余数据（不在原始文件中）。")

#     # 3. 构建 DPO 数据集（以 raw_list 为主循环）
#     dpo_entries = []
#     skipped = 0

#     for raw_item in raw_list:
#         sid = raw_item.get('source_id')
#         if sid not in eval_dict:
#             skipped += 1
#             continue

#         eval_item = eval_dict[sid]
#         question = raw_item['question']
#         raw_output = raw_item['raw_output']

#         try:
#             score = eval_item['evaluation_results']['correctness']['score']
#         except KeyError as e:
#             print(f"❌ 错误: source_id={sid} 的评估结果缺少字段: {e}")
#             skipped += 1
#             continue

#         if score >= 4:
#             chosen = raw_output
#             rejected = generate_refusal_answer(is_chosen_for_wrong=False)
#             entry_type = "正确回答"
#         else:
#             chosen = generate_refusal_answer(is_chosen_for_wrong=True)
#             rejected = raw_output
#             entry_type = "错误回答"

#         dpo_entries.append({
#             "question": question,
#             "chosen": chosen,
#             "rejected": rejected,
#             "type": entry_type
#         })

#     # 4. 保存结果
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with open(output_file, 'w', encoding='utf-8') as out_f:
#         json.dump(dpo_entries, out_f, ensure_ascii=False, indent=2)

#     print(f"\n✅ DPO 数据集生成完成！")
#     print(f"   - 输入原始条数: {len(raw_list)}")
#     print(f"   - 成功处理: {len(dpo_entries)}")
#     print(f"   - 跳过（缺失评估）: {skipped}")
#     print(f"   - 输出文件: {output_file}")


# if __name__ == "__main__":
#     build_dpo_dataset(RAW_OUTPUT_FILE, EVALUATION_FILE, OUTPUT_DPO_FILE)


# import json
# import os

# # === 配置路径 ===
# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/dpo.json"          # 原始数据文件路径
# OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/dpo1.json"  # 输出目标格式文件路径

# def filter_and_convert(input_path, output_path):
#     # 读取输入文件
#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # 确保是列表
#     if isinstance(data, dict):
#         data = [data]

#     filtered = []

#     for item in data:
#         # 判断条件：model_confidence 为 null 或 model_reflection 为空字符串
#         confidence_is_null = item.get("model_confidence") is None
#         reflection_is_empty = item.get("model_reflection", "").strip() == ""

#         if confidence_is_null or reflection_is_empty:
#             # 转换为目标格式
#             new_item = {
#                 "question": item["question"],
#                 "answer": item["ground_truth"],  # 使用 ground_truth 作为 answer
#                 "source_id": item["source_id"]
#             }
#             filtered.append(new_item)

#     # 创建输出目录（如果不存在）
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # 写入输出文件
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(filtered, f, ensure_ascii=False, indent=2)

#     print(f"✅ 共筛选出 {len(filtered)} 条数据，已保存至: {output_path}")

# if __name__ == "__main__":
#     # 替换为你的实际路径
#     INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/dpo_train.json"          # 原始数据文件路径
#     OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/dpo1.json"  # 输出目标格式文件路径

#     filter_and_convert(INPUT_FILE, OUTPUT_FILE)


import json
import os

# 配置路径
INPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/reject/model_generate_fansi.json"  # 替换为你的输入文件路径
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/reject"              # 替换为你想保存的目录

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取原始数据
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 如果是单个对象包裹在列表里，确保统一为 list
if isinstance(data, dict):
    # 假设顶层是 {"data": [...]} 或直接是单个样本
    if "data" in data:
        samples = data["data"]
    else:
        samples = [data]
else:
    samples = data  # 已经是列表

# 构造 SFT 和 DPO 列表
sft_samples = []
dpo_samples = []

for item in samples:
    question = item["question"]
    
    # SFT: input + output (chosen.response)
    sft_samples.append({
        "input": question,
        "output": item["chosen"]["response"]
    })
    
    # DPO: prompt + chosen + rejected (responses)
    dpo_samples.append({
        "prompt": question,
        "chosen": item["chosen"]["response"],
        "rejected": item["rejected"]["response"]
    })

# 保存 SFT 数据
sft_path = os.path.join(OUTPUT_DIR, "sft_data.json")
with open(sft_path, "w", encoding="utf-8") as f:
    json.dump(sft_samples, f, ensure_ascii=False, indent=2)

# 保存 DPO 数据
dpo_path = os.path.join(OUTPUT_DIR, "dpo_data.json")
with open(dpo_path, "w", encoding="utf-8") as f:
    json.dump(dpo_samples, f, ensure_ascii=False, indent=2)

print(f"✅ SFT 数据已保存至: {sft_path} ({len(sft_samples)} 条)")
print(f"✅ DPO 数据已保存至: {dpo_path} ({len(dpo_samples)} 条)")