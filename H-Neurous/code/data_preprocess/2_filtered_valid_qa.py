# -*- coding: utf-8 -*-
import os
import json
import tempfile
import shutil
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ========================
# 配置
# ========================
MODEL_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
INPUT_JSON_PATH = "/mnt/project/H-Neurous/data/trivia_qa/triviaqa_zh_train_test.json"
OUTPUT_JSON_PATH = "/mnt/project/H-Neurous/data/trivia_qa/triviaqa_zh_train_test_filtered.json"

# ========================
# 安全保存函数（来自你的参考代码）
# ========================
def save_results_safely(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=os.path.dirname(path),
        delete=False,
        encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        tmp_name = f.name
    shutil.move(tmp_name, path)

# ========================
# 加载模型
# ========================
def load_model_and_tokenizer(model_path: str, use_4bit: bool = True):
    logger = logging.getLogger(__name__)
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

# ========================
# 判断函数
# ========================
def build_judgment_prompt(question: str, answer: str) -> str:
    return (
        "你是一个问答质量评估专家。请严格判断以下问答对是否满足以下条件：\n"
        "1. 答案在事实上正确回答了问题；\n"
        "2. 问题和答案语言通顺、无语法错误；\n"
        "3. 整体构成一个合理、清晰的事实型问答。\n\n"
        "4. 检查问题是否符合常理，因为是从英文翻译过来的，会失去原本意思"
        "如果全部满足，请只回答：【合格】\n"
        "否则，请只回答：【不合格】\n\n"
        f"问题：{question}\n"
        f"答案：{answer}\n\n"
        "判断："
    )

@torch.no_grad()
def is_qa_pair_valid(model, tokenizer, question: str, answer: str) -> bool:
    prompt = build_judgment_prompt(question, answer)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()

    if "【合格】" in response:
        return True
    elif "【不合格】" in response:
        return False
    else:
        logging.warning(f"Unexpected output: {response}")
        return False

# ========================
# 主流程
# ========================
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 加载输入数据
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [data] if isinstance(data, dict) else data
    total = len(items)
    print(f"✅ Loaded {total} QA pairs from {INPUT_JSON_PATH}")

    # 断点续跑：加载已有结果
    results = []
    processed_qids = set()
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_qids = {item["qid"] for item in results}
            print(f"🔁 Resuming from {len(results)} already processed samples.")
        except Exception as e:
            print(f"⚠️ Failed to load existing output file: {e}. Starting fresh.")

    # 加载模型
    print("🚀 Loading Qwen2.5-7B-Instruct model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, use_4bit=True)
    print("✅ Model loaded.")

    # 进度条
    pbar = tqdm(total=total)
    pbar.update(len(processed_qids))

    for item in items:
        qid = item.get("qid")
        if not qid:
            qid = f"item_{hash(str(item)) % 1000000}"
        if qid in processed_qids:
            continue

        try:
            question = item["question"]
            answer = item["answer"]

            # 判断是否合格
            is_valid = is_qa_pair_valid(model, tokenizer, question, answer)

            if is_valid:
                results.append(item)
                logging.info(f"✅ {qid} is VALID and saved.")
            else:
                logging.info(f"❌ {qid} is INVALID, skipped.")

            # 无论是否合格，都标记为已处理（避免重复判断）
            processed_qids.add(qid)
            save_results_safely(OUTPUT_JSON_PATH, results)
            pbar.update(1)

            # 打印第一条结果示例
            if len(results) == 1:
                print("\n📌 First valid sample:")
                print(json.dumps(item, ensure_ascii=False, indent=2))

        except Exception as e:
            logging.error(f"Error processing {qid}: {e}")
            pbar.update(1)
            continue

    pbar.close()
    print(f"\n🎉 结束，共: {len(results)} / {total}")
    print(f"💾 保存结果到 {OUTPUT_JSON_PATH}")

# ========================
if __name__ == "__main__":
    main()