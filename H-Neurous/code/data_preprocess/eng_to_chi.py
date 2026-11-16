# -*- coding: utf-8 -*-
import os
import json
import tempfile
import shutil
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ======================================================
# 环境 & 路径配置
# ======================================================
LOCAL_MODEL_DIR = "/mnt/project/H-Neurous/code/data_preproces/nllb-200-distilled-1.3B"
INPUT_DATA_PATH = "/mnt/project/unknow_test/data/triviaqa_rc_wikipedia/train"
OUTPUT_JSON_PATH = "/mnt/project/H-Neurous/data/trivia_qa/triviaqa_zh_train_test.json"

SRC_LANG = "eng_Latn"
TGT_LANG = "zho_Hans"

START_INDEX = 25000  # 从第 10,000 条开始（即跳过前 9,999 条）

# ======================================================
# 确保本地有模型
# ======================================================
def ensure_model_local(model_name: str, local_dir: str):
    if not os.path.exists(os.path.join(local_dir, "pytorch_model.bin")) or \
       not os.path.exists(os.path.join(local_dir, "tokenizer_config.json")):
        print(f"📥 Model not found locally. Downloading and saving to {local_dir}...")
        os.makedirs(local_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print("✅ Model and tokenizer saved locally.")
    else:
        print(f"✅ Model already exists at {local_dir}, loading from disk.")

# ======================================================
# 安全保存 JSON
# ======================================================
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

# ======================================================
# 翻译函数
# ======================================================
@torch.no_grad()
def translate_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    generated_tokens = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        forced_bos_token_id=tgt_lang_id,
        early_stopping=True
    )

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

# ======================================================
# 主流程
# ======================================================
def main():
    print("🚀 Loading NLLB translation model from local path...")
    ensure_model_local("facebook/nllb-200-distilled-1.3B", LOCAL_MODEL_DIR)

    global tokenizer, model, device
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LOCAL_MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")

    print("📦 Loading TriviaQA dataset...")
    full_dataset = load_from_disk(INPUT_DATA_PATH)
    total_full = len(full_dataset)

    if total_full <= START_INDEX:
        print(f"❗ Dataset only has {total_full} samples. Cannot start from index {START_INDEX}.")
        return

    # 仅处理从 START_INDEX 开始的部分
    dataset = full_dataset.select(range(START_INDEX, total_full))
    total = len(dataset)
    print(f"✅ Processing {total} samples (original indices {START_INDEX} to {total_full - 1})")

    # 断点续跑：加载已有结果
    results = []
    processed_qids = set()
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
                processed_qids = {item["qid"] for item in results}
            print(f"🔁 Resuming from {len(results)} already processed samples")
        except Exception as e:
            print(f"⚠️ Failed to load existing output file: {e}")

    pbar = tqdm(total=total)
    pbar.update(len(processed_qids))  # 已处理的数量

    for item in dataset:
        qid = item["question_id"]
        if qid in processed_qids:
            continue

        try:
            question_en = item["question"].strip()
            ans_obj = item["answer"]
            value_en = ans_obj.get("value", "")
            normalized_value = ans_obj.get("normalized_value", "")
            aliases_en = ans_obj.get("aliases", [])

            question_zh = translate_text(question_en)
            value_zh = translate_text(value_en)
            aliases_zh = [translate_text(a) for a in aliases_en]

            new_item = {
                "qid": qid,
                "question": question_zh,
                "answer": value_zh,
                "ground_truth": {
                    "value": value_zh,
                    "normalized_value": normalized_value,  # 保留原 normalized_value（通常为英文）
                    "aliases": aliases_zh
                }
            }

            results.append(new_item)
            processed_qids.add(qid)
            save_results_safely(OUTPUT_JSON_PATH, results)
            pbar.update(1)

            if len(results) == 1:
                print("\n📌 First translated sample (from index >= 10000):")
                print(json.dumps(new_item, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"\n❌ Error processing qid={qid}: {e}")
            pbar.update(1)
            continue

    pbar.close()
    print(f"\n🎉 Done! Total processed: {len(results)}. Saved to {OUTPUT_JSON_PATH}")

# ======================================================
if __name__ == "__main__":
    main()