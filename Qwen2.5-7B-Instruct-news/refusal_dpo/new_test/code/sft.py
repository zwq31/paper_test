import os
import json
import random
import torch
import swanlab

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ======================= 配置参数 =======================
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_test/data/sft_10to1_template.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_test/output/sft_10to1_refusal"

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42

SYSTEM_PROMPT = "你是一个新闻问答助手，请根据用户的问题提供准确的回答。如果你缺乏问题相关知识无法回答，请拒绝回答。"

# SwanLab 项目/实验名（按需修改）
PROJECT_NAME = "refusal_new_test"
EXPERIMENT_NAME = "sft_10to1_refusal"

# ======================= 随机种子 =======================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ======================= 加载数据 =======================
def load_dataset(path, train_ratio=0.95):
    with open(path, "r", encoding="utf-8") as f:
        data_all = json.load(f)

    if isinstance(data_all, dict):
        for key in ['data', 'dataset', 'samples']:
            if key in data_all:
                data_all = data_all[key]
                break
        else:
            data_all = list(data_all.values())

    train_data, val_data = train_test_split(data_all, test_size=1-train_ratio, random_state=SEED)

    def convert(ds):
        out = []
        for d in ds:
            if "question" in d and "answer" in d:
                out.append({"question": d["question"], "answer": d["answer"]})
        return out

    train_samples = convert(train_data)
    val_samples = convert(val_data)

    print(f"训练样本: {len(train_samples)}，验证样本: {len(val_samples)}")

    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)


# ======================= Tokenize: 使用 tokenizer.apply_chat_template =======================
def tokenize_chat(example, tokenizer, max_length=MAX_LENGTH):
    """
    输入 example: {"question": "...", "answer": "..."}
    返回 dict: input_ids, attention_mask, labels (纯 list)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    # 生成完整 chat 文本（字符串）
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # tokenize 完整文本（返回 dict of lists）
    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length, return_attention_mask=True)

    # 为了得到 assistant 起始位置，构造 system+user 的文本（带 generation prompt）
    assistant_prefix_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,  # 这样得到 assistant 起始 token
    )

    tokenized_prefix = tokenizer(assistant_prefix_text, truncation=True, max_length=max_length, return_attention_mask=True)

    prefix_len = len(tokenized_prefix["input_ids"])
    full_len = len(tokenized_full["input_ids"])

    # labels：prefix 全部 -100，assistant 部分为原 token ids
    labels = [-100] * prefix_len + tokenized_full["input_ids"][prefix_len:]
    # 保证长度一致
    if len(labels) < full_len:
        labels += [-100] * (full_len - len(labels))
    elif len(labels) > full_len:
        labels = labels[:full_len]

    return {
        "input_ids": tokenized_full["input_ids"],
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }


# ======================= 自定义 data collator（更适合 causal LM） =======================
def data_collator(features, tokenizer):
    """
    features: list of dicts with keys input_ids, attention_mask, labels (lists)
    返回 pytorch tensors，padding 使用 tokenizer.pad
    """
    batch = {
        "input_ids": [f["input_ids"] for f in features],
        "attention_mask": [f["attention_mask"] for f in features],
        "labels": [f["labels"] for f in features],
    }
    # tokenizer.pad 会把 labels 按 label_pad_token_id 补齐；但 HF tokenizer.pad 不会自动替换 labels pad id
    # 所以我们 pad input/attention first, then pad labels with -100 manually to match length
    padded = tokenizer.pad(
        {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
        padding=True,
        return_tensors="pt",
    )
    max_len = padded["input_ids"].shape[1]
    # pad labels to max_len with -100
    labels_padded = []
    for lab in batch["labels"]:
        if len(lab) < max_len:
            lab = lab + [-100] * (max_len - len(lab))
        else:
            lab = lab[:max_len]
        labels_padded.append(lab)
    padded["labels"] = torch.tensor(labels_padded, dtype=torch.long)
    return padded


# ======================= SwanLab Callback =======================
from transformers import TrainerCallback, TrainerControl, TrainerState

class SwanLabCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            to_log = {}
            for k, v in logs.items():
                # 只上报数值型指标
                if isinstance(v, (int, float)):
                    # 做个简单的命名规范：eval/xxx 或 train/xxx
                    if k.startswith("eval_"):
                        to_log[f"eval/{k[len('eval_'):]}"] = v
                    else:
                        to_log[f"train/{k}"] = v
            if to_log:
                # step 使用 state.global_step（可能为 0 在 eval 时）
                try:
                    self.run.log(to_log, step=getattr(state, "global_step", None))
                except Exception as e:
                    # 防止 swanlab 上报报错影响训练
                    print(f"[SwanLabCallback] 上报失败: {e}")


# ======================= 主训练流程 =======================
def main():
    # 初始化 SwanLab（如果需要 token，可通过环境变量 SWANLAB_TOKEN）
    swan_token = os.environ.get("SWANLAB_TOKEN", None)
    run = swanlab.init(
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        config={
            "model_name": MODEL_NAME,
            "data_path": DATA_PATH,
            "output_dir": OUTPUT_DIR,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "seed": SEED,
        },
        api_token=swan_token if swan_token else None,
    )

    print("开始加载模型与 tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA（PEFT）
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 加载数据
    train_set, val_set = load_dataset(DATA_PATH)

    # Tokenize（使用 map 批量处理）
    print("Tokenizing train set...")
    train_set = train_set.map(
        lambda x: tokenize_chat(x, tokenizer),
        remove_columns=train_set.column_names,
        desc="Tokenize train"
    )
    print("Tokenizing val set...")
    val_set = val_set.map(
        lambda x: tokenize_chat(x, tokenizer),
        remove_columns=val_set.column_names,
        desc="Tokenize val"
    )

    # TrainingArguments（不要放 swanlab_project 等非法字段）
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        remove_unused_columns=False,  # 我们保留 input_ids/labels/attention_mask
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        run_name=EXPERIMENT_NAME,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to=[],  # 禁用 HF 内置 logger（我们自己上报）
    )

    # Trainer：使用自定义 data_collator（需要包装为 lambda 传入 tokenizer）
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=lambda features: data_collator(features, tokenizer),
    )

    # 添加 SwanLab callback（用于把 logs 上传到 SwanLab）
    trainer.add_callback(SwanLabCallback(run))

    # 训练
    print("开始训练...")
    try:
        trainer.train()
        print("训练结束，保存 LoRA adapter 与 tokenizer...")
        # 保存 PEFT adapter + tokenizer
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        # 记录完成状态
        run.log({"status": "finished"})
        print("保存完成。")
    except Exception as e:
        print(f"训练过程中出现异常: {e}")
        # 尝试保存当前 adapter
        try:
            model.save_pretrained(os.path.join(OUTPUT_DIR, "interrupted"))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "interrupted_tokenizer"))
            run.log({"status": "interrupted", "error": str(e)})
            print("已保存中断时模型状态。")
        except Exception as e2:
            print(f"中断保存失败: {e2}")
        raise e


if __name__ == "__main__":
    main()
