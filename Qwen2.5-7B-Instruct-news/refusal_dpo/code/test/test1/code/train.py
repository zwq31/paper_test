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
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =======================
# 配置参数
# =======================
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/reject/sft_data.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42

SYSTEM_PROMPT = """请你按照以下格式回答问题：
1. 先给出直接答案
2. 然后进行反思解释
3. 最后给出置信度评分

请严格按照以下格式输出：
答案: [你的答案]
反思: [你的反思过程]
置信度: [0-1之间的数值]"""

# =======================
# 随机种子
# =======================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =======================
# 加载数据并随机划分
# =======================
def load_dataset(path, train_ratio=0.9):
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

    def convert(dataset):
        samples = []
        for item in dataset:
            if 'input' in item and 'output' in item:
                samples.append({"input": item['input'], "output": item['output']})
        return samples

    train_samples = convert(train_data)
    val_samples = convert(val_data)

    print(f"训练样本数: {len(train_samples)}")
    print(f"验证样本数: {len(val_samples)}")

    if train_samples:
        print("训练样本示例:")
        print(f"Input: {train_samples[0]['input'][:100]}...")
        print(f"Output: {train_samples[0]['output'][:100]}...")

    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)

# =======================
# 手动构造 Chat 格式 Tokenize
# =======================
def tokenize(example, tokenizer, max_length=MAX_LENGTH):
    # 构造对话
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]

    # system + user 文本
    input_text = f"<|im_start|>system\n{conversation[0]['content']}<|im_end|>\n"
    input_text += f"<|im_start|>user\n{conversation[1]['content']}<|im_end|>\n"

    # assistant 文本
    assistant_text = f"<|im_start|>assistant\n{conversation[2]['content']}<|im_end|>"

    # 完整文本
    full_text = input_text + assistant_text

    # tokenize
    tokenized_full = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    tokenized_input = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_len = len(tokenized_input["input_ids"])
    full_len = len(tokenized_full["input_ids"])

    # 构造 labels: 输入部分为 -100，assistant 部分保留
    labels = [-100] * input_len + tokenized_full["input_ids"][input_len:]
    if len(labels) < full_len:
        labels += [-100] * (full_len - len(labels))
    elif len(labels) > full_len:
        labels = labels[:full_len]

    return {"input_ids": tokenized_full["input_ids"], "labels": labels}

# =======================
# 主训练流程
# =======================
def main():
    print("开始训练流程...")

     # 初始化SwanLab
    run = swanlab.init(
        project="model-generate-refusal",
        name=f"sft",
       
    )
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 加载模型
    print("加载模型和分词器...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA 配置
    print("配置LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.4f}%)")

    # 加载数据集
    print("加载数据集...")
    train_set, val_set = load_dataset(DATA_PATH)

    # Tokenize 数据集
    print("Tokenize 数据集...")
    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=train_set.column_names,
        desc="Tokenizing 训练集"
    )
    val_set = val_set.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=val_set.column_names,
        desc="Tokenizing 验证集"
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    # TrainingArguments
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
        report_to="swanlab",
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("开始训练...")

    try:
        trainer.train()
        print("保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("训练完成!")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


