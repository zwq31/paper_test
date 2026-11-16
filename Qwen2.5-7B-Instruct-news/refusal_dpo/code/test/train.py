#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# -------------------------
# 方案 B：使用 Instruction + Output 的 SFT 格式（ChatML）
# -------------------------
def build_chatml(instruction, output):
    return (
        "<|im_start|>user\n"
        f"{instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output}\n"
        "<|stop|>\n"
        "<|im_end|>"
    )

# -------------------------
# 加载数据
# -------------------------
def load_dataset(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)  # 你的文件是完整 JSON 数组
        for obj in raw_json:
            prompt = obj["prompt"]
            ans = obj["chosen"]

            # ChatML Instruction + Output
            full_text = build_chatml(prompt, ans)
            data.append({"text": full_text})
    return data

# -------------------------
# SwanLab 监控模块（保持你原来的逻辑）
# -------------------------
import swanlab
from swanlab.integration.huggingface import SwanLabCallback

# -------------------------
# 主训练函数
# -------------------------
def main():
    # 数据路径
    train_data_path = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/2leidantiao_train_data.json"
    raw_data = load_dataset(train_data_path)

    # Dataset 构建并拆分
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.shuffle(seed=42)
    split_datasets = dataset.train_test_split(test_size=0.1)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # -------------------------
    # 模型、Tokenizer、4bit 量化
    # -------------------------
    model_name = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 增加 stop token
    special_tokens = {"additional_special_tokens": ["<|stop|>"]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )

    # resize embeddings（新增 token 必须做）
    model.resize_token_embeddings(len(tokenizer))

    # -------------------------
    # LoRA 配置
    # -------------------------
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # -------------------------
    # Tokenize
    # -------------------------
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize, batched=False)
    eval_dataset = eval_dataset.map(tokenize, batched=False)

    # -------------------------
    # DataCollator
    # -------------------------
    def data_collator(features):
        batch = tokenizer.pad(features, padding="longest", return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # -------------------------
    # TrainingArguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir="/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/outputs/output_new_2lei",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=1,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        save_strategy="epoch",
        eval_strategy="epoch",
    )

    # -------------------------
    # 启动 SwanLab
    # -------------------------
    swanlab.init(
        project="reflex-confidence-model",
        config={"lr": 2e-5, "lora_r": 64},
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[SwanLabCallback()],
    )

    trainer.train()
    trainer.save_model("final-model")


if __name__ == "__main__":
    main()
