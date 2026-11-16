#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    Trainer
)
from trl import DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel

# ================== SwanLab ==================
import swanlab

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="sft_refusal_only.log",
    filemode="a",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ========== 基本配置 ==========
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/sft_dataset_class_1.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/outputs_5epoch/outputs_refusal_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 2048

# ========== 保存回调 ==========
class EpochSaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        save_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        model = kwargs.get('model', None)
        tokenizer = kwargs.get('tokenizer', None)
        if model is not None and tokenizer is not None:
            logger.info(f"💾 保存第 {epoch} 个epoch的拒答头权重到: {save_dir}")
            torch.save(model.refusal_head.state_dict(), os.path.join(save_dir, "refusal_head.pt"))
            tokenizer.save_pretrained(save_dir)
            logger.info(f"✅ 第 {epoch} 个epoch保存完成")

# ========== 数据 ==========
def load_and_prepare_datasets(data_path, train_ratio=0.9, seed=42):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        formatted_data.append({
            "messages": [
                {"role": "user", "content": f"回答以下问题：{item['question']}"},
                {"role": "assistant", "content": item.get("answer", "")}
            ],
            "refusal_label": int(item.get("refusal_label", 0))
        })

    dataset = Dataset.from_list(formatted_data).shuffle(seed=seed)
    split_idx = int(len(dataset) * train_ratio)
    return dataset.select(range(split_idx)), dataset.select(range(split_idx, len(dataset)))

def format_chat_prompts(examples, tokenizer):
    formatted_prompts, refusal_labels = [], []
    for msg, label in zip(examples["messages"], examples["refusal_label"]):
        formatted = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        formatted_prompts.append(formatted)
        refusal_labels.append(label)

    tokenized = tokenizer(
        formatted_prompts,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    tokenized["refusal_label"] = refusal_labels
    return tokenized

# ========== 模型定义 ==========
class RefusalOnlyModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.refusal_head = nn.Linear(base_model.config.hidden_size, 1)

        # 冻结主干，仅训练拒答头
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, refusal_label=None):
        outputs = self.base.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        cls_token = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        refusal_logits = self.refusal_head(cls_token).squeeze(-1)

        bce_loss = None
        if refusal_label is not None:
            refusal_label = refusal_label.float().to(refusal_logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(refusal_logits, refusal_label)
        return bce_loss, refusal_logits

    # ✅ 修复 gradient_checkpointing 报错
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self, **kwargs):
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()

# ========== Trainer ==========
class RefusalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        refusal_label = inputs.get("refusal_label")
        loss, refusal_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            refusal_label=refusal_label
        )

        if self.state.global_step % 10 == 0 and self.state.global_step > 0:
            logger.info(f"Step {self.state.global_step} | BCE Loss: {loss.item():.4f}")
            try:
                swanlab.log({
                    "train/bce_loss": loss.item(),
                    "train/step": self.state.global_step
                })
            except Exception:
                pass

        return (loss, refusal_logits) if return_outputs else loss

# ========== 主程序 ==========
def main():
    logger.info("加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        quantization_config=bnb_config,
        max_seq_length=MAX_LENGTH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    model = RefusalOnlyModel(base_model)

    train_dataset, eval_dataset = load_and_prepare_datasets(DATA_PATH)
    train_dataset = train_dataset.map(format_chat_prompts, batched=True, remove_columns=["messages"], fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(format_chat_prompts, batched=True, remove_columns=["messages"], fn_kwargs={"tokenizer": tokenizer})

    collator = DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant", tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        report_to=[],
        gradient_checkpointing=True,  # ✅ 保留，兼容 HF Trainer
    )

    global run
    run = swanlab.init(project="qwen2.5-refusal-only", name="refusal-head-only")

    epoch_callback = EpochSaveCallback()

    trainer = RefusalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[epoch_callback],
    )

    logger.info("开始训练拒答头...")
    trainer.train()

    if trainer.is_world_process_zero():
        save_dir = os.path.join(OUTPUT_DIR, "final-refusal-head")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.refusal_head.state_dict(), os.path.join(save_dir, "refusal_head.pt"))
        tokenizer.save_pretrained(save_dir)
        logger.info(f"✅ 拒答头已成功保存到: {save_dir}")

if __name__ == "__main__":
    main()
