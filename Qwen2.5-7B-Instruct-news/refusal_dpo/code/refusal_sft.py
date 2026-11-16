#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import random
import torch
import logging
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
from swanlab.integration.transformers import SwanLabCallback
from transformers import BitsAndBytesConfig
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

# ---------------- 日志配置 ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qlora_refusal_contrastive.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ---------------- 随机种子 ----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ---------------- 配置参数 ----------------
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/dpo_dataset_all_moban.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/outputs_5epoch/outputs_margin_contrastive"
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 2048
MARGIN = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- 数据加载 ----------------
def format_chat_prompt(item):
    return {
        "question": item["question"],
        "positive_answer": item["positive_answer"],
        "negative_answer": item["negative_answer"]
    }

logger.info(f"加载数据集: {DATA_PATH}")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
formatted_data = [format_chat_prompt(d) for d in raw_data]
dataset = Dataset.from_list(formatted_data)
dataset = dataset.shuffle(seed=42)

split_idx = int(len(dataset) * 0.9)
train_dataset = dataset.select(range(split_idx))
eval_dataset = dataset.select(range(split_idx, len(dataset)))
logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

# ---------------- 模型与 tokenizer ----------------
logger.info("加载模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    max_seq_length=MAX_LENGTH,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
)

# 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

# ---------------- SwanLab Callback ----------------
swanlab_callback = SwanLabCallback()

# ---------------- 数据预处理 ----------------
def preprocess_dataset(dataset):
    pos_texts = []
    neg_texts = []
    for item in dataset:
        pos_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"请回答以下问题：{item['question']}"},
             {"role": "assistant", "content": item["positive_answer"]}],
            tokenize=False, add_generation_prompt=True
        )
        neg_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"请回答以下问题：{item['question']}"},
             {"role": "assistant", "content": item["negative_answer"]}],
            tokenize=False, add_generation_prompt=True
        )
        pos_texts.append(pos_text)
        neg_texts.append(neg_text)
    pos_tokens = tokenizer(pos_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    neg_tokens = tokenizer(neg_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    return pos_tokens, neg_tokens

train_pos_tokens, train_neg_tokens = preprocess_dataset(train_dataset)
eval_pos_tokens, eval_neg_tokens = preprocess_dataset(eval_dataset)

# ---------------- DataLoader ----------------
class RefusalDataset(torch.utils.data.Dataset):
    def __init__(self, pos_tokens, neg_tokens):
        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens
    def __len__(self):
        return self.pos_tokens["input_ids"].size(0)
    def __getitem__(self, idx):
        return (
            {k: v[idx] for k, v in self.pos_tokens.items()},
            {k: v[idx] for k, v in self.neg_tokens.items()}
        )

train_loader = DataLoader(RefusalDataset(train_pos_tokens, train_neg_tokens),
                          batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(RefusalDataset(eval_pos_tokens, eval_neg_tokens),
                         batch_size=BATCH_SIZE, shuffle=False)

# ---------------- 优化器 ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ---------------- 训练循环 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
    for step, (pos_batch, neg_batch) in train_iter:
        optimizer.zero_grad()

        pos_input_ids = pos_batch["input_ids"].to(device)
        pos_attention_mask = pos_batch["attention_mask"].to(device)
        neg_input_ids = neg_batch["input_ids"].to(device)
        neg_attention_mask = neg_batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pos_logits = model(pos_input_ids, attention_mask=pos_attention_mask).logits
            neg_logits = model(neg_input_ids, attention_mask=neg_attention_mask).logits

            pos_log_probs = F.log_softmax(pos_logits, dim=-1)
            neg_log_probs = F.log_softmax(neg_logits, dim=-1)

            # 忽略 pad token（这里简化处理，实际可 mask 掉）
            pos_token_log_probs = pos_log_probs.gather(dim=-1, index=pos_input_ids.unsqueeze(-1)).squeeze(-1)
            neg_token_log_probs = neg_log_probs.gather(dim=-1, index=neg_input_ids.unsqueeze(-1)).squeeze(-1)

            pos_seq_ll = pos_token_log_probs.sum(dim=-1)
            neg_seq_ll = neg_token_log_probs.sum(dim=-1)

            loss = torch.clamp(MARGIN + neg_seq_ll - pos_seq_ll, min=0).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        train_iter.set_postfix({"loss": total_loss / (step + 1)})

    # ✅ 每轮保存 LoRA 适配器（仅保存 adapter，不合并）
    epoch_output_dir = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    model.save_pretrained(epoch_output_dir)          # 保存 adapter_model.safetensors 和 adapter_config.json
    tokenizer.save_pretrained(epoch_output_dir)
    logger.info(f"Epoch {epoch+1} LoRA 适配器已保存至: {epoch_output_dir}")

# ---------------- 训练完成，保存最终 LoRA ----------------
final_output_dir = os.path.join(OUTPUT_DIR, "final-model")
os.makedirs(final_output_dir, exist_ok=True)
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
logger.info(f"最终 LoRA 适配器已保存至: {final_output_dir}")