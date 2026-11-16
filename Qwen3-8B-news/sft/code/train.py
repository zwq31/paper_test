#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 保持Unsloth为第一个导入的库，适配低版本
import unsloth

import os
import json
import random
import numpy as np
import torch
import logging
from typing import Dict, Any

# 禁用不必要的日志，减少干扰
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("unsloth").setLevel(logging.INFO)
logging.getLogger("trl").setLevel(logging.ERROR)

# 环境变量配置（提前设置，避免冲突）
os.environ["UNSLOTH_DISABLE_VLLM_CHECK"] = "1"
os.environ["UNSLOTH_DISABLE_STRICT_CHECKS"] = "1" 
os.environ["WANDB_MODE"] = "disabled"  # 禁用WANDB，避免冲突

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
import swanlab

# 设置本地日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='finetune-cpt.log',
    filemode='a',
    encoding='utf-8' 
)
logger = logging.getLogger(__name__)

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 配置参数
MODEL_NAME = "/mnt/project/Qwen3-8B-news/models/Qwen3-8B"
OUTPUT_DIR = "/mnt/project/Qwen3-8B-news/models/Qwen3-8B-sft"
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.15
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 2048
WARMUP_STEPS = 100  # 替换废弃的warmup_ratio
WEIGHT_DECAY = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据格式化函数
def format_qa_item(item):
    return {
        "messages": [
            {"role": "user", "content": f"回答以下问题：{item['question']}"},
            {"role": "assistant", "content": item.get("answer", "")}
        ]
    }

# 加载数据集
def load_and_prepare_datasets():
    logger.info("加载数据集...")
    try:
        with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/val.json', 'r', encoding='utf-8') as f:
            val_data = json.load(f)
    except Exception as e:
        logger.error(f"加载数据集失败：{e}")
        raise

    train_formatted = [format_qa_item(item) for item in train_data if "question" in item]
    val_formatted = [format_qa_item(item) for item in val_data if "question" in item]

    train_dataset = Dataset.from_list(train_formatted).shuffle(seed=42)
    val_dataset = Dataset.from_list(val_formatted)

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    return train_dataset, val_dataset

# 格式化聊天模板
def format_chat_prompts(examples, tokenizer):
    formatted_prompts = []
    for messages in examples["messages"]:
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            formatted_prompts.append(formatted)
        except Exception as e:
            logger.warning(f"格式化样本失败：{e}，跳过该样本")
            formatted_prompts.append("")
    return {"text": formatted_prompts}

# 自定义SwanLab回调（简化版）
class CustomSwanLabCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step == 0:
            return
        swanlab_logs = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)) and not k.startswith("_"):
                if k.startswith("eval_"):
                    swanlab_logs[f"eval/{k.replace('eval_', '')}"] = v
                elif k in ["loss", "learning_rate", "grad_norm"]:
                    swanlab_logs[f"train/{k}"] = v
        if swanlab_logs:
            self.run.log(swanlab_logs, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.run.finish()

def main():
    # 初始化SwanLab
    logger.info("初始化SwanLab实验...")
    run = swanlab.init(
        project="qwen3-8b",
        name="sft",
        config={
            "model_name": MODEL_NAME,
            "output_dir": OUTPUT_DIR,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "seed": 42,
        },
        logdir=os.path.join(OUTPUT_DIR, "swanlab_logs"),
    )

    # 加载模型（适配低版本Unsloth）
    logger.info("加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # 配置LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=False,
        loftq_config=None,
    )
    logger.info(f"LoRA配置完成，可训练参数：{model.print_trainable_parameters()}")

    # 准备数据集
    train_dataset, eval_dataset = load_and_prepare_datasets()
    train_dataset = train_dataset.map(
        format_chat_prompts,
        batched=True,
        remove_columns=["messages"],
        fn_kwargs={"tokenizer": tokenizer}
    ).filter(lambda x: x["text"] != "")

    eval_dataset = eval_dataset.map(
        format_chat_prompts,
        batched=True,
        remove_columns=["messages"],
        fn_kwargs={"tokenizer": tokenizer}
    ).filter(lambda x: x["text"] != "")

    # 配置数据collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 训练参数（【核心修复】仅保留低版本transformers支持的参数）
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,  # 低版本支持warmup_steps
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(OUTPUT_DIR, "tensorboard_logs"),  # 低版本用logging_dir
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to=[],  # 禁用所有默认日志工具
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=2.0,
        weight_decay=WEIGHT_DECAY,
        # 【关键修改】仅保留低版本支持的push_to_hub=False，移除其他不支持的参数
        push_to_hub=False,
        disable_tqdm=False,
        remove_unused_columns=True,
    )

    # 初始化SFTTrainer（【关键】参数极简，避免Unsloth内部解析冲突）
    logger.info("初始化SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        packing=True,
        max_seq_length=MAX_LENGTH,
        # 移除所有可能导致冲突的冗余参数
        callbacks=[CustomSwanLabCallback(run)],
    )

    # 开始训练
    logger.info("开始训练...")
    try:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    except Exception as e:
        logger.error(f"训练过程出错：{e}")
        run.finish(status="failed")
        raise

    # 保存模型
    logger.info(f"保存最终模型到 {OUTPUT_DIR}")
    if trainer.is_world_process_zero():
        # 保存LoRA权重
        model.save_pretrained(os.path.join(OUTPUT_DIR, "lora-final"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora-final"))
        
        # 合并LoRA和基础模型（可选）
        logger.info("合并LoRA权重到基础模型...")
        model.save_pretrained_merged(
            save_directory=os.path.join(OUTPUT_DIR, "final-model"),
            model_name=MODEL_NAME,
            dtype="float16",
            save_method="safetensors",
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final-model"))

    logger.info("训练完成！")
    run.finish(status="success")

if __name__ == "__main__":
    main()