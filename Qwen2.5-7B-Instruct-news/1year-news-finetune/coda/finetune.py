#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unsloth
from unsloth import FastLanguageModel
import os
import json
import random
import numpy as np
import torch
import logging
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback  # ✅ 保留 SwanLab 集成
from trl import SFTTrainer  # 仅从 trl 导入 SFTTrainer
from transformers import TrainingArguments
from transformers import TrainerCallback  # 用于自定义回调
import swanlab  # ✅ 导入 swanlab

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='finetune-cpt.log',
    filemode='a',
    encoding='utf-8' 
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 配置参数
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-finetune-news-enhanced-1moban"
LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.15
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 2048
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_qa_item(item):
    return {
        "messages": [
            {"role": "user", "content": f"回答以下问题：{item['question']}"},
            {"role": "assistant", "content": item.get("answer", "")}
        ]
    }

def load_and_prepare_datasets():
    logger.info("加载数据集...")
    with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected/sft_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    train_formatted = [format_qa_item(item) for item in train_data]
    val_formatted = [format_qa_item(item) for item in val_data]

    train_dataset = Dataset.from_list(train_formatted).shuffle(seed=42)
    val_dataset = Dataset.from_list(val_formatted)

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    return train_dataset, val_dataset  # ✅ 修正：返回两个数据集

def format_chat_prompts(examples, tokenizer):
    formatted_prompts = []
    for messages in examples["messages"]:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_prompts.append(formatted)
    return {"text": formatted_prompts}

# ✅ 自定义 SwanLab 回调（适配 SFTTrainer）
class CustomSwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            swanlab_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if key.startswith("eval_"):
                        swanlab_logs[f"eval/{key[5:]}"] = value  # 如 eval_loss -> eval/loss
                    elif "loss" in key or "learning_rate" in key or "grad_norm" in key:
                        swanlab_logs[f"train/{key}"] = value
            if swanlab_logs:
                swanlab.log(swanlab_logs, step=state.global_step)

def main():
    # ✅ 初始化 SwanLab 实验
    run = swanlab.init(
        project="qwen2.5-sft-1news-data-enhanced",  # 项目名
        name="1moban-sft",  # 实验名
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
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "seed": 42,
        }
    )

    # 加载模型（4-bit + Unsloth）
    logger.info("加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 准备数据
    train_dataset, eval_dataset = load_and_prepare_datasets()  # ✅ 修正：获取 eval_dataset

    train_dataset = train_dataset.map(
        format_chat_prompts,
        batched=True,
        remove_columns=["messages"],
        fn_kwargs={"tokenizer": tokenizer}
    )
    eval_dataset = eval_dataset.map(
        format_chat_prompts,
        batched=True,
        remove_columns=["messages"],
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 构造 collator
    collator = None

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",  # ✅ 启用评估
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to=[],  # 不使用其他日志平台
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=2.0,
    )

    # 准备训练
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # ✅ 修正：传入 eval_dataset
        dataset_text_field="text",
        packing=True,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        # callbacks=[CustomSwanLabCallback()]  # ✅ 使用自定义回调
    )

    # ✅ 手动添加自定义回调（SwanLabCallback 集成可能需要特殊处理）
    trainer.add_callback(CustomSwanLabCallback())

    # 训练模型
    logger.info("开始训练...")
    train_result = trainer.train()

    # 保存模型
    logger.info(f"保存最终模型到 {OUTPUT_DIR}")
    if trainer.is_world_process_zero():
        print("Saving merged model...")
        model.save_pretrained_merged(
            save_directory=os.path.join(OUTPUT_DIR, "final-model"),
            model_name=MODEL_NAME,
            dtype="float16",
            save_method="safetensors",
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final-model"))
        print("Model saved.")

if __name__ == "__main__":
    main()


# import unsloth
# from unsloth import FastLanguageModel
# import os
# import json
# import random
# import numpy as np
# import torch
# import logging
# from datasets import Dataset
# from swanlab.integration.transformers import SwanLabCallback
# from trl import SFTTrainer
# from transformers import TrainingArguments
# from transformers import TrainerCallback
# import swanlab

# # ------------------ 日志 ------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='finetune-cpt.log',
#     filemode='a',
#     encoding='utf-8' 
# )
# logger = logging.getLogger(__name__)

# # ------------------ 随机种子 ------------------
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
# set_seed(42)

# # ------------------ 参数 ------------------
# MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years"
# OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-finetune-news-enhanced_2moban"

# LORA_RANK = 128
# LORA_ALPHA = 128
# LORA_DROPOUT = 0.15
# BATCH_SIZE = 32
# GRADIENT_ACCUMULATION_STEPS = 2
# LEARNING_RATE = 2e-4
# EPOCHS = 3
# MAX_LENGTH = 2048
# WARMUP_RATIO = 0.03
# WEIGHT_DECAY = 0.01

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ------------------ 样本格式化 ------------------
# def format_qa_item(item):
#     """
#     item['type'] -> "answered" 或 "refused"
#     """
#     if item["type"] == "answered":
#         system_content = """你是一名新闻事实问答助手。
# 规则：
# 1. 如果新闻中明确提供了答案，则直接回答。
# 2. 不要编造信息。"""
#         assistant_content = item["answer"]

#     elif item["type"] == "refused":
#         system_content = """你是一名新闻事实问答助手。
# 规则：
# 1. 如果新闻中没有提供所需信息，必须拒绝回答。
# 2. 拒绝时必须说：
# “抱歉，我无法回答这个问题，因为我缺乏相关知识，无法准确回答。”
# 禁止编造信息。"""
#         assistant_content = "抱歉，我无法回答这个问题，因为我缺乏相关知识，无法准确回答。"
#     else:
#         raise ValueError(f"未知样本类型: {item['type']}")

#     return {
#         "messages": [
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": item["question"]},
#             {"role": "assistant", "content": assistant_content}
#         ]
#     }

# # ------------------ 加载数据集 ------------------
# def load_and_prepare_datasets():
#     logger.info("加载数据集...")
#     with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected/sft_train.json', 'r', encoding='utf-8') as f:
#         train_data = json.load(f)
#     with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/val.json', 'r', encoding='utf-8') as f:
#         val_data = json.load(f)

#     train_formatted = [format_qa_item(item) for item in train_data]
#     val_formatted = [format_qa_item(item) for item in val_data]

#     train_dataset = Dataset.from_list(train_formatted).shuffle(seed=42)
#     val_dataset = Dataset.from_list(val_formatted)

#     logger.info(f"训练集大小: {len(train_dataset)}")
#     logger.info(f"验证集大小: {len(val_dataset)}")
#     return train_dataset, val_dataset

# # ------------------ Chat 模板转换 ------------------
# def format_chat_prompts(examples, tokenizer):
#     formatted_prompts = []
#     for messages in examples["messages"]:
#         formatted = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=False
#         )
#         formatted_prompts.append(formatted)
#     return {"text": formatted_prompts}

# # ------------------ SwanLab 回调 ------------------
# class CustomSwanLabCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             swanlab_logs = {}
#             for key, value in logs.items():
#                 if isinstance(value, (int, float)):
#                     if key.startswith("eval_"):
#                         swanlab_logs[f"eval/{key[5:]}"] = value
#                     elif "loss" in key or "learning_rate" in key or "grad_norm" in key:
#                         swanlab_logs[f"train/{key}"] = value
#             if swanlab_logs:
#                 swanlab.log(swanlab_logs, step=state.global_step)

# # ------------------ 主程序 ------------------
# def main():
#     # SwanLab 实验初始化
#     run = swanlab.init(
#         project="qwen2.5-sft-1news-data-enhanced",
#         name="2moban-sft",
#         config={
#             "model_name": MODEL_NAME,
#             "output_dir": OUTPUT_DIR,
#             "lora_rank": LORA_RANK,
#             "lora_alpha": LORA_ALPHA,
#             "lora_dropout": LORA_DROPOUT,
#             "batch_size": BATCH_SIZE,
#             "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
#             "learning_rate": LEARNING_RATE,
#             "epochs": EPOCHS,
#             "max_length": MAX_LENGTH,
#             "warmup_ratio": WARMUP_RATIO,
#             "weight_decay": WEIGHT_DECAY,
#             "seed": 42,
#         }
#     )

#     # ------------------ 加载模型 ------------------
#     logger.info("加载模型...")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=MODEL_NAME,
#         max_seq_length=MAX_LENGTH,
#         dtype=torch.bfloat16,
#         load_in_4bit=True,
#         attn_implementation="flash_attention_2",
#     )

#     model = FastLanguageModel.get_peft_model(
#         model,
#         r=LORA_RANK,
#         lora_alpha=LORA_ALPHA,
#         lora_dropout=LORA_DROPOUT,
#         bias="none",
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
#     )

#     # ------------------ 数据准备 ------------------
#     train_dataset, eval_dataset = load_and_prepare_datasets()

#     train_dataset = train_dataset.map(
#         format_chat_prompts,
#         batched=True,
#         remove_columns=["messages"],
#         fn_kwargs={"tokenizer": tokenizer}
#     )
#     eval_dataset = eval_dataset.map(
#         format_chat_prompts,
#         batched=True,
#         remove_columns=["messages"],
#         fn_kwargs={"tokenizer": tokenizer}
#     )

#     # ------------------ Collator ------------------
#     collator = None
#     # ------------------ 训练参数 ------------------
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=8,
#         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#         warmup_ratio=WARMUP_RATIO,
#         num_train_epochs=EPOCHS,
#         learning_rate=LEARNING_RATE,
#         logging_dir=os.path.join(OUTPUT_DIR, "logs"),
#         logging_steps=1,
#         save_strategy="epoch",
#         eval_strategy="epoch",
#         bf16=True,
#         tf32=True,
#         gradient_checkpointing=True,
#         report_to=[],
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#         max_grad_norm=2.0,
#     )

#     # ------------------ Trainer ------------------
#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         dataset_text_field="text",
#         packing=True,
#         tokenizer=tokenizer,
#         args=training_args,
#         data_collator=collator,
#     )
#     trainer.add_callback(CustomSwanLabCallback())

#     # ------------------ 开始训练 ------------------
#     logger.info("开始训练...")
#     train_result = trainer.train()

#     # ------------------ 保存模型 ------------------
#     logger.info(f"保存最终模型到 {OUTPUT_DIR}")
#     if trainer.is_world_process_zero():
#         print("Saving merged model...")
#         model.save_pretrained_merged(
#             save_directory=os.path.join(OUTPUT_DIR, "final-model"),
#             model_name=MODEL_NAME,
#             dtype="float16",
#             save_method="safetensors",
#             max_shard_size="10GB",
#         )
#         tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final-model"))
#         print("Model saved.")

# if __name__ == "__main__":
#     main()
