#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
# import wandb
import logging
from datasets import load_dataset,Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from unsloth import FastLanguageModel
from tqdm import tqdm
from trl import SFTTrainer
import os
# 设置日志
# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='finetune.log',
    filemode='a',
    encoding='utf-8' 
)

logger = logging.getLogger(__name__)


# 配置参数
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years"  # 预训练模型路径
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/output"  # 输出目录
LORA_RANK = 64  # LoRA秩
LORA_ALPHA = 128  # LoRA alpha
LORA_DROPOUT = 0.05  # LoRA dropout
BATCH_SIZE = 4  # 每个GPU的批次大小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
LEARNING_RATE = 2e-4  # 学习率
EPOCHS = 3  # 训练轮次
MAX_LENGTH = 2048  # 最大序列长度
SAVE_STEPS = 100  # 保存检查点的步数
EVAL_STEPS = 100  # 评估步数
WARMUP_RATIO = 0.03  # 热身比例
WEIGHT_DECAY = 0.01  # 权重衰减
USE_FLASH_ATTENTION = True  # 是否使用Flash Attention

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_qa_item(item):
    """
    将单个问答样本转换为 messages 格式
    """
    return {
        "messages": [
            {"role": "user", "content": f"回答以下问题：{item['question']}"},
            {"role": "assistant", "content": item.get("answer", "")}
        ]
    }

def load_and_prepare_datasets():
    """加载并准备数据集"""
    logger.info("加载数据集...")

    # 加载原始 JSON 数据
    with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # 格式化训练集与验证集
    train_formatted = [format_qa_item(item) for item in train_data]
    val_formatted = [format_qa_item(item) for item in val_data]

    # 构建 HuggingFace datasets.Dataset
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")

    return train_dataset, val_dataset

# def format_chat_prompts(tokenizer, sample):
#     """格式化对话格式"""
#     # 将消息列表转换为Qwen2.5模型所需的格式
#     formatted_messages = []
#     for message in sample["messages"]:
#         # 添加每条消息的角色和内容
#         formatted_messages.append({
#             "role": message["role"],
#             "content": message["content"]
#         })
    
#     # 使用unsloth的格式化方法处理消息
#     formatted_prompt = FastLanguageModel.get_formatted_prompt(formatted_messages)
    
#     # 使用tokenizer编码格式化后的提示
#     encoded = tokenizer(
#         formatted_prompt,
#         truncation=True,
#         max_length=2048,
#         padding=False,
#         return_tensors=None
#     )
    
#     return {
#         "input_ids": encoded["input_ids"],
#         "attention_mask": encoded["attention_mask"],
#         "labels": encoded["input_ids"].copy()
#     }
def format_chat_prompts(examples,tokenizer):
    formatted_prompts = []
    for messages in examples["messages"]:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 关键！仅格式化不编码
            add_generation_prompt=True  # 为assistant添加空内容占位符
        )
        formatted_prompts.append(formatted)
    return {"text": formatted_prompts}

def main():
    # 登录 Wandb
    # wandb.login(key="1b8adc705fb9b3e125c05f15107ad7c22c830811")
    # wandb.init(project="Qwen2.5-7B-instruct-qlora-news_dataset")
    

    # 初始化量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # 使用unsloth加载和优化模型
    logger.info("加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        quantization_config=bnb_config,
        device_map=None,  
        max_seq_length=MAX_LENGTH,
        attn_implementation="flash_attention_2",  # 使用新参数名
        dtype=torch.bfloat16,
    )
    
    model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 准备数据集
    train_dataset, eval_dataset = load_and_prepare_datasets()
    
    train_dataset = train_dataset.map(
    format_chat_prompts,
    batched=True,
    remove_columns=["messages"],
    fn_kwargs={"tokenizer": tokenizer}  # 关键：传递tokenizer参数
    )
    eval_dataset = eval_dataset.map(
        format_chat_prompts,
        batched=True,
        remove_columns=["messages"],
        fn_kwargs={"tokenizer": tokenizer}  # 关键：传递tokenizer参数
    )
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",          # 新参数名 
        eval_steps=EVAL_STEPS,        # 与 save_steps 保持一致
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to=[],
        # 增加评估相关参数  
        load_best_model_at_end=True,    # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss", # 使用eval_loss作为指标
        greater_is_better=False,        # loss越低越好
    )

    # 准备训练
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",  # 对应格式化函数生成的字段
        packing=True,  # 启用序列打包
        tokenizer=tokenizer,
        args=training_args
    )
    # 训练模型
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存最终模型到 {OUTPUT_DIR}")
    model.save_pretrained("../output/Qwen2.5-7B-Instruct-news-finetune-lora")
    tokenizer.save_pretrained("../output/Qwen2.5-7B-Instruct-news-finetune-lora")
    
    # 再次评估并记录最终结果
    eval_results = trainer.evaluate()
    logger.info(f"最终评估结果: {eval_results}")
    
    
    # # 关闭wandb
    # wandb.finish()

if __name__ == "__main__":
    main() 