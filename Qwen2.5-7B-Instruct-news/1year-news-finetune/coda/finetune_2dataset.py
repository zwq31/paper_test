#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unsloth
from unsloth import FastLanguageModel
import os
import json
import random
import numpy as np
import torch
# import wandb
import logging
from datasets import load_dataset, Dataset
from swanlab.integration.transformers import SwanLabCallback
from trl import DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    TrainerCallback
)

from tqdm import tqdm
from trl import SFTTrainer
import os
# 设置日志
# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='finetune_2dataset.log',
    filemode='a',
    encoding='utf-8' 
)

logger = logging.getLogger(__name__)

# 设置随机种子以确保实验可重复性
def set_seed(seed):
    random.seed(seed)           # Python内置随机数生成器
    np.random.seed(seed)        # NumPy随机数生成器
    torch.manual_seed(seed)     # PyTorch CPU随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU随机数生成器

# 设置随机种子为42，这是一个常用的基准值
set_seed(42)

# 配置参数
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years"  # 预训练模型路径
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/output-mol-cpt-drouput0.1-r128-bs16-epoch5-2dataset"  # 输出目录
LORA_RANK = 128  # LoRA秩
LORA_ALPHA = 128  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout
BATCH_SIZE =16  # 每个GPU的批次大小
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数
LEARNING_RATE = 2e-4  # 学习率
EPOCHS = 5  # 训练轮次
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

def format_magpie_item(item):
    """
    将magpie数据集样本转换为与第一个数据集相同的格式
    使用instruction和output字段
    """
    return {
        "messages": [
            {"role": "user", "content": f"回答以下问题：{item['instruction']}"},
            {"role": "assistant", "content": item.get("output", "")}
        ]
    }

def calculate_token_count(text, tokenizer):
    """计算文本的token数量"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def load_jsonl_file(file_path):
    """加载JSONL格式的文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    return data

def load_and_prepare_datasets(tokenizer):
    """加载并准备两个数据集，并确保第二个数据集采样相同token数的数据"""
    print("加载数据集...")

    # 加载第一个数据集
    with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/combined_qa.json', 'r', encoding='utf-8') as f:
        dataset1_data = json.load(f)
    
    # 计算第一个数据集的token总数
    dataset1_token_count = 0
    for item in dataset1_data:
        question_tokens = calculate_token_count(item['question'], tokenizer)
        answer_tokens = calculate_token_count(item.get('answer', ''), tokenizer)
        dataset1_token_count += question_tokens + answer_tokens
    
    print(f"第一个数据集token总数: {dataset1_token_count}")
    
    # 加载第二个数据集（magpie_general JSON文件）
    # 请替换为您的实际文件路径
    magpie_file_path = "/mnt/project/Qwen2.5-7B-Instruct-news/MOL-CPT/data/magpie_general_mol.jsonl"  # 请修改为实际路径
    magpie_data = load_jsonl_file(magpie_file_path)
    
    # 计算第二个数据集中每个样本的token数
    magpie_token_counts = []
    
    for item in magpie_data:
        instruction_tokens = calculate_token_count(item['instruction'], tokenizer)
        output_tokens = calculate_token_count(item.get('output', ''), tokenizer)
        total_tokens = instruction_tokens + output_tokens
        magpie_token_counts.append(total_tokens)
    
    print(f"第二个数据集总样本数: {len(magpie_data)}")
    print(f"第二个数据集平均每个样本token数: {np.mean(magpie_token_counts):.2f}")
    
    # 从第二个数据集中随机采样，使得总token数与第一个数据集相同
    sampled_indices = []
    current_token_count = 0
    target_token_count = dataset1_token_count
    
    # 随机打乱索引
    indices = list(range(len(magpie_data)))
    random.shuffle(indices)
    
    for idx in indices:
        if current_token_count + magpie_token_counts[idx] <= target_token_count:
            sampled_indices.append(idx)
            current_token_count += magpie_token_counts[idx]
        else:
            # 如果加上这个样本会超过目标token数，可以选择跳过或部分采样
            # 这里我们选择跳过，保持不超过目标token数
            continue
    
    print(f"从第二个数据集中采样了 {len(sampled_indices)} 个样本")
    print(f"采样后的第二个数据集token总数: {current_token_count}")
    print(f"目标token数: {target_token_count}")
    print(f"采样比例: {current_token_count/target_token_count*100:.2f}%")
    
    # 格式化两个数据集
    dataset1_formatted = [format_qa_item(item) for item in dataset1_data]
    dataset2_formatted = [format_magpie_item(magpie_data[i]) for i in sampled_indices]
    
    # 合并数据集
    combined_dataset = dataset1_formatted + dataset2_formatted
    
    # 打乱合并后的数据集
    random.shuffle(combined_dataset)
    
    # 构建 HuggingFace datasets.Dataset
    train_dataset = Dataset.from_list(combined_dataset)
    
    print(f"合并后训练集大小: {len(train_dataset)}")
    print(f"第一个数据集样本数: {len(dataset1_formatted)}")
    print(f"第二个数据集样本数: {len(dataset2_formatted)}")
    
    return train_dataset

def format_chat_prompts(examples, tokenizer):
    formatted_prompts = []
    for messages in examples["messages"]:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 关键！仅格式化不编码
            add_generation_prompt=False  # 为assistant添加空内容占位符
        )
        formatted_prompts.append(formatted)
    return {"text": formatted_prompts}

def main():
    # 登录 Wandb
    # wandb.login(key="1b8adc705fb9b3e125c05f15107ad7c22c830811")
    # wandb.init(project="Qwen2.5-7B-instruct-qlora-news_dataset")
    
    swanlab_callback = SwanLabCallback()
    # loss_callback = LossLoggingCallback()
    # 初始化量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # 使用unsloth加载和优化模型
    print("加载模型...")
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

    # 准备数据集（现在需要传入tokenizer用于计算token数）
    train_dataset = load_and_prepare_datasets(tokenizer)
    
    train_dataset = train_dataset.map(
    format_chat_prompts,
    batched=True,
    remove_columns=["messages"],
    fn_kwargs={"tokenizer": tokenizer}  # 关键：传递tokenizer参数
    )

    # 构造专门的 collator，只训练 assistant 部分
    response_template = "<|im_start|>assistant"   # assistant 开始标记
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  # 输出目录
        per_device_train_batch_size=BATCH_SIZE,  # 每个设备的训练批次大小
        # per_device_eval_batch_size=BATCH_SIZE,  # 每个设备的评估批次大小
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 梯度累积步数
        warmup_ratio=WARMUP_RATIO,  # 预热比例
        num_train_epochs=EPOCHS,  # 训练轮次
        learning_rate=LEARNING_RATE,  # 学习率
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),  # 日志目录
        logging_steps=1,  # 日志记录步数间隔
        save_strategy="epoch",           # 改为每轮保存
        # eval_strategy="epoch",           # 改为每轮评估
        bf16=True,  # 使用bfloat16精度
        tf32=True,  # 使用TF32精度
        gradient_checkpointing=True,  # 使用梯度检查点减少内存使用
        report_to=[],  # 不报告到任何平台
        # load_best_model_at_end=True, # 训练结束时加载最佳模型
        # metric_for_best_model="eval_loss",  # 最佳模型指标
        greater_is_better=False,  # 指标越小越好
        max_grad_norm=2.0,  # 梯度裁剪阈值，常用 1.0
    )

    # 准备训练
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        dataset_text_field="text",  # 对应格式化函数生成的字段
        packing=True,  # 启用序列打包
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[swanlab_callback],  # 添加自定义回调
        data_collator=collator
    )

    # 训练模型
    print("开始训练...")
    train_result = trainer.train()
    
    # 保存模型
    print(f"保存最终模型到 {OUTPUT_DIR}")
    # 训练完成后，主进程合并并保存完整模型
    if trainer.is_world_process_zero():
        print("Saving merged model...")
        model.save_pretrained_merged(
            save_directory=os.path.join(OUTPUT_DIR, "final-model"),
            model_name=MODEL_NAME,  # ✅ 必须指定原始基础模型
            dtype="float16",                         # 或 bfloat16
            save_method="safetensors",               # 推荐 "safetensors" 或 "merged_16bit"
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final-model"))
        print("Model saved.")
    
    # # 保存损失记录
    # loss_callback.save_losses(OUTPUT_DIR)
    
    # # 评估并记录最终结果
    # eval_results = trainer.evaluate()
    # print(f"最终评估结果: {eval_results}")
    
    # # 记录训练指标
    # metrics = train_result.metrics
    # metrics["eval_loss"] = eval_results["eval_loss"]
    
    # # 保存训练和评估指标
    # with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
    #     json.dump(metrics, f)

if __name__ == "__main__":
    main()