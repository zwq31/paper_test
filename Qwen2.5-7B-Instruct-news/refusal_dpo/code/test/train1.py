"""
训练Qwen2.5-7B模型，学习输出反思和置信度
使用LoRA、4-bit量化和SwanLab记录
"""

import os
import math
import sys
import json
import torch
from torch.utils.data import Dataset, random_split

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

try:
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not available. Install it with: pip install swanlab")

def set_random_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReflectionConfidenceDataset(Dataset):
    """包含反思和置信度的数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=1024, use_rejected_for_wrong_refusal=False):
        """
        Args:
            data_path: JSON文件路径，包含训练数据
            tokenizer: tokenizer对象
            max_length: 最大序列长度
            use_rejected_for_wrong_refusal: 对于"错误拒答"的情况，是否使用rejected作为训练目标
                                           如果True，则使用rejected（正确的回答）而不是chosen（错误的拒答）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_rejected_for_wrong_refusal = use_rejected_for_wrong_refusal
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"加载了 {len(self.data)} 个样本")
        if use_rejected_for_wrong_refusal:
            print("注意：对于'错误拒答'的情况，将使用rejected（正确的回答）作为训练目标")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取prompt
        prompt = item.get("prompt", "")
        
        # 决定使用chosen还是rejected
        # 对于"错误拒答"的情况，如果use_rejected_for_wrong_refusal=True，则使用rejected
        category = item.get("category", "")
        if category == "错误拒答" and self.use_rejected_for_wrong_refusal:
            # 使用rejected（正确的回答）作为训练目标
            target_text = item.get("rejected", item.get("chosen", ""))
        else:
            # 使用chosen作为训练目标
            target_text = item.get("chosen", "")
        
        # 构建输入文本：prompt + target_text
        full_text = prompt + target_text
        
        # Tokenize完整文本
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 创建labels（与input_ids相同，用于语言建模）
        labels = input_ids.clone()
        
        # 计算prompt的长度（在tokenize之后）
        # 使用更简单的方法：分别tokenize prompt和target_text
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt"
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]
        
        # 将prompt部分的labels设置为-100（不计算损失）
        # 由于tokenization的一致性，prompt应该出现在input_ids的开头
        # 但需要考虑可能因为截断而导致长度不同
        actual_prompt_length = min(prompt_length, len(input_ids))
        
        # 验证prompt是否匹配（前几个token应该匹配）
        prompt_tokens = prompt_encoding["input_ids"].squeeze(0)
        if len(prompt_tokens) > 0 and len(input_ids) >= len(prompt_tokens):
            # 检查前几个token是否匹配
            check_len = min(5, len(prompt_tokens), len(input_ids))
            if torch.equal(input_ids[:check_len], prompt_tokens[:check_len]):
                # 匹配，使用prompt_length
                labels[:actual_prompt_length] = -100
            else:
                # 不匹配，尝试找到prompt的位置
                # 简化：假设prompt在开头，但长度可能不同
                # 使用一个更保守的估计
                estimated_length = min(prompt_length - 1, len(input_ids) - 10)
                if estimated_length > 0:
                    labels[:estimated_length] = -100
        else:
            # 如果prompt为空或太短，使用全部作为target_text
            # 否则使用prompt_length
            if actual_prompt_length > 0:
                labels[:actual_prompt_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def split_dataset(dataset, val_split_ratio=0.1):
    """将数据集划分为训练集和验证集"""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split_ratio)
    train_size = dataset_size - val_size
    
    train_dataset, eval_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可重复性
    )
    
    print(f"数据集划分: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    return train_dataset, eval_dataset


# ============================================================================
# 配置参数（直接写在代码里，不需要外部传参）
# ============================================================================

# 数据参数
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/reflection_confidence_training_dataset.json"  # 训练数据路径
MAX_SEQ_LEN = 1024  # 最大序列长度
VAL_SPLIT_RATIO = 0.1  # 验证集划分比例

# 模型参数
MODEL_NAME_OR_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"  # 模型路径
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/outputs/output_reject"  # 输出目录

# 训练参数
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # 每设备训练批次大小
PER_DEVICE_EVAL_BATCH_SIZE = 1  # 每设备验证批次大小
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数
LEARNING_RATE = 2e-4  # 学习率
NUM_TRAIN_EPOCHS = 3  # 训练轮数
LR_SCHEDULER_TYPE = "cosine"  # 学习率调度器类型
NUM_WARMUP_STEPS = 100  # 预热步数
WEIGHT_DECAY = 0.01  # 权重衰减
LOGGING_STEPS = 10  # 日志记录的步数间隔

# LoRA参数
LORA_R = 64  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha
LORA_DROPOUT = 0.1  # LoRA dropout
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # LoRA目标模块

# 量化参数
USE_4BIT = True  # 是否使用4-bit量化
BNB_4BIT_COMPUTE_DTYPE = "float16"  # 4-bit量化的计算dtype
BNB_4BIT_QUANT_TYPE = "nf4"  # 4-bit量化类型（nf4或fp4）
USE_NESTED_QUANT = True  # 是否使用嵌套量化

# SwanLab参数
SWANLAB_PROJECT = "qwen-reflection-confidence"  # SwanLab项目名
SWANLAB_EXPERIMENT_NAME = "qwen2.5-7b-lora-4bit"  # SwanLab实验名

# 其他参数
SEED = 42  # 随机种子
GRADIENT_CHECKPOINTING = True  # 是否启用梯度检查点
FP16 = True  # 是否使用FP16训练
BF16 = False  # 是否使用BF16训练
DATALOADER_NUM_WORKERS = 4  # 数据加载器工作进程数
SAVE_TOTAL_LIMIT = 3  # 只保留最近3个检查点

# 关于rejected字段的使用说明：
# - 在SFT阶段（当前训练），rejected字段不会被使用
# - rejected字段主要用于：
#   1. Step 2 (奖励模型训练): 需要chosen和rejected的对比数据
#   2. Step 3 (RLHF训练): 需要chosen和rejected的对比数据
# - 对于"错误拒答"的情况，虽然数据中包含了rejected字段（正确的回答），
#   但在SFT训练中，我们只使用chosen字段（错误的拒答）进行训练
# - 如果您想在SFT阶段也利用rejected数据，可以：
#   1. 对于错误拒答的情况，使用rejected作为训练目标（而不是chosen）
#   2. 或者，同时训练chosen和rejected，让模型学习到应该使用rejected


def create_bnb_config():
    """创建BitsAndBytesConfig"""
    if not USE_4BIT:
        return None
    
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )
    
    return bnb_config


def create_lora_config():
    """创建LoRA配置"""
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return lora_config


def main():
    # 设置随机种子
    set_random_seed(SEED)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化SwanLab
    if SWANLAB_AVAILABLE:
        swanlab.init(
            project=SWANLAB_PROJECT,
            experiment_name=SWANLAB_EXPERIMENT_NAME,
            config={
                "model_name": MODEL_NAME_OR_PATH,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": LEARNING_RATE,
                "batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "num_epochs": NUM_TRAIN_EPOCHS,
                "max_seq_len": MAX_SEQ_LEN,
                "use_4bit": USE_4BIT,
                "val_split_ratio": VAL_SPLIT_RATIO,
            }
        )
        print("SwanLab initialized")
    else:
        print("SwanLab not available, skipping initialization")
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 创建量化配置
    bnb_config = create_bnb_config()
    
    # 加载模型
    print("Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        print("Using 4-bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        **model_kwargs
    )
    
    # 准备模型进行k-bit训练
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    
    # 创建LoRA配置并应用
    print("Applying LoRA...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    # 启用梯度检查点
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    
    # 加载数据集
    print("Loading datasets...")
    # 对于"错误拒答"的情况，是否使用rejected作为训练目标
    # 如果True，则使用rejected（正确的回答）而不是chosen（错误的拒答）
    # 如果False，则使用chosen（错误的拒答），让模型学习到这是错误的行为
    USE_REJECTED_FOR_WRONG_REFUSAL = True  # 设为True则使用rejected，False则使用chosen
    
    full_dataset = ReflectionConfidenceDataset(
        DATA_PATH,
        tokenizer,
        max_length=MAX_SEQ_LEN,
        use_rejected_for_wrong_refusal=USE_REJECTED_FOR_WRONG_REFUSAL
    )
    
    # 划分训练集和验证集
    train_dataset, eval_dataset = split_dataset(
        full_dataset, 
        val_split_ratio=VAL_SPLIT_RATIO
    )
    
    # 创建数据collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=NUM_WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        report_to="swanlab" if SWANLAB_AVAILABLE else "none",
        remove_unused_columns=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
    )
    
    # 创建callbacks
    callbacks = []
    if SWANLAB_AVAILABLE:
        callbacks.append(SwanLabCallback())
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 开始训练
    print("Starting training...")
    print(f"训练配置:")
    print(f"- 总轮次: {NUM_TRAIN_EPOCHS}")
    print(f"- 训练集大小: {len(train_dataset)}")
    print(f"- 验证集大小: {len(eval_dataset)}")
    print(f"- 验证集比例: {VAL_SPLIT_RATIO}")
    print(f"- 每轮保存模型")
    print(f"- 每轮验证模型")
    print(f"- 输出目录: {OUTPUT_DIR}")
    
    trainer.train()
    
    # 保存最终模型
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存LoRA权重
    model.save_pretrained(OUTPUT_DIR)
    
    print("Training completed!")
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()