import os

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()  # 修补DPOTrainer以支持Unsloth优化

import torch
import json
import random
import numpy as np
from torch.utils.data import Dataset
import transformers
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from datasets import Dataset as HFDataset
import swanlab
from tqdm import tqdm

# 设置随机种子以确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 硬编码配置参数
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/final_model"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/reject/dpo_data.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft_dpo"

# 训练参数
BETA = 0.3
LEARNING_RATE = 5e-6
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 3
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 512
WARMUP_RATIO = 0.1
EVAL_SPLIT = 0.1
SEED = 42

# LoRA参数
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
USE_GRADIENT_CHECKPOINTING = "unsloth"

# 其他参数
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.0
LOG_STEPS = 1

def load_and_prepare_datasets(data_path, eval_split=0.1):
    """加载数据并准备为HuggingFace Dataset格式"""
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为DPOTrainer需要的格式
    formatted_data = []
    for item in data:
        formatted_data.append({
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected'],
        })
    
    # # 随机打乱数据
    random.shuffle(formatted_data)
    
    # 划分训练集和评估集
    split_idx = int(len(formatted_data) * (1 - eval_split))
    train_data = formatted_data[:split_idx]
    eval_data = formatted_data[split_idx:]
    
    print(f"训练集大小: {len(train_data)}")
    print(f"评估集大小: {len(eval_data)}")
    
    # 转换为HuggingFace Dataset
    train_dataset = HFDataset.from_list(train_data)
    eval_dataset = HFDataset.from_list(eval_data)
    
    return train_dataset, eval_dataset

def monkey_patch_dpo_trainer_log():
    """猴子补丁修复DPOTrainer的log方法兼容性问题"""
    import trl
    original_log = trl.DPOTrainer.log
    
    def patched_log(self, logs, start_time=None):
        # 如果提供了start_time参数，忽略它
        if start_time is not None:
            return original_log(self, logs)
        else:
            return original_log(self, logs)
    
    # 应用猴子补丁
    trl.DPOTrainer.log = patched_log
    print("已应用DPOTrainer log方法猴子补丁")

def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 应用猴子补丁修复log方法
    monkey_patch_dpo_trainer_log()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化SwanLab
    run = swanlab.init(
        project="model-generate-refusal",
        name=f"sft-dpo",
        config={
            "model_name": MODEL_NAME,
            "data_path": DATA_PATH,
            "output_dir": OUTPUT_DIR,
            "beta": BETA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "warmup_ratio": WARMUP_RATIO,
            "eval_split": EVAL_SPLIT,
            "seed": SEED,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "max_grad_norm": MAX_GRAD_NORM,
            "label_smoothing": LABEL_SMOOTHING,
        }
    )
    
    # 加载和准备数据集
    train_dataset, eval_dataset = load_and_prepare_datasets(DATA_PATH, EVAL_SPLIT)
    
    # 使用Unsloth加载模型
    print(f"使用Unsloth加载模型: {MODEL_NAME}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.padding_side = "right"
    # 配置LoRA参数
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=SEED,
        max_seq_length=MAX_LENGTH,
    )
    
    # 创建DPOConfig
    dpo_config = DPOConfig(
    # 基础训练参数
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=LOG_STEPS,
    optim="adamw_8bit",
    seed=SEED,
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",          # 每轮保存
    eval_strategy="epoch",          # ✅ 每轮结束后运行验证集
    eval_steps=None,                # 与 eval_strategy 冲突，设为 None
    report_to=[],  
    remove_unused_columns=False,
    max_grad_norm=MAX_GRAD_NORM,
    label_smoothing_factor=LABEL_SMOOTHING,
    dataloader_pin_memory=False,
    
    # DPO特有参数
    beta=BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
)
    
    # 创建DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth会自动处理参考模型
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # 自定义回调函数来记录指标到SwanLab（简化版本）
    class SwanLabCallback(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                swanlab_logs = {}
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        if "eval_" in key:
                            swanlab_logs[f"eval/{key.replace('eval_', '')}"] = value
                        elif any(metric in key for metric in ["loss", "reward", "accuracy", "learning_rate"]):
                            swanlab_logs[f"train/{key}"] = value
                if swanlab_logs:
                    run.log(swanlab_logs, step=state.global_step)
    # 添加回调
    dpo_trainer.add_callback(SwanLabCallback())
    
    # 开始训练
    print("开始DPO训练...")
    try:
        train_result = dpo_trainer.train()
        
        # 保存最终模型
        print(f"保存最终模型到 {OUTPUT_DIR}")
        dpo_trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
        
        # 记录最终指标
        final_metrics = {
            "final/train_loss": train_result.metrics.get("train_loss", 0),
            "final/epoch": train_result.metrics.get("epoch", 0),
        }
        run.log(final_metrics)
        
        print("训练完成!")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        # 尝试保存当前进度
        try:
            dpo_trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted_model"))
            print("已保存中断时的模型状态")
        except:
            print("无法保存模型状态")
        
        # 检查是否是log方法的问题
        if "takes 2 positional arguments but 3 were given" in str(e):
            print("检测到log方法兼容性问题，建议升级或降级相关库版本")
            print("尝试执行以下命令解决兼容性问题：")
            print("pip install --upgrade unsloth")
            print("或")
            print("pip install transformers==4.45.0 trl==0.12.0")
        
        raise e
    
    return model, tokenizer

if __name__ == "__main__":
    main()