# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import os
# import json
# import random
# import torch
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling,
# )
# from peft import LoraConfig, get_peft_model

# # =======================
# # 配置参数
# # =======================
# MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
# DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/sft_refusal_training1.json"
# OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/outputs_5epoch/sft_laoshi_1"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# BATCH_SIZE = 1
# GRADIENT_ACCUMULATION_STEPS = 8
# LEARNING_RATE = 2e-5
# EPOCHS = 3
# MAX_LENGTH = 1024
# LORA_R = 16
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.05
# SEED = 42

# # =======================
# # 随机种子
# # =======================
# def set_seed(seed):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(SEED)

# # =======================
# # 加载数据并随机划分
# # =======================
# def load_dataset(path, train_ratio=0.9):
#     with open(path, "r", encoding="utf-8") as f:
#         data_all = json.load(f)

#     train_data, val_data = train_test_split(data_all, test_size=1-train_ratio, random_state=SEED)

#     def convert(dataset):
#         samples = []
#         for item in dataset:
#             prompt = item['instruction'].strip() + "\n" + item['input'].strip() + "\n"
#             output = item['output'].strip()
#             samples.append({"prompt": prompt, "output": output})
#         return samples

#     train_samples = convert(train_data)
#     val_samples = convert(val_data)

#     return Dataset.from_list(train_samples), Dataset.from_list(val_samples)

# # =======================
# # Tokenize，只计算 output 的 loss
# # =======================
# def tokenize(example, tokenizer, max_length=MAX_LENGTH):
#     prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
#     output_ids = tokenizer(example["output"], add_special_tokens=False)["input_ids"]

#     input_ids = prompt_ids + output_ids
#     labels = [-100]*len(prompt_ids) + output_ids

#     if len(input_ids) > max_length:
#         input_ids = input_ids[:max_length]
#         labels = labels[:max_length]

#     return {"input_ids": input_ids, "labels": labels}

# # =======================
# # 主训练流程
# # =======================
# def main():
#     # 量化配置
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )

#     # 加载模型
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token

#     # LoRA
#     lora_config = LoraConfig(
#         r=LORA_R,
#         lora_alpha=LORA_ALPHA,
#         lora_dropout=LORA_DROPOUT,
#         target_modules=["q_proj","k_proj","v_proj","o_proj"],
#         task_type="CAUSAL_LM"
#     )
#     model = get_peft_model(model, lora_config)

#     # 加载数据集
#     train_set, val_set = load_dataset(DATA_PATH)

#     # Tokenize
#     train_set = train_set.map(lambda x: tokenize(x, tokenizer), remove_columns=train_set.column_names)
#     val_set = val_set.map(lambda x: tokenize(x, tokenizer), remove_columns=val_set.column_names)

#     # Data collator
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     # TrainingArguments
#     args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#     num_train_epochs=EPOCHS,
#     learning_rate=LEARNING_RATE,
#     logging_steps=20,
#     save_strategy="epoch",       # 每轮结束保存一次
#     eval_strategy="epoch",       # 每轮结束验证一次
#     bf16=True,
#     report_to="swanlab"
# )


#     # Trainer
#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_set,
#         eval_dataset=val_set,
#         tokenizer=tokenizer,
#         data_collator=data_collator
#     )

#     trainer.train()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import random
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =======================
# 配置参数
# =======================
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/sft_refusal_training_fixed.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/outputs_5epoch/sft_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
EPOCHS = 3
MAX_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
SEED = 42

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

    train_data, val_data = train_test_split(data_all, test_size=1-train_ratio, random_state=SEED)

    def convert(dataset):
        samples = []
        for item in dataset:
            # 使用与推理完全一致的格式
            prompt = item['instruction'].strip() + item['input'].strip() + "\n"
            output = item['output'].strip()
            samples.append({"prompt": prompt, "output": output})
        return samples

    train_samples = convert(train_data)
    val_samples = convert(val_data)

    print(f"训练样本数: {len(train_samples)}")
    print(f"验证样本数: {len(val_samples)}")
    
    if train_samples:
        print("训练样本示例:")
        print(f"Prompt: {train_samples[0]['prompt'][:100]}...")
        print(f"Output: {train_samples[0]['output'][:100]}...")

    return Dataset.from_list(train_samples), Dataset.from_list(val_samples)

# =======================
# Tokenize，只计算 output 的 loss
# =======================
def tokenize(example, tokenizer, max_length=MAX_LENGTH):
    prompt_text = example["prompt"]
    output_text = example["output"]
    
    # 分别编码prompt和output
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    
    # 合并输入
    input_ids = prompt_ids + output_ids + [tokenizer.eos_token_id]
    # 只计算output部分的loss
    labels = [-100] * len(prompt_ids) + output_ids + [tokenizer.eos_token_id]
    
    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    
    # 确保长度一致
    assert len(input_ids) == len(labels), f"长度不一致: input_ids={len(input_ids)}, labels={len(labels)}"
    
    return {
        "input_ids": input_ids, 
        "labels": labels,
        "attention_mask": [1] * len(input_ids)  # 添加attention_mask
    }

# =======================
# 主训练流程
# =======================
def main():
    print("开始训练流程...")
    
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

    # 准备模型用于k-bit训练
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA配置
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
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"可训练参数: {name}")
    
    print(f"可训练参数: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.4f}%)")

    # 加载数据集
    print("加载数据集...")
    train_set, val_set = load_dataset(DATA_PATH)

    # Tokenize
    print("Tokenize数据集...")
    train_set = train_set.map(
        lambda x: tokenize(x, tokenizer), 
        remove_columns=train_set.column_names,
        desc="Tokenizing训练集"
    )
    val_set = val_set.map(
        lambda x: tokenize(x, tokenizer), 
        remove_columns=val_set.column_names,
        desc="Tokenizing验证集"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8
    )

    # TrainingArguments - 简化配置
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
        # 简化配置
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # 禁用可能引起问题的功能
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        # 优化设置
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # 使用原生Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("开始训练...")
    
    # 检查模型和设备
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"使用BF16: {args.bf16}")
    
    try:
        trainer.train()
        
        # 保存最终模型
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