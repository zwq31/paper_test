import os
import time
import torch
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
import wandb
from sklearn.model_selection import train_test_split

# 登录 Wandb
wandb.login(key="1b8adc705fb9b3e125c05f15107ad7c22c830811")
wandb.init(project="DeepSeek-R1-Distill-Llama-8B-Qlora-1.0")

# 初始化Accelerator
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=16,
)

# 步骤 1：数据集划分
# 加载原始数据集
dataset = load_from_disk("/mnt/project/deepseek-r1-distill-llama-8b-qlora/data/medical-o1-reasoning-SFT-zh")

# 将 Hugging Face Dataset 转换为 Pandas DataFrame 以便使用 train_test_split
df = dataset.to_pandas()

# 划分数据集
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)  # 80% 训练集，20% 临时数据集
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 10% 验证集，10% 测试集

# 重置索引，避免多余的索引列
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# 将划分后的数据集转换回 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)
test_dataset = Dataset.from_pandas(test_data)

# 将划分后的数据集保存
train_dataset.save_to_disk("/mnt/project/deepseek-r1-distill-llama-8b-qlora/data/train_dataset")
valid_dataset.save_to_disk("/mnt/project/deepseek-r1-distill-llama-8b-qlora/data/valid_dataset")
test_dataset.save_to_disk("/mnt/project/deepseek-r1-distill-llama-8b-qlora/data/test_dataset")

# 步骤 2：加载模型和数据
model_path = "/mnt/project/deepseek-r1-distill-llama-8b-qlora/model/DeepSeek-R1-Distill-Llama-8B"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    max_seq_length=2048,
    padding_side="right",
    use_fast=True
)

# 确保设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 添加量化配置对象（更规范）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map={"": accelerator.process_index}  # 显式指定设备映射
)

model.config.use_cache = False  # DeepSpeed需要禁用缓存

# 数据预处理
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}</think>
{}"""

EOS_TOKEN = tokenizer.eos_token  # 必须加上 EOS_TOKEN

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]

    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)

    model_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=2048,
        truncation=True,
        return_tensors="pt"
    )

    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs

# 处理数据集
train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=["Question", "Complex_CoT", "Response"])
valid_dataset = valid_dataset.map(formatting_prompts_func, batched=True, remove_columns=["Question", "Complex_CoT", "Response"])
test_dataset = test_dataset.map(formatting_prompts_func, batched=True, remove_columns=["Question", "Complex_CoT", "Response"])

# 步骤 3：Lora 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 步骤 4：设置训练参数
args = TrainingArguments(
    output_dir="../output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    report_to="wandb",
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4
)

# 使用 DataCollatorForSeq2Seq 作为数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=2048)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_collator)

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=16,
)
# 准备模型、优化器、数据加载器
model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(model, optimizer, train_dataloader, valid_dataloader)

# 步骤 5：训练循环
start_time = time.time()

def save_epoch_model(epoch):
    """每个epoch结束时只保存LoRA适配器"""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            epoch_output_dir = f"../output/epoch_{epoch}"
            os.makedirs(epoch_output_dir, exist_ok=True)

            # 只保存LoRA适配器
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(epoch_output_dir)
            print(f"Epoch {epoch} LoRA适配器已保存至: {epoch_output_dir}")

        except Exception as e:
            print(f"保存失败: {str(e)}")
        
        finally:
            torch.cuda.empty_cache()

def save_final_model():
    """训练结束后合并并保存完整模型"""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            final_dir = "../output/final_model"
            os.makedirs(final_dir, exist_ok=True)

            # 1. 保存最终LoRA适配器
            lora_dir = os.path.join(final_dir, "lora_adapters")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(lora_dir)

            # 2. 合并模型
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            merged_model = PeftModel.from_pretrained(
                base_model,
                lora_dir,
                device_map="auto"
            ).merge_and_unload()

            # 3. 保存完整模型
            merged_model.save_pretrained(
                final_dir,
                safe_serialization=True,
                max_shard_size="100GB"
            )
            tokenizer.save_pretrained(final_dir)
            print(f"最终模型已保存至: {final_dir}")

        except Exception as e:
            print(f"最终模型保存失败: {str(e)}")
        
        finally:
            if 'base_model' in locals():
                del base_model
            if 'merged_model' in locals():
                del merged_model
            torch.cuda.empty_cache()

# 训练循环
for epoch in range(args.num_train_epochs):
    # ===== 训练阶段 =====
    model.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

        # 日志记录
        if step % args.logging_steps == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch, "step": step})

    # ===== 验证阶段 =====
    model.eval()
    eval_loss = 0.0
    for batch in valid_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
    
    avg_eval_loss = eval_loss / len(valid_dataloader)
    wandb.log({"eval_loss": avg_eval_loss, "epoch": epoch})

    # ===== 保存模型 =====
    accelerator.wait_for_everyone()
    save_epoch_model(epoch)  # 只保存LoRA适配器

# ===== 最终保存 =====
save_final_model()  # 训练结束后合并保存完整模型


