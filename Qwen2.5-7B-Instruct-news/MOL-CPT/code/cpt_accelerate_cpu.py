import json
import torch
import os
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm import tqdm
import time
import torch.distributed as dist
from itertools import zip_longest
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
# --- 1. 初始化 Accelerator ---
accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")
print(f"[Rank {accelerator.process_index}] Hello from {torch.cuda.get_device_name()} (device={torch.cuda.current_device()})")
# --- 2. 加载模型和tokenizer ---
MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("加载学生模型...")
student_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

print("加载教师模型...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="cpu",  # 强制加载到CPU
)
for param in teacher_model.parameters():
    param.requires_grad = False
teacher_model.config.use_cache = False
teacher_model.eval()  # 教师模型只做推理，不训练


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"

# --- 3. 加载LoRA配置 ---
lora_config = LoraConfig(
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
student_model = get_peft_model(student_model, lora_config)
student_model.gradient_checkpointing_enable()
student_model.config.use_cache = False  # 确保 gradient checkpointing 生效

# --- 4. 统计token和加载数据函数 ---
def count_tokens_and_size(path, tokenizer, name=""):
    total_tokens, total_bytes, count = 0, 0, 0
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] {name} 数据第 {count} 条无法解析：{e}")
                continue
            if name == "domain":
                text = sample.get("text", "")
            else:
                text = sample.get("instruction", "") + sample.get("output", "")
            text = text.strip()
            if not text:
                continue
            total_tokens += len(tokenizer(text).input_ids)
            total_bytes += len(text.encode('utf-8'))
            items.append(sample)
            count += 1
    print(f"✅ {name} 加载完毕：{count} 条样本，{total_tokens} tokens，{round(total_tokens/1024/1024, 2)} MB")
    return items, total_tokens, total_bytes

# --- 5. 数据集类 ---
class MoLDatasetSingleType(TorchDataset):
    def __init__(self, data, tokenizer, type, max_seq_len=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.type = type  # 'general' or 'domain'
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.type == 'domain':
            full_text = sample['text'].strip() + EOS_TOKEN
            input_ids = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len).input_ids
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                "is_domain_batch": 1,
            }
        else:
            prompt = self.tokenizer.apply_chat_template([
                {"role": "user", "content": sample["instruction"].strip()}
            ], tokenize=False)
            prompt_ids = self.tokenizer(prompt, truncation=False).input_ids
            response_ids = self.tokenizer(sample["output"].strip() + EOS_TOKEN, truncation=False).input_ids
            input_ids = (prompt_ids + response_ids)[:self.max_seq_len]
            labels = ([-100] * len(prompt_ids) + response_ids)[:self.max_seq_len]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                "is_domain_batch": 0,
            }

# --- 6. 自定义 collate_fn ---
def mol_collate_fn(batch):
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-100
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0
    )
    is_domain_batch = torch.stack([
        torch.tensor(b["is_domain_batch"], dtype=torch.bool) for b in batch
    ])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "is_domain_batch": is_domain_batch,
    }

# --- 7. 加载数据集和dataloader ---
data_paths = {
    "general": "../data/magpie_general_mol1.jsonl",
    "domain": "../data/news_3year1.jsonl",
}
general_data, general_tokens, _ = count_tokens_and_size(data_paths["general"], tokenizer, "general")
domain_data, domain_tokens, _ = count_tokens_and_size(data_paths["domain"], tokenizer, "domain")

# 保证token数平衡
min_tokens = min(general_tokens, domain_tokens)

def truncate_to_tokens(data, target_tokens, mode):
    total, kept = 0, []
    for s in data:
        t = s.get("text", "").strip() if mode == "domain" else s.get("instruction", "").strip() + s.get("output", "").strip()
        n = len(tokenizer(t, add_special_tokens=False).input_ids)
        if total + n > target_tokens:
            break
        total += n
        kept.append(s)
    return kept, total

general_data, _ = truncate_to_tokens(general_data, min_tokens, "general")
domain_data, _ = truncate_to_tokens(domain_data, min_tokens, "domain")

print(f"✅ 加载数据完成：通用样本 {len(general_data)} 条，领域样本 {len(domain_data)} 条")

batch_size = 1  # 你可以根据显存调整
num_train_epochs = 3

# drop_last=True 保证每步4卡同步，最后不足4个样本舍弃
general_dataset = MoLDatasetSingleType(general_data, tokenizer, type='general')
domain_dataset = MoLDatasetSingleType(domain_data, tokenizer, type='domain')

general_loader = DataLoader(
    general_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=mol_collate_fn,
    drop_last=False
)
domain_loader = DataLoader(
    domain_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=mol_collate_fn,
    drop_last=False
)

# --- 8. 优化器和学习率调度器 ---
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
num_training_steps = num_train_epochs * (len(general_loader) + len(domain_loader))
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# --- 9. Accelerator管理 ---
student_model, optimizer, general_loader, domain_loader, lr_scheduler = accelerator.prepare(
    student_model, optimizer, general_loader, domain_loader, lr_scheduler
)

# --- 10. 训练循环 ---
def train_one_batch(batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, accelerator, step_losses, general_kl_log, domain_ce_log, loader_name):
    if len(is_domain) == 0:
        if accelerator.is_main_process:
            tqdm.write(f"[{loader_name}] 跳过空batch")
        return
    with accelerator.accumulate(student_model):
        outputs = student_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        if isinstance(outputs, tuple):
            student_logits = outputs[1]
        else:
            student_logits = outputs.logits
        # teacher临时转到student_model.device
        teacher_device = student_model.device
        teacher_model.to(teacher_device)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=batch["input_ids"].to(teacher_device),
                attention_mask=batch["attention_mask"].to(teacher_device),
            )
            teacher_logits = teacher_outputs.logits
        teacher_model.to("cpu")
        torch.cuda.empty_cache()
        losses = torch.zeros(len(is_domain), device=student_model.device)
        ce_losses = torch.zeros(len(is_domain), device=student_model.device)
        kl_losses = torch.zeros(len(is_domain), device=student_model.device)
        loss_count = 0
        for i in range(len(is_domain)):
            mask = (batch["attention_mask"][i] == 1) & (batch["labels"][i] != -100)
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) == 0:
                continue
            s_logits = student_logits[i, valid_indices]
            t_logits = teacher_logits[i, valid_indices]
            ce_loss = torch.nn.functional.cross_entropy(
                s_logits,
                batch["labels"][i, valid_indices],
                reduction='mean'
            )
            student_log_probs = torch.nn.functional.log_softmax(s_logits, dim=-1)
            teacher_probs = torch.nn.functional.softmax(t_logits, dim=-1)
            kl_per_token = torch.nn.functional.kl_div(
                input=student_log_probs,
                target=teacher_probs,
                reduction="none",
                log_target=False
            ).sum(dim=-1)
            kl_loss = torch.mean(kl_per_token)
            if loader_name == "General":
                losses[i] = alpha * ce_loss + (1 - alpha) * kl_loss
                general_kl_log.append((step_count, kl_loss.item()))
            else:
                losses[i] = (1 - alpha) * ce_loss + alpha * kl_loss
                domain_ce_log.append((step_count, ce_loss.item()))
            ce_losses[i] = ce_loss
            kl_losses[i] = kl_loss
            loss_count += 1
        if loss_count == 0:
            if accelerator.is_main_process:
                tqdm.write(f"[{loader_name}] 跳过无有效token的batch")
            return
        loss = losses.mean()
        ce_loss_mean = ce_losses.mean().item()
        kl_loss_mean = kl_losses.mean().item()
        step_losses.append((step_count, loss.item(), ce_loss_mean, kl_loss_mean))
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        del student_logits, teacher_logits, teacher_outputs, outputs
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            tqdm.write(f"[{loader_name}] step={step_count+1}, loss={loss.item()}, ce={ce_loss_mean}, kl={kl_loss_mean}")

alpha = 0  # 你原先 Trainer 传的alpha参数
student_model.train()

step_losses = []  # 记录每一步的(step, total_loss, ce_loss, kl_loss)
step_count = 0
general_kl_log = []  # 记录通用样本的(step, kl_loss)
domain_ce_log = []   # 记录领域样本的(step, ce_loss)

start_time = time.time()  # 记录训练开始时间

for epoch in range(num_train_epochs):
    general_iter = iter(general_loader)
    domain_iter = iter(domain_loader)
    finished = [False, False]
    loader_names = ["General", "Domain"]
    loader_iters = [general_iter, domain_iter]
    loader_idx = 0
    while not all(finished):
        try:
            batch = next(loader_iters[loader_idx])
            if batch is None:
                finished[loader_idx] = True
                loader_idx = 1 - loader_idx
                continue
        except StopIteration:
            finished[loader_idx] = True
            loader_idx = 1 - loader_idx
            continue
        is_domain = batch.pop("is_domain_batch")
        train_one_batch(
            batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, accelerator,
            step_losses, general_kl_log, domain_ce_log, loader_names[loader_idx]
        )
        step_count += 1
        loader_idx = 1 - loader_idx
    # 每个epoch保存一次模型（保存LoRA权重）
    save_path = f"../output/mol_pretrain_epoch_{epoch+1}"
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(student_model)
    unwrapped_model.save_pretrained(save_path, save_dtype=torch.float16)
    tokenizer.save_pretrained(save_path)

# --- 11. 训练结束保存最终模型 ---
save_path = "../output/final"
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(student_model)
unwrapped_model.save_pretrained(save_path, save_dtype=torch.float16)
tokenizer.save_pretrained(save_path)

# 保存loss到文件，包含step, total_loss, ce_loss, kl_loss
with open("loss_log.txt", "w") as f:
    f.write("step,total_loss,ce_loss,kl_loss\n")
    for s, total, ce, kl in step_losses:
        f.write(f"{s},{total},{ce},{kl}\n")

# 保存通用数据KL损失日志
with open("general_kl_log.txt", "w") as f:
    f.write("step,kl_loss\n")
    for s, kl in general_kl_log:
        f.write(f"{s},{kl}\n")
if general_kl_log:
    avg_kl = sum([kl for _, kl in general_kl_log]) / len(general_kl_log)
    print(f"通用样本KL散度均值: {avg_kl:.6f}")
else:
    print("无通用样本KL散度记录。")

# 保存领域数据CE损失日志
with open("domain_ce_log.txt", "w") as f:
    f.write("step,ce_loss\n")
    for s, ce in domain_ce_log:
        f.write(f"{s},{ce}\n")
if domain_ce_log:
    avg_ce = sum([ce for _, ce in domain_ce_log]) / len(domain_ce_log)
    print(f"领域样本交叉熵损失均值: {avg_ce:.6f}")
else:
    print("无领域样本交叉熵损失记录。")

print("训练结束")
torch.distributed.destroy_process_group() 