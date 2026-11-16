#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import logging
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from trl import DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from transformers import Trainer

# ================== SwanLab ==================
import swanlab

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="sft_class.log",
    filemode="a",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# ========== 固定随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ========== 基本配置 ==========
MODEL_NAME = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/5epoch_data/sft_dataset_class_1.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/outputs_5epoch/outputs_dual_sft"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_LENGTH = 2048
ALPHA = 10  # BCE权重

# ========== 自定义回调，用于在每个epoch结束时保存模型 ==========
class EpochSaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        save_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch}")
        os.makedirs(save_dir, exist_ok=True)

        # 获取模型和tokenizer
        model = kwargs.get('model', None)
        tokenizer = kwargs.get('tokenizer', None)
        
        if model is not None and tokenizer is not None:
            # 保存LoRA adapter
            logger.info(f"💾 正在保存第 {epoch} 个epoch的LoRA adapter权重到: {save_dir}")
            model.base.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            # 保存拒答头
            torch.save(model.refusal_head.state_dict(), os.path.join(save_dir, "refusal_head.pt"))
            logger.info(f"✅ 第 {epoch} 个epoch保存完成")

# ========== 数据处理 ==========
def load_and_prepare_datasets(data_path, train_ratio=0.9, seed=42):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        formatted_data.append({
            "messages": [
                {"role": "user", "content": f"回答以下问题：{item['question']}"},
                {"role": "assistant", "content": item.get("answer", "")}
            ],
            "refusal_label": int(item.get("refusal_label", 0))
        })

    dataset = Dataset.from_list(formatted_data).shuffle(seed=seed)
    split_idx = int(len(dataset) * train_ratio)
    return dataset.select(range(split_idx)), dataset.select(range(split_idx, len(dataset)))


def format_chat_prompts(examples, tokenizer):
    formatted_prompts, refusal_labels = [], []
    for msg, label in zip(examples["messages"], examples["refusal_label"]):
        formatted = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        formatted_prompts.append(formatted)
        refusal_labels.append(label)

    tokenized = tokenizer(
        formatted_prompts,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )
    tokenized["refusal_label"] = refusal_labels
    return tokenized

# ========== 模型定义 ==========
class DualLossModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.refusal_head = nn.Linear(base_model.config.hidden_size, 1)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.base, "enable_input_require_grads"):
            self.base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def forward(self, input_ids, attention_mask=None, labels=None, refusal_label=None):
        outputs = self.base.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        cls_token = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        refusal_logits = self.refusal_head(cls_token).squeeze(-1)

        # CE Loss
        lm_logits = self.base.lm_head(hidden_states)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = None
        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # BCE Loss
        bce_loss = None
        if refusal_label is not None:
            refusal_label = refusal_label.float().to(refusal_logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(refusal_logits, refusal_label)

        return ce_loss, bce_loss, refusal_logits


# ========== 自定义 Trainer ==========
class DualLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        refusal_label = inputs.get("refusal_label")

        ce_loss, bce_loss, refusal_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
            refusal_label=refusal_label
        )

        # 计算总损失
        total_loss = 0
        if ce_loss is not None:
            total_loss += ce_loss
        if bce_loss is not None:
            total_loss += ALPHA * bce_loss

        # ====== 打印 / 记录每步损失 ======
        # 为防止过多输出，每隔一定步数打印一次
        if self.state.global_step % 10 == 0 and self.state.global_step > 0:
            logger.info(
                f"Step {self.state.global_step} | "
                f"CE Loss: {ce_loss.item() if ce_loss is not None else 0:.4f} | "
                f"BCE Loss: {bce_loss.item() if bce_loss is not None else 0:.4f} | "
                f"Total: {total_loss.item():.4f}"
            )

            # ✅ SwanLab 可视化
            try:
                swanlab.log({
                    "train/ce_loss": ce_loss.item() if ce_loss is not None else 0,
                    "train/bce_loss": bce_loss.item() if bce_loss is not None else 0,
                    "train/total_loss": total_loss.item(),
                    "train/step": self.state.global_step
                })
            except Exception as e:
                pass  # 防止 swanlab 初始化前报错

        return (total_loss, (ce_loss, bce_loss)) if return_outputs else total_loss



# ========== 主程序 ==========
def main():
    logger.info("加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        quantization_config=bnb_config,
        max_seq_length=MAX_LENGTH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    base_model = FastLanguageModel.get_peft_model(
        base_model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = DualLossModel(base_model)

    train_dataset, eval_dataset = load_and_prepare_datasets(DATA_PATH)
    train_dataset = train_dataset.map(format_chat_prompts, batched=True, remove_columns=["messages"], fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(format_chat_prompts, batched=True, remove_columns=["messages"], fn_kwargs={"tokenizer": tokenizer})

    collator = DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant", tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=True,
        report_to=[],
        gradient_checkpointing=True,
    )

    global run
    run = swanlab.init(project="qwen2.5-dual-loss-sft", name="dual-loss-sft", config={"alpha": ALPHA})

    # 创建回调实例
    epoch_save_callback = EpochSaveCallback()

    trainer = DualLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[epoch_save_callback],  # 添加回调
    )

    logger.info("开始训练...")
    trainer.train()

    # ✅ 最后保存完整模型（不合并，直接保存LoRA权重）
    if trainer.is_world_process_zero():
        save_dir = os.path.join(OUTPUT_DIR, "final-model")
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        try:
            # 直接保存LoRA adapter，不进行合并
            model.base.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            torch.save(model.refusal_head.state_dict(), os.path.join(save_dir, "refusal_head.pt"))
            
            # 保存完整的模型配置信息
            config_info = {
                "model_type": "DualLossModel",
                "base_model": MODEL_NAME,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "refusal_head_dim": model.refusal_head.in_features
            }
            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
                
            logger.info(f"✅ 模型及拒答头已成功保存到: {save_dir}")
            logger.info("💡 注意：这是LoRA adapter，使用时需要加载到基础模型上")
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()