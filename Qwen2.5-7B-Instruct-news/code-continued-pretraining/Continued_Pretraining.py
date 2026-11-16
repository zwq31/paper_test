from unsloth import FastLanguageModel
import torch

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
MODEL_NAME = "../model/Qwen2.5-7B-Instruct"  # 模型名称

import csv
from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    def __init__(self, file_path):
        self.file_path = file_path
        self.writer = None
        self.step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # 初始化 CSV 文件
        self.file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['step', 'loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.writer.writerow([state.global_step, logs["loss"]])
            self.file.flush()  # 实时写入磁盘

    def on_train_end(self, args, state, control, **kwargs):
        self.file.close()
loss_logger = LossLoggerCallback("../output-continued-pretraining/3years/training_logs.csv")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME, 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

import wandb
import os
os.environ["WANDB_API_KEY"] = "1b8adc705fb9b3e125c05f15107ad7c22c830811"
os.environ["WANDB_MODE"] = "offline"
wandb.init(project="news_3year", name="news_3year")

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import Dataset
# 1. 读取整个 txt 文件为字符串
with open("../data-continued-pretraining/news_3year.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# 2. 用 <|endoftext|> 分割为样本
samples = [s.strip() for s in full_text.split("<|endoftext|>") if s.strip()]

# 3. 构造 Dataset 对象
dataset = Dataset.from_dict({"text": samples})

EOS_TOKEN = "<|im_end|>"
def formatting_prompts_func(examples):
    return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,
    callbacks=[loss_logger],
    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        warmup_ratio = 0.1,
        num_train_epochs = 3,
        save_strategy="epoch",
        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "../output-continued-pretraining/3years/",
        report_to = "wandb", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()