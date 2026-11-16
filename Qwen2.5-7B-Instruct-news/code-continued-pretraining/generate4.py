import torch
from peft import PeftModel
from unsloth import FastLanguageModel
from tqdm import tqdm
from accelerate import Accelerator

# 初始化加速器，自动处理多卡逻辑
accelerator = Accelerator()

# 模型路径配置
MODEL_NAME = "../model/Qwen2.5-7B-Instruct"
LORA_PATH = "../output-continued-pretraining/3years/checkpoint-1980"

# 加载模型和 tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, LORA_PATH)

# 准备模型用于推理（必须放在 accelerator.prepare 前）
FastLanguageModel.for_inference(model)
model.eval()

# 使用 accelerate 分布模型到多个 GPU
model = accelerator.prepare(model)

device = accelerator.device  # 自动获取当前设备（GPU）

# 文件路径
input_file = "../data-continued-pretraining/news_3year.txt"
output_file = "../data-continued-pretraining/generated_news_3year.txt"

# 读取样本
def parse_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        blocks = f.read().split("}")
    samples = []
    for block in blocks:
        block = block.strip()
        if block:
            try:
                lines = block.split("\n")
                time_line = lines[0].strip()
                title_line = lines[1].strip()
                if time_line.startswith("新闻发表时间：") and title_line.startswith("新闻标题："):
                    time = time_line.replace("新闻发表时间：", "").strip()
                    title = title_line.replace("新闻标题：", "").strip()
                    samples.append((time, title))
            except:
                continue
    return samples

samples = parse_training_data(input_file)
total_samples = len(samples)
print(f"📊 共找到 {total_samples} 条新闻标题需要生成内容。")

# 将数据在多个 GPU 上分片（每个 GPU 只处理一部分）
from torch.utils.data import Dataset, DataLoader

class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = NewsDataset(samples)
dataloader = DataLoader(dataset, batch_size=1)  # batch_size=1 保证每条独立处理

# 使用 accelerator 自动分发数据
dataloader = accelerator.prepare(dataloader)

# 多卡推理主循环
with open(output_file, "w", encoding="utf-8") as out_f:  # 清空文件或新建
    pass  # 确保输出文件为空

for idx, (time, title) in enumerate(tqdm(dataloader, desc="生成中")):
    prompt = f"新闻发表时间：{time}\n新闻标题：{title}\n新闻内容："
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_content = generated.replace(prompt, "").strip()

    # 保存结果（只在主进程写入，避免冲突）
    if accelerator.is_main_process:
        with open(output_file, "a", encoding="utf-8") as out_f:
            out_f.write(f"新闻发表时间：{time}\n")
            out_f.write(f"新闻标题：{title}\n")
            out_f.write(f"新闻内容：{generated_content}\n")
            out_f.write("}\n\n")

    # 每个样本都同步一下（可选）
    accelerator.wait_for_everyone()

# 主进程打印完成信息
if accelerator.is_main_process:
    print(f"✅ 已完成，共处理了 {total_samples} 条数据，并已保存到 {output_file}")