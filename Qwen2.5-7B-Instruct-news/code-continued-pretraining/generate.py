
import torch
from peft import PeftModel
from unsloth import FastLanguageModel
from tqdm import tqdm

# 模型路径配置
MODEL_NAME = "../model/Qwen2.5-7B-Instruct"
LORA_PATH = "../output-continued-pretraining/3years/checkpoint-1980"

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)
FastLanguageModel.for_inference(model)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文件路径
input_file = "../data-continued-pretraining/news_3year.txt"  # 替换为你的实际路径
output_file = "../data-continued-pretraining/generated_news_3year.txt"

# 读取样本
def parse_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        blocks = f.read().split("<|endoftext|>")
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

# 逐条生成并立即写入
for idx, (time, title) in enumerate(samples):
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

    # 立即保存
    with open(output_file, "a", encoding="utf-8") as out_f:
        out_f.write(f"新闻发表时间：{time}\n")
        out_f.write(f"新闻标题：{title}\n")
        out_f.write(f"新闻内容：{generated_content}\n")
        out_f.write("<|endoftext|>\n")

    print(f"✅ 已生成并保存样本 {idx + 1}")
