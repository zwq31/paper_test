from unsloth import FastLanguageModel

MODEL_NAME = "../model/Qwen2.5-7B-Instruct"

# 注意这里同时接收model和tokenizer两个对象
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# 读取文本文件
with open("../data-continued-pretraining/news_3year.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# 按 <|endoftext|> 分割样本
samples = [s.strip() for s in full_text.split("<|endoftext|>") if s.strip()]

# 添加结尾标记，和训练时一样
EOS_TOKEN = "<|im_end|>"
samples = [sample + EOS_TOKEN for sample in samples]

# 批量编码，快速计算token数量
batch_encoding = tokenizer(
    samples,
    add_special_tokens=False,  # 训练时自己加了EOS，不要额外加特殊token
    padding=False,             # 不做padding，节省时间和内存
    truncation=False,          # 不截断，完整编码
)

# 统计总token数
total_tokens = sum(len(ids) for ids in batch_encoding["input_ids"])

print(f"训练数据总 token 数：{total_tokens}")
