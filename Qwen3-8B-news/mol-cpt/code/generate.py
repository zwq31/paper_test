from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "/mnt/project/Qwen3-8B-news/models/Qwen3-8B"

# 1. 配置 4-bit 量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 启用 4-bit 加载
    bnb_4bit_quant_type="nf4",      # 使用 NF4 数据类型 (通常效果最好)
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算数据类型 (建议与模型权重一致，通常为 bf16)
    bnb_4bit_use_double_quant=True, # 启用双重量化以节省更多显存
)

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

# 3. 加载模型 (整合了你的额外参数和量化配置)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,       # <--- 关键：传入量化配置
    device_map="auto",                    # 自动分配设备 (量化后通常必须用 auto 或 manual)
    torch_dtype=torch.bfloat16,           # 基础精度
    trust_remote_code=True,               # 你的要求
    ignore_mismatched_sizes=True,         # 你的要求：忽略尺寸不匹配警告
)





# 5. 准备模型输入
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]

# 应用聊天模板
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True 
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 6. 生成文本
# 注意：量化模型的生成参数可能与全精度略有不同，max_new_tokens 保持不变
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2400,
    pad_token_id=tokenizer.eos_token_id # 建议显式指定 pad_token_id 以避免警告
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# 7. 解析思考内容 (保持原有逻辑)
thinking_content = ""
content = ""

try:
    # rindex finding 151668 (</think>)
    # 注意：确保 151668 确实是该模型对应的 </think> token ID，不同 tokenizer 版本可能不同
    if 151668 in output_ids:
        index = len(output_ids) - output_ids[::-1].index(151668)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    else:
        # 如果没有找到思维标签，全部视为普通内容
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
except ValueError:
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("Thinking content:", thinking_content)
print("Content:", content)