import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========= 配置 =========
base_model_name_or_path = "/mnt/project/Qwen3-8B-news/models/Qwen3-8B"
lora_model_path = "/mnt/project/Qwen3-8B-news/outputs/mol_cpt/mol_pretrain_epoch_0"
save_path = "/mnt/project/Qwen3-8B-news/outputs/mol_cpt/Qwen3-8B-1epoch"

# ========= 加载基础模型 =========
print(f">>> 加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name_or_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

# ========= 加载 LoRA =========
print(f">>> 加载 LoRA 参数...")
model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    # adapter_name="default"
)

# ========= 合并权重 =========
print(">>> 合并 LoRA 到基础模型...")
model = model.merge_and_unload()  # 合并后仍在 GPU 1

# ========= 保存 =========
print(">>> 保存合并后的模型...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 模型已保存到 {save_path}")


