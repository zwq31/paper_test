import re
import torch
import os
import wandb
from accelerate import Accelerator
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from unsloth import FastLanguageModel, PatchFastRL,is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
os.environ["VLLM_USE_V1"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 禁用异步 CUDA 调用
os.environ["TORCH_USE_CUDA_DSA"] = "1"   # 启用动态共享分配器

# 登录 Wandb
wandb.login(key="1b8adc705fb9b3e125c05f15107ad7c22c830811")
wandb.init(project="Qwen2.5-1.5B-lora-medical-GRPO")

# 初始化 Accelerator
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=16,
)

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/mnt/project/Qwen2.5-1.5B-lora/output/Qwen2.5-1.5B-lora-medical-o1/Qwen2.5-1.5B-lora",
    max_seq_length=1024,
    device_map={"": accelerator.process_index}, # 显式指定设备映射
    fast_inference=True,  # 启用 vLLM 快速推理
    dtype=torch.bfloat16,   # 强制使用 bfloat16
    load_in_4bit=False, 
    gpu_memory_utilization=0.9,  # 显存利用率

)

# 数据预处理
# 定义了一个模板，要求模型输出的回答必须包含 <reasoning> 和 <answer> 标签。
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 一个模板字符串，用于将推理过程和答案格式化为 XML 格式。
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# 从给定文本中提取 <answer> 标签内的内容。
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# 从给定文本中提取 #### 后的内容（如果存在）
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# 加载并预处理 GSM8K 数据集
def get_gsm8k_questions() -> Dataset:
    data =load_from_disk("/mnt/project/Qwen2.5-1.5B-lora/data/gsm8k/train")
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# 奖励函数
# 评估生成的回答是否与标准答案完全一致
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# 检查提取的答案是否为数字
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# 严格检查回答是否符合指定的 XML 格式
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# 宽松检查回答是否包含 <reasoning> 和 <answer> 标签
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# 评估生成的回答是否符合特定的 XML 格式，并对格式的正确性和完整性进行打分
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

# 根据回答中 XML 标签的数量和质量计算奖励
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# 训练配置
PatchFastRL("GRPO", FastLanguageModel)
training_args = GRPOConfig(
    output_dir = "../output/Qwen2.5-1.5B-lora-medical-GRPO",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, 
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = True,
    num_generations = 1, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    num_train_epochs = 3, # Set to 1 for a full training run
    save_strategy="epoch",
    max_grad_norm = 0.1,
    report_to = "wandb", 
    dataloader_num_workers=4,  # 增加数据加载线程
    ddp_find_unused_parameters=False,  # 分布式训练优化
)

# 初始化 Trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
# 初始化 Accelerator
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=16,
)
# 使用 Accelerator 准备组件
model, tokenizer = accelerator.prepare(model, tokenizer)
# 分布式训练准备
trainer.train_dataloader = accelerator.prepare_data_loader(trainer.train_dataloader)
trainer.model, trainer.accelerator = model, accelerator

# 开始训练
accelerator.wait_for_everyone()
trainer.train()

# 保存最终模型
if accelerator.is_main_process:
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
