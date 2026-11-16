import wandb
wandb.login(key="1b8adc705fb9b3e125c05f15107ad7c22c830811")
wandb.init(project="Llama-3.2-1B-GRPO-gsm8k")
##获取数据集
import re
import torch
from datasets import load_dataset, Dataset,load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_from_disk('/mnt/llama_GRPO_project/data/gsm8k')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()
dataset


#设置奖励函数组
# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

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

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# GRPO训练
# 使用 expanduser 来获取绝对路径
import os
model_path = os.path.expanduser('/mnt/llama_GRPO_project/models')
output_dir="/mnt/llama_GRPO_project/outputs/Llama-3.2-1B-GRPO"
run_name="Llama-3.2-1B-GRPO-gsm8k"

training_args = GRPOConfig(
    # ===== 硬件与分布式配置 =====
    per_device_train_batch_size=2,      # 从1→2（单卡batch_size）
    gradient_accumulation_steps=8,      # 从4→8 → 总batch=2×8×4=64
    fp16=False,                        # 禁用fp16
    bf16=True,                         # 优先使用bf16（Ampere架构优化）
    gradient_checkpointing=True,        # 显存优化关键！
    ddp_find_unused_parameters=False,   # 多GPU必需参数

    # ===== 优化器配置 =====
    learning_rate=2e-5,                # 从5e-6→2e-5（适合1B模型）
    adam_beta1=0.9,
    adam_beta2=0.95,                   # 从0.99→0.95（更稳定）
    weight_decay=0.1,
    max_grad_norm=1.0,                 # 从0.1→1.0（放松梯度裁剪）

    # ===== 训练控制 =====
    num_train_epochs=3,                # 从1→3（更多迭代）
    warmup_ratio=0.05,                 # 从0.1→5%（更快进入训练）
    lr_scheduler_type='cosine',
    logging_steps=10,                  # 从1→10（减少日志频率）

    # ===== 生成配置 =====
    use_vllm=True,                     # 启用vLLM加速生成
    vllm_gpu_memory_utilization=0.85,  # 提升显存利用率
    vllm_tensor_parallel_size=4,       # 使用4卡并行生成
    num_generations=32,                # 从16→32（更准的策略评估）

    # ===== 其他配置 =====
    output_dir=output_dir,
    run_name=run_name,
    report_to="wandb",
    save_steps=500,                    # 从100→500（减少保存频率）
    log_on_each_node=False
)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained( model_path)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(output_dir)
