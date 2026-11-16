import time
import bert_score
import torch
import numpy as np
import evaluate
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    AutoModel
)
from transformers import AutoModel
from datasets import load_from_disk
from torch.utils.data import DataLoader
import wandb

# 初始化Accelerator
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=1,  # 测试不需要梯度累积
)

# # WandB仅在主进程初始化
if accelerator.is_main_process:
    wandb.login(key="1b8adc705fb9b3e125c05f15107ad7c22c830811")
    wandb.init(project="DeepSeek-R1-Distill-Llama-8B-Qlora-1")

# 步骤1：加载模型和测试数据
final_model_path = "/mnt/project/deepseek-r1-distill-llama-8b-qlora/model/DeepSeek-R1-Distill-Llama-8B"

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 分布式加载模型
with accelerator.main_process_first():
    tokenizer = AutoTokenizer.from_pretrained(final_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        final_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.process_index},
        trust_remote_code=True
    )
model.eval()

# 步骤2：加载并处理测试数据
test_dataset = load_from_disk(
    "/mnt/project/deepseek-r1-distill-llama-8b-qlora/data/test_dataset"
)
# 测试prompt模板
test_prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Please answer the following medical question.

### Question:
{}

### Response:
<think>
"""

# 数据预处理函数
def prepare_test_example(example):
    full_prompt = test_prompt_template.format(example["Question"])
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024  # 保留生成空间
    )
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "reference": example["Response"]
    }
class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # 提取reference并保留原始文本
        references = [feature.pop("reference") for feature in features]
        # 处理模型输入字段
        batch = super().__call__(features, return_tensors=return_tensors)
        # 将reference以列表形式添加回batch
        batch["references"] = references
        return batch

# 分布式数据处理
with accelerator.main_process_first():
    processed_dataset = test_dataset.map(
        prepare_test_example,
        remove_columns=["Question", "Complex_CoT", "Response"]
    )

# 创建分布式DataLoader
test_dataloader = DataLoader(
    processed_dataset,
    batch_size=32,
    collate_fn=CustomDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=2048,
    ),
    shuffle=False,
    num_workers=0
)
model, test_dataloader = accelerator.prepare(model, test_dataloader)

# 生成参数配置
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "repetition_penalty": 1.1,
    "use_cache": True,  # 确保启用KV缓存
    "num_beams": 1,  # 保持为1以最大化速度
    "output_scores": False  # 关闭分数计算
}

# 步骤3：分布式生成预测
predictions = []
references = []
start_time = time.time()

for batch in tqdm(test_dataloader, desc="Generating", disable=not accelerator.is_local_main_process):
    with torch.inference_mode():
        # 将数据移到对应设备
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device)
        }
        
        # 生成回答
        generated = model.module.generate(
            **inputs,
            **generation_config
        )
        
        # 解码预测
        outputs = generated[:, inputs["input_ids"].shape[1]:]
        batch_predictions = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        # 处理预测结果
        processed_predictions = []
        for pred in batch_predictions:
            try:
                _, answer_part = pred.split("</think>", 1)
                processed_predictions.append(answer_part.strip())
            except ValueError:
                processed_predictions.append(pred.strip())
        
        # 收集结果
        predictions.extend(processed_predictions)
        # 获取该batch对应的references
        batch_references = batch["references"]
        # 收集结果时使用正确的字段
        references.extend(batch_references)

# 步骤4：收集所有结果到主进程
all_predictions = accelerator.gather_for_metrics(predictions)
all_references = accelerator.gather_for_metrics(references)

# 步骤5：主进程计算指标
if accelerator.is_main_process:
    # 初始化评估指标
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    # 计算指标
    rouge_results = rouge.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=True
    )
    # 手动添加模型路径到 model2layers 映射
    bert_score.utils.model2layers["/mnt/project/bert-base-chinese"] = 12  # 替换为实际层数

    # 调用 bertscore 并指定模型和分词器
    bertscore_results = bertscore.compute(
        predictions=all_predictions,
        references=all_references,
        lang="zh",
        model_type="/mnt/project/bert-base-chinese",
        batch_size=16,
        rescale_with_baseline=False
)
    
    # 计算耗时
    total_time = time.time() - start_time
    samples_per_sec = len(all_predictions) / total_time

    # 记录到WandB
    wandb.log({
        "basemodel_test/rouge1": rouge_results["rouge1"],
        "basemodel_test/rouge2": rouge_results["rouge2"],
        "basemodel_test/rougeL": rouge_results["rougeL"],
        "basemodel_test/bert_p": np.mean(bertscore_results["precision"]),
        "basemodel_test/bert_r": np.mean(bertscore_results["recall"]),
        "basemodel_test/bert_f1": np.mean(bertscore_results["f1"]),
        "basemodel_test/samples_per_sec": samples_per_sec,
        "basemodel_test/total_samples": len(all_predictions),
        "basemodel_test/total_time": total_time
    })
    
    # 打印结果
    print(f"\nDistributed Evaluation Results (4x3090):")
    print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
    print(f"BERTScore F1: {np.mean(bertscore_results['f1']):.4f}")
    print(f"Throughput: {samples_per_sec:.2f} samples/sec")
    
    wandb.finish()

# 同步所有进程
accelerator.wait_for_everyone()