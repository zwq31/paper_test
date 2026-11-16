

# # =================================================加入自评置信度推理========================================================
# import os
# import json
# import torch
# import logging
# import random
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch.nn.functional as F

# # ----------------------------
# # 日志配置
# # ----------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='generate_train.log',
#     filemode='a',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# # ----------------------------
# # 配置路径和参数
# # ----------------------------
# MODEL_BASE_PATH = "/mnt/project/Qwen3-8B-news/models/Qwen3-8B-cpt-sft"
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen3-8B-news/sft/code/data/train/train.json"

# MAX_LENGTH = 4096
# MAX_NEW_TOKENS = 2048

# # ----------------------------
# # 拒绝阈值
# # ----------------------------
# PPL_THRESHOLD = 1.1022038499105375
# MIN_TOKEN_PROB_THRESHOLD = 0.3661533556634178
# SELF_CONF_THRESHOLD = 0.44508323935997046  # 自报置信度阈值

# # ----------------------------
# # 拒答模板
# # ----------------------------
# REJECTION_RESPONSES = [
#     "抱歉，我无法回答这个问题。"
# ]

# # ----------------------------
# # 构建知识自信度 prompt
# # ----------------------------
# def build_confidence_messages(question):
#     messages = [
#         {
#             "role": "system",
#             "content": "你是一个新闻知识自省助手，你的任务是根据你对问题所掌握的事实知识，判断自己能否准确回答问题，并输出置信度（0.00-1.00）"
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"请仅输出你对下面问题的知识自信度（0.00~1.00），表示你对能否准确回答该问题的自信程度\n\n"
#                 f"问题：{question}\n\n"
#                 "严格按照下列格式要求输出：\n"
#                 "置信度: [0-1之间的数字]\n"
#                 "要求：置信度能准确反映你对问题的了解程度，1表示非常了解且非常有把握回答正确。0表示没有把握回答正确，如果你不确定，输出一个较低的置信度\n"
#                 "不要回答问题，只输出置信度。不能省略置信度标识"
#             )
#         }
#     ]
#     return messages

# # ----------------------------
# # 构建回答 prompt
# # ----------------------------
# def build_answer_messages(question):
#     messages = [
#         {
#             "role": "system",
#             "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
#         },
#         {
#             "role": "user",
#             "content": (
#                 f"请回答下面的问题：\n\n"
#                 f"问题：{question}"
#             )
#         }
#     ]
#     return messages

# # ----------------------------
# # 解析模型自报置信度
# # ----------------------------
# def parse_self_confidence(model_answer):
#     try:
#         lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
#         self_conf = None
        
#         # 优先查找含 '置信度:' 的行
#         for line in lines:
#             if "置信度" in line:
#                 val_str = line.split(':')[-1].strip()
#                 try:
#                     self_conf = float(val_str)
#                 except ValueError:
#                     logger.warning(f"置信度值无法转为浮点数: '{val_str}'")
#                     self_conf = None
#                 return self_conf
        
#         # 如果没有找到含 '置信度' 的行，尝试最后一行直接转换为数字
#         if lines:
#             try:
#                 self_conf = float(lines[-1])
#             except ValueError:
#                 logger.warning(f"最后一行无法转为浮点数: '{lines[-1]}'")
#                 self_conf = None
        
#         return self_conf
#     except Exception as e:
#         logger.warning(f"解析自报置信度失败: {e}")
#         return None


# # ----------------------------
# # 计算 ppl / min_token_prob
# # ----------------------------
# def compute_ppl_minprob(generated_tokens, generate_logits):
#     try:
#         if generate_logits.dim() == 3:
#             generate_logits = generate_logits.squeeze(1)
#         min_len = min(generate_logits.size(0), len(generated_tokens))
#         if min_len == 0:
#             return float('inf'), 0.0
#         generate_logits = generate_logits[:min_len]
#         generated_tokens = generated_tokens[:min_len]

#         token_probs = []
#         for i, token_id in enumerate(generated_tokens):
#             try:
#                 logits_float = generate_logits[i].float()
#                 token_prob = F.softmax(logits_float, dim=-1)[token_id].item()
#                 token_probs.append(token_prob)
#             except IndexError:
#                 continue

#         min_token_prob = min(token_probs) if token_probs else 0.0

#         if token_probs:
#             safe_probs = np.clip(token_probs, 1e-15, 1.0)
#             nll = -np.mean(np.log(safe_probs))
#             ppl = np.exp(nll)
#             if np.isnan(ppl) or np.isinf(ppl):
#                 ppl = float('inf')
#         else:
#             ppl = float('inf')

#         return ppl, min_token_prob
#     except Exception as e:
#         logger.error(f"PPL/min_token_prob计算失败: {e}")
#         return float('inf'), 0.0

# # ----------------------------
# # 加载模型
# # ----------------------------
# def load_model(model_path):
#     logger.info("加载模型和tokenizer...")
    
#     # 配置4bit量化
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4"
#     )
    
#     # 加载tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         padding_side="left"
#     )
    
#     # 设置pad_token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # 加载模型
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         quantization_config=quantization_config,
#         device_map="auto",
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16
#     )
    
#     model.eval()
#     logger.info("模型加载完成")
#     return model, tokenizer

# # ----------------------------
# # 生成置信度
# # ----------------------------
# def generate_confidence(model, tokenizer, question):
#     # 构建置信度生成消息列表
#     messages = build_confidence_messages(question)
    
#     # 使用apply_chat_template构建文本
#     prompt_text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False
#     )
    
#     # 编码输入
#     inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     prompt_len = inputs['input_ids'].size(1)
    
#     # 生成置信度
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=10,  # 置信度输出很短
#             return_dict_in_generate=True,
#             output_scores=False,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     # 解码生成结果
#     generated_tokens = outputs.sequences[0, prompt_len:]
#     confidence_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
#     # 解析置信度
#     self_conf = parse_self_confidence(confidence_output)
    
#     return self_conf, confidence_output

# # ----------------------------
# # 生成答案
# # ----------------------------
# def generate_answer(model, tokenizer, question):
#     # 构建回答消息列表
#     messages = build_answer_messages(question)
    
#     # 使用apply_chat_template构建文本
#     prompt_text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False
#     )
    
#     # 编码输入
#     inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     prompt_len = inputs['input_ids'].size(1)
    
#     # 生成回答
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             return_dict_in_generate=True,
#             output_scores=True,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     # 解码生成答案
#     generated_tokens = outputs.sequences[0, prompt_len:]
#     raw_model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
#     # 计算ppl和min_token_prob
#     ppl, min_token_prob = float('inf'), 0.0
#     if outputs.scores:
#         generate_logits = torch.stack(outputs.scores, dim=0)
#         ppl, min_token_prob = compute_ppl_minprob(generated_tokens, generate_logits)
    
#     return raw_model_output, ppl, min_token_prob

# # ----------------------------
# # 主函数
# # ----------------------------
# def main():
#     model, tokenizer = load_model(MODEL_BASE_PATH)
    
#     with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
#         input_data = json.load(f)
    
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
#     with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
#         f_out.write("[\n")
#         for idx, item in enumerate(tqdm(input_data)):
#             question = item.get("question", "").strip()
#             if not question:
#                 continue
            
#             try:
#                 # 第一步：生成置信度
#                 self_conf, confidence_output = generate_confidence(model, tokenizer, question)
                
#                 # 第二步：生成答案
#                 raw_model_output, ppl, min_token_prob = generate_answer(model, tokenizer, question)
                
#                 # 决策是否拒答
#                 should_answer = (
#                     ppl <= PPL_THRESHOLD and
#                     min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
#                     (self_conf is not None and self_conf >= SELF_CONF_THRESHOLD)
#                 )
#                 final_answer = raw_model_output if should_answer else random.choice(REJECTION_RESPONSES)
                
#                 output_item = {
#                     "source_id": item.get("source_id", idx),
#                     "question": question,
#                     "standard_answer": item.get("answer", ""),
#                     "model_answer": raw_model_output,  # 注意：这里保存的是原始答案输出
#                     "refusal_decision": {
#                         "ppl": round(ppl, 4) if np.isfinite(ppl) else None,
#                         "min_token_prob": round(min_token_prob, 4),
#                         "self_reported_confidence": round(self_conf, 4) if isinstance(self_conf, (int, float)) else None,
#                         "confidence_output": confidence_output,  # 保存置信度生成的原始输出
#                         "should_answer": bool(should_answer),
#                         "final_model_answer": final_answer
#                     }
#                 }
                
#                 if idx > 0:
#                     f_out.write(",\n")
#                 f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
#                 f_out.flush()
                
#             except Exception as e:
#                 logger.error(f"第{idx}条数据处理失败: {e}")
#                 continue
#         f_out.write("\n]")
    
#     logger.info(f"推理完成，结果保存至 {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import torch
import logging
from typing import Dict, Any

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='inference.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 禁用 transformers 警告
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# 确保 Unsloth 是第一个导入（如果仍需其推理优化）
try:
    import unsloth
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning("Unsloth not available, falling back to standard transformers.")

if not UNSLOTH_AVAILABLE:
    from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置路径
MODEL_PATH = "/mnt/project/Qwen3-8B-news/models/Qwen3-8B-cpt-sft"  # 合并后的模型路径
INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train_data.json"
OUTPUT_FILE = "/mnt/project/Qwen3-8B-news/sft/code/data/train/train.jsonl"  # 每行一个 JSON 对象

MAX_LENGTH = 2048
MAX_NEW_TOKENS = 512

def load_model_and_tokenizer():
    logger.info(f"Loading model from {MODEL_PATH}...")
    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,  # 推理建议不用 4bit（除非显存不足）
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        FastLanguageModel.for_inference(model)  # 启用推理优化
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        model.eval()
    return model, tokenizer

def format_prompt(question: str, tokenizer) -> str:
    """根据 Qwen 聊天模板构造 prompt"""
    messages = [
        {"role": "user", "content": f"回答以下问题：{question}"}
    ]
    # 使用 tokenizer 的 chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    return prompt

def inference_single_sample(model, tokenizer, question: str) -> str:
    prompt = format_prompt(question,tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH - MAX_NEW_TOKENS).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer()
    logger.info("Model loaded successfully.")

    # 加载输入数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples for inference.")

    # 逐样本推理并保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for idx, item in enumerate(data):
            if "question" not in item:
                logger.warning(f"Skipping sample {idx}: missing 'question'")
                continue

            question = item["question"]
            try:
                answer = inference_single_sample(model, tokenizer, question)
                result = {
                    "id": idx,
                    "question": question,
                    "predicted_answer": answer,
                    "ground_truth": item.get("answer", "")
                }
                # 立即写入一行（JSONL 格式）
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()  # 确保写入磁盘
                logger.info(f"Processed sample {idx}: {question[:50]}... -> {answer[:50]}...")
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                result = {
                    "id": idx,
                    "question": question,
                    "predicted_answer": "",
                    "error": str(e)
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

    logger.info(f"Inference completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()