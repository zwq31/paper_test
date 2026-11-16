# import os
# import json
# import torch
# import logging
# import random
# import numpy as np
# from tqdm import tqdm
# from unsloth import FastLanguageModel
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
# MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/final_model"
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/reject/sft/test_data.json"

# MAX_LENGTH = 4096
# MAX_NEW_TOKENS = 2048

# # 拒绝阈值
# PPL_THRESHOLD = 1.1071248144943788
# MIN_TOKEN_PROB_THRESHOLD = 0.337702810936907

# # 拒答模板
# REJECTION_RESPONSES = [
#     "抱歉，我无法回答这个问题。"
# ]

# # ----------------------------
# # Chat 风格 prompt 构建
# # ----------------------------
# def build_chat_prompt(question):
#     """构建 system + user prompt"""
#     messages = [
#         {
#             "role": "system",
#             "content": """你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"""
#         },
#         {"role": "user", "content": question}
#     ]
#     return messages


# # ----------------------------
# # 计算 ppl、min_token_prob 
# # ----------------------------
# def compute_ppl_minprob_selfcertainty(generated_tokens, generate_logits):
#     try:
#         if generate_logits.dim() == 3:
#             generate_logits = generate_logits.squeeze(1)
#         min_len = min(generate_logits.size(0), len(generated_tokens))
#         if min_len == 0:
#             return float('inf'), 0.0, 0.0

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
#         logger.error(f"综合指标计算失败: {e}")
#         return float('inf'), 0.0, 0.0

# # ----------------------------
# # 加载模型
# # ----------------------------
# def load_model(model_path):
#     logger.info("加载 Unsloth 模型和 tokenizer...")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=model_path,
#         max_seq_length=MAX_LENGTH,
#         dtype=torch.bfloat16,
#         load_in_4bit=True
#     )
#     model.eval()
#     return model, tokenizer

# # ----------------------------
# # 生成 chat 风格回答
# # ----------------------------
# def generate_chat_answer(model, tokenizer, question):
#     messages = build_chat_prompt(question)
#     prompt_text = ""
#     for msg in messages:
#         if msg['role'] == 'system':
#             prompt_text += f"<|im_start|>system\n{msg['content']}\n<|im_end|>\n"
#         elif msg['role'] == 'user':
#             prompt_text += f"<|im_start|>user\n{msg['content']}\n<|im_end|>\n"
    
#     prompt_text += "<|im_start|>assistant\n"

#     inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     prompt_ids = inputs['input_ids']
#     prompt_len = prompt_ids.size(1)

#     # 生成
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             return_dict_in_generate=True,
#             output_scores=True,
#             do_sample=False,
#             temperature=0.7,
#             top_p=0.9,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )

#     # 解码
#     generated_tokens = outputs.sequences[0, prompt_len:]
#     model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

#     # ppl/min_token_prob
#     ppl, min_token_prob = float('inf'), 0.0
#     if outputs.scores:
#         generate_logits = torch.stack(outputs.scores, dim=0)
#         ppl, min_token_prob = compute_ppl_minprob_selfcertainty(generated_tokens, generate_logits)

#     # 决定是否拒答
#     should_answer = (
#         ppl <= PPL_THRESHOLD and
#         min_token_prob >= MIN_TOKEN_PROB_THRESHOLD
#     )
#     final_answer = model_answer if should_answer else random.choice(REJECTION_RESPONSES)

#     return {
#         "original_model_answer": model_answer,
#         "final_answer": final_answer,
#         "should_answer": should_answer,
#         "ppl": ppl,
#         "min_token_prob": min_token_prob,
#     }

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
#                 result = generate_chat_answer(model, tokenizer, question)
#                 output_item = {
#                     "source_id": item.get("source_id", idx),
#                     "question": question,
#                     "standard_answer": item.get("answer", ""),
#                     "model_answer": result["original_model_answer"],
#                     "refusal_decision": {
#                         "ppl": round(result["ppl"], 4),
#                         "min_token_prob": round(result["min_token_prob"], 4),
#                         "should_answer": bool(result["should_answer"]),
#                         "final_model_answer": result["final_answer"]
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


# =================================================加入自评置信度推理========================================================
import os
import json
import torch
import logging
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_train.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ----------------------------
# 配置路径和参数
# ----------------------------
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected/unmentioned_qa/unmentioned_qa.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/unmention/unmention.json"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# 拒绝阈值
# ----------------------------
PPL_THRESHOLD = 1.1022038499105375
MIN_TOKEN_PROB_THRESHOLD = 0.3661533556634178
SELF_CONF_THRESHOLD = 0.44508323935997046  # 自报置信度阈值

# ----------------------------
# 拒答模板
# ----------------------------
REJECTION_RESPONSES = [
    "抱歉，我无法回答这个问题。"
]

# ----------------------------
# 构建知识自信度 prompt
# ----------------------------
def build_confidence_messages(question):
    messages = [
        {
            "role": "system",
            "content": "你是一个新闻知识自省助手，你的任务是根据你对问题所掌握的事实知识，判断自己能否准确回答问题，并输出置信度（0.00-1.00）"
        },
        {
            "role": "user",
            "content": (
                f"请仅输出你对下面问题的知识自信度（0.00~1.00），表示你对能否准确回答该问题的自信程度\n\n"
                f"问题：{question}\n\n"
                "严格按照下列格式要求输出：\n"
                "置信度: [0-1之间的数字]\n"
                "要求：置信度能准确反映你对问题的了解程度，1表示非常了解且非常有把握回答正确。0表示没有把握回答正确，如果你不确定，输出一个较低的置信度\n"
                "不要回答问题，只输出置信度。不能省略置信度标识"
            )
        }
    ]
    return messages

# ----------------------------
# 构建回答 prompt
# ----------------------------
def build_answer_messages(question):
    messages = [
        {
            "role": "system",
            "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
        },
        {
            "role": "user",
            "content": (
                f"请回答下面的问题：\n\n"
                f"问题：{question}"
            )
        }
    ]
    return messages

# ----------------------------
# 解析模型自报置信度
# ----------------------------
def parse_self_confidence(model_answer):
    try:
        lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
        self_conf = None
        
        # 优先查找含 '置信度:' 的行
        for line in lines:
            if "置信度" in line:
                val_str = line.split(':')[-1].strip()
                try:
                    self_conf = float(val_str)
                except ValueError:
                    logger.warning(f"置信度值无法转为浮点数: '{val_str}'")
                    self_conf = None
                return self_conf
        
        # 如果没有找到含 '置信度' 的行，尝试最后一行直接转换为数字
        if lines:
            try:
                self_conf = float(lines[-1])
            except ValueError:
                logger.warning(f"最后一行无法转为浮点数: '{lines[-1]}'")
                self_conf = None
        
        return self_conf
    except Exception as e:
        logger.warning(f"解析自报置信度失败: {e}")
        return None


# ----------------------------
# 计算 ppl / min_token_prob
# ----------------------------
def compute_ppl_minprob(generated_tokens, generate_logits):
    try:
        if generate_logits.dim() == 3:
            generate_logits = generate_logits.squeeze(1)
        min_len = min(generate_logits.size(0), len(generated_tokens))
        if min_len == 0:
            return float('inf'), 0.0
        generate_logits = generate_logits[:min_len]
        generated_tokens = generated_tokens[:min_len]

        token_probs = []
        for i, token_id in enumerate(generated_tokens):
            try:
                logits_float = generate_logits[i].float()
                token_prob = F.softmax(logits_float, dim=-1)[token_id].item()
                token_probs.append(token_prob)
            except IndexError:
                continue

        min_token_prob = min(token_probs) if token_probs else 0.0

        if token_probs:
            safe_probs = np.clip(token_probs, 1e-15, 1.0)
            nll = -np.mean(np.log(safe_probs))
            ppl = np.exp(nll)
            if np.isnan(ppl) or np.isinf(ppl):
                ppl = float('inf')
        else:
            ppl = float('inf')

        return ppl, min_token_prob
    except Exception as e:
        logger.error(f"PPL/min_token_prob计算失败: {e}")
        return float('inf'), 0.0

# ----------------------------
# 加载模型
# ----------------------------
def load_model(model_path):
    logger.info("加载模型和tokenizer...")
    
    # 配置4bit量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    model.eval()
    logger.info("模型加载完成")
    return model, tokenizer

# ----------------------------
# 生成置信度
# ----------------------------
def generate_confidence(model, tokenizer, question):
    # 构建置信度生成消息列表
    messages = build_confidence_messages(question)
    
    # 使用apply_chat_template构建文本
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
    # 生成置信度
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # 置信度输出很短
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成结果
    generated_tokens = outputs.sequences[0, prompt_len:]
    confidence_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 解析置信度
    self_conf = parse_self_confidence(confidence_output)
    
    return self_conf, confidence_output

# ----------------------------
# 生成答案
# ----------------------------
def generate_answer(model, tokenizer, question):
    # 构建回答消息列表
    messages = build_answer_messages(question)
    
    # 使用apply_chat_template构建文本
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成答案
    generated_tokens = outputs.sequences[0, prompt_len:]
    raw_model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 计算ppl和min_token_prob
    ppl, min_token_prob = float('inf'), 0.0
    if outputs.scores:
        generate_logits = torch.stack(outputs.scores, dim=0)
        ppl, min_token_prob = compute_ppl_minprob(generated_tokens, generate_logits)
    
    return raw_model_output, ppl, min_token_prob

# ----------------------------
# 主函数
# ----------------------------
def main():
    model, tokenizer = load_model(MODEL_BASE_PATH)
    
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        f_out.write("[\n")
        for idx, item in enumerate(tqdm(input_data)):
            question = item.get("question", "").strip()
            if not question:
                continue
            
            try:
                # 第一步：生成置信度
                self_conf, confidence_output = generate_confidence(model, tokenizer, question)
                
                # 第二步：生成答案
                raw_model_output, ppl, min_token_prob = generate_answer(model, tokenizer, question)
                
                # 决策是否拒答
                should_answer = (
                    ppl <= PPL_THRESHOLD and
                    min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
                    (self_conf is not None and self_conf >= SELF_CONF_THRESHOLD)
                )
                final_answer = raw_model_output if should_answer else random.choice(REJECTION_RESPONSES)
                
                output_item = {
                    "source_id": item.get("source_id", idx),
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "model_answer": raw_model_output,  # 注意：这里保存的是原始答案输出
                    "raw_model_output": raw_model_output,
                    "refusal_decision": {
                        "ppl": round(ppl, 4) if np.isfinite(ppl) else None,
                        "min_token_prob": round(min_token_prob, 4),
                        "self_reported_confidence": round(self_conf, 4) if isinstance(self_conf, (int, float)) else None,
                        "confidence_output": confidence_output,  # 保存置信度生成的原始输出
                        "should_answer": bool(should_answer),
                        "final_model_answer": final_answer
                    }
                }
                
                if idx > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
                f_out.flush()
                
            except Exception as e:
                logger.error(f"第{idx}条数据处理失败: {e}")
                continue
        f_out.write("\n]")
    
    logger.info(f"推理完成，结果保存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()





# # ===================================================拒答头sft推理===============================================================
# import os
# import json
# import torch
# import logging
# import random
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer
# from unsloth import FastLanguageModel
# import torch.nn.functional as F
# import torch.nn as nn

# # ---------------- 日志配置 ----------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='enhance_existing_data.log',
#     filemode='a',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# # ---------------- 配置参数 ----------------
# MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
# REFUSAL_HEAD_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/outputs_5epoch/outputs_refusal_only/final-refusal-head/refusal_head.pt"
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/test_data_refusal_only.json"

# MAX_LENGTH = 4096
# MAX_NEW_TOKENS = 2048

# # ---------------- 拒答阈值 ----------------
# PPL_THRESHOLD = 2.2
# MIN_TOKEN_PROB_THRESHOLD = 0.0969
# TEMPERATURE = 1.5  # 温度校准

# # ---------------- DualLoss 推理模型 ----------------
# class DualLossModel(nn.Module):
#     """Base模型 + 拒答头"""
#     def __init__(self, base_model, hidden_size):
#         super().__init__()
#         self.base = base_model
#         self.refusal_head = nn.Linear(hidden_size, 1)

#     def forward(self, input_ids, attention_mask=None):
#         # 直接调用 self.base 而不是 self.base.model
#         outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
#         hidden_states = outputs.hidden_states[-1]   # [B, seq_len, hidden_size]
#         cls_token = hidden_states[:, -1, :]         # 最后token
#         refusal_logits = self.refusal_head(cls_token).squeeze(-1)
#         return refusal_logits

# # ---------------- EnhancedRefusalGenerator ----------------
# class EnhancedRefusalGenerator:
#     """增强型拒答生成器：PPL + min_token_prob + 拒答头概率"""
#     def __init__(self, model, tokenizer,
#                  ppl_threshold=PPL_THRESHOLD,
#                  min_prob_threshold=MIN_TOKEN_PROB_THRESHOLD,
#                  temperature=TEMPERATURE):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.ppl_threshold = ppl_threshold
#         self.min_prob_threshold = min_prob_threshold
#         self.temperature = temperature

#         self.rejection_responses = [
#             "很抱歉，关于这个问题我目前无法提供确切信息。",
#             "我不确定这个问题的答案。",
#             "这个问题超出了我的知识范围。",
#             "我没有足够的信息来回答这个问题。"
#         ]

#     # --------- 计算各指标 ---------
#     def compute_metrics(self, generated_tokens, generate_logits):
#         if generate_logits.dim() == 3:
#             generate_logits = generate_logits.squeeze(1)
#         min_len = min(generate_logits.size(0), len(generated_tokens))
#         if min_len == 0:
#             return float('inf'), 0.0, 0.0, 0.0, 0.0
#         generate_logits = generate_logits[:min_len]
#         generated_tokens = generated_tokens[:min_len]

#         token_probs, topk_deltas = [], []
#         for i, token_id in enumerate(generated_tokens):
#             logits_float = generate_logits[i].float()
#             logits_scaled = logits_float / self.temperature
#             probs_scaled = F.softmax(logits_scaled, dim=-1)
#             token_probs.append(probs_scaled[token_id].item())
#             top2_probs, _ = torch.topk(probs_scaled, k=2)
#             topk_deltas.append((top2_probs[0] - top2_probs[1]).item())

#         min_token_prob = min(token_probs) if token_probs else 0.0


#         safe_probs = np.clip(token_probs, 1e-15, 1.0)
#         nll = -np.mean(np.log(safe_probs)) if len(safe_probs) > 0 else float('inf')
#         ppl = np.exp(nll) if np.isfinite(nll) else float('inf')

#         return ppl, min_token_prob

#     # --------- Prompt拼接 ---------
#     def format_prompt(self, question):
#         return [{"role": "user", "content": f"回答以下问题：{question}"}]

#     # --------- 提取模型回答 ---------
#     def extract_assistant_response(self, full_text, prompt_text):
#         if "<|im_start|>assistant" in full_text:
#             start_idx = full_text.find("<|im_start|>assistant")
#             assistant_part = full_text[start_idx:]
#             end_idx = assistant_part.find("<|im_end|>")
#             if end_idx != -1:
#                 response = assistant_part[len("<|im_start|>assistant"):end_idx].strip()
#                 return response
#         if prompt_text in full_text:
#             response = full_text.replace(prompt_text, "").strip()
#             response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
#             return response
#         return full_text.strip()

#     # --------- 生成答案 + 拒答概率 ---------
#     def generate(self, question):
#         messages = self.format_prompt(question)
#         prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#         inputs = {k: v.to(self.model.base.device) for k, v in inputs.items()}
#         prompt_len = inputs['input_ids'].size(1)

#         # --------- 生成文本 ---------
#         with torch.no_grad():
#             outputs = self.model.base.generate(
#                 **inputs,
#                 max_new_tokens=MAX_NEW_TOKENS,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 do_sample=False,
#                 temperature=0.7,
#                 top_p=0.9,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )

#         generated_tokens = outputs.sequences[0, prompt_len:]
#         full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
#         model_answer = self.extract_assistant_response(full_text, prompt_text)

#         # --------- 计算指标 ---------
#         ppl, min_token_prob = 0,0
#         if outputs.scores:
#             try:
#                 generate_logits = torch.stack(outputs.scores, dim=0)
#                 ppl, min_token_prob= self.compute_metrics(
#                     generated_tokens, generate_logits
#                 )
#             except Exception as e:
#                 logger.warning(f"计算指标失败: {e}")

#         # --------- 拒答头概率 ---------
#         refusal_logits = self.model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
#         refusal_prob = torch.sigmoid(refusal_logits).item()

#         # --------- 决策逻辑 ---------
#         should_answer = (
#             ppl <= self.ppl_threshold and
#             min_token_prob >= self.min_prob_threshold and
#             refusal_prob < 0.5   # 拒答概率阈值，可调整
#         )

#         final_answer = model_answer
#         refusal_applied = False
#         if not should_answer:
#             final_answer = random.choice(self.rejection_responses)
#             refusal_applied = True

#         return {
#             "ppl": ppl,
#             "min_token_prob": min_token_prob,
#             "refusal_prob": refusal_prob,
#             "original_model_answer": model_answer,
#             "final_answer": final_answer,
#             "should_answer": should_answer,
#             "refusal_applied": refusal_applied
#         }

# # ---------------- 模型加载 ----------------
# def load_model(model_base_path, refusal_head_path):
#     logger.info("加载模型...")
#     base_model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=model_base_path,
#         max_seq_length=MAX_LENGTH,
#         dtype=torch.bfloat16,
#         load_in_4bit=True,
#     )

#     hidden_size = base_model.config.hidden_size
#     model = DualLossModel(base_model, hidden_size)

#     # 加载 refusal_head 权重
#     refusal_state_dict = torch.load(refusal_head_path, map_location="cpu")
#     model.refusal_head.load_state_dict(refusal_state_dict)

#     # 移动 refusal_head 到 base_model 的设备并转为 bfloat16
#     device = model.base.device
#     model.refusal_head.to(device=device, dtype=torch.bfloat16)

#     model.eval()
#     if hasattr(model.base, "gradient_checkpointing_disable"):
#         model.base.gradient_checkpointing_disable()  

#     return model, tokenizer

# # ---------------- 主函数 ----------------
# def main():
#     model, tokenizer = load_model(MODEL_BASE_PATH, REFUSAL_HEAD_PATH)
#     generator = EnhancedRefusalGenerator(model, tokenizer)

#     with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
#         input_data = json.load(f)

#     with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
#         f_out.write("[\n")
#         for idx, item in enumerate(tqdm(input_data)):
#             question = item.get("question", "")
#             if not question:
#                 continue
#             result = generator.generate(question)
#             output_item = {
#                 "source_id": item.get("source_id", idx),
#                 "question": question,
#                 "standard_answer": item.get("answer", ""),
#                 "model_answer": result["original_model_answer"],
#                 "refusal_decision": {
#                     "ppl": round(result["ppl"], 4),
#                     "min_token_prob": round(result["min_token_prob"], 4),
#                     "refusal_prob": round(result["refusal_prob"], 4),
#                     "should_answer": bool(result["should_answer"]),
#                     "final_model_answer": result["final_answer"]
#                 }
#             }
#             if idx > 0:
#                 f_out.write(",\n")
#             f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
#         f_out.write("\n]")

# if __name__ == "__main__":
#     main()


# ==========================================================chat模板=========================================================
# import os
# import json
# import torch
# import logging
# import random
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import torch.nn.functional as F

# # 日志配置
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='inference_qwen_chat_with_metrics.log',
#     filemode='a',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# # ==============================
# # 配置路径
# # ==============================
# MODEL_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/base_model_dpo_train_2lei_model/final_model"
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/output/base_model_dpo_train_2lei_model/test_output.json"

# MAX_NEW_TOKENS = 1024
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ==============================
# # 拒绝阈值（请根据你的训练结果调整）
# # ==============================
# PPL_THRESHOLD = 2.2
# MIN_TOKEN_PROB_THRESHOLD = 0.0969
# SELF_CERTAINTY_THRESHOLD = 17.90

# # ==============================
# # System Prompt（必须与训练一致！）
# # ==============================
# SYSTEM_PROMPT = """请你按照以下格式回答问题：
# 1. 先给出直接答案
# 2. 然后进行反思解释
# 3. 最后给出置信度评分

# 请严格按照以下格式输出：
# 答案: [你的答案]
# 反思: [你的反思过程]
# 置信度: [0-1之间的数值]"""

# # 拒绝回答模板（仅用于最终输出）
# REJECTION_RESPONSES = [
#     "很抱歉，关于这个问题我目前无法提供确切信息。",
#     "我不确定这个问题的答案。",
#     "这个问题超出了我的知识范围。",
#     "我没有足够的信息来回答这个问题。"
# ]


# def extract_from_output(text):
#     """从模型输出中提取答案、反思、置信度"""
#     answer = ""
#     reflection = ""
#     confidence = 0.0

#     text = text.replace("：", ":")

#     if "答案:" in text:
#         start = text.find("答案:") + len("答案:")
#         end = len(text)
#         for kw in ["反思:", "置信度:"]:
#             pos = text.find(kw, start)
#             if pos != -1:
#                 end = min(end, pos)
#         answer = text[start:end].strip()

#     if "反思:" in text:
#         start = text.find("反思:") + len("反思:")
#         end = text.find("置信度:", start)
#         if end == -1:
#             end = len(text)
#         reflection = text[start:end].strip()

#     if "置信度:" in text:
#         start = text.find("置信度:") + len("置信度:")
#         rest = text[start:].strip()
#         import re
#         match = re.search(r'([0-9]*\.?[0-9]+)', rest)
#         if match:
#             try:
#                 confidence = float(match.group(1))
#             except (ValueError, TypeError):
#                 confidence = 0.0

#     if not answer and not reflection:
#         answer = text.strip()

#     return answer, reflection, confidence


# def compute_self_certainty(logits):
#     """计算自我确定性"""
#     try:
#         if logits.dim() == 3:
#             logits = logits.squeeze(1)
#         logits = logits.float()
#         logsumexp = torch.logsumexp(logits, dim=-1)
#         mean_logits = logits.mean(dim=-1)
#         self_certainty = logsumexp - mean_logits
#         if torch.isnan(self_certainty).any() or torch.isinf(self_certainty).any():
#             probs = F.softmax(logits, dim=-1)
#             entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
#             self_certainty = -entropy
#         result = self_certainty.mean().item()
#         if np.isinf(result) or np.isnan(result):
#             return 0.0
#         return result
#     except Exception as e:
#         logger.warning(f"自我确定性计算失败: {e}")
#         return 0.0


# def compute_ppl_minprob_selfcertainty(generated_tokens, generate_logits):
#     """计算ppl、min_token_prob和self_certainty"""
#     try:
#         if generate_logits.dim() == 3:
#             generate_logits = generate_logits.squeeze(1)
#         min_len = min(generate_logits.size(0), len(generated_tokens))
#         if min_len == 0:
#             return float('inf'), 0.0, 0.0
#         generate_logits = generate_logits[:min_len]
#         generated_tokens = generated_tokens[:min_len]

#         self_certainty = compute_self_certainty(generate_logits)

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
#             if np.isinf(ppl) or np.isnan(ppl):
#                 ppl = float('inf')
#         else:
#             ppl = float('inf')

#         return ppl, min_token_prob, self_certainty

#     except Exception as e:
#         logger.error(f"综合指标计算失败: {e}")
#         return float('inf'), 0.0, 0.0


# def main():
#     logger.info("正在加载模型和分词器...")

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True
#     )

#     if os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json")):
#         logger.info("检测到 LoRA 权重，正在加载...")
#         model = PeftModel.from_pretrained(base_model, MODEL_PATH)
#         logger.info("LoRA 加载完成")
#     else:
#         model = base_model
#         logger.info("未检测到 LoRA，使用基础模型")

#     model.eval()
#     logger.info("模型加载完成")

#     with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
#         test_data = json.load(f)
#     logger.info(f"共加载 {len(test_data)} 条测试数据")

#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

#     with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
#         f_out.write("[\n")

#         for idx, item in enumerate(tqdm(test_data)):
#             question = item.get("question", "").strip()
#             if not question:
#                 continue

#             messages = [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": question}
#             ]

#             input_ids = tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 return_tensors="pt"
#             ).to(model.device)

#             with torch.no_grad():
#                 outputs = model.generate(
#                     input_ids=input_ids,
#                     max_new_tokens=MAX_NEW_TOKENS,
#                     do_sample=False,
#                     temperature=0.7,
#                     top_p=0.9,
#                     repetition_penalty=1.2,
#                     pad_token_id=tokenizer.pad_token_id,
#                     eos_token_id=tokenizer.eos_token_id,
#                     return_dict_in_generate=True,
#                     output_scores=True
#                 )

#             prompt_len = input_ids.shape[1]
#             generated_tokens = outputs.sequences[0][prompt_len:]
#             model_answer_raw = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

#             # >>>>>>>>>> 关键：先从原始输出提取结构化字段 <<<<<<<<<<
#             extracted_answer, extracted_reflection, extracted_confidence = extract_from_output(model_answer_raw)

#             # 计算不确定性指标
#             ppl, min_token_prob, self_certainty = float('inf'), 0.0, 0.0
#             if hasattr(outputs, 'scores') and outputs.scores:
#                 generate_logits = torch.stack(outputs.scores, dim=0)
#                 ppl, min_token_prob, self_certainty = compute_ppl_minprob_selfcertainty(generated_tokens, generate_logits)

#             # 决策是否拒答（仅用于 final_model_answer）
#             should_answer = (
#                 ppl <= PPL_THRESHOLD and
#                 min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
#                 self_certainty >= SELF_CERTAINTY_THRESHOLD
#             )

#             if not should_answer:
#                 final_model_answer = random.choice(REJECTION_RESPONSES)
#             else:
#                 final_model_answer = model_answer_raw

#             # 构建输出项
#             output_item = {
#                 "source_id": item.get("source_id", idx),
#                 "question": question,
#                 "standard_answer": item.get("answer", ""),
#                 "model_answer_raw": model_answer_raw,
#                 "extracted_answer": extracted_answer,
#                 "extracted_reflection": extracted_reflection,
#                 "extracted_confidence": extracted_confidence,
#                 "refusal_decision": {
#                     "ppl": round(ppl, 4) if ppl != float('inf') else "inf",
#                     "min_token_prob": round(min_token_prob, 4),
#                     "self_certainty": round(self_certainty, 4),
#                     "should_answer": bool(should_answer),
#                     "final_model_answer": final_model_answer,
#                 }
#             }

#             # 写入文件
#             if idx > 0:
#                 f_out.write(",\n")
#             f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
#             f_out.flush()

#         f_out.write("\n]")

#     logger.info(f"推理完成，结果已保存至: {OUTPUT_PATH}")


# if __name__ == "__main__":
#     main()