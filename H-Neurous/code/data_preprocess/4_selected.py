# import json
# import re
# import os

# def normalize_text(text):
#     """标准化：转小写 + 去除标点和多余空格"""
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r'[^\w\s]', '', text)  # 移除所有非字母/数字/空格字符（包括·、-、()等）
#     text = re.sub(r'\s+', ' ', text).strip().lower()
#     return text

# def is_correct(model_answer, ground_truth):
#     """
#     判断 model_answer 是否正确：
#     - 匹配 ground_truth["value"] 或 aliases 中任意一个（归一化后互相包含即算对）
#     """
#     pred_norm = normalize_text(model_answer)
#     if not pred_norm:
#         return False

#     candidates = [ground_truth["value"]]
#     candidates.extend(ground_truth.get("aliases", []))

#     for cand in candidates:
#         cand_norm = normalize_text(cand)
#         if not cand_norm:
#             continue
#         # 双向包含：预测包含标准答案，或标准答案包含预测（适用于“乞力马扎罗山” vs “吉利曼贾罗山”这类近似）
#         if cand_norm in pred_norm or pred_norm in cand_norm:
#             return True
#     return False

# def main():
#     input_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/inference1.json"
#     output_dir = "/mnt/project/H-Neurous/data/trivia_qa/inference"
#     os.makedirs(output_dir, exist_ok=True)

#     output_all_correct = os.path.join(output_dir, "faithful.json")
#     output_all_wrong = os.path.join(output_dir, "hallucination.json")

#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     all_correct = []
#     all_wrong = []

#     for sample in data:
#         model_answers = sample["model_answers"]  # list of 10 strings
#         ground_truth = sample["ground_truth"]

#         # 判断每个回答是否正确
#         correctness = [is_correct(ans, ground_truth) for ans in model_answers]

#         # 构造新样本（只保留一个 model_answer）
#         new_sample = {
#             "qid": sample["qid"],
#             "question": sample["question"],
#             "answer": sample["answer"],
#             "model_answer": model_answers[0],  # 选第一个作为代表
#             "ground_truth": ground_truth
#         }

#         if all(correctness):  # 10次全对
#             all_correct.append(new_sample)
#         elif not any(correctness):  # 10次全错
#             all_wrong.append(new_sample)

#     # 保存全对
#     with open(output_all_correct, "w", encoding="utf-8") as f:
#         json.dump(all_correct, f, ensure_ascii=False, indent=2)

#     # 保存全错
#     with open(output_all_wrong, "w", encoding="utf-8") as f:
#         json.dump(all_wrong, f, ensure_ascii=False, indent=2)

#     print(f"✅ 全对样本数: {len(all_correct)} → 已保存至 {output_all_correct}")
#     print(f"✅ 全错样本数: {len(all_wrong)} → 已保存至 {output_all_wrong}")

# if __name__ == "__main__":
#     main()
# ============================================================字符串匹配正确=======================================
# import json
# import re
# import os

# def normalize_text(text):
#     """标准化：转小写 + 去除标点和多余空格"""
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r'[^\w\s]', '', text)  # 移除所有非字母/数字/空格字符（包括·、-、()等）
#     text = re.sub(r'\s+', ' ', text).strip().lower()
#     return text

# def is_correct(model_answer, ground_truth):
#     """
#     判断 model_answer 是否正确：
#     - 匹配 ground_truth["value"] 或 aliases 中任意一个（归一化后互相包含即算对）
#     """
#     pred_norm = normalize_text(model_answer)
#     if not pred_norm:
#         return False

#     candidates = [ground_truth["value"]]
#     candidates.extend(ground_truth.get("aliases", []))

#     for cand in candidates:
#         cand_norm = normalize_text(cand)
#         if not cand_norm:
#             continue
#         # 双向包含：预测包含标准答案，或标准答案包含预测
#         if cand_norm in pred_norm or pred_norm in cand_norm:
#             return True
#     return False

# def main():
#     input_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/inference1.json"
#     output_dir = "/mnt/project/H-Neurous/data/trivia_qa/inference"
#     os.makedirs(output_dir, exist_ok=True)

#     output_all_correct = os.path.join(output_dir, "faithful.json")
#     output_all_wrong = os.path.join(output_dir, "hallucination.json")

#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     all_correct = []
#     all_wrong = []

#     for sample in data:
#         model_answers = sample["model_answers"]  # list of 10 strings
#         ground_truth = sample["ground_truth"]

#         # 判断每个回答是否正确
#         correctness = [is_correct(ans, ground_truth) for ans in model_answers]

#         # 保留原始样本结构（不修改任何字段）
#         if all(correctness):  # 10次全对
#             all_correct.append(sample)
#         elif not any(correctness):  # 10次全错
#             all_wrong.append(sample)

#     # 保存全对
#     with open(output_all_correct, "w", encoding="utf-8") as f:
#         json.dump(all_correct, f, ensure_ascii=False, indent=2)

#     # 保存全错
#     with open(output_all_wrong, "w", encoding="utf-8") as f:
#         json.dump(all_wrong, f, ensure_ascii=False, indent=2)

#     print(f"✅ 全对样本数: {len(all_correct)} → 已保存至 {output_all_correct}")
#     print(f"✅ 全错样本数: {len(all_wrong)} → 已保存至 {output_all_wrong}")

# if __name__ == "__main__":
#     main() 

# ==========================================大模型判断正确性====================================
# import os
# import json
# import torch
# import gc
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from tqdm import tqdm

# # ==================== 配置 ====================
# class Config:
#     model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
#     input_data_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/faithful.json"  # 必须含 aliases 和 model_answers
#     output_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/faithful1.json"
    
#     max_new_tokens = 64
#     use_4bit = True


# # ==================== 加载模型 ====================
# def load_model_and_tokenizer(config):
#     print("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(
#         config.model_path,
#         trust_remote_code=True,
#         use_fast=True
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     print("Loading model...")
#     bnb_config = None
#     if config.use_4bit:
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )
    
#     model = AutoModelForCausalLM.from_pretrained(
#         config.model_path,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True
#     )
#     model.eval()
#     return model, tokenizer


# # ==================== 构造判断 Prompt（含 aliases）====================
# def build_judge_prompt(question: str, gold_answer: str, aliases, model_answers):
#     # 处理别名列表为空的情况
#     if not aliases:
#         aliases_str = "无"
#     else:
#         # 转为字符串列表，避免 JSON 格式干扰
#         aliases_str = ", ".join(str(a) for a in aliases)
    
#     answers_text = "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(model_answers))
    
#     prompt = (
#         f"问题：{question}\n"
#         f"标准答案：{gold_answer}\n"
#         f"可接受的别名包括：[{aliases_str}]\n"
#         f"模型生成的10个答案如下：\n{answers_text}\n\n"
#         "请严格判断：以上10个答案是否每一个都正确表达了标准答案或其任一别名的核心事实？\n"
#         "允许表述不同、包含额外说明（如'导演是...'），但不能包含错误信息或与标准答案无关的内容。\n"
#         "只要有一个答案错误或不匹配，就视为不全对。\n"
#         "请仅回答一个字：是 或 否"
#     )
#     return prompt


# # ==================== 解析模型输出-10个全对 ====================
# def parse_judge_output(output: str) -> bool:
#     output_clean = output.strip().replace(" ", "").replace("。", "")
#     if output_clean.startswith("是"):
#         return True
#     if output_clean.startswith("否"):
#         return False
#     # 保守策略：无法确定即判错
#     return False


# # ==================== 主函数 ====================
# # ==================== 主函数（流式写入版）====================
# def main():
#     config = Config()
#     model, tokenizer = load_model_and_tokenizer(config)
    
#     with open(config.input_data_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    
#     written_count = 0  # 已写入样本数
    
#     with open(config.output_path, "w", encoding="utf-8") as f_out:
#         f_out.write("[\n")
        
#         for idx, sample in enumerate(tqdm(data, desc="LLM judging with aliases")):
#             required_keys = ["qid", "question", "answer", "model_answers"]
#             if not all(k in sample for k in required_keys):
#                 continue
            
#             question = sample["question"]
#             gold_answer = sample["answer"]
#             aliases = sample.get("aliases", [])
#             model_answers = sample["model_answers"]
            
#             if len(model_answers) != 10:
#                 continue
            
#             # 构建判断 prompt
#             judge_prompt = build_judge_prompt(question, gold_answer, aliases, model_answers)
            
#             # 构造聊天消息
#             chat_msgs = [
#                 {
#                     "role": "system",
#                     "content": "你是一个严谨的事实一致性验证专家。请根据标准答案及其别名，判断所有生成答案是否全部正确。"
#                 },
#                 {
#                     "role": "user",
#                     "content": judge_prompt
#                 }
#             ]
            
#             inputs = tokenizer.apply_chat_template(
#                 chat_msgs,
#                 tokenize=True,
#                 add_generation_prompt=True,
#                 return_tensors="pt"
#             ).to(model.device)
            
#             # 生成判断
#             with torch.no_grad():
#                 outputs = model.generate(
#                     inputs,
#                     max_new_tokens=config.max_new_tokens,
#                     do_sample=False,
#                     temperature=0.0,
#                     top_p=1.0,
#                     pad_token_id=tokenizer.pad_token_id,
#                     eos_token_id=tokenizer.eos_token_id
#                 )
            
#             generated_tokens = outputs[0][inputs.shape[1]:]
#             judge_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
#             # 解析判断结果
#             all_correct = parse_judge_output(judge_text)
            
#             if all_correct:
#                 new_sample = {
#                     "qid": sample["qid"],
#                     "question": sample["question"],
#                     "answer": sample["answer"],
#                     "model_answer": model_answers[0],  # 任选第一个
#                     "ground_truth": sample["ground_truth"]
#                 }
                
#                 # 流式写入：加逗号（如果不是第一个）
#                 if written_count > 0:
#                     f_out.write(",\n")
#                 f_out.write(json.dumps(new_sample, ensure_ascii=False, indent=2))
#                 f_out.flush()
#                 written_count += 1
            
#             # 显存清理
#             if (idx + 1) % 10 == 0:
#                 gc.collect()
#                 torch.cuda.empty_cache()
        
#         f_out.write("\n]\n")
    
#     print(f"✅ 完成！共保留 {written_count} 个样本，已保存至: {config.output_path}")


# if __name__ == "__main__":
#     main()

# ==================== 解析模型输出-10个全错 ====================
import os
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ==================== 配置 ====================
class Config:
    model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
    input_data_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/hallucination.json"  # 必须含 aliases 和 model_answers
    output_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/hallucination1.json"
    
    max_new_tokens = 64
    use_4bit = True


# ==================== 加载模型 ====================
def load_model_and_tokenizer(config):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


# ==================== 构造判断 Prompt：是否 10 个全错？====================
def build_judge_prompt(question: str, gold_answer: str, aliases, model_answers):
    if not aliases:
        aliases_str = "无"
    else:
        aliases_str = ", ".join(str(a) for a in aliases)
    
    answers_text = "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(model_answers))
    
    prompt = (
        f"问题：{question}\n"
        f"标准答案：{gold_answer}\n"
        f"可接受的别名包括：[{aliases_str}]\n"
        f"模型生成的10个答案如下：\n{answers_text}\n\n"
        "请严格判断：以上10个答案是否**全部错误**？\n"
        "即：没有任何一个答案正确表达了标准答案或其任一别名的核心事实。\n"
        "只要有一个答案正确（即使其他都错），就不算‘全部错误’。\n"
        "请仅回答一个字：是 或 否"
    )
    return prompt


# ==================== 解析 LLM 输出：是否全错？====================
def parse_judge_output(output: str) -> bool:
    output_clean = output.strip().replace(" ", "").replace("。", "")
    if output_clean.startswith("是"):
        return True   # 是 → 全错
    if output_clean.startswith("否"):
        return False  # 否 → 不全错（至少有一个对）
    return False      # 默认保守：不认为全错


# ==================== 主函数 ====================
def main():
    config = Config()
    model, tokenizer = load_model_and_tokenizer(config)
    
    with open(config.input_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    written_count = 0

    with open(config.output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        
        for idx, sample in enumerate(tqdm(data, desc="LLM judging all-wrong samples")):
            required_keys = ["qid", "question", "answer", "model_answers", "ground_truth"]
            if not all(k in sample for k in required_keys):
                continue
            
            question = sample["question"]
            gold_answer = sample["answer"]
            aliases = sample.get("aliases", [])
            model_answers = sample["model_answers"]
            
            if len(model_answers) != 10:
                continue
            
            # 构建判断 prompt（是否全错）
            judge_prompt = build_judge_prompt(question, gold_answer, aliases, model_answers)
            
            # 构造聊天消息
            chat_msgs = [
                {
                    "role": "system",
                    "content": "你是一个严谨的事实一致性验证专家。请判断以下10个答案是否全部错误。"
                },
                {
                    "role": "user",
                    "content": judge_prompt
                }
            ]
            
            inputs = tokenizer.apply_chat_template(
                chat_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            # 生成判断
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_tokens = outputs[0][inputs.shape[1]:]
            judge_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # 判断是否全错
            all_wrong = parse_judge_output(judge_text)
            
            # 仅当 LLM 判定“全错”时才保存
            if all_wrong:
                new_sample = {
                    "qid": sample["qid"],
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "model_answer": model_answers[0],  # 第一个生成答案
                    "ground_truth": sample["ground_truth"]
                    # 可选：记录 LLM 判断原文用于调试
                    # "llm_judge_raw": judge_text
                }
                
                if written_count > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(new_sample, ensure_ascii=False, indent=2))
                f_out.flush()
                written_count += 1
            
            # 每处理 10 个样本清理一次显存
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        f_out.write("\n]\n")
    
    print(f"✅ 完成！共保留 {written_count} 个‘10个答案全错’的样本，已保存至: {config.output_path}")


if __name__ == "__main__":
    main()