import os
import json
import torch
import logging
import unsloth
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel

import os


# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate1.log',
    filemode='a',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)


# 配置参数
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"  # 基础模型路径
MODEL_CPT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-3years"  # 预训练模型路径
MODEL_MOL_CPT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years"  # MOL预训练模型路径
MODEL_CPT_QLORA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/output-cpt-drouput0.1-r128-bs16-epoch5/final_model"  # 预训练+QLORA微调模型路径
MODEL_MOL_CPT_QLORA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"  # MOL预训练+QLORA微调模型路径
MODEL_CPT_QLORA_2DATASET_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/output-cpt-drouput0.1-r128-bs16-epoch5/final_model_2dataset"  # 预训练+QLORA2个数据集微调模型路径
MODEL_MOL_CPT_QLORA_DPO_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/outputs_sft_dpo/final_model1"

TEST_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train.json"  # 测试集路径
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/mol_cpt_qlora_sft_dpo_5epoch.json"  # 输出结果路径
MAX_LENGTH = 2048  # 最大输入长度
MAX_NEW_TOKENS = 2048  # 最大生成的新token数量
BATCH_SIZE = 16  # 推理批次大小


def load_model(model_name):
    """使用Unsloth加载模型和分词器"""
    logger.info("正在使用Unsloth加载模型和分词器...")
    
    # 使用Unsloth加载基础模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    
    model.eval()  # 设置为评估模式
    
    return model, tokenizer


def format_prompt(question):
    """格式化提问"""
    # 使用与训练时相同的格式
    messages = [
        {"role": "user", "content": f"回答以下问题：{question}"}
    ]
    return messages


def generate_answer(model, tokenizer, question):
    """使用Unsloth生成回答"""
    messages = format_prompt(question)
    prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # 关键！仅格式化不编码
            add_generation_prompt=True  # 为assistant添加空内容占位符
        )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 
    
    # 使用模型生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # 使用贪婪解码
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的回答
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # 提取助手的回答部分
    assistant_start = generated_text.rfind("<|im_start|>assistant\n")
    if assistant_start != -1:
        # 截取助手部分的回答
        assistant_text = generated_text[assistant_start + len("<|im_start|>assistant\n"):]
        # 去除结束标记
        end_token_pos = assistant_text.find("<|im_end|>")
        if end_token_pos != -1:
            assistant_text = assistant_text[:end_token_pos]
        return assistant_text.strip()
    else:
        # 如果找不到标记，直接返回生成的文本
        return generated_text.replace(prompt, "").strip()


# 加载模型和分词器
# model_base, _ = load_model(MODEL_BASE_PATH)
# model_cpt, tokenizer = load_model(MODEL_CPT_PATH)
# model_mol_cpt, tokenizer = load_model(MODEL_MOL_CPT_PATH)
# model_cpt_qlora, tokenizer = load_model(MODEL_CPT_QLORA_PATH)
# model_mol_cpt_qlora, tokenizer = load_model(MODEL_MOL_CPT_QLORA_PATH)
# model_cpt_qlora_2dataset, tokenizer = load_model(MODEL_CPT_QLORA_2DATASET_PATH)
model_mol_cpt_qlora_dpo, tokenizer = load_model(MODEL_MOL_CPT_QLORA_DPO_PATH)


# 加载测试数据
logger.info(f"正在加载测试数据: {TEST_DATA_PATH}")
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 初始化结果列表
results = []


logger.info(f"开始对{len(test_data)}条测试数据进行推理并实时保存...")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write("[\n")  # 开始 JSON 数组
    for idx, item in enumerate(tqdm(test_data)):
        question = item["question"]

        # 生成回答
        # base_model_answer = generate_answer(model_base, tokenizer, question)
        # cpt_model_answer = generate_answer(model_cpt, tokenizer, question)
        # mol_cpt_model_answer = generate_answer(model_mol_cpt, tokenizer, question)
        # cpt_qlora_model_answer = generate_answer(model_cpt_qlora, tokenizer, question)
        # mol_cpt_qlora_model_answer = generate_answer(model_mol_cpt_qlora, tokenizer, question)
        # cpt_qlora_2dataset_model_answer = generate_answer(model_cpt_qlora_2dataset, tokenizer, question)
        mol_cpt_qlora_dpo_model_answer = generate_answer(model_mol_cpt_qlora_dpo, tokenizer, question)

        

        # 创建结果字典
        result = {
            "question": item["question"],
            "answer": item["answer"],
            # "cpt_qlora_2dataset_model_answer": cpt_qlora_2dataset_model_answer,
            # "base_model_answer": item["base_model_answer"],
            # "cpt_model_answer": item["cpt_model_answer"],
            # "mol_cpt_model_answer": item["mol_cpt_model_answer"],
            # "cpt_qlora_model_answer": cpt_qlora_model_answer,
            # "mol_cpt_qlora_model_answer": mol_cpt_qlora_model_answer,
            "mol_cpt_qlora_sft_dpo_model_answer": mol_cpt_qlora_dpo_model_answer,
            "source_id": item.get("source_id"),
        }

        # 如果不是第一条，先写逗号换行
        if idx > 0:
            f.write(",\n")

        # 写入 JSON 对象
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
        f.flush()

    f.write("\n]")  # 结束 JSON 数组

logger.info("所有操作已完成，结果已实时保存为 JSON 数组。")

# import os
# import json
# import torch
# import logging
# from tqdm import tqdm
# from unsloth import FastLanguageModel

# # 设置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='inference_only.log',
#     filemode='w',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# # 配置路径
# MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
# MODEL_CPT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-3years"
# MODEL_MOL_CPT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years"
# MODEL_CPT_QLORA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/output-cpt-drouput0.1-r128-bs16-epoch5/final_model"
# MODEL_MOL_CPT_QLORA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
# TEST_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/datasets_fact_qa/domain_test_result.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/datasets_fact_qa/cpt_domain_test_result.json"

# MAX_LENGTH = 2048
# MAX_NEW_TOKENS = 2048

# # ==================== 新增：置信度提示====================
# CONFIDENCE_PROMPT = (
#     "请回答以下问题：{question}。只有在你非常确定答案正确时才回答。"
#     "如果你不确定、信息不足或无法验证，请回答“我不知道”。"
#     "避免猜测，因为错误回答比不回答更糟糕。"
# )
# # ========================================================================

# def load_model(model_path):
#     """加载模型和分词器"""
#     logger.info(f"正在加载模型: {model_path}")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=model_path,
#         max_seq_length=MAX_LENGTH,
#         dtype=torch.bfloat16,
#         load_in_4bit=True,
#     )
#     model.eval()
#     return model, tokenizer

# def format_prompt(tokenizer, question, use_confidence=False):
#     """格式化提示词"""
#     if use_confidence:
#         question = CONFIDENCE_PROMPT.format(question=question)
    
#     messages = [{"role": "user", "content": question}]
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     return prompt

# def generate_response(model, tokenizer, prompt):
#     """生成回答"""
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,
#             temperature=0.7,
#             top_p=0.9,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
#     assistant_start = full_text.rfind("<|im_start|>assistant\n")
    
#     if assistant_start != -1:
#         answer = full_text[assistant_start + len("<|im_start|>assistant\n"):]
#         end_pos = answer.find("<|im_end|>")
#         if end_pos != -1:
#             answer = answer[:end_pos]
#         return answer.strip()
#     else:
#         return full_text.replace(prompt, "").strip()

# # ========== 加载所有模型 ==========
# logger.info("开始加载模型...")
# # model_base, tokenizer = load_model(MODEL_BASE_PATH)
# # model_cpt, _ = load_model(MODEL_CPT_PATH)
# # model_mol_cpt, _ = load_model(MODEL_MOL_CPT_PATH)
# # model_cpt_qlora, _ = load_model(MODEL_CPT_QLORA_PATH)
# model_mol_cpt_qlora, tokenizer = load_model(MODEL_MOL_CPT_QLORA_PATH)
# logger.info("所有模型加载完成。")

# # ========== 加载测试数据 ==========
# logger.info(f"加载测试数据: {TEST_DATA_PATH}")
# with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
#     test_data = json.load(f)

# # ========== 推理并保存 ==========
# logger.info(f"开始推理，共 {len(test_data)} 条数据...")

# with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
#     f.write("[\n")
#     for idx, item in enumerate(tqdm(test_data)):
#         question = item["question"]

#         # 生成各模型回答（使用置信度提示）
#         # base_answer = generate_response(model_base, tokenizer, format_prompt(tokenizer, question, use_confidence=True))
#         # cpt_answer = generate_response(model_cpt, tokenizer, format_prompt(tokenizer, question, use_confidence=True))
#         # mol_cpt_answer = generate_response(model_mol_cpt, tokenizer, format_prompt(tokenizer, question, use_confidence=True))
#         # cpt_qlora_answer = generate_response(model_cpt_qlora, tokenizer, format_prompt(tokenizer, question, use_confidence=True))
#         mol_cpt_qlora_model_answer_prompt = generate_response(model_mol_cpt_qlora, tokenizer, format_prompt(tokenizer, question, use_confidence=True))

#         # 构造结果
#         result = {
#             "question": question,
#             "answer": item["answer"],
#             "base_model_answer": item["base_model_answer"],
#             "cpt_model_answer": item["cpt_model_answer"],
#             "mol_cpt_model_answer": item["mol_cpt_model_answer"],
#             "cpt_qlora_model_answer": item["cpt_qlora_model_answer"],
#             "mol_cpt_qlora_model_answer": item["mol_cpt_qlora_model_answer"],
#             "mol_cpt_qlora_model_answer_prompt":mol_cpt_qlora_model_answer_prompt,
#             "source_id": item.get("source_id")
#         }

#         # 写入文件
#         if idx > 0:
#             f.write(",\n")
#         f.write(json.dumps(result, ensure_ascii=False, indent=2))
#         f.flush()

#     f.write("\n]")

# logger.info(f"推理完成，结果已保存至: {OUTPUT_PATH}")