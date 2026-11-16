# # # =============================================================召回数据分析===============================================================
# import json

# # 读取你的 JSON 文件（假设是列表）
# with open('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_yuanshi/test_v2/test_claude_nodecide/evaluation_results1.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 初始化计数器
# non_refusal_wrong_total = 0
# non_refusal_wrong_retrieved_ground_truth = 0

# refusal_total = 0
# refusal_retrieved_ground_truth = 0

# for item in data:
#     # 提取关键字段
#     refusal = item['evaluation_results']['correctness']['refusal']
#     score = item['evaluation_results']['correctness']['score']
#     source_doc = item.get('source_doc', {})
    
#     # 跳过缺失 source_doc 的样本（可选）
#     if not source_doc:
#         continue

#     ground_truth_title = source_doc['title']
#     ground_truth_date = source_doc['pub_date']

#     # 获取召回文档列表
#     retrieved_docs = item.get('retrieval_info', {}).get('retrieved_docs', [])

#     # 判断是否召回了标准文档（title + pub_date 完全匹配）
#     retrieved_ground_truth = any(
#         doc.get('title') == ground_truth_title and
#         doc.get('publish_date') == ground_truth_date
#         for doc in retrieved_docs
#     )

#     # 分类统计
#     if refusal:
#         refusal_total += 1
#         if retrieved_ground_truth:
#             refusal_retrieved_ground_truth += 1
#     else:
#         if score < 4:  # 回答错误
#             non_refusal_wrong_total += 1
#             if retrieved_ground_truth:
#                 non_refusal_wrong_retrieved_ground_truth += 1

# # 输出结果
# print("📊 错误原因分析统计结果：\n")

# print(f"1. 非拒答但回答错误 (refusal=False 且 score < 4):")
# print(f"   - 总数: {non_refusal_wrong_total}")
# print(f"   - 召回了标准文档: {non_refusal_wrong_retrieved_ground_truth}")
# print(f"   - 未召回标准文档: {non_refusal_wrong_total - non_refusal_wrong_retrieved_ground_truth}\n")

# print(f"2. 拒答 (refusal=True):")
# print(f"   - 总数: {refusal_total}")
# print(f"   - 召回了标准文档: {refusal_retrieved_ground_truth}")
# print(f"   - 未召回标准文档: {refusal_total - refusal_retrieved_ground_truth}\n")

# ==============================================================召回数据分析大模型===============================================================
# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import concurrent.futures
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from openai import OpenAI
from collections import Counter

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='recall_analysis_multithread.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 配置参数
INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk_api/ds_450/evaluation_results.json"
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk_api/ds_450/evaluation_results_analysis"
NON_REFUSAL_FILE = os.path.join(OUTPUT_DIR, "non_refusal_errors_with_recall.json")
REFUSAL_FILE = os.path.join(OUTPUT_DIR, "refusal_errors_with_recall.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "recall_analysis_summary.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化OpenAI客户端
client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def load_data() -> List[Dict[str, Any]]:
    """加载数据"""
    logger.info(f"加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_recall_success(question: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
    """
    判断召回是否成功（召回的文档中是否有相关信息）
    返回: True=召回成功（有相关信息）, False=召回失败（无相关信息）
    """
    if not retrieved_docs:
        return False
    
    # 构建召回的文档内容字符串
    docs_content = []
    for i, doc in enumerate(retrieved_docs[:5], 1):
        title = doc.get("title", "")
        content_preview = doc.get("content_preview", "")
        docs_content.append(f"【文档{i}】{title}\n{content_preview}")
    
    retrieved_content = "\n\n".join(docs_content)
    
    prompt = f"""请严格分析以下问题及其召回的文档，判断召回的文档中是否包含**完整且足够精确的信息**来回答问题。

## 问题：
{question}

## 召回的文档：
{retrieved_content}

## 评估标准（严格标准）：
1. **关键实体匹配**：召回的文档必须包含问题中提到的关键实体（如具体的人名、地名、组织名、时间、数字、活动名称等）
2. **信息完整性**：文档中的信息必须能够**直接回答**问题，而不是仅提供模糊的背景信息
3. **精确性要求**：对于涉及具体数值、日期、名称的问题，必须有准确对应的信息
4. **否定情况示例**：
   - 文档只提到相关话题，但没有具体信息
   - 文档提到相关实体，但没有回答问题所需的具体属性
   - 文档信息模糊，无法直接支持回答问题

## 判断逻辑：
- 如果**任何**一个召回的文档中包含能够直接回答问题所需的所有关键信息，则召回成功
- 如果所有文档都缺乏回答问题所需的关键信息，或者信息不完整、不精确，则召回失败

## 示例说明：
示例1：
问题："永嘉县在优化法治营商环境中推行了哪种执法模式？"
要求：文档中必须明确提到"永嘉县在优化法治营商环境中推行了XX执法模式"的具体语句

示例2：
问题："温州北站改建后的站房总建筑面积是多少平方米？"
要求：文档中必须明确提到"温州北站改建后的站房总建筑面积是XX平方米"的具体语句

请严格按照以下JSON格式回答：
{{
  "recall_success": true/false,  // 召回的文档中是否包含足够精确和完整的信息来回答问题
}}

请先分析问题需要的关键信息，然后检查文档是否包含这些信息。"""
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {"role": "system", "content": "你是一个专业的评估专家，判断召回的文档是否包含问题相关信息。仔细检查文档内容，不要只看标题。只返回JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 提取JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            
            if "recall_success" not in result:
                raise ValueError("返回结果缺少recall_success字段")
            
            return result["recall_success"]
            
        except Exception as e:
            if attempt == 2:
                logger.warning(f"解析失败，使用保守估计: False, 错误: {e}")
                return False  # 保守估计，认为召回失败
    
    return False

def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个样本"""
    try:
        # 检查召回是否成功
        recall_success = check_recall_success(
            question=sample.get("question", ""),
            retrieved_docs=sample.get("refusal_decision", {}).get("retrieved_docs_preview", [])
        )
        
        # 创建新样本对象，添加recall_success字段
        sample_copy = sample.copy()
        sample_copy["recall_success"] = recall_success
        
        logger.debug(f"处理样本 {sample.get('source_id')}: recall_success={recall_success}")
        
        return sample_copy
        
    except Exception as e:
        logger.error(f"处理样本 {sample.get('source_id')} 失败: {e}")
        # 返回一个标记失败的样本
        sample_copy = sample.copy()
        sample_copy["recall_success"] = False
        sample_copy["analysis_error"] = str(e)
        return sample_copy

def analyze_samples_multithread(samples: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
    """多线程分析样本"""
    if not samples:
        return []
    
    logger.info(f"开始多线程分析 {len(samples)} 个样本，线程数: {max_workers}")
    
    analyzed_samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_sample = {
            executor.submit(process_single_sample, sample): sample 
            for sample in samples
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(samples), desc="分析进度") as pbar:
            for future in concurrent.futures.as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    analyzed_samples.append(result)
                except Exception as e:
                    logger.error(f"样本 {sample.get('source_id')} 处理异常: {e}")
                    # 添加一个默认结果
                    sample_copy = sample.copy()
                    sample_copy["recall_success"] = False
                    sample_copy["analysis_error"] = str(e)
                    analyzed_samples.append(sample_copy)
                finally:
                    pbar.update(1)
    
    logger.info(f"多线程分析完成，成功处理 {len(analyzed_samples)} 个样本")
    return analyzed_samples

def analyze_and_save_samples_multithread(max_workers: int = 5):
    """多线程分析样本并保存结果"""
    # 加载数据
    data = load_data()
    
    # 分类样本
    non_refusal_errors = []  # refusal=false, score<4
    refusal_samples = []      # refusal=true
    
    for sample in data:
        try:
            correctness = sample.get("evaluation_results", {}).get("correctness", {})
            refusal = correctness.get("refusal", False)
            score = correctness.get("score", 5)
            
            if not refusal and score < 4:
                non_refusal_errors.append(sample)
            elif refusal:
                refusal_samples.append(sample)
        except Exception as e:
            logger.error(f"分类失败: {e}")
    
    logger.info(f"非拒答错误样本: {len(non_refusal_errors)}")
    logger.info(f"拒答样本: {len(refusal_samples)}")
    
    # 多线程分析非拒答错误样本
    logger.info("开始多线程分析非拒答错误样本...")
    non_refusal_with_recall = analyze_samples_multithread(non_refusal_errors, max_workers)
    
    # 多线程分析拒答样本
    logger.info("开始多线程分析拒答样本...")
    refusal_with_recall = analyze_samples_multithread(refusal_samples, max_workers)
    
    # 计算统计信息
    non_refusal_stats = {
        "recall_success": sum(1 for s in non_refusal_with_recall if s.get("recall_success", False)),
        "recall_failure": sum(1 for s in non_refusal_with_recall if not s.get("recall_success", False))
    }
    
    refusal_stats = {
        "recall_success": sum(1 for s in refusal_with_recall if s.get("recall_success", False)),
        "recall_failure": sum(1 for s in refusal_with_recall if not s.get("recall_success", False))
    }
    
    # 保存结果
    logger.info("保存分析结果...")
    with open(NON_REFUSAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(non_refusal_with_recall, f, ensure_ascii=False, indent=2)
    logger.info(f"非拒答错误样本已保存: {NON_REFUSAL_FILE}")
    
    with open(REFUSAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(refusal_with_recall, f, ensure_ascii=False, indent=2)
    logger.info(f"拒答样本已保存: {REFUSAL_FILE}")
    
    # 生成统计摘要
    generate_summary(
        non_refusal_with_recall, 
        refusal_with_recall,
        non_refusal_stats,
        refusal_stats
    )
    
    return non_refusal_with_recall, refusal_with_recall

def generate_summary(non_refusal_data, refusal_data, non_refusal_stats, refusal_stats):
    """生成统计摘要"""
    
    def calculate_rates(stats, total):
        if total == 0:
            return "0.0%", "0.0%"
        success_rate = (stats["recall_success"] / total * 100)
        failure_rate = (stats["recall_failure"] / total * 100)
        return f"{success_rate:.1f}%", f"{failure_rate:.1f}%"
    
    non_refusal_total = len(non_refusal_data)
    refusal_total = len(refusal_data)
    
    non_refusal_success_rate, non_refusal_failure_rate = calculate_rates(
        non_refusal_stats, non_refusal_total
    )
    
    refusal_success_rate, refusal_failure_rate = calculate_rates(
        refusal_stats, refusal_total
    )
    
    # 检查是否有分析错误的样本
    non_refusal_errors = [s for s in non_refusal_data if s.get("analysis_error")]
    refusal_errors = [s for s in refusal_data if s.get("analysis_error")]
    
    summary = {
        "总体统计": {
            "总样本数": len(non_refusal_data) + len(refusal_data),
            "非拒答错误样本数": non_refusal_total,
            "拒答样本数": refusal_total,
            "分析错误样本数": len(non_refusal_errors) + len(refusal_errors)
        },
        "非拒答错误样本分析": {
            "总数": non_refusal_total,
            "召回成功数": non_refusal_stats["recall_success"],
            "召回失败数": non_refusal_stats["recall_failure"],
            "召回成功率": non_refusal_success_rate,
            "召回失败率": non_refusal_failure_rate,
            "分析错误数": len(non_refusal_errors),
            "错误类型分析": {
                "召回失败导致的错误": non_refusal_failure_rate,
                "提取错误导致的错误": non_refusal_success_rate
            }
        },
        "拒答样本分析": {
            "总数": refusal_total,
            "召回成功数": refusal_stats["recall_success"],
            "召回失败数": refusal_stats["recall_failure"],
            "召回成功率": refusal_success_rate,
            "召回失败率": refusal_failure_rate,
            "分析错误数": len(refusal_errors),
            "拒答类型分析": {
                "正确拒答（召回失败）": refusal_failure_rate,
                "错误拒答（召回成功）": refusal_success_rate
            }
        },
        "性能指标": {
            "非拒答错误中提取错误比例": non_refusal_success_rate,
            "拒答样本中错误拒答比例": refusal_success_rate,
            "总体召回失败率": f"{((non_refusal_stats['recall_failure'] + refusal_stats['recall_failure']) / (non_refusal_total + refusal_total) * 100):.1f}%"
        }
    }
    
    # 保存摘要
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"统计摘要已保存: {SUMMARY_FILE}")
    
    # 打印关键结果
    logger.info("\n" + "="*60)
    logger.info("召回分析结果摘要 (多线程版本)")
    logger.info("="*60)
    logger.info(f"非拒答错误样本分析 ({non_refusal_total}个):")
    logger.info(f"  - 召回失败: {non_refusal_stats['recall_failure']} ({non_refusal_failure_rate})")
    logger.info(f"  - 提取错误: {non_refusal_stats['recall_success']} ({non_refusal_success_rate})")
    if non_refusal_errors:
        logger.info(f"  - 分析错误: {len(non_refusal_errors)}个")
    
    logger.info(f"\n拒答样本分析 ({refusal_total}个):")
    logger.info(f"  - 召回失败（正确拒答）: {refusal_stats['recall_failure']} ({refusal_failure_rate})")
    logger.info(f"  - 召回成功（错误拒答）: {refusal_stats['recall_success']} ({refusal_success_rate})")
    if refusal_errors:
        logger.info(f"  - 分析错误: {len(refusal_errors)}个")
    
    logger.info("\n" + "="*60)
    logger.info("关键发现:")
    logger.info("1. 非拒答错误中，提取错误占比: " + non_refusal_success_rate)
    logger.info("2. 拒答样本中，错误拒答占比: " + refusal_success_rate)
    logger.info("3. 总体召回失败率: " + summary["性能指标"]["总体召回失败率"])
    logger.info("="*60)
    
    return summary

def main():
    """主函数"""
    logger.info("开始多线程召回成功率分析...")
    
    try:
        # 设置线程数（根据你的网络和API限制调整）
        # 建议开始使用3-5个线程，避免触发API限制
        max_workers = 5
        
        non_refusal_data, refusal_data = analyze_and_save_samples_multithread(max_workers)
        
        logger.info("分析完成！")
        logger.info(f"非拒答错误样本文件: {NON_REFUSAL_FILE}")
        logger.info(f"拒答样本文件: {REFUSAL_FILE}")
        logger.info(f"统计摘要文件: {SUMMARY_FILE}")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()

# ============================================================评估错误原因============================================================
# import os
# import json
# import logging
# import concurrent.futures
# from typing import List, Dict, Any
# from tqdm import tqdm
# from openai import OpenAI

# # ==================== 配置 ====================
# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test1/test_nodecide/evaluation_results_analysis/refusal.json"
# OUTPUT_FILE = INPUT_FILE.replace(".json", "_error_type_classified.json")
# OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename=os.path.join(OUTPUT_DIR, 'error_type_classification.log'),
#     filemode='a',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# client = OpenAI(
#     api_key="__SET_VIA_ENVIRONMENT__",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )

# MODEL_NAME = "deepseek-v3"
# MAX_WORKERS = 5


# # ==================== 构建你指定的 Prompt ====================
# def build_error_type_prompt(question: str, reference_answer: str, model_answer: str, retrieved_docs: List[Dict]) -> str:
#     doc_texts = []
#     for i, doc in enumerate(retrieved_docs[:5], 1):
#         title = doc.get("title", "").strip()
#         content = doc.get("content_preview", "").strip()
#         if content:
#             doc_texts.append(f"【文档{i}】{title}\n{content}")
    
#     retrieved_content = "\n\n".join(doc_texts) if doc_texts else "无召回内容"

#     # 注意：所有字面量的 { 和 } 必须写成 {{ 和 }}
#     return f"""你是一个新闻问答系统评估专家。

# 你的任务不是判断模型是否“像参考答案”，
# 而是判断模型回答是否正确使用了召回文档中的事实性证据。

# 注意：
# - 参考答案只是构造测试集时使用的一个示例答案，
#   在某些问题中，文档中可能存在多个同样合法的答案。
# - 判断时应以“召回文档中的事实”为最高依据，
#   而不是参考答案的表述。

# ## 错误类型定义
# 0. **证据缺失 (evidence_absent)**  
#    在召回的文档中，不存在能够直接回答该问题的明确事实性证据，
#    即使模型给出了一个看似合理的回答，
#    也无法在文档中找到事实支持。

# 1. **证据漂移 (evidence_drift)**  
#    召回文档中存在能够直接回答问题的关键信息，
#    但模型未使用最相关的证据，或抄错/提取错误了关键事实
#    （如日期、数字、名称、事件对应关系等）。

# 2. **多答案选择 (multiple_valid_answers)**  
#    召回文档中存在多个互不冲突、均可回答该问题的事实性答案，
#    参考答案只是其中之一，而模型回答基于文档中的另一个合法答案，
#    从文档角度看是正确且合理的。

# 3. **信息遗漏 (partial_information)**  
#    模型回答基于正确的文档证据，
#    但仅覆盖了部分关键信息，遗漏了问题所要求的重要细节，
#    或表述过于笼统、模糊。

# ## 输入信息
# **问题**:
# {question}

# **参考答案（仅供理解问题，不作为唯一正确标准）**:
# {reference_answer}

# **模型回答**:
# {model_answer}

# **召回的文档**:
# {retrieved_content}

# ## 判断步骤（严格按顺序）
# 1. 判断召回文档中是否存在能够直接回答问题的明确事实性证据。
#    - 若不存在，直接归类为：证据缺失（evidence_absent），停止判断。
# 2. 若存在证据，优先判断模型回答是否发生了证据漂移：
#    - 是否忽略了最直接的答案证据
#    - 是否使用了不恰当的句子
#    - 是否抄错了关键事实
#    若是，归类为：证据漂移。
# 3. 若未发生证据漂移，判断问题是否存在多个合法答案，
#    且模型回答是否命中了其中任意一个，
#    若是，归类为：多答案选择。
# 4. 若模型回答基于正确证据，但信息不完整或表述模糊，
#    归类为：信息遗漏。

# ## 输出要求
# 严格输出以下 JSON，不要任何额外文本：
# {{
#   "type": "evidence_absent/evidence_drift/multiple_valid_answers/partial_information",
#   "reason": "简短原因"
# }}
# """


# # ==================== 分类并映射 correctness ====================
# def classify_and_update(sample: Dict[str, Any]) -> Dict[str, Any]:
#     try:
#         if not sample.get("recall_success", False):
#             return sample  # 跳过 recall 失败的样本

#         question = sample.get("question", "")
#         reference_answer = sample.get("answer", "")  # 仅用于提示，不影响判断逻辑
#         model_answer = sample.get("model_answer", "")
#         retrieved_docs = sample.get("retrieval_info", {}).get("retrieved_docs", [])

#         prompt = build_error_type_prompt(question, reference_answer, model_answer, retrieved_docs)

#         for attempt in range(3):
#             try:
#                 response = client.chat.completions.create(
#                     model=MODEL_NAME,
#                     messages=[
#                         {"role": "system", "content": "你是一个严谨的评估专家。只输出合法JSON，不要解释、不要Markdown、不要多余文本。"},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=200,
#                     temperature=0.0,
#                     timeout=30
#                 )

#                 text = response.choices[0].message.content.strip()
#                 # 清理可能的 Markdown
#                 if text.startswith("```json"):
#                     text = text[7:].split("```")[0]
#                 elif text.startswith("```"):
#                     text = text[3:].split("```")[0]

#                 result = json.loads(text)
#                 error_type = result.get("type", "").strip()
#                 reason = result.get("reason", "").strip()

#                 # 映射 correctness
#                 if error_type == "multiple_valid_answers":
#                     correct = True
#                     score = 5
#                     error_types_list = []
#                 elif error_type == "evidence_drift":
#                     correct = False
#                     score = 2
#                     error_types_list = ["evidence_drift"]
#                 elif error_type == "partial_information":
#                     correct = False
#                     score = 3
#                     error_types_list = ["partial_information"]
#                 elif error_type == "evidence_absent":
#                     correct = False
#                     score = 1
#                     error_types_list = ["evidence_absent"]
#                 else:
#                     # fallback
#                     correct = False
#                     score = 1
#                     error_types_list = ["unknown"]
#                     reason = f"未知类型: {error_type}"

#                 new_correctness = {
#                     "refusal": len(model_answer.strip()) < 5 or any(kw in model_answer for kw in ["无法回答", "不知道", "不清楚"]),
#                     "correct": correct,
#                     "score": score,
#                     "error_types": error_types_list,
#                     "reason": reason
#                 }

#                 updated = sample.copy()
#                 updated["evaluation_results"]["correctness"] = new_correctness
#                 updated["error_type_classification"] = error_type
#                 return updated

#             except Exception as e:
#                 if attempt == 2:
#                     logger.warning(f"样本 {sample.get('source_id')} 分类失败: {e}")
#                     sample["classification_error"] = str(e)
#                     return sample
#                 continue

#     except Exception as e:
#         logger.error(f"处理异常 {sample.get('source_id')}: {e}")
#         sample["classification_error"] = str(e)
#         return sample


# # ==================== 主函数 ====================
# def main():
#     with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     to_eval = [s for s in data if s.get("recall_success") is True]
#     skip = [s for s in data if s.get("recall_success") is not True]

#     logger.info(f"总样本数: {len(data)}，需分类样本数: {len(to_eval)}")

#     updated_samples = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(classify_and_update, s): s for s in to_eval}
#         with tqdm(total=len(to_eval), desc="错误类型分类") as pbar:
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     res = future.result()
#                     updated_samples.append(res)
#                 except Exception as e:
#                     s = futures[future]
#                     logger.error(f"线程异常 {s.get('source_id')}: {e}")
#                     s["classification_error"] = "Thread crash"
#                     updated_samples.append(s)
#                 pbar.update(1)

#     final_data = updated_samples + skip

#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         json.dump(final_data, f, ensure_ascii=False, indent=2)

#     logger.info(f"✅ 分类完成，结果保存至: {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()
# =============================================================数据补充===============================================================
# import json
# from pymilvus import connections, Collection, DataType

# # === 1. 连接 Milvus ===
# print("🔌 正在连接 Milvus...")
# connections.connect(alias="default", host="122.226.69.234", port="3042")
# collection = Collection("news_collection_v2")

# # 构建 doc_id -> full_doc 的缓存（避免重复查询）
# doc_cache = {}

# def get_full_doc_by_doc_id(doc_id: str):
#     if doc_id in doc_cache:
#         return doc_cache[doc_id]
    
#     # Milvus 精确查询：使用 term 查询
#     expr = f'doc_id == "{doc_id}"'
#     try:
#         results = collection.query(
#             expr=expr,
#             output_fields=["title", "content", "publish_date_str"]
#         )
#         if results:
#             full_doc = results[0]
#             doc_cache[doc_id] = full_doc
#             return full_doc
#         else:
#             print(f"⚠️ 未找到 doc_id: {doc_id}")
#             return None
#     except Exception as e:
#         print(f"❌ 查询 doc_id {doc_id} 出错: {e}")
#         return None

# # === 2. 读取原始 JSON 文件 ===
# input_file = '/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test1/test/evaluation_results.json'  # 替换为你的文件名
# output_file = '/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test1/test/evaluation_results.json'

# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# print(f"📄 共加载 {len(data)} 条样本")

# # === 3. 遍历并补充完整 content ===
# total_retrieved_docs = 0
# filled_count = 0

# for item in data:
#     retrieved_docs = item.get("retrieval_info", {}).get("retrieved_docs", [])
#     for doc in retrieved_docs:
#         total_retrieved_docs += 1
#         doc_id = doc.get("doc_id")
#         if not doc_id:
#             continue
        
#         full_doc = get_full_doc_by_doc_id(doc_id)
#         if full_doc:
#             # 替换 preview 为完整 content
#             doc["content_preview"] = full_doc["content"]  # 新增完整 content 字段
#             # 可选：删除 content_preview 或保留
#             # del doc["content_preview"]
#             filled_count += 1
#         else:
#             doc["content_preview"] = ""  # 或保留原 preview

# print(f"✅ 成功填充 {filled_count} / {total_retrieved_docs} 个召回文档的完整内容")

# # === 4. 保存结果 ===
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"💾 结果已保存至 {output_file}")


# =====================================================================投票===========================================================
# import json
# import os

# # ============================================================================
# # 配置：替换为你自己的三个 JSON 文件路径
# # ============================================================================
# JSON_FILES = [
#     "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test/test_time_nodecide_relaxed/evaluation_results1.json",
#     "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test/test_time_nodecide_relaxed_qwen/evaluation_results.json",
#     "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test/test_time_nodecide/evaluation_results.json"
# ]

# OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test/test_nodecide_toupiao"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_consensus.json")
# AMBIGUOUS_OUTPUT = os.path.join(OUTPUT_DIR, "ambiguous_cases.json")


# def load_samples_by_key(filepath):
#     """
#     加载 JSON 文件，返回字典：
#     key: (str(source_id), question)
#     value: 原始样本 dict
#     """
#     with open(filepath, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     sample_dict = {}
#     for item in data:
#         sid = str(item.get("source_id", ""))
#         q = item.get("question", "").strip()
#         key = (sid, q)
#         # 如果 key 重复，后面的会覆盖前面的（可加日志警告）
#         sample_dict[key] = item
#     return sample_dict

# def extract_correctness(sample):
#     corr = sample["evaluation_results"]["correctness"]
#     return {
#         "refusal": bool(corr["refusal"]),
#         "score": float(corr["score"])
#     }

# def main():
#     # 加载三个模型的结果，按 (source_id, question) 建索引
#     model_datas = [load_samples_by_key(fp) for fp in JSON_FILES]
    
#     # 获取所有唯一键（并集）
#     all_keys = set()
#     for md in model_datas:
#         all_keys.update(md.keys())
    
#     # 获取三个模型都包含的键（交集）
#     common_keys = set(model_datas[0].keys())
#     for md in model_datas[1:]:
#         common_keys &= set(md.keys())
    
#     consensus = []
#     ambiguous = []

#     # 处理三个模型都有的样本（可能一致）
#     for key in sorted(common_keys):
#         candidates = [md[key] for md in model_datas]
#         corr_list = [extract_correctness(cand) for cand in candidates]

#         refusals = [c["refusal"] for c in corr_list]
#         scores = [c["score"] for c in corr_list]

#         # 规则1: 全拒答
#         if all(r is True for r in refusals):
#             consensus.append(candidates[0])
#         # 规则2: 全非拒答 + 全高分 (>3)
#         elif all(r is False for r in refusals) and all(s > 3 for s in scores):
#             consensus.append(candidates[0])
#         # 规则3: 全非拒答 + 全低分 (<=3)
#         elif all(r is False for r in refusals) and all(s <= 3 for s in scores):
#             consensus.append(candidates[0])
#         else:
#             # 规则4: 不一致 → ambiguous
#             ambiguous.append(candidates[0])

#     # 处理缺失样本（只在部分模型中出现）
#     missing_keys = all_keys - common_keys
#     for key in sorted(missing_keys):
#         # 任选一个存在的作为代表
#         rep = None
#         for md in model_datas:
#             if key in md:
#                 rep = md[key]
#                 break
#         if rep:
#             ambiguous.append(rep)

#     # 保存结果
#     with open(FINAL_OUTPUT, 'w', encoding='utf-8') as f:
#         json.dump(consensus, f, ensure_ascii=False, indent=2)
    
#     with open(AMBIGUOUS_OUTPUT, 'w', encoding='utf-8') as f:
#         json.dump(ambiguous, f, ensure_ascii=False, indent=2)

#     total_processed = len(consensus) + len(ambiguous)
#     print(f"✅ 主结果（一致判定）: {len(consensus)} 条")
#     print(f"⚠️ 争议/缺失/不一致: {len(ambiguous)} 条")
#     print(f"📊 总处理样本数（基于 question + source_id）: {total_processed}")
#     print(f"💾 结果已保存至: {OUTPUT_DIR}")

# if __name__ == "__main__":
#     main()

# ==================================================合并数据用于分析召回==================================
# # import json
# import json
# import sys

# def merge_by_question(file_eval_path, file_retrieval_path, output_path):
#     """
#     通过 question 字段合并评估结果文件和检索详情文件。
    
#     参数:
#         file_eval_path: 包含 evaluation_results 的文件路径 (文件 A)
#         file_retrieval_path: 包含 retrieval_info 的文件路径 (文件 B)
#         output_path: 输出文件路径
#     """
    
#     print(f"正在读取评估结果文件: {file_eval_path}")
#     try:
#         with open(file_eval_path, 'r', encoding='utf-8') as f:
#             data_eval = json.load(f)
#     except Exception as e:
#         print(f"❌ 读取评估文件失败: {e}")
#         return

#     print(f"正在读取检索详情文件: {file_retrieval_path}")
#     try:
#         with open(file_retrieval_path, 'r', encoding='utf-8') as f:
#             data_retrieval = json.load(f)
#     except Exception as e:
#         print(f"❌ 读取检索文件失败: {e}")
#         return

#     # 确保数据是列表格式
#     if not isinstance(data_eval, list):
#         print("⚠️ 警告: 评估文件不是列表格式，尝试包裹为列表")
#         data_eval = [data_eval]
    
#     if not isinstance(data_retrieval, list):
#         print("⚠️ 警告: 检索文件不是列表格式，尝试包裹为列表")
#         data_retrieval = [data_retrieval]

#     # 建立检索文件的索引：Key = question, Value = 整条记录
#     # 注意：如果有重复的 question，后面的会覆盖前面的。如果数据有重复问题，需要更复杂的逻辑。
#     retrieval_map = {}
#     for item in data_retrieval:
#         q = item.get('question')
#         if q:
#             retrieval_map[q] = item
#         else:
#             print(f"⚠️ 警告: 检索文件中有一条数据缺少 'question' 字段，已跳过: {item.get('source_id', 'Unknown')}")

#     merged_data = []
#     matched_count = 0
#     missing_count = 0

#     print(f"开始合并... (评估数据量: {len(data_eval)}, 检索数据量: {len(data_retrieval)})")

#     for eval_item in data_eval:
#         question = eval_item.get('question')
        
#         if not question:
#             print(f"⚠️ 警告: 评估文件中有一条数据缺少 'question' 字段，已跳过: {eval_item.get('source_id', 'Unknown')}")
#             continue

#         # 构建基础结果 (基于评估文件)
#         # 目标格式要求: source_id, question, answer, model_answer, refusal_decision, evaluation_results
        
#         # 注意：用户提供的示例中，文件A的answer字段叫 "answer"，文件B叫 "standard_answer"
#         # 目标格式统一叫 "answer"。这里优先使用文件A的 "answer"，如果没有则尝试文件B的 "standard_answer"
#         final_answer = eval_item.get('answer')
        
#         merged_item = {
#             "source_id": eval_item.get('source_id'),
#             "question": question,
#             "answer": final_answer,
#             "model_answer": eval_item.get('model_answer'),
#             "evaluation_results": eval_item.get('evaluation_results'),
#             "refusal_decision": None  # 默认置空
#         }

#         # 尝试在检索文件中查找匹配的 question
#         if question in retrieval_map:
#             retrieval_item = retrieval_map[question]
            
#             # 提取 retrieval_info 并重命名为 refusal_decision
#             retrieval_info = retrieval_item.get('retrieval_info')
            
#             if retrieval_info:
#                 merged_item['refusal_decision'] = retrieval_info
#                 matched_count += 1
#             else:
#                 # 如果找到了记录但没有 retrieval_info 字段
#                 merged_item['refusal_decision'] = {}
#                 print(f"⚠️ 提示: 问题 '{question[:20]}...' 在检索文件中存在，但缺少 'retrieval_info' 字段")
#         else:
#             missing_count += 1
#             print(f"⚠️ 未匹配: 问题 '{question[:30]}...' 在检索文件中未找到。")

#         merged_data.append(merged_item)

#     # 保存结果
#     print(f"\n合并完成:")
#     print(f"  - 成功匹配: {matched_count} 条")
#     print(f"  - 未匹配 (缺失检索信息): {missing_count} 条")
#     print(f"  - 总输出条数: {len(merged_data)} 条")
    
#     print(f"正在写入输出文件: {output_path}")
#     try:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(merged_data, f, ensure_ascii=False, indent=2)
#         print("✅ 文件保存成功!")
#     except Exception as e:
#         print(f"❌ 保存文件失败: {e}")

# if __name__ == "__main__":
#     # 使用方法：
#     # python merge_script.py <评估文件路径> <检索文件路径> <输出文件路径>
    
#     if len(sys.argv) == 4:
#         eval_file = sys.argv[1]
#         retrieval_file = sys.argv[2]
#         out_file = sys.argv[3]
#     else:
#         # 如果没有命令行参数，请在此处修改默认文件名
#         print("未提供命令行参数，请使用默认文件名或按格式运行: python script.py eval.json retrieval.json output.json")
#         eval_file = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk_api/ds_450_notime/evaluation_results.json"       # 替换为你的文件 A 路径
#         retrieval_file = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk_api/ds_450_notime.json"   # 替换为你的文件 B 路径
#         out_file = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk_api/ds_450_notime/evaluation_results.json"              # 替换为你想要的输出路径

#     merge_by_question(eval_file, retrieval_file, out_file)