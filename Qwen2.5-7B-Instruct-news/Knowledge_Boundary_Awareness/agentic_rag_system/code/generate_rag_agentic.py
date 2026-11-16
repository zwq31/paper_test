"""
使用Agentic RAG的新闻问答系统主程序
集成查询改写、迭代检索、答案验证等高级功能
"""

import os
import json
import torch
import logging
import asyncio
import re  # 添加这行导入
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# 导入自定义模块
from utils.keywords_extractor import NewsKeywordExtractor, load_stopwords
from utils.time_extractor import extract_time_from_question
from utils.rag_utils import (
    build_rag_answer_messages, get_random_rejection_response
)
from utils.model_utils import load_model
from agentic_rag_system import AgenticRAGSystem
from advanced_reranker import get_reranker

# 导入原有的检索管理器
from generate_rag_time import HybridRetrievalManager

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_agentic_rag.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 控制台输出
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# ----------------------------
# 配置路径和参数
# ----------------------------
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/agentic_rag_system/data/test/test.json"
STOPWORDS_PATH = "/mnt/project/milvus/data/hit_stopwords.txt"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# Milvus配置
MILVUS_HOST = "122.226.69.234"
MILVUS_PORT = "3042"

# ----------------------------
# LangChain LLM包装器 (增强版)
# ----------------------------
class AsyncHuggingFaceLLM(Runnable):
    """异步HuggingFace LLM包装器,兼容LangChain Runnable接口"""
    
    def __init__(self, model, tokenizer, max_new_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        logger.info(f"LLM包装器初始化: max_new_tokens={max_new_tokens}")
    
    def invoke(self, input_data: Any, config: Dict = None) -> str:
        """LangChain兼容的同步调用方法"""
        prompt = self._extract_prompt(input_data)
        return self._generate_sync(prompt)
    
    async def ainvoke(self, input_data: Any, config: Dict = None) -> str:
        """LangChain兼容的异步调用方法"""
        prompt = self._extract_prompt(input_data)
        return await self._generate_async(prompt)
    
    def _extract_prompt(self, input_data: Any) -> str:
        """从输入数据中提取提示文本"""
        if isinstance(input_data, dict):
            # 检查不同的键
            if "prompt" in input_data:
                prompt = input_data["prompt"]
            elif "text" in input_data:
                prompt = input_data["text"]
            elif "question" in input_data:
                prompt = input_data["question"]
            elif "messages" in input_data:
                # 处理聊天格式
                messages = input_data["messages"]
                if messages and isinstance(messages, list):
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except Exception as e:
                        logger.warning(f"应用聊天模板失败: {e}, 使用原始消息")
                        prompt = "\n".join([str(msg) for msg in messages])
                else:
                    prompt = str(messages)
            else:
                # 尝试获取第一个字符串值
                for key, value in input_data.items():
                    if isinstance(value, str) and value.strip():
                        prompt = value
                        break
                else:
                    # 没有找到字符串值，使用整个字典
                    prompt = json.dumps(input_data, ensure_ascii=False)
        elif isinstance(input_data, str):
            prompt = input_data
        else:
            prompt = str(input_data)
        
        logger.debug(f"LLM输入提取: {prompt[:200]}...")
        return prompt
    
    def _generate_sync(self, prompt: str) -> str:
        """同步生成"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].size(1):],
                skip_special_tokens=True
            )
            
            # 清理生成文本
            generated_text = generated_text.strip()
            logger.debug(f"LLM生成完成: {generated_text[:200]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return f"生成失败: {str(e)[:100]}"
    
    async def _generate_async(self, prompt: str) -> str:
        """异步生成"""
        def _generate():
            return self._generate_sync(prompt)
        
        try:
            result = await asyncio.to_thread(_generate)
            return result
        except Exception as e:
            logger.error(f"异步生成失败: {e}")
            return f"异步生成失败: {str(e)[:100]}"

# ----------------------------
# 答案生成函数
# ----------------------------
async def generate_answer_with_docs(
    model,
    tokenizer,
    question: str,
    documents: List[Dict]
) -> str:
    """使用检索文档生成答案"""
    if not documents:
        logger.warning(f"没有检索到文档来回答问题: {question}")
        return get_random_rejection_response()
    
    try:
        # 构建消息
        messages = build_rag_answer_messages(question, documents)
        
        # 应用chat template
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 异步生成答案
        def _generate():
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            prompt_len = inputs['input_ids'].size(1)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated_tokens = outputs[0, prompt_len:]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return answer.strip()
        
        answer = await asyncio.to_thread(_generate)
        logger.info(f"生成答案完成，长度: {len(answer)}")
        return answer
        
    except Exception as e:
        logger.error(f"生成答案失败: {e}", exc_info=True)
        return f"生成答案时出现错误: {str(e)[:100]}"

# ----------------------------
# 清理验证结果
# ----------------------------
def clean_verification_result(verification: Dict) -> Dict:
    """清理验证结果，移除JSON标记等"""
    if not verification:
        return verification
    
    # 清理reasoning字段
    if "reasoning" in verification and verification["reasoning"]:
        reasoning = verification["reasoning"]
        # 移除可能的JSON代码块标记
        reasoning = reasoning.replace("```json", "").replace("```", "").strip()
        # 移除多余的空格和换行
        reasoning = re.sub(r'\s+', ' ', reasoning)
        verification["reasoning"] = reasoning
    
    # 确保confidence是浮点数
    if "confidence" in verification:
        try:
            verification["confidence"] = float(verification["confidence"])
        except (ValueError, TypeError):
            verification["confidence"] = 0.0
    
    return verification

# ----------------------------
# 主函数
# ----------------------------
async def process_single_question(
    item: Dict,
    idx: int,
    agentic_rag: AgenticRAGSystem,
    model,
    tokenizer
) -> Dict:
    """处理单个问题"""
    question = item.get("question", "").strip()
    if not question:
        logger.warning(f"第{idx}条数据问题为空，跳过")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"处理第 {idx+1} 个问题: {question}")
    logger.info(f"{'='*60}")
    
    try:
        # 提取时间信息
        time_range = extract_time_from_question(question)
        if time_range:
            logger.info(f"提取到时间范围: {time_range}")
        
        # 定义答案生成函数
        async def gen_func(q, docs):
            return await generate_answer_with_docs(model, tokenizer, q, docs)
        
        # 执行Agentic RAG
        logger.info("开始Agentic RAG处理...")
        result = await agentic_rag.agentic_retrieve_and_answer(
            question=question,
            time_range=time_range,
            model_generate_func=gen_func
        )
        
        # 清理验证结果
        verification = result.get("process_metadata", {}).get("verification", {})
        if verification:
            verification = clean_verification_result(verification)
        
        # 构建输出
        output_item = {
            "source_id": item.get("source_id", idx),
            "question": question,
            "standard_answer": item.get("answer", ""),
            "model_answer": result.get("final_answer", "无答案").strip(),
            "confidence": float(result.get("confidence", 0.0)),
            "retrieved_docs_count": len(result.get("retrieved_docs", [])),
            "retrieved_docs_preview": [
                {
                    "doc_id": doc.get("doc_id", f"doc_{i}"),
                    "title": doc.get("title", "无标题")[:50],
                    "publish_date": doc.get("publish_date_str", doc.get("publish_date", "")),
                    "rrf_score": round(float(doc.get("rrf_score", 0)), 4)
                }
                for i, doc in enumerate(result.get("retrieved_docs", [])[:5])
            ],
            "process_metadata": {
                "query_plan": result.get("process_metadata", {}).get("query_plan", None),
                "retrieval": result.get("process_metadata", {}).get("retrieval", []),
                "verification": verification
            }
        }
        
        # 记录处理结果
        confidence = result.get("confidence", 0.0)
        docs_count = len(result.get("retrieved_docs", []))
        logger.info(f"第 {idx+1} 个问题处理完成 - 置信度: {confidence:.3f}, 检索文档数: {docs_count}")
        return output_item
        
    except Exception as e:
        logger.error(f"第{idx}条数据处理失败: {e}", exc_info=True)
        # 返回错误信息但结构完整
        return {
            "source_id": item.get("source_id", idx),
            "question": question,
            "standard_answer": item.get("answer", ""),
            "model_answer": f"处理错误: {str(e)[:100]}",
            "confidence": 0.0,
            "retrieved_docs_count": 0,
            "retrieved_docs_preview": [],
            "process_metadata": {
                "error": str(e),
                "query_plan": None,
                "retrieval": [],
                "verification": {
                    "is_supported": False,
                    "confidence": 0.0,
                    "missing_info": ["处理过程中出现异常"],
                    "reasoning": f"异常: {str(e)[:100]}"
                }
            }
        }


async def main():
    """主函数 - 处理一条保存一条（标准 JSON 格式）"""
    import jieba
    
    logger.info("初始化分词器...")
    jieba.initialize()
    
    # 加载停用词和关键词提取器
    stopwords = load_stopwords(STOPWORDS_PATH)
    keyword_extractor = NewsKeywordExtractor(stopwords)
    logger.info(f"已加载 {len(stopwords)} 个停用词")
    
    # 加载模型
    logger.info("加载语言模型...")
    model, tokenizer = load_model(MODEL_BASE_PATH)
    logger.info(f"模型加载完成，设备: {model.device}")
    
    # 初始化检索管理器
    logger.info("初始化检索系统...")
    retrieval_manager = HybridRetrievalManager(MILVUS_HOST, MILVUS_PORT)
    retrieval_manager.initialize()
    logger.info("检索系统初始化完成")
    
    # 创建LangChain LLM包装器
    llm_wrapper = AsyncHuggingFaceLLM(model, tokenizer, max_new_tokens=512)
    logger.info("LangChain LLM包装器创建完成")
    
    # 初始化重排序器
    logger.info("初始化重排序器...")
    reranker = get_reranker("BAAI/bge-reranker-large")
    logger.info("重排序器初始化完成")
    
    # 初始化Agentic RAG系统
    logger.info("初始化Agentic RAG系统...")
    agentic_rag = AgenticRAGSystem(
        llm=llm_wrapper,
        retrieval_manager=retrieval_manager,
        keyword_extractor=keyword_extractor,
        reranker=reranker
    )
    logger.info("Agentic RAG系统初始化完成")
    
    # 加载输入数据
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    logger.info(f"共加载 {len(input_data)} 个问题")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    logger.info(f"输出目录已创建: {os.path.dirname(OUTPUT_PATH)}")
    
    # 初始化结果列表
    results = []

    # 如果输出文件已存在，尝试加载已有结果以支持断点续跑
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    results = existing
                    logger.info(f"检测到已有结果文件，已加载 {len(results)} 条结果")
                else:
                    logger.warning("现有输出文件格式异常，将重新开始")
        except Exception as e:
            logger.warning(f"加载已有结果失败: {e}，将重新开始")

    # 获取已处理的 source_id 集合（用于跳过）
    processed_source_ids = {item.get("source_id") for item in results if item.get("source_id") is not None}
    logger.info(f"已处理 {len(processed_source_ids)} 个不同source_id")

    # 处理所有问题（支持断点续跑）
    for idx, item in enumerate(tqdm(input_data, desc="处理问题")):
        source_id = item.get("source_id", idx)
        
        # 跳过已处理的项（断点续跑）
        if source_id in processed_source_ids:
            logger.info(f"跳过已处理的 source_id: {source_id}")
            continue

        output_item = await process_single_question(item, idx, agentic_rag, model, tokenizer)
        if output_item:
            results.append(output_item)
            # ✅ 每处理一条，立即保存当前全部结果（覆盖写入）
            try:
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
                    json.dump(results, f_out, ensure_ascii=False, indent=2)
                logger.info(f"✅ 已保存第 {len(results)} 条结果 (source_id={source_id})")
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
        else:
            logger.warning(f"⚠️  跳过第 {idx} 条问题（处理失败）")

    logger.info(f"🎉 处理完成! 共处理 {len(results)} 个问题")
    logger.info(f"📁 结果保存至: {OUTPUT_PATH}")
    
    # 统计信息
    if results:
        # 只计算有置信度的结果
        confidences = [r["confidence"] for r in results if r["confidence"] > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            logger.info(f"📊 平均置信度: {avg_confidence:.3f} (基于{len(confidences)}个有效结果)")
        
        # 统计成功和失败的情况
        success_count = sum(1 for r in results if r.get("confidence", 0) > 0)
        error_count = sum(1 for r in results if "处理错误" in r.get("model_answer", ""))
        logger.info(f"📈 成功: {success_count}, 错误: {error_count}, 总计: {len(results)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
    except Exception as e:
        logger.error(f"主程序执行失败: {e}", exc_info=True)