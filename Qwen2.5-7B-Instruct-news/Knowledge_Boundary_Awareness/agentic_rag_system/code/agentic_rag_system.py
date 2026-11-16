"""
Agentic RAG System - 智能检索增强生成系统
实现查询改写、HyDE、自我反思等高级功能
"""

import json
import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ==================== 数据结构定义 ====================

class QueryType(Enum):
    """查询类型"""
    FACTUAL = "factual"  # 事实性问题
    TEMPORAL = "temporal"  # 时间相关问题
    COMPARATIVE = "comparative"  # 比较性问题
    CAUSAL = "causal"  # 因果性问题
    OPINION = "opinion"  # 观点性问题


@dataclass
class RetrievalResult:
    """检索结果"""
    documents: List[Dict[str, Any]]
    retrieval_method: str
    confidence: float
    metadata: Dict[str, Any]


class QueryPlan(BaseModel):
    """查询计划"""
    query_type: str = Field(description="查询类型")
    sub_queries: List[str] = Field(description="子查询列表", default_factory=list)
    search_strategy: str = Field(description="搜索策略", default="hybrid")
    needs_rewrite: bool = Field(description="是否需要改写", default=True)
    needs_hyde: bool = Field(description="是否需要HyDE", default=False)


class AnswerVerification(BaseModel):
    """答案验证结果"""
    is_supported: bool = Field(description="答案是否被支持", default=False)
    confidence: float = Field(description="置信度0-1", default=0.0)
    missing_info: List[str] = Field(description="缺失的信息", default_factory=list)
    reasoning: str = Field(description="推理过程", default="")

# ==================== 自定义解析器 ====================

class SafeJsonOutputParser(JsonOutputParser):
    """安全的JSON输出解析器，处理解析失败的情况"""
    
    def parse(self, text: str) -> Dict:
        try:
            # 尝试直接解析
            return super().parse(text)
        except Exception as e:
            logger.warning(f"JSON解析失败: {e}, 尝试清理文本...")
            # 清理文本，移除可能的markdown代码块标记
            cleaned_text = text.strip()
            
            # 移除 ```json 和 ``` 标记
            cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)
            cleaned_text = re.sub(r'```json\n', '', cleaned_text)
            cleaned_text = re.sub(r'```\n', '', cleaned_text)
            
            # 尝试找到第一个 { 和最后一个 }
            start = cleaned_text.find('{')
            end = cleaned_text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = cleaned_text[start:end]
                try:
                    return json.loads(json_str)
                except Exception as e2:
                    logger.warning(f"清理后JSON解析仍失败: {e2}")
            
            # 如果还是失败，返回默认结构
            logger.warning(f"无法解析JSON，返回默认结构。原始文本开头: {text[:200]}")
            return {
                "error": "解析失败",
                "raw_text": text[:500]
            }
    
    async def aparse(self, text: str) -> Dict:
        return self.parse(text)


# ==================== 专用解析器 ====================

class QueryRewriterParser(SafeJsonOutputParser):
    """查询改写专用解析器"""
    
    def parse(self, text: str) -> Dict:
        try:
            result = super().parse(text)
            
            # 确保改写查询是完整的
            if "rewritten_queries" in result:
                queries = result["rewritten_queries"]
                cleaned_queries = []
                
                for i, q in enumerate(queries):
                    if isinstance(q, str) and q.strip():
                        q_clean = q.strip()
                        # 确保查询是完整的句子
                        if len(q_clean) < 5:  # 太短，可能是截断的
                            logger.warning(f"查询太短或被截断: '{q_clean}'，跳过")
                            continue
                        
                        # 如果查询不以标点结束，添加句号
                        if q_clean and q_clean[-1] not in ['.', '?', '!', '。', '？', '！']:
                            q_clean = q_clean + '。'
                        
                        cleaned_queries.append(q_clean)
                
                # 如果清理后没有查询，使用原问题
                if not cleaned_queries:
                    cleaned_queries = ["原始问题"]
                
                result["rewritten_queries"] = cleaned_queries[:3]  # 最多3个
            
            return result
        except Exception as e:
            logger.warning(f"查询改写解析失败: {e}")
            return {
                "rewritten_queries": ["原始问题"],
                "key_entities": []
            }


# ==================== Agentic RAG 主类 ====================

class AgenticRAGSystem:
    """智能RAG系统"""
    
    def __init__(self, llm, retrieval_manager, keyword_extractor, reranker=None):
        """
        初始化Agentic RAG系统
        
        Args:
            llm: 语言模型（必须是Runnable）
            retrieval_manager: 检索管理器
            keyword_extractor: 关键词提取器
            reranker: 重排序器(可选)
        """
        self.llm = llm
        self.retrieval_manager = retrieval_manager
        self.keyword_extractor = keyword_extractor
        self.reranker = reranker
        
        # 初始化各个Agent组件
        self._init_query_analyzer()
        self._init_query_rewriter()
        self._init_hyde_generator()
        self._init_answer_verifier()
        self._init_query_decomposer()
    
    def _init_query_analyzer(self):
        """初始化查询分析器 - 使用RunnableSequence"""
        template = """你是一个查询分析专家。分析以下问题并输出JSON格式的查询计划。

问题: {question}

请分析:
1. 问题类型 (factual/temporal/comparative/causal/opinion)
2. 是否需要分解为子问题
3. 推荐的搜索策略
4. 是否需要查询改写
5. 是否适合使用HyDE

输出严格的JSON格式:
{{
    "query_type": "类型",
    "sub_queries": ["子问题1", "子问题2"],
    "search_strategy": "vector/keyword/hybrid",
    "needs_rewrite": true/false,
    "needs_hyde": true/false
}}

请确保输出有效的JSON格式，不要有其他文本。"""
        
        prompt = PromptTemplate.from_template(template)
        parser = SafeJsonOutputParser()
        
        self.query_analyzer = prompt | self.llm | parser
    
    def _init_query_rewriter(self):
        """初始化查询改写器 - 使用RunnableSequence"""
        template = """你是一个查询改写专家。根据原始问题生成3个不同的改写版本。

原始问题: {question}

改写要求:
1. 版本1: 保留原意，优化表达方式
2. 版本2: 使用同义词和相关术语扩展
3. 版本3: 从不同角度重新表述，突出关键信息

请确保每个改写版本都是完整的查询语句，能够独立使用。

输出JSON格式:
{{
    "rewritten_queries": [
        "改写版本1 - 完整查询语句",
        "改写版本2 - 完整查询语句", 
        "改写版本3 - 完整查询语句"
    ],
    "key_entities": ["关键实体1", "关键实体2"]
}}

请确保JSON格式正确，每个改写版本都是完整的句子。"""
        
        prompt = PromptTemplate.from_template(template)
        parser = QueryRewriterParser()
        
        self.query_rewriter = prompt | self.llm | parser
    
    def _init_hyde_generator(self):
        """初始化HyDE生成器 - 使用RunnableSequence"""
        template = """你是一个新闻文档生成专家。根据问题生成一篇假设性的新闻文档。

问题: {question}

生成一篇200-300字的新闻摘要，包含可能回答该问题的信息。
注意:
- 使用新闻语言风格
- 包含关键事实和细节
- 保持客观中立

假设性文档:"""
        
        prompt = PromptTemplate.from_template(template)
        parser = StrOutputParser()
        
        self.hyde_generator = prompt | self.llm | parser
    
    def _init_answer_verifier(self):
        """初始化答案验证器 - 使用RunnableSequence"""
        template = """你是一个答案验证专家。验证答案是否被检索到的文档充分支持。

问题: {question}

生成的答案: {answer}

检索到的文档:
{contexts}

请按照以下标准验证答案：
1. 检查答案中的每个关键事实是否都在文档中有明确支持
2. 标记任何没有文档支持的声明
3. 列出缺失的关键信息
4. 给出0-1之间的整体置信度分数

输出必须严格按照以下JSON格式，不要有任何额外的文本：
{{
    "is_supported": true或false,
    "confidence": 0.0到1.0之间的数字,
    "missing_info": ["缺失信息1", "缺失信息2"],
    "reasoning": "详细的推理过程，解释为什么给出这个判断"
}}

请确保只输出JSON，不要有其他内容。"""
        
        prompt = PromptTemplate.from_template(template)
        parser = SafeJsonOutputParser()
        
        self.answer_verifier = prompt | self.llm | parser
    
    def _init_query_decomposer(self):
        """初始化查询分解器 - 使用RunnableSequence"""
        template = """你是一个问题分解专家。将复杂问题分解为简单的子问题。

问题: {question}

如果这是一个复杂问题，将其分解为2-4个简单子问题。
如果是简单问题，返回原问题。

输出JSON格式:
{{
    "is_complex": true/false,
    "sub_questions": ["子问题1", "子问题2", ...]
}}

请确保输出有效的JSON格式，不要有其他文本。"""
        
        prompt = PromptTemplate.from_template(template)
        parser = SafeJsonOutputParser()
        
        self.query_decomposer = prompt | self.llm | parser
    
    # ==================== 核心流程方法 ====================
    
    async def analyze_query(self, question: str) -> QueryPlan:
        """分析查询"""
        try:
            result = await self.query_analyzer.ainvoke({"question": question})
            
            if not result or "error" in result:
                logger.warning(f"查询分析失败，使用默认计划: {result}")
                return QueryPlan(
                    query_type="factual",
                    sub_queries=[question],
                    search_strategy="hybrid",
                    needs_rewrite=True,
                    needs_hyde=False
                )
            
            # 确保所有字段都存在
            query_type = result.get("query_type", "factual")
            sub_queries = result.get("sub_queries", [])
            search_strategy = result.get("search_strategy", "hybrid")
            needs_rewrite = bool(result.get("needs_rewrite", True))
            needs_hyde = bool(result.get("needs_hyde", False))
            
            # 如果子查询为空，使用原问题
            if not sub_queries:
                sub_queries = [question]
            
            logger.info(f"查询分析结果: 类型={query_type}, 子查询数={len(sub_queries)}")
            return QueryPlan(
                query_type=query_type,
                sub_queries=sub_queries,
                search_strategy=search_strategy,
                needs_rewrite=needs_rewrite,
                needs_hyde=needs_hyde
            )
        except Exception as e:
            logger.error(f"查询分析失败: {e}", exc_info=True)
            return QueryPlan(
                query_type="factual",
                sub_queries=[question],
                search_strategy="hybrid",
                needs_rewrite=True,
                needs_hyde=False
            )
    
    async def rewrite_query(self, question: str, context: str = "") -> List[str]:
        """改写查询"""
        try:
            result = await self.query_rewriter.ainvoke({
                "question": question,
                "retrieval_context": context or "首次检索"
            })
            
            if not result or "error" in result:
                logger.warning(f"查询改写失败: {result}")
                return [question]
            
            queries = result.get("rewritten_queries", [])
            if not queries:
                queries = [question]
            
            logger.info(f"查询改写: {len(queries)}个版本")
            return queries
        except Exception as e:
            logger.error(f"查询改写失败: {e}")
            return [question]
    
    async def generate_hyde(self, question: str) -> str:
        """生成假设性文档"""
        try:
            hyde_doc = await self.hyde_generator.ainvoke({"question": question})
            logger.info(f"HyDE生成: {len(hyde_doc)}字")
            return hyde_doc
        except Exception as e:
            logger.error(f"HyDE生成失败: {e}")
            return question
    
    async def decompose_query(self, question: str) -> List[str]:
        """分解复杂查询"""
        try:
            result = await self.query_decomposer.ainvoke({"question": question})
            
            if not result or "error" in result:
                logger.warning(f"查询分解失败: {result}")
                return [question]
            
            if result.get("is_complex", False):
                sub_questions = result.get("sub_questions", [question])
                logger.info(f"问题分解: {len(sub_questions)}个子问题")
                return sub_questions
            return [question]
        except Exception as e:
            logger.error(f"查询分解失败: {e}")
            return [question]
    
    async def verify_answer_with_retry(
        self, 
        question: str, 
        answer: str, 
        contexts: List[Dict],
        max_retries: int = 2
    ) -> AnswerVerification:
        """带重试机制的答案验证"""
        for attempt in range(max_retries):
            try:
                verification = await self.verify_answer(question, answer, contexts)
                
                # 如果验证结果看起来合理，直接返回
                if verification.reasoning and len(verification.reasoning) > 10:
                    return verification
                    
                # 否则重试
                logger.warning(f"验证结果不完整，重试 {attempt + 1}/{max_retries}")
                await asyncio.sleep(0.5)  # 短暂延迟
                
            except Exception as e:
                logger.error(f"验证尝试 {attempt + 1} 失败: {e}")
                if attempt == max_retries - 1:  # 最后一次尝试也失败
                    return AnswerVerification(
                        is_supported=False,
                        confidence=0.0,
                        missing_info=[f"验证失败: {str(e)[:100]}"],
                        reasoning=f"验证过程出现异常: {str(e)[:100]}"
                    )
                await asyncio.sleep(1)  # 重试前等待
        
        # 不应该执行到这里
        return AnswerVerification(
            is_supported=False,
            confidence=0.0,
            missing_info=["验证过程异常"],
            reasoning="验证过程出现未知异常"
        )
    
    async def verify_answer(
        self, 
        question: str, 
        answer: str, 
        contexts: List[Dict]
    ) -> AnswerVerification:
        """验证答案"""
        try:
            if not contexts:
                logger.warning("验证答案时没有提供上下文文档")
                return AnswerVerification(
                    is_supported=False,
                    confidence=0.0,
                    missing_info=["没有检索到相关文档"],
                    reasoning="没有检索到相关文档用于验证答案"
                )
            
            # 构建上下文文本
            contexts_text = "\n\n".join([
                f"文档{i+1} (标题: {doc.get('title', '无标题')}):\n{doc.get('content', '')[0:300]}..."
                for i, doc in enumerate(contexts[:5])
            ])
            
            logger.info(f"开始验证答案，上下文长度: {len(contexts_text)}")
            
            result_dict = await self.answer_verifier.ainvoke({
                "question": question,
                "answer": answer,
                "contexts": contexts_text
            })
            
            # 确保result_dict不是None
            if not result_dict:
                logger.error("答案验证返回了None")
                return AnswerVerification(
                    is_supported=False,
                    confidence=0.0,
                    missing_info=["验证过程返回空结果"],
                    reasoning="答案验证过程返回了空结果"
                )
            
            # 处理可能的错误
            if "error" in result_dict:
                logger.warning(f"答案验证返回错误: {result_dict['error']}")
                return AnswerVerification(
                    is_supported=False,
                    confidence=0.0,
                    missing_info=["验证过程出错"],
                    reasoning=f"验证出错: {result_dict.get('error', '未知错误')}"
                )
            
            # 确保所有必需的字段都存在
            is_supported = result_dict.get("is_supported", False)
            confidence = result_dict.get("confidence", 0.0)
            missing_info = result_dict.get("missing_info", [])
            reasoning = result_dict.get("reasoning", "无详细推理")
            
            # 清理推理文本
            reasoning = reasoning.strip()
            # 移除可能的JSON标记
            reasoning = re.sub(r'```json\s*', '', reasoning)
            reasoning = re.sub(r'```\s*', '', reasoning)
            
            # 验证confidence在0-1范围内
            try:
                confidence = float(confidence)
                if confidence < 0 or confidence > 1:
                    confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.0
            
            # 确保missing_info是列表
            if not isinstance(missing_info, list):
                missing_info = [str(missing_info)] if missing_info else []
            
            logger.info(f"答案验证完成: 是否支持={is_supported}, 置信度={confidence}")
            
            return AnswerVerification(
                is_supported=is_supported,
                confidence=confidence,
                missing_info=missing_info,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"答案验证过程失败: {e}", exc_info=True)
            return AnswerVerification(
                is_supported=False,
                confidence=0.0,
                missing_info=[f"验证过程异常: {str(e)[:100]}"],
                reasoning=f"答案验证过程出现异常: {str(e)[:100]}"
            )
    
    # ==================== 迭代检索流程 ====================
    
    async def iterative_retrieval(
        self,
        question: str,
        time_range: Optional[Dict] = None,
        max_iterations: int = 2  # 减少到2次迭代
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        迭代检索流程
        
        Returns:
            (最终文档列表, 元数据)
        """
        all_documents = []
        seen_doc_ids = set()
        metadata = {
            "iterations": [],
            "total_queries": 0
        }
        
        current_query = question
        
        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"迭代 {iteration + 1}/{max_iterations}")
            logger.info(f"当前查询: {current_query}")
            logger.info(f"{'='*60}")
            
            # 1. 查询改写
            rewritten_queries = await self.rewrite_query(current_query)
            metadata["total_queries"] += len(rewritten_queries)
            
            # 2. 对每个改写查询进行检索
            iteration_docs = []
            for rq in rewritten_queries:
                try:
                    # 混合检索
                    hybrid_results = self.retrieval_manager.hybrid_search(
                        rq, 
                        self.keyword_extractor, 
                        time_range
                    )
                    
                    # 去重并收集
                    for doc in hybrid_results.get("final_results", [])[:10]:
                        doc_id = doc.get("doc_id")
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            iteration_docs.append(doc)
                            all_documents.append(doc)
                except Exception as e:
                    logger.error(f"检索失败: {e}")
            
            # 记录本次迭代信息
            metadata["iterations"].append({
                "iteration": iteration + 1,
                "queries": rewritten_queries,
                "docs_found": len(iteration_docs)
            })
            
            logger.info(f"第{iteration+1}轮检索到 {len(iteration_docs)} 个新文档")
            
            # 3. 判断是否需要继续
            if len(all_documents) >= 15:  # 收集足够文档
                logger.info("已收集足够文档，停止迭代")
                break
            
            if len(iteration_docs) < 2:  # 新文档太少
                logger.info("新文档数量太少，停止迭代")
                break
            
            # 4. 基于当前文档调整查询(用于下一轮)
            if iteration < max_iterations - 1 and iteration_docs:
                # 提取文档中的关键信息作为下一轮查询的上下文
                context = " ".join([
                    doc.get("title", "")[:50] 
                    for doc in iteration_docs[:2]
                ])
                current_query = f"{question} {context}"  # 结合原问题和上下文
        
        # 按分数排序
        all_documents.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        
        logger.info(f"迭代检索完成: 共{len(all_documents)}个文档")
        return all_documents[:15], metadata  # 返回top15
    
    # ==================== 主入口 ====================
    
    async def agentic_retrieve_and_answer(
        self,
        question: str,
        time_range: Optional[Dict] = None,
        model_generate_func=None
    ) -> Dict[str, Any]:
        """
        Agentic RAG主流程
        
        Args:
            question: 问题
            time_range: 时间范围
            model_generate_func: 生成答案的函数
            
        Returns:
            完整结果字典
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"开始Agentic RAG处理: {question}")
        logger.info(f"{'='*80}")
        
        result = {
            "question": question,
            "final_answer": "",
            "confidence": 0.0,
            "retrieved_docs": [],
            "process_metadata": {
                "query_plan": None,
                "retrieval": [],
                "verification": None
            }
        }
        
        try:
            # 步骤1: 查询分析
            logger.info("步骤1: 查询分析...")
            query_plan = await self.analyze_query(question)
            result["process_metadata"]["query_plan"] = query_plan.dict() if hasattr(query_plan, 'dict') else query_plan
            
            # 步骤2: 问题分解(如果需要)
            if query_plan.sub_queries and len(query_plan.sub_queries) > 1:
                questions_to_process = query_plan.sub_queries
                logger.info(f"使用分解的子问题: {questions_to_process}")
            else:
                questions_to_process = [question]
            
            # 步骤3: 迭代检索
            all_retrieved_docs = []
            retrieval_metadata = []
            
            for sub_q in questions_to_process:
                logger.info(f"处理子问题: {sub_q}")
                
                # 3a. HyDE增强(如果需要)
                if query_plan.needs_hyde:
                    logger.info("应用HyDE增强...")
                    hyde_doc = await self.generate_hyde(sub_q)
                    # 使用HyDE文档检索
                    try:
                        hyde_results = self.retrieval_manager.hybrid_search(
                            hyde_doc,
                            self.keyword_extractor,
                            time_range
                        )
                        hyde_docs = hyde_results.get("final_results", [])[:5]
                        all_retrieved_docs.extend(hyde_docs)
                        logger.info(f"HyDE检索到 {len(hyde_docs)} 个文档")
                    except Exception as e:
                        logger.error(f"HyDE检索失败: {e}")
                
                # 3b. 迭代检索
                logger.info("执行迭代检索...")
                docs, meta = await self.iterative_retrieval(
                    sub_q,
                    time_range,
                    max_iterations=2  # 减少迭代次数避免过慢
                )
                all_retrieved_docs.extend(docs)
                retrieval_metadata.append(meta)
            
            # 去重并排序
            seen_ids = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                doc_id = doc.get("doc_id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
            
            unique_docs.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
            
            # 步骤4: 高级重排序
            if self.reranker and unique_docs:
                logger.info("应用高级重排序...")
                try:
                    unique_docs = self.reranker.rerank_comprehensive(
                        query=question,
                        documents=unique_docs,
                        top_k=15,
                        use_cross_encoder=True,
                        use_diversity=True,
                        use_temporal=True
                    )
                except Exception as e:
                    logger.error(f"重排序失败: {e}")
            
            final_docs = unique_docs[:10]  # Top10文档
            
            result["retrieved_docs"] = final_docs
            result["process_metadata"]["retrieval"] = retrieval_metadata
            
            logger.info(f"最终检索到 {len(final_docs)} 个高质量文档")
            
            # 步骤5: 生成答案
            if final_docs and model_generate_func:
                logger.info("生成答案...")
                try:
                    answer = await model_generate_func(question, final_docs)
                    result["final_answer"] = answer
                    
                    # 步骤6: 答案验证（使用重试机制）
                    logger.info("验证答案...")
                    verification = await self.verify_answer_with_retry(
                        question,
                        answer,
                        final_docs,
                        max_retries=2
                    )
                    result["confidence"] = verification.confidence
                    result["process_metadata"]["verification"] = verification.dict() if hasattr(verification, 'dict') else verification
                    
                    # 步骤7: 如果置信度低,再检索一轮
                    if verification.confidence < 0.7 and verification.missing_info:
                        logger.info(f"置信度较低({verification.confidence})，基于缺失信息再检索一轮")
                        missing_info = verification.missing_info
                        
                        for missing in missing_info[:2]:  # 最多2个缺失信息
                            additional_docs, _ = await self.iterative_retrieval(
                                f"{question} {missing}",
                                time_range,
                                max_iterations=1
                            )
                            final_docs.extend(additional_docs[:3])
                        
                        # 重新生成答案
                        if final_docs:
                            answer = await model_generate_func(question, final_docs[:10])
                            result["final_answer"] = answer
                            result["retrieved_docs"] = final_docs[:10]
                except Exception as e:
                    logger.error(f"生成或验证答案失败: {e}")
                    result["final_answer"] = f"生成答案失败: {str(e)[:100]}"
                    result["confidence"] = 0.0
            
            else:
                logger.warning("没有检索到足够文档或缺少生成函数")
                result["final_answer"] = "抱歉，未能检索到足够的相关信息。"
                result["confidence"] = 0.0
            
        except Exception as e:
            logger.error(f"Agentic RAG处理失败: {e}", exc_info=True)
            result["final_answer"] = f"抱歉，处理过程中出现错误: {str(e)[:100]}"
            result["confidence"] = 0.0
            result["error"] = str(e)
        
        return result