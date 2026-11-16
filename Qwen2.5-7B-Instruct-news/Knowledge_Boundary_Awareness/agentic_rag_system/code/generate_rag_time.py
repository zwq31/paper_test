# generate_rag_time.py
import os 
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入自定义模块
from utils.keywords_extractor import NewsKeywordExtractor, load_stopwords, extract_keywords_from_question
from utils.time_extractor import extract_time_from_question
from utils.rag_utils import (
    build_confidence_messages, build_answer_messages, build_rag_answer_messages,
    parse_self_confidence, get_random_rejection_response, should_answer_directly, DEFAULT_THRESHOLDS
)
from utils.model_utils import compute_ppl_minprob, load_model

# 导入RAG相关库
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_train_time.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ----------------------------
# 配置路径和参数
# ----------------------------
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_2/test/test_time.json"
STOPWORDS_PATH = "/mnt/project/milvus/data/hit_stopwords.txt"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# Milvus配置（使用自建Milvus）
# ----------------------------
MILVUS_HOST = "122.226.69.234"
MILVUS_PORT = "3042"
MILVUS_COLLECTION_NAME = "news_collection_all"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
TOP_K = 10  # 每种检索方法取top10，用于融合

# ----------------------------
# RAG相关函数
# ----------------------------
class HybridRetrievalManager:
    """混合检索管理器"""
    
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port
        self.connection = None
        self.collection = None
        self.embedding_model = None
        
    def initialize(self):
        """初始化连接和模型"""
        try:
            # 初始化Milvus连接
            logger.info(f"正在连接 Milvus ({self.host}:{self.port})...")
            
            self.connection = connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            # 获取Collection
            self.collection = Collection(MILVUS_COLLECTION_NAME)
            logger.info(f"已连接到集合: {MILVUS_COLLECTION_NAME}")
            
            # 初始化嵌入模型
            logger.info("加载嵌入模型...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("向量数据库和嵌入模型初始化成功")
            
        except Exception as e:
            logger.error(f"混合检索管理器初始化失败: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        if self.embedding_model is None:
            raise ValueError("嵌入模型未初始化")
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def vector_search(self, question: str, time_range: Dict[str, Any] = None, top_k: int = TOP_K) -> List[Dict]:
        """
        向量检索
        """
        if self.collection is None or self.embedding_model is None:
            raise ValueError("检索管理器未初始化")
        
        try:
            # 生成问题嵌入
            question_embedding = self.get_embedding(question)
            
            # 构建搜索参数
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10, "ef": 50}
            }
            
            # 构建搜索表达式 - 只有当有时间范围且需要过滤时才添加
            expr = None
            if time_range and time_range.get("has_specific_date", False):
                start_ts = time_range.get("start_timestamp")
                end_ts = time_range.get("end_timestamp")
                if start_ts and end_ts:
                    expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
            
            # 执行向量搜索
            results = self.collection.search(
                data=[question_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["doc_id", "title", "content", "publish_time", "publish_date_str", "region", "keywords"],
                consistency_level="Eventually"
            )
            
            # 格式化结果
            formatted_results = []
            if results:
                for hits in results:
                    for hit in hits:
                        entity = hit.entity
                        doc_info = {
                            "id": hit.id,
                            "doc_id": entity.get("doc_id", ""),
                            "score": hit.score,
                            "title": entity.get("title", ""),
                            "content": entity.get("content", ""),
                            "publish_time": entity.get("publish_time", 0),
                            "publish_date_str": entity.get("publish_date_str", ""),
                            "region": entity.get("region", ""),
                            "keywords": entity.get("keywords", []),
                            "retrieval_type": "vector"
                        }
                        formatted_results.append(doc_info)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def keyword_search(self, question: str, keyword_extractor: NewsKeywordExtractor, 
                      time_range: Dict[str, Any] = None, top_k: int = TOP_K) -> List[Dict]:
        """
        关键字检索
        """
        if self.collection is None:
            raise ValueError("检索管理器未初始化")
        
        try:
            # 提取关键词
            keywords = extract_keywords_from_question(question, keyword_extractor)
            if not keywords:
                return []
            
            # 构建关键词过滤表达式
            keyword_expr_parts = []
            for keyword in keywords[:5]:  # 使用前5个关键词
                keyword_expr_parts.append(f"array_contains(keywords, '{keyword}')")
            
            if not keyword_expr_parts:
                return []
            
            keyword_expr = " or ".join(keyword_expr_parts)
            
            # 构建时间过滤表达式
            time_expr = ""
            if time_range and time_range.get("has_specific_date", False):
                start_ts = time_range.get("start_timestamp")
                end_ts = time_range.get("end_timestamp")
                if start_ts and end_ts:
                    time_expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
            
            # 组合表达式
            if time_expr:
                expr = f"({keyword_expr}) and ({time_expr})"
            else:
                expr = keyword_expr
            
            # 执行查询
            query_results = self.collection.query(
                expr=expr,
                output_fields=["doc_id", "title", "content", "publish_time", "publish_date_str", "region", "keywords"],
                limit=top_k
            )
            
            # 格式化结果
            formatted_results = []
            base_score = 0.5  # 关键字检索的基础分数
            for i, result in enumerate(query_results):
                # 分数递减
                score = base_score - (i * 0.05)
                if score < 0.1:
                    score = 0.1
                    
                doc_info = {
                    "id": i,
                    "doc_id": result.get("doc_id", ""),
                    "score": score,
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "publish_time": result.get("publish_time", 0),
                    "publish_date_str": result.get("publish_date_str", ""),
                    "region": result.get("region", ""),
                    "keywords": result.get("keywords", []),
                    "retrieval_type": "keyword"
                }
                formatted_results.append(doc_info)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"关键字检索失败: {e}")
            return []
    
    def reciprocal_rank_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        倒数排名融合
        """
        # 创建文档ID到结果的映射
        doc_id_to_results = {}
        
        # 处理向量检索结果
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.get("doc_id", "")
            if doc_id:
                if doc_id not in doc_id_to_results:
                    doc_id_to_results[doc_id] = {
                        "doc_info": result,
                        "vector_rank": rank,
                        "keyword_rank": float('inf')
                    }
                else:
                    doc_id_to_results[doc_id]["vector_rank"] = rank
        
        # 处理关键字检索结果
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result.get("doc_id", "")
            if doc_id:
                if doc_id not in doc_id_to_results:
                    doc_id_to_results[doc_id] = {
                        "doc_info": result,
                        "vector_rank": float('inf'),
                        "keyword_rank": rank
                    }
                else:
                    doc_id_to_results[doc_id]["keyword_rank"] = rank
        
        # 计算RRF分数
        rrf_results = []
        for doc_id, info in doc_id_to_results.items():
            # 计算RRF分数：1 / (k + rank)
            vector_rrf = 1 / (k + info["vector_rank"]) if info["vector_rank"] != float('inf') else 0
            keyword_rrf = 1 / (k + info["keyword_rank"]) if info["keyword_rank"] != float('inf') else 0
            
            # 总RRF分数
            total_rrf_score = vector_rrf + keyword_rrf
            
            # 创建融合结果
            rrf_result = {
                "doc_info": info["doc_info"],
                "vector_rank": info["vector_rank"],
                "keyword_rank": info["keyword_rank"],
                "vector_rrf": vector_rrf,
                "keyword_rrf": keyword_rrf,
                "rrf_score": total_rrf_score
            }
            rrf_results.append(rrf_result)
        
        # 按RRF分数降序排序
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # 提取文档信息并设置新的综合分数
        final_results = []
        for i, rrf_result in enumerate(rrf_results):
            doc_info = rrf_result["doc_info"].copy()
            # 使用RRF分数作为新的综合分数
            doc_info["score"] = rrf_result["rrf_score"]
            doc_info["rrf_score"] = rrf_result["rrf_score"]
            doc_info["vector_rank"] = rrf_result["vector_rank"]
            doc_info["keyword_rank"] = rrf_result["keyword_rank"]
            doc_info["vector_rrf"] = rrf_result["vector_rrf"]
            doc_info["keyword_rrf"] = rrf_result["keyword_rrf"]
            final_results.append(doc_info)
        
        return final_results
    
    def hybrid_search(self, question: str, keyword_extractor: NewsKeywordExtractor, 
                     time_range: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        混合检索
        """
        # 分别执行两种检索，各取10个结果用于融合
        vector_results = self.vector_search(question, time_range, TOP_K)
        keyword_results = self.keyword_search(question, keyword_extractor, time_range, TOP_K)
        
        # 使用倒数排名融合（RRF）进行结果融合
        rrf_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
        return {
            "vector_results": vector_results,
            "keyword_results": keyword_results,
            "final_results": rrf_results
        }

# 全局检索管理器
retrieval_manager = None

# ----------------------------
# 生成函数
# ----------------------------
def generate_confidence(model, tokenizer, question):
    """生成置信度"""
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
    
    logger.info(f"生成置信度完成: {self_conf}")
    return self_conf, confidence_output

def generate_answer_with_context(model, tokenizer, question, contexts=None):
    """
    生成答案，可以指定上下文（用于RAG）
    """
    # 构建消息
    if contexts:
        messages = build_rag_answer_messages(question, contexts)
    else:
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
# 初始化函数
# ----------------------------
def initialize_retrieval():
    """初始化检索系统"""
    global retrieval_manager
    
    logger.info("初始化混合检索系统...")
    try:
        retrieval_manager = HybridRetrievalManager(MILVUS_HOST, MILVUS_PORT)
        retrieval_manager.initialize()
        logger.info("混合检索系统初始化成功")
        return True
    except Exception as e:
        logger.error(f"混合检索系统初始化失败: {e}")
        return False

# ----------------------------
# 主函数
# ----------------------------
def main():
    # 初始化jieba
    import jieba
    jieba.initialize()
    
    # 加载停用词和关键词提取器
    stopwords = load_stopwords(STOPWORDS_PATH)
    keyword_extractor = NewsKeywordExtractor(stopwords)
    logger.info(f"已加载 {len(stopwords)} 个停用词")
    
    # 加载语言模型
    model, tokenizer = load_model(MODEL_BASE_PATH)
    
    # 初始化检索系统
    retrieval_enabled = initialize_retrieval()
    if not retrieval_enabled:
        logger.warning("检索系统初始化失败，将只使用模型直接推理")
    
    # 加载输入数据
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 处理每个问题
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        f_out.write("[\n")
        for idx, item in enumerate(tqdm(input_data)):
            question = item.get("question", "").strip()
            if not question:
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"处理第 {idx+1} 个问题: {question}")
            logger.info(f"{'='*60}")
            
            try:
                # 第一步：从问题中提取时间信息
                time_range = extract_time_from_question(question)
                
                # 第二步：生成置信度
                self_conf, confidence_output = generate_confidence(model, tokenizer, question)
                
                # 第三步：生成原始答案（不带RAG）
                raw_model_output, ppl, min_token_prob = generate_answer_with_context(
                    model, tokenizer, question, contexts=None
                )
                
                # 第四步：决策流程
                decision = should_answer_directly(ppl, min_token_prob, self_conf, DEFAULT_THRESHOLDS)
                
                final_answer = ""
                use_retrieval = False
                retrieved_docs = []
                
                if decision:
                    # 直接使用模型答案
                    final_answer = raw_model_output
                    decision_reason = "满足阈值条件，直接使用模型答案"
                    logger.info("决策: 满足阈值条件，直接使用模型答案")
                else:
                    # 使用检索
                    use_retrieval = True
                    decision_reason = "未满足阈值条件，启用混合检索"
                    logger.info("决策: 未满足阈值条件，启用混合检索")
                    
                    if retrieval_enabled and retrieval_manager:
                        # 执行混合检索
                        hybrid_results = retrieval_manager.hybrid_search(question, keyword_extractor, time_range)
                        all_retrieved_docs = hybrid_results["final_results"]
                        
                        # 只保存top5文档
                        retrieved_docs = all_retrieved_docs[:5]
                        
                        if retrieved_docs:
                            # 使用检索到的文档生成答案
                            rag_answer, rag_ppl, rag_min_prob = generate_answer_with_context(
                                model, tokenizer, question, contexts=retrieved_docs
                            )
                            final_answer = rag_answer
                            decision_reason += "，检索到相关信息，基于检索结果生成答案"
                            logger.info("决策: 检索到相关信息，基于检索结果生成答案")
                            
                            # 更新ppl和min_token_prob为RAG生成的值
                            ppl, min_token_prob = rag_ppl, rag_min_prob
                        else:
                            # 没有相关支持信息，拒绝回答
                            final_answer = get_random_rejection_response()
                            decision_reason += "，未检索到相关信息，拒绝回答"
                            logger.info("决策: 未检索到相关信息，拒绝回答")
                    else:
                        # 检索系统不可用，拒绝回答
                        final_answer = get_random_rejection_response()
                        decision_reason += "，检索系统不可用，拒绝回答"
                        logger.info("决策: 检索系统不可用，拒绝回答")
                
                # 构建输出项
                output_item = {
                    "source_id": item.get("source_id", idx),
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "model_answer": final_answer,
                    "raw_model_output": raw_model_output,
                    "retrieval_decision": {
                        "ppl": round(ppl, 4) if np.isfinite(ppl) else None,
                        "min_token_prob": round(min_token_prob, 4),
                        "self_reported_confidence": round(self_conf, 4) if isinstance(self_conf, (int, float)) else None,
                        "confidence_output": confidence_output,
                        "should_answer_directly": bool(decision),
                        "use_retrieval": use_retrieval,
                        "retrieved_docs_preview": [
                            {
                                "doc_id": doc.get("doc_id", ""),
                                "title": doc.get("title", "")[:50] + "..." if len(doc.get("title", "")) > 50 else doc.get("title", ""),
                                "content_preview": doc.get("content", "")[:100] + "..." if len(doc.get("content", "")) > 100 else doc.get("content", ""),
                                "publish_date": doc.get("publish_date_str", ""),
                                "retrieval_type": doc.get("retrieval_type", "unknown"),
                                "rrf_score": round(doc.get("rrf_score", 0), 4),
                            }
                            for doc in retrieved_docs[:5]
                        ],
                        "decision_reason": decision_reason,
                        "final_model_answer": final_answer
                    }
                }
                
                # 写入输出文件
                if idx > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
                f_out.flush()
                
                logger.info(f"第 {idx+1} 个问题处理完成")
                
            except Exception as e:
                logger.error(f"第{idx}条数据处理失败: {e}", exc_info=True)
                continue
        
        f_out.write("\n]")
    
    logger.info(f"推理完成，结果保存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()