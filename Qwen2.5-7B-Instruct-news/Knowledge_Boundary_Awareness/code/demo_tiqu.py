# ==========================================
# 增强型RAG问答系统 - 两阶段生成版本
# generate_rag_two_stage.py
# ==========================================

import os 
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from datetime import datetime, timedelta
import jieba
import jieba.posseg as pseg

# RAG相关库
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer, CrossEncoder

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_rag_two_stage.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 同时输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# ----------------------------
# 配置路径和参数
# ----------------------------
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test2/test_two_stage_nodecide.json"
STOPWORDS_PATH = "/mnt/project/milvus/data/hit_stopwords.txt"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# Milvus配置
# ----------------------------
MILVUS_HOST = "122.226.69.234"
MILVUS_PORT = "3042"
MILVUS_COLLECTION_NAME = "news_collection_v2"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 检索参数
TOP_K_RETRIEVAL = 20  # 初始检索数量
TOP_K_FINAL = 5       # 最终返回数量

# ----------------------------
# 停用词加载
# ----------------------------
def load_stopwords(stopwords_path: str) -> set:
    """加载停用词"""
    stopwords = set()
    
    base_stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', '那', '但', '中', '从', '对', '而', '且', '或', '之', '为', '以',
        '及', '将', '并', '等', '个', '年', '月', '日', '时', '与', '于', '就', '又',
        '还', '没', '不', '都', '好', '很', '得', '多', '最', '更'
    }
    
    if os.path.exists(stopwords_path):
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and len(word) > 1:
                    stopwords.add(word)
        stopwords.update(base_stopwords)
        logger.info(f"已加载 {len(stopwords)} 个停用词")
    else:
        stopwords = base_stopwords
        logger.warning("使用内置停用词")
    return stopwords

STOPWORDS = load_stopwords(STOPWORDS_PATH)

# ----------------------------
# 时间提取
# ----------------------------
def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
    """
    从问题中提取时间信息
    """
    
    # 1. 处理绝对时间
    date_patterns = [
        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', 'ymd'),   # 年月日
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ymd'),       # 2023-01-15
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', 'ymd'),       # 2023/01/15
        (r'(\d{4})年(\d{1,2})月', 'ym'),               # 年月
        (r'(\d{4})-(\d{1,2})', 'ym'),                  # 2023-01
        (r'(\d{4})/(\d{1,2})', 'ym'),                  # 2023/01
        (r'(\d{4})年', 'y'),                           # 只有年份
    ]
    
    for pattern, date_type in date_patterns:
        match = re.search(pattern, question)
        if match:
            try:
                groups = match.groups()
                
                if date_type == 'ymd':  # 年月日 - 前后1个月
                    year, month, day = map(int, groups)
                    extracted_date = datetime(year, month, day)
                    days_range = 30
                    need_filter = True
                    
                elif date_type == 'ym':  # 年月 - 前后2个月
                    year, month = map(int, groups)
                    extracted_date = datetime(year, month, 1)
                    days_range = 60
                    need_filter = True
                    
                elif date_type == 'y':  # 只有年份 - 前后1年
                    year = int(groups[0])
                    extracted_date = datetime(year, 6, 30)  # 年中
                    days_range = 365
                    need_filter = True
                
                if need_filter:
                    start_date = extracted_date - timedelta(days=days_range)
                    end_date = extracted_date + timedelta(days=days_range)
                    
                    start_timestamp = int(start_date.timestamp())
                    end_timestamp = int(end_date.timestamp())
                    
                    logger.info(f"提取时间 [{date_type}]: {extracted_date.strftime('%Y-%m-%d')}")
                    
                    return {
                        "extracted_date": extracted_date,
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                        "start_date_str": start_date.strftime("%Y-%m-%d"),
                        "end_date_str": end_date.strftime("%Y-%m-%d"),
                        "has_specific_date": True,
                        "date_type": date_type,
                        "range_days": days_range * 2
                    }
                
            except Exception as e:
                logger.warning(f"时间解析失败: {e}")
                continue
    
    # 2. 处理相对时间
    relative_time_patterns = {
        r'今天|当天': (0, 1),
        r'昨天|昨日': (-1, 1),
        r'前天': (-2, 1),
        r'大前天': (-3, 1),
        r'上周|上星期': (-7, 7),
        r'本周|这周': (-3, 3),
        r'上个月|上月': (-30, 30),
        r'这个月|本月': (-15, 15),
        r'去年|上一年': (-365, 365),
        r'今年|本年': (-180, 180),
        r'最近|近期': (-30, 0),
    }
    
    current_date = datetime.now()
    for pattern, (days_offset, days_range) in relative_time_patterns.items():
        if re.search(pattern, question):
            center_date = current_date + timedelta(days=days_offset)
            start_date = center_date - timedelta(days=abs(days_range))
            end_date = center_date + timedelta(days=abs(days_range))
            
            logger.info(f"相对时间提取: {pattern}")
            
            return {
                "extracted_date": center_date,
                "start_timestamp": int(start_date.timestamp()),
                "end_timestamp": int(end_date.timestamp()),
                "start_date_str": start_date.strftime("%Y-%m-%d"),
                "end_date_str": end_date.strftime("%Y-%m-%d"),
                "has_specific_date": True,
                "is_relative": True,
                "relative_pattern": pattern
            }
    
    logger.info("未提取到时间信息，不进行时间过滤")
    return None

# ----------------------------
# 关键词提取
# ----------------------------
def extract_keywords_enhanced(question: str, stopwords: set, top_k: int = 15) -> List[str]:
    """
    增强的关键词提取 - 使用词性加权
    """
    if not question or len(question.strip()) == 0:
        return []

    # 词性权重
    POS_WEIGHTS = {
        'n': 2.0,   # 普通名词
        'nr': 3.0,  # 人名
        'ns': 3.0,  # 地名
        'nt': 3.0,  # 机构名
        'nz': 2.5,  # 其他专名
        'v': 1.8,   # 动词
        'vn': 2.0,  # 动名词
        'a': 1.3,   # 形容词
        'an': 1.5,  # 名形词
        'i': 1.5,   # 成语
    }
    
    # 词性标注
    words_with_pos = pseg.cut(question)
    
    word_scores = {}
    for word, pos in words_with_pos:
        w = word.strip()
        
        # 基本过滤
        if len(w) <= 1:
            continue
        if w in stopwords:
            continue
        if w.isdigit():
            continue
        if not any('\u4e00' <= c <= '\u9fff' or c.isalpha() for c in w):
            continue
        
        # 获取词性权重
        pos_prefix = pos[0] if pos else 'x'
        weight = POS_WEIGHTS.get(pos_prefix, 0.5)
        
        # 多字专有名词额外加权
        if len(w) >= 3 and pos_prefix in ['nr', 'ns', 'nt', 'nz']:
            weight *= 1.2
        
        # 累加分数
        if w in word_scores:
            word_scores[w] += weight
        else:
            word_scores[w] = weight
    
    # 按分数排序
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_words[:top_k]]
    
    logger.info(f"增强关键词提取: {keywords}")
    return keywords

# ----------------------------
# 重排序器类
# ----------------------------
class EnhancedReranker:
    """增强的重排序器"""
    
    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"加载重排序模型: {model_name}")
        try:
            self.reranker = CrossEncoder(model_name, max_length=512)
            logger.info("重排序模型加载完成")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            self.reranker = None
    
    def rerank(
        self, 
        question: str, 
        documents: List[Dict], 
        top_k: int = 5,
        time_decay: bool = True,
        current_timestamp: Optional[int] = None
    ) -> List[Dict]:
        """
        重排序文档
        """
        if not documents or self.reranker is None:
            return documents[:top_k]
        
        # 准备query-document对
        pairs = []
        for doc in documents:
            doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"[:500]
            pairs.append([question, doc_text])
        
        # 计算相关性分数
        logger.info(f"重排序 {len(documents)} 个文档...")
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]
        
        # 添加分数
        for doc, score in zip(documents, rerank_scores):
            doc['rerank_score'] = float(score)
            
            # 时间衰减
            if time_decay and current_timestamp:
                publish_time = doc.get('publish_time', 0)
                if publish_time > 0:
                    days_diff = (current_timestamp - publish_time) / 86400
                    # 每30天衰减5%，最小保留50%权重
                    time_factor = max(0.5, 1.0 - (days_diff / 30) * 0.05)
                    doc['final_score'] = score * time_factor
                else:
                    doc['final_score'] = score
            else:
                doc['final_score'] = score
        
        # 排序
        ranked_docs = sorted(documents, key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"重排序完成，返回top-{top_k}")
        for i, doc in enumerate(ranked_docs[:top_k]):
            logger.info(f"  Rank {i+1}: {doc.get('title', '')[:50]}... "
                       f"(分数: {doc['final_score']:.4f})")
        
        return ranked_docs[:top_k]

# ----------------------------
# 混合检索管理器
# ----------------------------
class HybridRetrievalManager:
    """混合检索管理器"""
    
    def __init__(self, host: str, port: str, use_reranker: bool = True):
        self.host = host
        self.port = port
        self.connection = None
        self.collection = None
        self.embedding_model = None
        self.reranker = None
        self.use_reranker = use_reranker
        
    def initialize(self):
        """初始化连接、模型和重排序器"""
        try:
            # 连接Milvus
            logger.info(f"连接 Milvus ({self.host}:{self.port})...")
            self.connection = connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            self.collection = Collection(MILVUS_COLLECTION_NAME)
            logger.info(f"已连接到集合: {MILVUS_COLLECTION_NAME}")
            
            # 加载embedding模型
            logger.info("加载embedding模型...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # 加载重排序器
            if self.use_reranker:
                try:
                    self.reranker = EnhancedReranker(RERANKER_MODEL)
                except Exception as e:
                    logger.warning(f"重排序器加载失败: {e}, 将不使用重排序")
                    self.reranker = None
            
            logger.info("检索系统初始化完成")
            
        except Exception as e:
            logger.error(f"检索系统初始化失败: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        if self.embedding_model is None:
            raise ValueError("嵌入模型未初始化")
        
        # 为查询添加prompt
        query_prompt = "为这个问题生成搜索表示: "
        embedding = self.embedding_model.encode(query_prompt + text)
        return embedding.tolist()
    
    def vector_search(
        self, 
        question: str, 
        time_range: Optional[Dict[str, Any]] = None, 
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """向量检索"""
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
            
            # 构建时间过滤表达式
            expr = None
            if time_range and time_range.get("has_specific_date", False):
                start_ts = time_range.get("start_timestamp")
                end_ts = time_range.get("end_timestamp")
                if start_ts and end_ts:
                    expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
                    logger.info(f"向量检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")
            
            # 执行搜索
            results = self.collection.search(
                data=[question_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["doc_id", "title", "content", "publish_time", 
                              "publish_date_str", "region", "keywords"],
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
            
            logger.info(f"向量检索完成: {len(formatted_results)} 个文档")
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def keyword_search(
        self, 
        question: str, 
        time_range: Optional[Dict[str, Any]] = None, 
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """关键字检索"""
        if self.collection is None:
            raise ValueError("检索管理器未初始化")
        
        try:
            # 提取关键词
            keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
            if not keywords:
                logger.warning("未提取到关键词")
                return []
            
            # 构建关键词表达式
            keyword_expr_parts = []
            for keyword in keywords[:5]:
                # 对关键词进行简单的转义处理
                safe_keyword = keyword.replace("'", "''").replace('"', '""')
                keyword_expr_parts.append(f"array_contains(keywords, '{safe_keyword}')")
            
            if not keyword_expr_parts:
                return []
            
            keyword_expr = " or ".join(keyword_expr_parts)
            
            # 构建时间过滤
            time_expr = ""
            if time_range and time_range.get("has_specific_date", False):
                start_ts = time_range.get("start_timestamp")
                end_ts = time_range.get("end_timestamp")
                if start_ts and end_ts:
                    time_expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
                    logger.info(f"关键字检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")
            
            # 组合表达式
            if time_expr:
                expr = f"({keyword_expr}) and ({time_expr})"
            else:
                expr = keyword_expr
            
            # 执行查询
            query_results = self.collection.query(
                expr=expr,
                output_fields=["doc_id", "title", "content", "publish_time", 
                              "publish_date_str", "region", "keywords"],
                limit=top_k
            )
            
            logger.info(f"关键字检索完成: {len(query_results)} 个文档")
            
            # 格式化结果
            formatted_results = []
            base_score = 0.5
            for i, result in enumerate(query_results):
                score = base_score - (i * 0.02)
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
    
    def reciprocal_rank_fusion(
        self, 
        vector_results: List[Dict], 
        keyword_results: List[Dict], 
        k: int = 60
    ) -> List[Dict]:
        """倒数排名融合（RRF）"""
        logger.info(f"开始RRF融合，k={k}")
        
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
            vector_rrf = 1 / (k + info["vector_rank"]) if info["vector_rank"] != float('inf') else 0
            keyword_rrf = 1 / (k + info["keyword_rank"]) if info["keyword_rank"] != float('inf') else 0
            total_rrf_score = vector_rrf + keyword_rrf
            
            rrf_result = {
                "doc_info": info["doc_info"],
                "vector_rank": info["vector_rank"],
                "keyword_rank": info["keyword_rank"],
                "rrf_score": total_rrf_score
            }
            rrf_results.append(rrf_result)
        
        # 排序
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # 提取文档
        final_results = []
        for rrf_result in rrf_results:
            doc_info = rrf_result["doc_info"].copy()
            doc_info["score"] = rrf_result["rrf_score"]
            doc_info["rrf_score"] = rrf_result["rrf_score"]
            doc_info["vector_rank"] = rrf_result["vector_rank"]
            doc_info["keyword_rank"] = rrf_result["keyword_rank"]
            final_results.append(doc_info)
        
        logger.info(f"RRF融合完成: {len(final_results)} 个文档")
        return final_results
    
    def hybrid_search(
        self, 
        question: str, 
        time_range: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        混合检索流程
        """
        logger.info(f"开始混合检索: {question}")
        
        # 1. 分别检索
        vector_results = self.vector_search(question, time_range, TOP_K_RETRIEVAL)
        keyword_results = self.keyword_search(question, time_range, TOP_K_RETRIEVAL)
        
        logger.info(f"初始检索: 向量{len(vector_results)}, 关键词{len(keyword_results)}")
        
        # 2. RRF融合
        fused_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
        logger.info(f"RRF融合后: {len(fused_results)} 个文档")
        
        # 3. 重排序
        if self.reranker and fused_results:
            import time
            current_ts = int(time.time())
            final_results = self.reranker.rerank(
                question, 
                fused_results, 
                top_k=TOP_K_FINAL,
                time_decay=True,
                current_timestamp=current_ts
            )
        else:
            final_results = fused_results[:TOP_K_FINAL]
        
        logger.info(f"最终返回: {len(final_results)} 个文档")
        
        return {
            "all_retrieved": {
                "vector_results": vector_results[:10],
                "keyword_results": keyword_results[:10],
                "fused_results": fused_results[:10]
            },
            "final_results": final_results
        }

# 全局检索管理器
retrieval_manager = None

# ----------------------------
# 两阶段生成器
# ----------------------------
class TwoStageGenerator:
    """两阶段生成器"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_answer(self, question: str, contexts: List[Dict]) -> Tuple[str, str]:
        """两阶段生成答案"""
        # 第一阶段：提取关键信息
        logger.info("第一阶段：提取关键信息...")
        extraction_prompt = self._build_extraction_prompt(question, contexts)
        extracted_info = self._call_model(extraction_prompt)
        
        # 第二阶段：基于提取信息生成最终答案
        logger.info("第二阶段：生成最终答案...")
        final_prompt = self._build_final_answer_prompt(question, extracted_info)
        final_answer = self._call_model(final_prompt)
        
        return final_answer, extracted_info
    
    def _build_extraction_prompt(self, question: str, contexts: List[Dict]) -> str:
        """构建信息提取prompt"""
        if not contexts:
            return f"""问题：{question}

没有相关资料。"""

        context_text = "\n".join([
            f"【文档{i+1}】\n"
            f"标题: {ctx.get('title', '')}\n"
            f"内容: {ctx.get('content', '')[:1000]}\n"
            f"{'='*50}"
            for i, ctx in enumerate(contexts[:5])
        ])
        
        return f"""请从以下资料中提取回答"{question}"所需的关键信息：

资料：
{context_text}

请提取：
1. **直接答案**：直接回答问题的信息
2. **支持证据**：支持答案的具体事实、数据、时间、人物等
3. **来源文档**：每条信息的来源文档编号
4. **矛盾点**：如果多个资料有矛盾，记录矛盾内容

请用结构化方式提取：
【直接答案】
[答案要点]

【支持证据】
1. [证据1] (来源: 文档X)
2. [证据2] (来源: 文档Y)

【信息充足性】
- 充足 / 部分充足 / 不足"""
    
    def _build_final_answer_prompt(self, question: str, extracted_info: str) -> str:
        """构建最终答案prompt"""
        return f"""基于以下提取的信息，回答问题：{question}

提取的信息：
{extracted_info}

请生成最终答案：
1. 只使用提取信息中的事实
2. 明确引用来源（如"根据文档1"）
3. 如果信息不足或矛盾，明确指出
4. 保持简洁准确

最终答案："""
    
    def _call_model(self, prompt: str, temperature: float = 0.0) -> str:
        """调用模型生成"""
        messages = [
            {"role": "system", "content": "你是一个信息提取和回答生成助手。"},
            {"role": "user", "content": prompt}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=4096, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0, inputs['input_ids'].size(1):]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return result

# ----------------------------
# 模型加载
# ----------------------------
def load_model(model_path):
    """加载模型和tokenizer"""
    logger.info("加载模型和tokenizer...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
# 初始化检索系统
# ----------------------------
def initialize_retrieval():
    """初始化检索系统"""
    global retrieval_manager
    
    logger.info("初始化混合检索系统...")
    try:
        retrieval_manager = HybridRetrievalManager(
            MILVUS_HOST, 
            MILVUS_PORT,
            use_reranker=True  # 启用重排序
        )
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
    # 初始化
    jieba.initialize()
    
    # 加载模型
    model, tokenizer = load_model(MODEL_BASE_PATH)
    
    # 初始化检索系统
    retrieval_enabled = initialize_retrieval()
    if not retrieval_enabled:
        logger.error("检索系统初始化失败，无法进行RAG问答")
        return
    
    # 创建两阶段生成器
    generator = TwoStageGenerator(model, tokenizer)
    logger.info("使用两阶段生成方案")
    
    # 加载输入数据
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 处理每个问题
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        f_out.write("[\n")
        for idx, item in enumerate(tqdm(input_data, desc="处理问题")):
            question = item.get("question", "").strip()
            if not question:
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"处理第 {idx+1} 个问题: {question}")
            logger.info(f"{'='*60}")
            
            try:
                # 1. 提取时间信息
                logger.info("提取时间信息...")
                time_range = extract_time_from_question(question)
                
                # 2. 提取关键词
                extracted_keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
                
                # 3. 执行混合检索
                logger.info("开始混合检索...")
                hybrid_results = retrieval_manager.hybrid_search(question, time_range)
                retrieved_docs = hybrid_results["final_results"]
                
                has_supporting_info = len(retrieved_docs) > 0
                logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
                
                # 4. 生成答案（两阶段方案）
                answer, extracted_info = generator.generate_answer(
                    question, retrieved_docs
                )
                
                # 构建结果
                retrieval_info = {
                    "has_supporting_info": has_supporting_info,
                    "extracted_keywords": extracted_keywords,
                    "retrieved_docs_count": len(retrieved_docs),
                    "generation_strategy": "two_stage"
                }
                
                # 记录提取信息
                retrieval_info["extracted_info"] = extracted_info[:500] + "..." if len(extracted_info) > 500 else extracted_info
                
                # 添加检索文档预览
                retrieval_info["retrieved_docs_preview"] = [
                    {
                        "doc_id": doc.get("doc_id", ""),
                        "title": doc.get("title", ""),  # ← 完整标题，不再截断
                        "content_preview": doc.get("content", ""),  # ← 完整内容，不再截断
                        "publish_date": doc.get("publish_date_str", ""),
                        "retrieval_type": doc.get("retrieval_type", "unknown"),
                        "final_score": float(round(doc.get("final_score", 0), 4)),
                        "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
                    }
                    for doc in retrieved_docs
                ]
                
                # 添加时间信息
                if time_range:
                    retrieval_info["time_extraction"] = {
                        "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d") if time_range.get("extracted_date") else "N/A",
                        "date_type": time_range.get("date_type", "relative"),
                        "range_days": time_range.get("range_days", "N/A"),
                        "start_date": time_range.get("start_date_str"),
                        "end_date": time_range.get("end_date_str")
                    }
                
                output_item = {
                    "source_id": item.get("source_id", idx),
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "model_answer": answer,
                    "retrieval_info": retrieval_info
                }
                
                # 写入
                if idx > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
                f_out.flush()
                
                logger.info(f"第 {idx+1} 个问题处理完成")
                logger.info(f"答案: {answer[:100]}...")
                
            except Exception as e:
                logger.error(f"第{idx}条数据处理失败: {e}", exc_info=True)
                continue
        
        f_out.write("\n]")
    
    logger.info(f"✅ 两阶段生成RAG问答完成，结果保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()