# ==========================================
# 优化后的问答系统完整代码
# generate_train_optimized.py
# ==========================================

import os 
import json
import torch
import logging
import random
import numpy as np
import time  # 新增：导入time模块
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
import re
from datetime import datetime, timedelta
import jieba
import jieba.posseg as pseg
from collections import Counter

# RAG相关库
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer, CrossEncoder

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_train_optimized.log',
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
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk/test_qwen2.5_450.json"
STOPWORDS_PATH = "/mnt/project/milvus/data/hit_stopwords.txt"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# Milvus配置
# ----------------------------
MILVUS_HOST = "122.226.69.234"
MILVUS_PORT = "3042"
MILVUS_COLLECTION_NAME = "news_collection_450"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 检索参数
TOP_K_RETRIEVAL = 20  # 初始检索数量(增加)
TOP_K_FINAL = 5       # 最终返回数量

# ----------------------------
# 拒绝阈值
# ----------------------------
PPL_THRESHOLD = 1.1022038499105375
MIN_TOKEN_PROB_THRESHOLD = 0.3661533556634178
SELF_CONF_THRESHOLD = 0.44508323935997046

# ----------------------------
# 拒答模板
# ----------------------------
REJECTION_RESPONSES = [
    "抱歉，我无法回答这个问题。",
    "根据现有的信息，我无法回答这个问题。",
    "我暂时没有足够的信息来回答这个问题。"
]

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
# 改进的时间提取
# ----------------------------
def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
    """
    从问题中提取时间信息
    
    改进点:
    1. 只有年份: 前后1年范围
    2. 年月: 前后2个月范围
    3. 年月日: 前后1个月范围
    4. 相对时间: 智能处理
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
                    logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} (共{days_range*2}天)")
                    
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
            logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
            
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
# 改进的关键词提取
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
        
        Args:
            question: 问题
            documents: 待排序文档列表
            top_k: 返回前k个
            time_decay: 是否考虑时间衰减
            current_timestamp: 当前时间戳
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
# 改进的混合检索管理器
# ----------------------------
class ImprovedHybridRetrievalManager:
    """改进的混合检索管理器"""
    
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
        """获取文本的嵌入向量，使用优化的prompt"""
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
            
            # logger.info(f"关键词检索使用: {keywords[:5]}")
            
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
    
    def hybrid_search_v2(
        self, 
        question: str, 
        time_range: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        改进的混合检索流程
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
        
        # 新增：过滤出分数大于0.1的文档
        filtered_results = [doc for doc in final_results if doc.get('final_score', 0) > 0.1]
        
        logger.info(f"重排序后top-{TOP_K_FINAL}: {len(final_results)} 个文档")
        logger.info(f"分数过滤后(>0.1): {len(filtered_results)} 个文档")
        
        return {
            "all_retrieved": {
                "vector_results": vector_results[:10],
                "keyword_results": keyword_results[:10],
                "fused_results": fused_results[:10],
                "reranked_results": final_results
            },
            "final_results": final_results,
            "filtered_results": filtered_results  # 新增：过滤后的结果
        }

# 全局检索管理器
retrieval_manager = None

# ----------------------------
# Prompt构建函数
# ----------------------------
def build_confidence_messages(question):
    """构建置信度评估prompt"""
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

def build_answer_messages(question):
    """构建普通回答prompt"""
    messages = [
        {
            "role": "system",
            "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
        },
        {
            "role": "user",
            "content": f"请回答下面的问题：\n\n问题：{question}"
        }
    ]
    return messages

def build_rag_answer_messages_v2(question: str, contexts: List[Dict]) -> List[Dict]:
    """
    构建RAG Prompt
    """
    if not contexts:
        logger.warning("无检索结果，使用基础prompt")
        return [
            {
                "role": "system",
                "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答“基于现有资料，我无法回答这个问题”。"
            },
            {
                "role": "user", 
                "content": f"问题: {question}"
            }
        ]
    
    # 构建上下文
    context_parts = []
    for i, doc in enumerate(contexts, 1):
        title = doc.get("title", f"文档{i}")
        content = doc.get("content", "")[:800]
        pub_date = doc.get("publish_date_str", "未知")
        region = doc.get("region", "未知")
        
        context_parts.append(
            f"【文档{i}】\n"
            f"标题: {title}\n"
            f"时间: {pub_date}\n"
            f"地区: {region}\n"
            f"内容: {content}\n"
        )
    
    context_text = "\n".join(context_parts)
    
    # 保持简洁的系统提示，专注于角色和核心原则
    system_prompt = """你是专业的新闻问答助手，严格基于参考资料回答问题。"""

    # 增强的用户提示，引导逐步推理
    user_prompt = f"""# 参考资料
{context_text}

# 需要回答的问题
{question}

# 回答要求
请按照以下步骤思考并回答：

## 步骤1：分析问题要点
- 理解问题的核心是什么
- 识别关键信息：人物、事件、时间、地点、数据等

## 步骤2：定位相关证据
- 在参考资料中找出与问题唯一或最相关的所有片段
- 注意信息的时效性和相关性， 若资料存在多篇内容冲突，优先选择发布时间最新的资料内容作答

## 步骤3：基于证据回答
- **严格只使用参考资料中的信息**
- 将找到的证据自然整合到回答中
- 明确引用信息来源（例如："根据文档3的报道..."）
- 如果不同来源有冲突，如实说明

## 步骤4：最终检查
- 确认每个事实都有参考资料支持
- 检查是否直接回答了问题
- 确保回答简洁清晰

# 回答格式
- 如果根据参考资料能回答：直接给出答案
- 如果完全无法回答："基于现有资料，我无法回答这个问题"
"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# ----------------------------
# 辅助函数
# ----------------------------
def parse_self_confidence(model_answer):
    """解析模型自报置信度"""
    try:
        lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
        self_conf = None
        
        for line in lines:
            if "置信度" in line:
                val_str = line.split(':')[-1].strip()
                try:
                    self_conf = float(val_str)
                except ValueError:
                    logger.warning(f"置信度值无法转为浮点数: '{val_str}'")
                    self_conf = None
                return self_conf
        
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

def compute_ppl_minprob(generated_tokens, generate_logits):
    """计算困惑度和最小词元概率"""
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
        retrieval_manager = ImprovedHybridRetrievalManager(
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
# 生成函数（添加时间统计参数）
# ----------------------------
def generate_confidence(model, tokenizer, question, start_time=None):
    """生成置信度"""
    messages = build_confidence_messages(question)
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs.sequences[0, prompt_len:]
    confidence_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    self_conf = parse_self_confidence(confidence_output)
    
    logger.info(f"生成置信度: {self_conf}, 原始输出: {confidence_output}")
    
    # 如果传入了开始时间，计算生成时间
    if start_time:
        generation_time = time.perf_counter() - start_time
        logger.info(f"置信度生成耗时: {generation_time:.3f}秒")
        return self_conf, confidence_output, generation_time
    else:
        return self_conf, confidence_output

def generate_answer_with_context(model, tokenizer, question, contexts=None, start_time=None):
    """生成答案"""
    if contexts:
        logger.info(f"使用RAG模式生成答案，上下文数量: {len(contexts)}")
        messages = build_rag_answer_messages_v2(question, contexts)
    else:
        logger.info("使用普通模式生成答案")
        messages = build_answer_messages(question)
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
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
    
    generated_tokens = outputs.sequences[0, prompt_len:]
    raw_model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    ppl, min_token_prob = float('inf'), 0.0
    if outputs.scores:
        generate_logits = torch.stack(outputs.scores, dim=0)
        ppl, min_token_prob = compute_ppl_minprob(generated_tokens, generate_logits)
    
    logger.info(f"生成答案完成，ppl: {ppl:.4f}, min_token_prob: {min_token_prob:.4f}")
    
    # 如果传入了开始时间，计算生成时间
    if start_time:
        generation_time = time.perf_counter() - start_time
        logger.info(f"答案生成耗时: {generation_time:.3f}秒")
        return raw_model_output, ppl, min_token_prob, generation_time
    else:
        return raw_model_output, ppl, min_token_prob

# ----------------------------
# 主函数（添加时间统计）
# ----------------------------
def main():
    # 初始化
    jieba.initialize()
    
    # 加载模型
    model, tokenizer = load_model(MODEL_BASE_PATH)
    
    # 初始化检索系统
    retrieval_enabled = initialize_retrieval()
    if not retrieval_enabled:
        logger.warning("检索系统初始化失败，将只使用模型直接推理")
    
    # 加载输入数据
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 新增：时间统计变量
    total_samples = len(input_data)
    total_processing_time = 0.0
    total_confidence_time = 0.0
    total_answer_time = 0.0
    total_retrieval_time = 0.0
    total_rag_answer_time = 0.0
    sample_times = []  # 存储每个样本的详细时间
    
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
                # 记录样本开始处理时间
                sample_start_time = time.perf_counter()
                
                # 1. 提取时间信息
                logger.info("提取时间信息...")
                time_extract_start = time.perf_counter()
                time_range = extract_time_from_question(question)
                time_extract_time = time.perf_counter() - time_extract_start
                
                # 2. 生成置信度
                logger.info("生成置信度...")
                confidence_start_time = time.perf_counter()
                self_conf, confidence_output, confidence_time = generate_confidence(
                    model, tokenizer, question, start_time=confidence_start_time
                )
                total_confidence_time += confidence_time
                
                # 3. 生成原始答案
                logger.info("生成原始答案...")
                answer_start_time = time.perf_counter()
                raw_model_output, ppl, min_token_prob, answer_time = generate_answer_with_context(
                    model, tokenizer, question, contexts=None, start_time=answer_start_time
                )
                total_answer_time += answer_time
                
                # 4. 决策
                logger.info("进行决策判断...")
                should_answer_directly = (
                    ppl <= PPL_THRESHOLD and
                    min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
                    (self_conf is not None and self_conf >= SELF_CONF_THRESHOLD)
                )
                
                logger.info(f"阈值判断: ppl={ppl:.4f}<={PPL_THRESHOLD}, "
                           f"min_prob={min_token_prob:.4f}>={MIN_TOKEN_PROB_THRESHOLD}, "
                           f"conf={self_conf}>={SELF_CONF_THRESHOLD if self_conf else 'None'}")
                
                final_answer = ""
                use_retrieval = False
                retrieved_docs = []
                has_supporting_info = False
                extracted_keywords = []
                retrieval_time = 0.0
                rag_answer_time = 0.0
                total_retrieved_docs = 0
                filtered_docs_count = 0
                
                if should_answer_directly:
                    final_answer = raw_model_output
                    decision_reason = "满足阈值条件，直接使用模型答案"
                    logger.info("决策: 直接回答")
                else:
                    use_retrieval = True
                    decision_reason = "未满足阈值条件，启用混合检索"
                    logger.info("决策: 使用RAG")
                    
                    if retrieval_enabled and retrieval_manager:
                        # 提取关键词
                        keyword_extract_start = time.perf_counter()
                        extracted_keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
                        keyword_extract_time = time.perf_counter() - keyword_extract_start
                        
                        # 执行混合检索
                        logger.info("开始混合检索...")
                        retrieval_start_time = time.perf_counter()
                        hybrid_results = retrieval_manager.hybrid_search_v2(question, time_range)
                        retrieval_end_time = time.perf_counter()
                        retrieval_time = retrieval_end_time - retrieval_start_time
                        total_retrieval_time += retrieval_time
                        
                        # 获取过滤后的结果（分数大于0.1的）
                        filtered_docs = hybrid_results.get("filtered_results", [])
                        all_docs = hybrid_results.get("final_results", [])
                        
                        # 使用过滤后的文档作为上下文
                        retrieved_docs = filtered_docs
                        total_retrieved_docs = len(all_docs)
                        filtered_docs_count = len(filtered_docs)
                        
                        has_supporting_info = len(retrieved_docs) > 0
                        logger.info(f"检索到 {len(all_docs)} 个文档，其中 {len(filtered_docs)} 个分数大于0.1")
                        logger.info(f"检索耗时: {retrieval_time:.3f}秒")
                        
                        if has_supporting_info:
                            # 基于检索结果生成答案
                            logger.info("基于检索结果生成答案...")
                            rag_answer_start_time = time.perf_counter()
                            rag_answer, rag_ppl, rag_min_prob, rag_answer_time = generate_answer_with_context(
                                model, tokenizer, question, contexts=retrieved_docs, start_time=rag_answer_start_time
                            )
                            total_rag_answer_time += rag_answer_time
                            final_answer = rag_answer
                            decision_reason += "，检索到相关信息，基于检索结果生成答案"
                            
                            ppl, min_token_prob = rag_ppl, rag_min_prob
                        else:
                            final_answer = random.choice(REJECTION_RESPONSES)
                            decision_reason += "，未检索到相关信息，拒绝回答"
                    else:
                        final_answer = random.choice(REJECTION_RESPONSES)
                        decision_reason += "，检索系统不可用，拒绝回答"
                
                # 计算样本总处理时间
                sample_end_time = time.perf_counter()
                sample_processing_time = sample_end_time - sample_start_time
                total_processing_time += sample_processing_time
                
                # 构建结果
                retrieval_decision = {
                    "ppl": float(round(ppl, 4)) if np.isfinite(ppl) else None,
                    "min_token_prob": float(round(min_token_prob, 4)),
                    "self_reported_confidence": float(round(self_conf, 4)) if isinstance(self_conf, (int, float)) else None,
                    "confidence_output": confidence_output,
                    "should_answer_directly": bool(should_answer_directly),
                    "use_retrieval": use_retrieval,
                    "decision_reason": decision_reason,
                    "final_model_answer": final_answer,
                    "filtered_docs_count": filtered_docs_count,    # 新增：过滤后的文档数
                }
                
                # 新增：添加详细时间统计
                retrieval_decision["time_stats"] = {
                    "total_processing_time": round(sample_processing_time, 3),
                    "confidence_generation_time": round(confidence_time, 3),
                    "answer_generation_time": round(answer_time, 3),
                    "time_extraction_time": round(time_extract_time, 3),
                    "keyword_extraction_time": round(keyword_extract_time, 3) if use_retrieval else 0,
                    "retrieval_time": round(retrieval_time, 3) if use_retrieval else 0,
                    "rag_answer_generation_time": round(rag_answer_time, 3) if use_retrieval and has_supporting_info else 0,
                }
                
                if use_retrieval:
                    if extracted_keywords:
                        retrieval_decision["extracted_keywords"] = extracted_keywords
                    retrieval_decision["filtered_docs_count"] = filtered_docs_count

                
                # 只保存过滤后的文档预览
                retrieval_decision["retrieved_docs_preview"] = [
                    {
                        "doc_id": doc.get("doc_id", ""),
                        "title": doc.get("title", ""),
                        "content_preview": doc.get("content", ""),
                        "publish_date": doc.get("publish_date_str", ""),
                        "retrieval_type": doc.get("retrieval_type", "unknown"),
                        "final_score": float(round(doc.get("final_score", 0), 4)),
                        "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
                    }
                    for doc in retrieved_docs  # 使用过滤后的文档
                ]
                
                # 添加时间信息
                if time_range:
                    retrieval_decision["time_extraction"] = {
                        "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d"),
                        "date_type": time_range.get("date_type", "relative"),
                        "range_days": time_range.get("range_days", "N/A"),
                        "start_date": time_range.get("start_date_str"),
                        "end_date": time_range.get("end_date_str")
                    }
                
                output_item = {
                    "source_id": item.get("source_id", idx),
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "model_answer": final_answer,
                    "raw_model_output": raw_model_output,
                    "retrieval_decision": retrieval_decision
                }
                
                # 存储样本时间详情
                sample_times.append({
                    "sample_id": idx,
                    "question": question[:50] + "..." if len(question) > 50 else question,
                    "total_time": sample_processing_time,
                    "confidence_time": confidence_time,
                    "answer_time": answer_time,
                    "retrieval_time": retrieval_time,
                    "rag_answer_time": rag_answer_time,
                    "use_retrieval": use_retrieval,
                    "retrieved_docs_total": total_retrieved_docs,
                    "filtered_docs_count": filtered_docs_count,
                    "final_answer_length": len(final_answer)
                })
                
                # 写入
                if idx > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
                f_out.flush()
                
                logger.info(f"第 {idx+1} 个问题处理完成，总耗时: {sample_processing_time:.3f}秒")
                
            except Exception as e:
                logger.error(f"第{idx}条数据处理失败: {e}", exc_info=True)
                continue
        
        f_out.write("\n]")
    
    # 新增：计算并保存时间统计
    if total_samples > 0:
        avg_processing_time = total_processing_time / total_samples
        avg_confidence_time = total_confidence_time / total_samples
        avg_answer_time = total_answer_time / total_samples
        avg_retrieval_time = total_retrieval_time / total_samples if total_retrieval_time > 0 else 0
        avg_rag_answer_time = total_rag_answer_time / total_samples if total_rag_answer_time > 0 else 0
        
        # 统计使用RAG的样本数量
        rag_samples = sum(1 for sample in sample_times if sample["use_retrieval"])
        direct_answer_samples = total_samples - rag_samples
        
        time_stats = {
            "total_samples": total_samples,
            "total_processing_time_seconds": round(total_processing_time, 3),
            "average_processing_time_seconds": round(avg_processing_time, 3),
            "breakdown": {
                "confidence_generation": {
                    "total_seconds": round(total_confidence_time, 3),
                    "average_seconds": round(avg_confidence_time, 3),
                    "percentage": round((total_confidence_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                },
                "answer_generation": {
                    "total_seconds": round(total_answer_time, 3),
                    "average_seconds": round(avg_answer_time, 3),
                    "percentage": round((total_answer_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                },
                "retrieval": {
                    "total_seconds": round(total_retrieval_time, 3),
                    "average_seconds": round(avg_retrieval_time, 3) if rag_samples > 0 else 0,
                    "percentage": round((total_retrieval_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0,
                    "rag_samples_count": rag_samples
                },
                "rag_answer_generation": {
                    "total_seconds": round(total_rag_answer_time, 3),
                    "average_seconds": round(avg_rag_answer_time, 3) if rag_samples > 0 else 0,
                    "percentage": round((total_rag_answer_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                }
            },
            "decision_statistics": {
                "direct_answer_samples": direct_answer_samples,
                "rag_answer_samples": rag_samples,
                "direct_answer_percentage": round((direct_answer_samples / total_samples) * 100, 2) if total_samples > 0 else 0,
                "rag_answer_percentage": round((rag_samples / total_samples) * 100, 2) if total_samples > 0 else 0
            },
            "sample_details": sample_times,
            "performance_summary": {
                "samples_per_hour": round(3600 / avg_processing_time, 2) if avg_processing_time > 0 else 0,
                "direct_answer_speed": f"{avg_confidence_time + avg_answer_time:.3f}s/answer",
                "rag_speed": f"{avg_confidence_time + avg_answer_time + avg_retrieval_time + avg_rag_answer_time:.3f}s/answer" if rag_samples > 0 else "N/A",
                "retrieval_efficiency": f"{avg_retrieval_time:.3f}s/query" if rag_samples > 0 else "N/A"
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        
        # 输出时间统计到日志
        logger.info("\n" + "="*60)
        logger.info("时间统计摘要:")
        logger.info("="*60)
        logger.info(f"处理样本总数: {total_samples}")
        logger.info(f"决策统计: 直接回答 {direct_answer_samples} 个，RAG回答 {rag_samples} 个")
        logger.info(f"总处理时间: {total_processing_time:.2f}秒")
        logger.info(f"平均处理时间: {avg_processing_time:.3f}秒/样本")
        logger.info(f"  - 置信度生成: {avg_confidence_time:.3f}秒 ({time_stats['breakdown']['confidence_generation']['percentage']}%)")
        logger.info(f"  - 答案生成: {avg_answer_time:.3f}秒 ({time_stats['breakdown']['answer_generation']['percentage']}%)")
        if rag_samples > 0:
            logger.info(f"  - 检索时间: {avg_retrieval_time:.3f}秒 ({time_stats['breakdown']['retrieval']['percentage']}%)")
            logger.info(f"  - RAG答案生成: {avg_rag_answer_time:.3f}秒 ({time_stats['breakdown']['rag_answer_generation']['percentage']}%)")
        logger.info(f"处理速度: {time_stats['performance_summary']['samples_per_hour']} 样本/小时")
    
    logger.info(f"✅ 推理完成，结果保存至 {OUTPUT_PATH}")
    logger.info(f"日志文件: generate_train_optimized.log")


if __name__ == "__main__":
    main()
# ====================================================优化时间过滤====================================

# ==========================================
# 优化后的问答系统完整代码
# generate_train_optimized.py
# ==========================================

# import os 
# import json
# import torch
# import logging
# import random
# import numpy as np
# import time
# from tqdm import tqdm
# from typing import List, Dict, Tuple, Optional, Any
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch.nn.functional as F
# import re
# from datetime import datetime, timedelta
# import jieba
# import jieba.posseg as pseg
# from collections import Counter

# # RAG相关库
# from pymilvus import connections, Collection
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # ----------------------------
# # 日志配置
# # ----------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='generate_train_optimized.log',
#     filemode='a',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# # 同时输出到控制台
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# logger.addHandler(console_handler)

# # ----------------------------
# # 配置路径和参数
# # ----------------------------
# MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk/test_qwen2.5_450_youhuatime.json"
# STOPWORDS_PATH = "/mnt/project/milvus/data/hit_stopwords.txt"

# MAX_LENGTH = 4096
# MAX_NEW_TOKENS = 2048

# # ----------------------------
# # Milvus配置
# # ----------------------------
# MILVUS_HOST = "122.226.69.234"
# MILVUS_PORT = "3042"
# MILVUS_COLLECTION_NAME = "news_collection_450"
# EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
# RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# # 检索参数
# TOP_K_RETRIEVAL = 20  # 初始检索数量
# TOP_K_FINAL = 5       # 最终返回数量

# # ----------------------------
# # 拒绝阈值
# # ----------------------------
# PPL_THRESHOLD = 1.1022038499105375
# MIN_TOKEN_PROB_THRESHOLD = 0.3661533556634178
# SELF_CONF_THRESHOLD = 0.44508323935997046

# # ----------------------------
# # 拒答模板
# # ----------------------------
# REJECTION_RESPONSES = [
#     "抱歉，我无法回答这个问题。",
#     "根据现有的信息，我无法回答这个问题。",
#     "我暂时没有足够的信息来回答这个问题。"
# ]

# # ----------------------------
# # 停用词加载
# # ----------------------------
# def load_stopwords(stopwords_path: str) -> set:
#     """加载停用词"""
#     stopwords = set()
    
#     base_stopwords = {
#         '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
#         '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
#         '自己', '这', '那', '但', '中', '从', '对', '而', '且', '或', '之', '为', '以',
#         '及', '将', '并', '等', '个', '年', '月', '日', '时', '与', '于', '就', '又',
#         '还', '没', '不', '都', '好', '很', '得', '多', '最', '更'
#     }
    
#     if os.path.exists(stopwords_path):
#         with open(stopwords_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 word = line.strip()
#                 if word and len(word) > 1:
#                     stopwords.add(word)
#         stopwords.update(base_stopwords)
#         logger.info(f"已加载 {len(stopwords)} 个停用词")
#     else:
#         stopwords = base_stopwords
#         logger.warning("使用内置停用词")
#     return stopwords

# STOPWORDS = load_stopwords(STOPWORDS_PATH)

# # ----------------------------
# # 改进的时间提取（优化点）
# # ----------------------------
# def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
#     """
#     从问题中提取时间信息（优化版）
    
#     改进点:
#     1. 支持更多绝对时间格式和相对时间表达（如“近一周”、“上半年”）
#     2. 相对时间优先基于已提取的绝对时间计算（若存在）
#     3. 动态调整时间窗口（根据时效性关键词）
#     """
    
#     # 先尝试提取绝对时间，作为可能的基准日期
#     base_date = None
#     absolute_date_info = None
    
#     # 绝对时间模式
#     date_patterns = [
#         (r'(\d{4})年(\d{1,2})月(\d{1,2})日', 'ymd'),
#         (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ymd'),
#         (r'(\d{4})/(\d{1,2})/(\d{1,2})', 'ymd'),
#         (r'(\d{4})年(\d{1,2})月', 'ym'),
#         (r'(\d{4})-(\d{1,2})', 'ym'),
#         (r'(\d{4})/(\d{1,2})', 'ym'),
#         (r'(\d{4})年', 'y'),
#         # 季度处理（简化：映射到该季度的中间月份）
#         (r'(\d{4})年(第[一二三四]季度|Q[1-4])', 'quarter'),
#         (r'(\d{4})年上半年', 'half1'),
#         (r'(\d{4})年下半年', 'half2'),
#     ]
    
#     for pattern, date_type in date_patterns:
#         match = re.search(pattern, question)
#         if match:
#             groups = match.groups()
#             try:
#                 if date_type == 'ymd':
#                     year, month, day = map(int, groups)
#                     extracted_date = datetime(year, month, day)
#                     days_range = 30
#                 elif date_type == 'ym':
#                     year, month = map(int, groups)
#                     extracted_date = datetime(year, month, 1)
#                     days_range = 60
#                 elif date_type == 'y':
#                     year = int(groups[0])
#                     extracted_date = datetime(year, 6, 30)  # 年中
#                     days_range = 365
#                 elif date_type == 'quarter':
#                     year = int(groups[0])
#                     quarter_str = groups[1]
#                     # 映射到季度的第二个月中间（例如Q1→2月15日）
#                     quarter_map = {'第一':1, '第二':2, '第三':3, '第四':4, 'Q1':1, 'Q2':2, 'Q3':3, 'Q4':4}
#                     q = quarter_map.get(quarter_str, 1)
#                     month = q*3 - 1  # 2,5,8,11
#                     day = 15
#                     extracted_date = datetime(year, month, day)
#                     days_range = 45  # 前后1.5个月
#                 elif date_type == 'half1':
#                     year = int(groups[0])
#                     extracted_date = datetime(year, 4, 1)  # 4月1日
#                     days_range = 90
#                 elif date_type == 'half2':
#                     year = int(groups[0])
#                     extracted_date = datetime(year, 10, 1)  # 10月1日
#                     days_range = 90
#                 else:
#                     continue
                
#                 # 记录绝对时间作为基准
#                 base_date = extracted_date
#                 absolute_date_info = {
#                     "extracted_date": extracted_date,
#                     "date_type": date_type,
#                     "range_days": days_range
#                 }
#                 # 继续查找相对时间（可能包含“去年”等词）
#                 break  # 找到第一个绝对时间即可
#             except Exception as e:
#                 logger.warning(f"绝对时间解析失败: {e}")
#                 continue
    
#     # 相对时间模式（优化点：增加更多模式）
#     relative_patterns = {
#         # 基础相对词
#         r'今天|当日|当天': (0, 1),
#         r'昨天|昨日': (-1, 1),
#         r'前天': (-2, 1),
#         r'大前天': (-3, 1),
#         r'上周|上星期': (-7, 7),
#         r'本周|这周': (-3, 3),
#         r'下周|下星期': (7, 7),
#         r'上个月|上月': (-30, 30),
#         r'这个月|本月': (-15, 15),
#         r'下个月|下月': (30, 30),
#         r'去年|上一年': (-365, 365),
#         r'今年|本年': (-180, 180),
#         r'明年|下一年': (365, 365),
#         # 近期表达
#         r'近(一周|一星期|七天?)': (-7, 7),
#         r'近(一个月|三十天?)': (-30, 30),
#         r'近(三个月|九十天?)': (-90, 90),
#         r'近(半年|六个月)': (-180, 180),
#         r'最近|近期|近来': (-30, 0),  # 仅向前
#         r'最新': (-7, 0),              # 仅向前，窗口更小
#     }
    
#     # 判断是否存在时效性关键词，用于动态调整窗口
#     urgency_keywords = ['最新', '最近', '近期', '今天', '昨日', '昨日']
#     has_urgency = any(kw in question for kw in urgency_keywords)
    
#     # 遍历相对时间模式
#     for pattern, (center_offset, base_range) in relative_patterns.items():
#         match = re.search(pattern, question)
#         if match:
#             # 确定基准日期：优先使用已提取的绝对时间，否则当前时间
#             if base_date is not None:
#                 ref_date = base_date
#                 logger.info(f"相对时间 '{pattern}' 基于绝对时间 {ref_date.strftime('%Y-%m-%d')} 计算")
#             else:
#                 ref_date = datetime.now()
#                 logger.info(f"相对时间 '{pattern}' 基于当前时间 {ref_date.strftime('%Y-%m-%d')} 计算")
            
#             center_date = ref_date + timedelta(days=center_offset)
            
#             # 动态调整窗口：若含有时效性关键词，缩小窗口
#             range_days = base_range
#             if has_urgency and pattern in ['最近|近期|近来', '最新']:
#                 range_days = min(range_days, 7)  # 时效性问题窗口不超过7天
#             elif has_urgency and pattern in ['近一周']:
#                 range_days = 3  # 更严格
            
#             start_date = center_date - timedelta(days=abs(range_days))
#             end_date = center_date + timedelta(days=abs(range_days))
            
#             logger.info(f"相对时间提取: {pattern}, 窗口大小: {range_days*2}天")
#             logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
            
#             return {
#                 "extracted_date": center_date,
#                 "reference_timestamp": int(center_date.timestamp()),  # 用于衰减的参考时间
#                 "start_timestamp": int(start_date.timestamp()),
#                 "end_timestamp": int(end_date.timestamp()),
#                 "start_date_str": start_date.strftime("%Y-%m-%d"),
#                 "end_date_str": end_date.strftime("%Y-%m-%d"),
#                 "has_specific_date": True,
#                 "is_relative": True,
#                 "relative_pattern": pattern,
#                 "range_days": range_days * 2
#             }
    
#     # 如果没有相对时间，但有绝对时间，返回绝对时间范围
#     if absolute_date_info:
#         extracted_date = absolute_date_info["extracted_date"]
#         days_range = absolute_date_info["range_days"]
        
#         # 如果有时效性关键词，适当缩小窗口
#         if has_urgency:
#             days_range = min(days_range, 30)  # 最多30天
        
#         start_date = extracted_date - timedelta(days=days_range)
#         end_date = extracted_date + timedelta(days=days_range)
        
#         logger.info(f"绝对时间提取 [{absolute_date_info['date_type']}]: {extracted_date.strftime('%Y-%m-%d')}")
#         logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} (共{days_range*2}天)")
        
#         return {
#             "extracted_date": extracted_date,
#             "reference_timestamp": int(extracted_date.timestamp()),
#             "start_timestamp": int(start_date.timestamp()),
#             "end_timestamp": int(end_date.timestamp()),
#             "start_date_str": start_date.strftime("%Y-%m-%d"),
#             "end_date_str": end_date.strftime("%Y-%m-%d"),
#             "has_specific_date": True,
#             "date_type": absolute_date_info["date_type"],
#             "range_days": days_range * 2
#         }
    
#     logger.info("未提取到时间信息，不进行时间过滤")
#     return None

# # ----------------------------
# # 改进的关键词提取
# # ----------------------------
# def extract_keywords_enhanced(question: str, stopwords: set, top_k: int = 15) -> List[str]:
#     """
#     增强的关键词提取 - 使用词性加权
#     """
#     if not question or len(question.strip()) == 0:
#         return []

#     # 词性权重
#     POS_WEIGHTS = {
#         'n': 2.0,   # 普通名词
#         'nr': 3.0,  # 人名
#         'ns': 3.0,  # 地名
#         'nt': 3.0,  # 机构名
#         'nz': 2.5,  # 其他专名
#         'v': 1.8,   # 动词
#         'vn': 2.0,  # 动名词
#         'a': 1.3,   # 形容词
#         'an': 1.5,  # 名形词
#         'i': 1.5,   # 成语
#     }
    
#     # 词性标注
#     words_with_pos = pseg.cut(question)
    
#     word_scores = {}
#     for word, pos in words_with_pos:
#         w = word.strip()
        
#         # 基本过滤
#         if len(w) <= 1:
#             continue
#         if w in stopwords:
#             continue
#         if w.isdigit():
#             continue
#         if not any('\u4e00' <= c <= '\u9fff' or c.isalpha() for c in w):
#             continue
        
#         # 获取词性权重
#         pos_prefix = pos[0] if pos else 'x'
#         weight = POS_WEIGHTS.get(pos_prefix, 0.5)
        
#         # 多字专有名词额外加权
#         if len(w) >= 3 and pos_prefix in ['nr', 'ns', 'nt', 'nz']:
#             weight *= 1.2
        
#         # 累加分数
#         if w in word_scores:
#             word_scores[w] += weight
#         else:
#             word_scores[w] = weight
    
#     # 按分数排序
#     sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
#     keywords = [word for word, score in sorted_words[:top_k]]
    
#     logger.info(f"增强关键词提取: {keywords}")
#     return keywords

# # ----------------------------
# # 重排序器类（优化点：修改衰减基准）
# # ----------------------------
# class EnhancedReranker:
#     """增强的重排序器"""
    
#     def __init__(self, model_name: str = RERANKER_MODEL):
#         logger.info(f"加载重排序模型: {model_name}")
#         try:
#             self.reranker = CrossEncoder(model_name, max_length=512)
#             logger.info("重排序模型加载完成")
#         except Exception as e:
#             logger.error(f"重排序模型加载失败: {e}")
#             self.reranker = None
    
#     def rerank(
#         self, 
#         question: str, 
#         documents: List[Dict], 
#         top_k: int = 5,
#         time_decay: bool = True,
#         reference_timestamp: Optional[int] = None  # 优化点：改为参考时间戳
#     ) -> List[Dict]:
#         """
#         重排序文档
        
#         Args:
#             question: 问题
#             documents: 待排序文档列表
#             top_k: 返回前k个
#             time_decay: 是否考虑时间衰减
#             reference_timestamp: 参考时间戳（通常为问题中提取的中心时间）
#         """
#         if not documents or self.reranker is None:
#             return documents[:top_k]
        
#         # 准备query-document对
#         pairs = []
#         for doc in documents:
#             doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"[:500]
#             pairs.append([question, doc_text])
        
#         # 计算相关性分数
#         logger.info(f"重排序 {len(documents)} 个文档...")
#         try:
#             rerank_scores = self.reranker.predict(pairs)
#         except Exception as e:
#             logger.error(f"重排序失败: {e}")
#             return documents[:top_k]
        
#         # 添加分数
#         for doc, score in zip(documents, rerank_scores):
#             doc['rerank_score'] = float(score)
            
#             # 时间衰减（优化点：基于参考时间戳，而不是当前时间）
#             if time_decay and reference_timestamp is not None:
#                 publish_time = doc.get('publish_time', 0)
#                 if publish_time > 0:
#                     # 计算与参考时间的绝对天数差
#                     days_diff = abs(publish_time - reference_timestamp) / 86400
#                     # 每30天衰减5%，最小保留50%权重
#                     time_factor = max(0.5, 1.0 - (days_diff / 30) * 0.05)
#                     doc['final_score'] = score * time_factor
#                 else:
#                     doc['final_score'] = score
#             else:
#                 doc['final_score'] = score
        
#         # 排序
#         ranked_docs = sorted(documents, key=lambda x: x['final_score'], reverse=True)
        
#         logger.info(f"重排序完成，返回top-{top_k}")
#         for i, doc in enumerate(ranked_docs[:top_k]):
#             logger.info(f"  Rank {i+1}: {doc.get('title', '')[:50]}... "
#                        f"(分数: {doc['final_score']:.4f})")
        
#         return ranked_docs[:top_k]

# # ----------------------------
# # 改进的混合检索管理器（优化点：传递参考时间戳）
# # ----------------------------
# class ImprovedHybridRetrievalManager:
#     """改进的混合检索管理器"""
    
#     def __init__(self, host: str, port: str, use_reranker: bool = True):
#         self.host = host
#         self.port = port
#         self.connection = None
#         self.collection = None
#         self.embedding_model = None
#         self.reranker = None
#         self.use_reranker = use_reranker
        
#     def initialize(self):
#         """初始化连接、模型和重排序器"""
#         try:
#             # 连接Milvus
#             logger.info(f"连接 Milvus ({self.host}:{self.port})...")
#             self.connection = connections.connect(
#                 alias="default",
#                 host=self.host,
#                 port=self.port
#             )
            
#             self.collection = Collection(MILVUS_COLLECTION_NAME)
#             logger.info(f"已连接到集合: {MILVUS_COLLECTION_NAME}")
            
#             # 加载embedding模型
#             logger.info("加载embedding模型...")
#             self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
#             # 加载重排序器
#             if self.use_reranker:
#                 try:
#                     self.reranker = EnhancedReranker(RERANKER_MODEL)
#                 except Exception as e:
#                     logger.warning(f"重排序器加载失败: {e}, 将不使用重排序")
#                     self.reranker = None
            
#             logger.info("检索系统初始化完成")
            
#         except Exception as e:
#             logger.error(f"检索系统初始化失败: {e}")
#             raise
    
#     def get_embedding(self, text: str) -> List[float]:
#         """获取文本的嵌入向量，使用优化的prompt"""
#         if self.embedding_model is None:
#             raise ValueError("嵌入模型未初始化")
        
#         # 为查询添加prompt
#         query_prompt = "为这个问题生成搜索表示: "
#         embedding = self.embedding_model.encode(query_prompt + text)
#         return embedding.tolist()
    
#     def vector_search(
#         self, 
#         question: str, 
#         time_range: Optional[Dict[str, Any]] = None, 
#         top_k: int = TOP_K_RETRIEVAL
#     ) -> List[Dict]:
#         """向量检索"""
#         if self.collection is None or self.embedding_model is None:
#             raise ValueError("检索管理器未初始化")
        
#         try:
#             # 生成问题嵌入
#             question_embedding = self.get_embedding(question)
            
#             # 构建搜索参数
#             search_params = {
#                 "metric_type": "IP",
#                 "params": {"nprobe": 10, "ef": 50}
#             }
            
#             # 构建时间过滤表达式
#             expr = None
#             if time_range and time_range.get("has_specific_date", False):
#                 start_ts = time_range.get("start_timestamp")
#                 end_ts = time_range.get("end_timestamp")
#                 if start_ts and end_ts:
#                     expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
#                     logger.info(f"向量检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")
            
#             # 执行搜索
#             results = self.collection.search(
#                 data=[question_embedding],
#                 anns_field="embedding",
#                 param=search_params,
#                 limit=top_k,
#                 expr=expr,
#                 output_fields=["doc_id", "title", "content", "publish_time", 
#                               "publish_date_str", "region", "keywords"],
#                 consistency_level="Eventually"
#             )
            
#             # 格式化结果
#             formatted_results = []
#             if results:
#                 for hits in results:
#                     for hit in hits:
#                         entity = hit.entity
#                         doc_info = {
#                             "id": hit.id,
#                             "doc_id": entity.get("doc_id", ""),
#                             "score": hit.score,
#                             "title": entity.get("title", ""),
#                             "content": entity.get("content", ""),
#                             "publish_time": entity.get("publish_time", 0),
#                             "publish_date_str": entity.get("publish_date_str", ""),
#                             "region": entity.get("region", ""),
#                             "keywords": entity.get("keywords", []),
#                             "retrieval_type": "vector"
#                         }
#                         formatted_results.append(doc_info)
            
#             logger.info(f"向量检索完成: {len(formatted_results)} 个文档")
#             return formatted_results
            
#         except Exception as e:
#             logger.error(f"向量检索失败: {e}")
#             return []
    
#     def keyword_search(
#         self, 
#         question: str, 
#         time_range: Optional[Dict[str, Any]] = None, 
#         top_k: int = TOP_K_RETRIEVAL
#     ) -> List[Dict]:
#         """关键字检索"""
#         if self.collection is None:
#             raise ValueError("检索管理器未初始化")
        
#         try:
#             # 提取关键词
#             keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
#             if not keywords:
#                 logger.warning("未提取到关键词")
#                 return []
            
#             # 构建关键词表达式
#             keyword_expr_parts = []
#             for keyword in keywords[:5]:
#                 safe_keyword = keyword.replace("'", "''").replace('"', '""')
#                 keyword_expr_parts.append(f"array_contains(keywords, '{safe_keyword}')")
            
#             if not keyword_expr_parts:
#                 return []
            
#             keyword_expr = " or ".join(keyword_expr_parts)
            
#             # 构建时间过滤
#             time_expr = ""
#             if time_range and time_range.get("has_specific_date", False):
#                 start_ts = time_range.get("start_timestamp")
#                 end_ts = time_range.get("end_timestamp")
#                 if start_ts and end_ts:
#                     time_expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
#                     logger.info(f"关键字检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")
            
#             # 组合表达式
#             if time_expr:
#                 expr = f"({keyword_expr}) and ({time_expr})"
#             else:
#                 expr = keyword_expr
            
#             # 执行查询
#             query_results = self.collection.query(
#                 expr=expr,
#                 output_fields=["doc_id", "title", "content", "publish_time", 
#                               "publish_date_str", "region", "keywords"],
#                 limit=top_k
#             )
            
#             logger.info(f"关键字检索完成: {len(query_results)} 个文档")
            
#             # 格式化结果
#             formatted_results = []
#             base_score = 0.5
#             for i, result in enumerate(query_results):
#                 score = base_score - (i * 0.02)
#                 if score < 0.1:
#                     score = 0.1
                    
#                 doc_info = {
#                     "id": i,
#                     "doc_id": result.get("doc_id", ""),
#                     "score": score,
#                     "title": result.get("title", ""),
#                     "content": result.get("content", ""),
#                     "publish_time": result.get("publish_time", 0),
#                     "publish_date_str": result.get("publish_date_str", ""),
#                     "region": result.get("region", ""),
#                     "keywords": result.get("keywords", []),
#                     "retrieval_type": "keyword"
#                 }
#                 formatted_results.append(doc_info)
            
#             return formatted_results
            
#         except Exception as e:
#             logger.error(f"关键字检索失败: {e}")
#             return []
    
#     def reciprocal_rank_fusion(
#         self, 
#         vector_results: List[Dict], 
#         keyword_results: List[Dict], 
#         k: int = 60
#     ) -> List[Dict]:
#         """倒数排名融合（RRF）"""
#         logger.info(f"开始RRF融合，k={k}")
        
#         doc_id_to_results = {}
        
#         # 处理向量检索结果
#         for rank, result in enumerate(vector_results, 1):
#             doc_id = result.get("doc_id", "")
#             if doc_id:
#                 if doc_id not in doc_id_to_results:
#                     doc_id_to_results[doc_id] = {
#                         "doc_info": result,
#                         "vector_rank": rank,
#                         "keyword_rank": float('inf')
#                     }
#                 else:
#                     doc_id_to_results[doc_id]["vector_rank"] = rank
        
#         # 处理关键字检索结果
#         for rank, result in enumerate(keyword_results, 1):
#             doc_id = result.get("doc_id", "")
#             if doc_id:
#                 if doc_id not in doc_id_to_results:
#                     doc_id_to_results[doc_id] = {
#                         "doc_info": result,
#                         "vector_rank": float('inf'),
#                         "keyword_rank": rank
#                     }
#                 else:
#                     doc_id_to_results[doc_id]["keyword_rank"] = rank
        
#         # 计算RRF分数
#         rrf_results = []
#         for doc_id, info in doc_id_to_results.items():
#             vector_rrf = 1 / (k + info["vector_rank"]) if info["vector_rank"] != float('inf') else 0
#             keyword_rrf = 1 / (k + info["keyword_rank"]) if info["keyword_rank"] != float('inf') else 0
#             total_rrf_score = vector_rrf + keyword_rrf
            
#             rrf_result = {
#                 "doc_info": info["doc_info"],
#                 "vector_rank": info["vector_rank"],
#                 "keyword_rank": info["keyword_rank"],
#                 "rrf_score": total_rrf_score
#             }
#             rrf_results.append(rrf_result)
        
#         # 排序
#         rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
#         # 提取文档
#         final_results = []
#         for rrf_result in rrf_results:
#             doc_info = rrf_result["doc_info"].copy()
#             doc_info["score"] = rrf_result["rrf_score"]
#             doc_info["rrf_score"] = rrf_result["rrf_score"]
#             doc_info["vector_rank"] = rrf_result["vector_rank"]
#             doc_info["keyword_rank"] = rrf_result["keyword_rank"]
#             final_results.append(doc_info)
        
#         logger.info(f"RRF融合完成: {len(final_results)} 个文档")
#         return final_results
    
#     def hybrid_search_v2(
#         self, 
#         question: str, 
#         time_range: Optional[Dict] = None
#     ) -> Dict[str, Any]:
#         """
#         改进的混合检索流程（优化点：传递参考时间戳给重排序）
#         """
#         logger.info(f"开始混合检索: {question}")
        
#         # 1. 分别检索
#         vector_results = self.vector_search(question, time_range, TOP_K_RETRIEVAL)
#         keyword_results = self.keyword_search(question, time_range, TOP_K_RETRIEVAL)
        
#         logger.info(f"初始检索: 向量{len(vector_results)}, 关键词{len(keyword_results)}")
        
#         # 2. RRF融合
#         fused_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
#         logger.info(f"RRF融合后: {len(fused_results)} 个文档")
        
#         # 3. 重排序（优化点：使用参考时间戳）
#         if self.reranker and fused_results:
#             # 获取参考时间戳：优先使用 time_range 中的 reference_timestamp，否则当前时间
#             if time_range and 'reference_timestamp' in time_range:
#                 ref_ts = time_range['reference_timestamp']
#             else:
#                 ref_ts = int(time.time())  # 回退到当前时间
#                 logger.info("未提取到时间信息，使用当前时间作为衰减基准")
            
#             final_results = self.reranker.rerank(
#                 question, 
#                 fused_results, 
#                 top_k=TOP_K_FINAL,
#                 time_decay=True,
#                 reference_timestamp=ref_ts
#             )
#         else:
#             final_results = fused_results[:TOP_K_FINAL]
        
#         # 过滤出分数大于0.1的文档
#         filtered_results = [doc for doc in final_results if doc.get('final_score', 0) > 0.1]
        
#         logger.info(f"重排序后top-{TOP_K_FINAL}: {len(final_results)} 个文档")
#         logger.info(f"分数过滤后(>0.1): {len(filtered_results)} 个文档")
        
#         return {
#             "all_retrieved": {
#                 "vector_results": vector_results[:10],
#                 "keyword_results": keyword_results[:10],
#                 "fused_results": fused_results[:10],
#                 "reranked_results": final_results
#             },
#             "final_results": final_results,
#             "filtered_results": filtered_results
#         }

# # 全局检索管理器
# retrieval_manager = None

# # ----------------------------
# # Prompt构建函数
# # ----------------------------
# def build_confidence_messages(question):
#     """构建置信度评估prompt"""
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

# def build_answer_messages(question):
#     """构建普通回答prompt"""
#     messages = [
#         {
#             "role": "system",
#             "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
#         },
#         {
#             "role": "user",
#             "content": f"请回答下面的问题：\n\n问题：{question}"
#         }
#     ]
#     return messages

# def build_rag_answer_messages_v2(question: str, contexts: List[Dict]) -> List[Dict]:
#     """
#     构建RAG Prompt
#     """
#     if not contexts:
#         logger.warning("无检索结果，使用基础prompt")
#         return [
#             {
#                 "role": "system",
#                 "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答“基于现有资料，我无法回答这个问题”。"
#             },
#             {
#                 "role": "user", 
#                 "content": f"问题: {question}"
#             }
#         ]
    
#     # 构建上下文
#     context_parts = []
#     for i, doc in enumerate(contexts, 1):
#         title = doc.get("title", f"文档{i}")
#         content = doc.get("content", "")[:800]
#         pub_date = doc.get("publish_date_str", "未知")
#         region = doc.get("region", "未知")
        
#         context_parts.append(
#             f"【文档{i}】\n"
#             f"标题: {title}\n"
#             f"时间: {pub_date}\n"
#             f"地区: {region}\n"
#             f"内容: {content}\n"
#         )
    
#     context_text = "\n".join(context_parts)
    
#     # 保持简洁的系统提示，专注于角色和核心原则
#     system_prompt = """你是专业的新闻问答助手，严格基于参考资料回答问题。"""

#     # 增强的用户提示，引导逐步推理
#     user_prompt = f"""# 参考资料
# {context_text}

# # 需要回答的问题
# {question}

# # 回答要求
# 请按照以下步骤思考并回答：

# ## 步骤1：分析问题要点
# - 理解问题的核心是什么
# - 识别关键信息：人物、事件、时间、地点、数据等

# ## 步骤2：定位相关证据
# - 在参考资料中找出与问题唯一或最相关的所有片段
# - 注意信息的时效性和相关性， 若资料存在多篇内容冲突，优先选择发布时间最新的资料内容作答

# ## 步骤3：基于证据回答
# - **严格只使用参考资料中的信息**
# - 将找到的证据自然整合到回答中
# - 明确引用信息来源（例如："根据文档3的报道..."）
# - 如果不同来源有冲突，如实说明

# ## 步骤4：最终检查
# - 确认每个事实都有参考资料支持
# - 检查是否直接回答了问题
# - 确保回答简洁清晰

# # 回答格式
# - 如果根据参考资料能回答：直接给出答案
# - 如果完全无法回答："基于现有资料，我无法回答这个问题"
# """
    
#     return [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]

# # ----------------------------
# # 辅助函数
# # ----------------------------
# def parse_self_confidence(model_answer):
#     """解析模型自报置信度"""
#     try:
#         lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
#         self_conf = None
        
#         for line in lines:
#             if "置信度" in line:
#                 val_str = line.split(':')[-1].strip()
#                 try:
#                     self_conf = float(val_str)
#                 except ValueError:
#                     logger.warning(f"置信度值无法转为浮点数: '{val_str}'")
#                     self_conf = None
#                 return self_conf
        
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

# def compute_ppl_minprob(generated_tokens, generate_logits):
#     """计算困惑度和最小词元概率"""
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
# # 模型加载
# # ----------------------------
# def load_model(model_path):
#     """加载模型和tokenizer"""
#     logger.info("加载模型和tokenizer...")
    
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4"
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         padding_side="left"
#     )
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
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
# # 初始化检索系统
# # ----------------------------
# def initialize_retrieval():
#     """初始化检索系统"""
#     global retrieval_manager
    
#     logger.info("初始化混合检索系统...")
#     try:
#         retrieval_manager = ImprovedHybridRetrievalManager(
#             MILVUS_HOST, 
#             MILVUS_PORT,
#             use_reranker=True
#         )
#         retrieval_manager.initialize()
#         logger.info("混合检索系统初始化成功")
#         return True
#     except Exception as e:
#         logger.error(f"混合检索系统初始化失败: {e}")
#         return False

# # ----------------------------
# # 生成函数（添加时间统计参数）
# # ----------------------------
# def generate_confidence(model, tokenizer, question, start_time=None):
#     """生成置信度"""
#     messages = build_confidence_messages(question)
    
#     prompt_text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     prompt_len = inputs['input_ids'].size(1)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=10,
#             return_dict_in_generate=True,
#             output_scores=False,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     generated_tokens = outputs.sequences[0, prompt_len:]
#     confidence_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
#     self_conf = parse_self_confidence(confidence_output)
    
#     logger.info(f"生成置信度: {self_conf}, 原始输出: {confidence_output}")
    
#     if start_time:
#         generation_time = time.perf_counter() - start_time
#         logger.info(f"置信度生成耗时: {generation_time:.3f}秒")
#         return self_conf, confidence_output, generation_time
#     else:
#         return self_conf, confidence_output

# def generate_answer_with_context(model, tokenizer, question, contexts=None, start_time=None):
#     """生成答案"""
#     if contexts:
#         logger.info(f"使用RAG模式生成答案，上下文数量: {len(contexts)}")
#         messages = build_rag_answer_messages_v2(question, contexts)
#     else:
#         logger.info("使用普通模式生成答案")
#         messages = build_answer_messages(question)
    
#     prompt_text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     prompt_len = inputs['input_ids'].size(1)
    
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
    
#     generated_tokens = outputs.sequences[0, prompt_len:]
#     raw_model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
#     ppl, min_token_prob = float('inf'), 0.0
#     if outputs.scores:
#         generate_logits = torch.stack(outputs.scores, dim=0)
#         ppl, min_token_prob = compute_ppl_minprob(generated_tokens, generate_logits)
    
#     logger.info(f"生成答案完成，ppl: {ppl:.4f}, min_token_prob: {min_token_prob:.4f}")
    
#     if start_time:
#         generation_time = time.perf_counter() - start_time
#         logger.info(f"答案生成耗时: {generation_time:.3f}秒")
#         return raw_model_output, ppl, min_token_prob, generation_time
#     else:
#         return raw_model_output, ppl, min_token_prob

# # ----------------------------
# # 主函数（添加时间统计）
# # ----------------------------
# def main():
#     # 初始化
#     jieba.initialize()
    
#     # 加载模型
#     model, tokenizer = load_model(MODEL_BASE_PATH)
    
#     # 初始化检索系统
#     retrieval_enabled = initialize_retrieval()
#     if not retrieval_enabled:
#         logger.warning("检索系统初始化失败，将只使用模型直接推理")
    
#     # 加载输入数据
#     with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
#         input_data = json.load(f)
    
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
#     # 时间统计变量
#     total_samples = len(input_data)
#     total_processing_time = 0.0
#     total_confidence_time = 0.0
#     total_answer_time = 0.0
#     total_retrieval_time = 0.0
#     total_rag_answer_time = 0.0
#     sample_times = []
    
#     # 处理每个问题
#     with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
#         f_out.write("[\n")
#         for idx, item in enumerate(tqdm(input_data, desc="处理问题")):
#             question = item.get("question", "").strip()
#             if not question:
#                 continue
            
#             logger.info(f"\n{'='*60}")
#             logger.info(f"处理第 {idx+1} 个问题: {question}")
#             logger.info(f"{'='*60}")
            
#             try:
#                 # 记录样本开始处理时间
#                 sample_start_time = time.perf_counter()
                
#                 # 1. 提取时间信息（优化点：现在返回更多字段）
#                 logger.info("提取时间信息...")
#                 time_extract_start = time.perf_counter()
#                 time_range = extract_time_from_question(question)
#                 time_extract_time = time.perf_counter() - time_extract_start
                
#                 # 2. 生成置信度
#                 logger.info("生成置信度...")
#                 confidence_start_time = time.perf_counter()
#                 self_conf, confidence_output, confidence_time = generate_confidence(
#                     model, tokenizer, question, start_time=confidence_start_time
#                 )
#                 total_confidence_time += confidence_time
                
#                 # 3. 生成原始答案
#                 logger.info("生成原始答案...")
#                 answer_start_time = time.perf_counter()
#                 raw_model_output, ppl, min_token_prob, answer_time = generate_answer_with_context(
#                     model, tokenizer, question, contexts=None, start_time=answer_start_time
#                 )
#                 total_answer_time += answer_time
                
#                 # 4. 决策
#                 logger.info("进行决策判断...")
#                 should_answer_directly = (
#                     ppl <= PPL_THRESHOLD and
#                     min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
#                     (self_conf is not None and self_conf >= SELF_CONF_THRESHOLD)
#                 )
                
#                 logger.info(f"阈值判断: ppl={ppl:.4f}<={PPL_THRESHOLD}, "
#                            f"min_prob={min_token_prob:.4f}>={MIN_TOKEN_PROB_THRESHOLD}, "
#                            f"conf={self_conf}>={SELF_CONF_THRESHOLD if self_conf else 'None'}")
                
#                 final_answer = ""
#                 use_retrieval = False
#                 retrieved_docs = []
#                 has_supporting_info = False
#                 extracted_keywords = []
#                 retrieval_time = 0.0
#                 rag_answer_time = 0.0
#                 total_retrieved_docs = 0
#                 filtered_docs_count = 0
                
#                 if should_answer_directly:
#                     final_answer = raw_model_output
#                     decision_reason = "满足阈值条件，直接使用模型答案"
#                     logger.info("决策: 直接回答")
#                 else:
#                     use_retrieval = True
#                     decision_reason = "未满足阈值条件，启用混合检索"
#                     logger.info("决策: 使用RAG")
                    
#                     if retrieval_enabled and retrieval_manager:
#                         # 提取关键词
#                         keyword_extract_start = time.perf_counter()
#                         extracted_keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
#                         keyword_extract_time = time.perf_counter() - keyword_extract_start
                        
#                         # 执行混合检索（优化点：自动传递 time_range）
#                         logger.info("开始混合检索...")
#                         retrieval_start_time = time.perf_counter()
#                         hybrid_results = retrieval_manager.hybrid_search_v2(question, time_range)
#                         retrieval_end_time = time.perf_counter()
#                         retrieval_time = retrieval_end_time - retrieval_start_time
#                         total_retrieval_time += retrieval_time
                        
#                         # 获取过滤后的结果
#                         filtered_docs = hybrid_results.get("filtered_results", [])
#                         all_docs = hybrid_results.get("final_results", [])
                        
#                         retrieved_docs = filtered_docs
#                         total_retrieved_docs = len(all_docs)
#                         filtered_docs_count = len(filtered_docs)
                        
#                         has_supporting_info = len(retrieved_docs) > 0
#                         logger.info(f"检索到 {len(all_docs)} 个文档，其中 {len(filtered_docs)} 个分数大于0.1")
#                         logger.info(f"检索耗时: {retrieval_time:.3f}秒")
                        
#                         if has_supporting_info:
#                             # 基于检索结果生成答案
#                             logger.info("基于检索结果生成答案...")
#                             rag_answer_start_time = time.perf_counter()
#                             rag_answer, rag_ppl, rag_min_prob, rag_answer_time = generate_answer_with_context(
#                                 model, tokenizer, question, contexts=retrieved_docs, start_time=rag_answer_start_time
#                             )
#                             total_rag_answer_time += rag_answer_time
#                             final_answer = rag_answer
#                             decision_reason += "，检索到相关信息，基于检索结果生成答案"
                            
#                             ppl, min_token_prob = rag_ppl, rag_min_prob
#                         else:
#                             final_answer = random.choice(REJECTION_RESPONSES)
#                             decision_reason += "，未检索到相关信息，拒绝回答"
#                     else:
#                         final_answer = random.choice(REJECTION_RESPONSES)
#                         decision_reason += "，检索系统不可用，拒绝回答"
                
#                 # 计算样本总处理时间
#                 sample_end_time = time.perf_counter()
#                 sample_processing_time = sample_end_time - sample_start_time
#                 total_processing_time += sample_processing_time
                
#                 # 构建结果
#                 retrieval_decision = {
#                     "ppl": float(round(ppl, 4)) if np.isfinite(ppl) else None,
#                     "min_token_prob": float(round(min_token_prob, 4)),
#                     "self_reported_confidence": float(round(self_conf, 4)) if isinstance(self_conf, (int, float)) else None,
#                     "confidence_output": confidence_output,
#                     "should_answer_directly": bool(should_answer_directly),
#                     "use_retrieval": use_retrieval,
#                     "decision_reason": decision_reason,
#                     "final_model_answer": final_answer,
#                     "filtered_docs_count": filtered_docs_count,
#                 }
                
#                 # 添加详细时间统计
#                 retrieval_decision["time_stats"] = {
#                     "total_processing_time": round(sample_processing_time, 3),
#                     "confidence_generation_time": round(confidence_time, 3),
#                     "answer_generation_time": round(answer_time, 3),
#                     "time_extraction_time": round(time_extract_time, 3),
#                     "keyword_extraction_time": round(keyword_extract_time, 3) if use_retrieval else 0,
#                     "retrieval_time": round(retrieval_time, 3) if use_retrieval else 0,
#                     "rag_answer_generation_time": round(rag_answer_time, 3) if use_retrieval and has_supporting_info else 0,
#                 }
                
#                 if use_retrieval:
#                     if extracted_keywords:
#                         retrieval_decision["extracted_keywords"] = extracted_keywords
#                     retrieval_decision["filtered_docs_count"] = filtered_docs_count
                
#                 # 只保存过滤后的文档预览
#                 retrieval_decision["retrieved_docs_preview"] = [
#                     {
#                         "doc_id": doc.get("doc_id", ""),
#                         "title": doc.get("title", ""),
#                         "content_preview": doc.get("content", ""),
#                         "publish_date": doc.get("publish_date_str", ""),
#                         "retrieval_type": doc.get("retrieval_type", "unknown"),
#                         "final_score": float(round(doc.get("final_score", 0), 4)),
#                         "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
#                     }
#                     for doc in retrieved_docs
#                 ]
                
#                 # 添加时间信息
#                 if time_range:
#                     retrieval_decision["time_extraction"] = {
#                         "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d") if time_range.get("extracted_date") else None,
#                         "date_type": time_range.get("date_type", time_range.get("relative_pattern", "unknown")),
#                         "range_days": time_range.get("range_days", "N/A"),
#                         "start_date": time_range.get("start_date_str"),
#                         "end_date": time_range.get("end_date_str")
#                     }
                
#                 output_item = {
#                     "source_id": item.get("source_id", idx),
#                     "question": question,
#                     "standard_answer": item.get("answer", ""),
#                     "model_answer": final_answer,
#                     "raw_model_output": raw_model_output,
#                     "retrieval_decision": retrieval_decision
#                 }
                
#                 # 存储样本时间详情
#                 sample_times.append({
#                     "sample_id": idx,
#                     "question": question[:50] + "..." if len(question) > 50 else question,
#                     "total_time": sample_processing_time,
#                     "confidence_time": confidence_time,
#                     "answer_time": answer_time,
#                     "retrieval_time": retrieval_time,
#                     "rag_answer_time": rag_answer_time,
#                     "use_retrieval": use_retrieval,
#                     "retrieved_docs_total": total_retrieved_docs,
#                     "filtered_docs_count": filtered_docs_count,
#                     "final_answer_length": len(final_answer)
#                 })
                
#                 # 写入
#                 if idx > 0:
#                     f_out.write(",\n")
#                 f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
#                 f_out.flush()
                
#                 logger.info(f"第 {idx+1} 个问题处理完成，总耗时: {sample_processing_time:.3f}秒")
                
#             except Exception as e:
#                 logger.error(f"第{idx}条数据处理失败: {e}", exc_info=True)
#                 continue
        
#         f_out.write("\n]")
    
#     # 时间统计汇总
#     if total_samples > 0:
#         avg_processing_time = total_processing_time / total_samples
#         avg_confidence_time = total_confidence_time / total_samples
#         avg_answer_time = total_answer_time / total_samples
#         avg_retrieval_time = total_retrieval_time / total_samples if total_retrieval_time > 0 else 0
#         avg_rag_answer_time = total_rag_answer_time / total_samples if total_rag_answer_time > 0 else 0
        
#         rag_samples = sum(1 for sample in sample_times if sample["use_retrieval"])
#         direct_answer_samples = total_samples - rag_samples
        
#         time_stats = {
#             "total_samples": total_samples,
#             "total_processing_time_seconds": round(total_processing_time, 3),
#             "average_processing_time_seconds": round(avg_processing_time, 3),
#             "breakdown": {
#                 "confidence_generation": {
#                     "total_seconds": round(total_confidence_time, 3),
#                     "average_seconds": round(avg_confidence_time, 3),
#                     "percentage": round((total_confidence_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
#                 },
#                 "answer_generation": {
#                     "total_seconds": round(total_answer_time, 3),
#                     "average_seconds": round(avg_answer_time, 3),
#                     "percentage": round((total_answer_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
#                 },
#                 "retrieval": {
#                     "total_seconds": round(total_retrieval_time, 3),
#                     "average_seconds": round(avg_retrieval_time, 3) if rag_samples > 0 else 0,
#                     "percentage": round((total_retrieval_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0,
#                     "rag_samples_count": rag_samples
#                 },
#                 "rag_answer_generation": {
#                     "total_seconds": round(total_rag_answer_time, 3),
#                     "average_seconds": round(avg_rag_answer_time, 3) if rag_samples > 0 else 0,
#                     "percentage": round((total_rag_answer_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
#                 }
#             },
#             "decision_statistics": {
#                 "direct_answer_samples": direct_answer_samples,
#                 "rag_answer_samples": rag_samples,
#                 "direct_answer_percentage": round((direct_answer_samples / total_samples) * 100, 2) if total_samples > 0 else 0,
#                 "rag_answer_percentage": round((rag_samples / total_samples) * 100, 2) if total_samples > 0 else 0
#             },
#             "sample_details": sample_times,
#             "performance_summary": {
#                 "samples_per_hour": round(3600 / avg_processing_time, 2) if avg_processing_time > 0 else 0,
#                 "direct_answer_speed": f"{avg_confidence_time + avg_answer_time:.3f}s/answer",
#                 "rag_speed": f"{avg_confidence_time + avg_answer_time + avg_retrieval_time + avg_rag_answer_time:.3f}s/answer" if rag_samples > 0 else "N/A",
#                 "retrieval_efficiency": f"{avg_retrieval_time:.3f}s/query" if rag_samples > 0 else "N/A"
#             },
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         logger.info("\n" + "="*60)
#         logger.info("时间统计摘要:")
#         logger.info("="*60)
#         logger.info(f"处理样本总数: {total_samples}")
#         logger.info(f"决策统计: 直接回答 {direct_answer_samples} 个，RAG回答 {rag_samples} 个")
#         logger.info(f"总处理时间: {total_processing_time:.2f}秒")
#         logger.info(f"平均处理时间: {avg_processing_time:.3f}秒/样本")
#         logger.info(f"  - 置信度生成: {avg_confidence_time:.3f}秒 ({time_stats['breakdown']['confidence_generation']['percentage']}%)")
#         logger.info(f"  - 答案生成: {avg_answer_time:.3f}秒 ({time_stats['breakdown']['answer_generation']['percentage']}%)")
#         if rag_samples > 0:
#             logger.info(f"  - 检索时间: {avg_retrieval_time:.3f}秒 ({time_stats['breakdown']['retrieval']['percentage']}%)")
#             logger.info(f"  - RAG答案生成: {avg_rag_answer_time:.3f}秒 ({time_stats['breakdown']['rag_answer_generation']['percentage']}%)")
#         logger.info(f"处理速度: {time_stats['performance_summary']['samples_per_hour']} 样本/小时")
    
#     logger.info(f"✅ 推理完成，结果保存至 {OUTPUT_PATH}")
#     logger.info(f"日志文件: generate_train_optimized.log")


# if __name__ == "__main__":
#     main()



# =================================================bge-m3=================================================
# ==========================================
# 优化后的问答系统完整代码（支持BGE-M3稠密+稀疏向量）
# generate_train_optimized.py
# ==========================================

import os
import json
import torch
import logging
import random
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
import re
from datetime import datetime, timedelta

# RAG相关库
from pymilvus import connections, Collection
from sentence_transformers import CrossEncoder
from pymilvus.model.hybrid import BGEM3EmbeddingFunction  # BGE-M3嵌入函数

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_train_optimized.log',
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
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk/test_qwen2.5_450.json"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# Milvus配置（使用新建的稠密+稀疏集合）
# ----------------------------
MILVUS_HOST = "122.226.69.234"
MILVUS_PORT = "3042"
MILVUS_COLLECTION_NAME = "news_collection_m3"          # 使用新集合
EMBEDDING_MODEL = "BAAI/bge-m3"                         # BGE-M3模型名称
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 检索参数
TOP_K_RETRIEVAL = 20  # 初始检索数量(增加)
TOP_K_FINAL = 5       # 最终返回数量

# ----------------------------
# 拒绝阈值
# ----------------------------
PPL_THRESHOLD = 1.1022038499105375
MIN_TOKEN_PROB_THRESHOLD = 0.3661533556634178
SELF_CONF_THRESHOLD = 0.44508323935997046

# ----------------------------
# 拒答模板
# ----------------------------
REJECTION_RESPONSES = [
    "抱歉，我无法回答这个问题。",
    "根据现有的信息，我无法回答这个问题。",
    "我暂时没有足够的信息来回答这个问题。"
]

# ----------------------------
# 改进的时间提取（保持不变）
# ----------------------------
def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
    """
    从问题中提取时间信息
    """
    date_patterns = [
        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', 'ymd'),
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ymd'),
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', 'ymd'),
        (r'(\d{4})年(\d{1,2})月', 'ym'),
        (r'(\d{4})-(\d{1,2})', 'ym'),
        (r'(\d{4})/(\d{1,2})', 'ym'),
        (r'(\d{4})年', 'y'),
    ]

    for pattern, date_type in date_patterns:
        match = re.search(pattern, question)
        if match:
            try:
                groups = match.groups()

                if date_type == 'ymd':
                    year, month, day = map(int, groups)
                    extracted_date = datetime(year, month, day)
                    days_range = 30
                elif date_type == 'ym':
                    year, month = map(int, groups)
                    extracted_date = datetime(year, month, 1)
                    days_range = 60
                elif date_type == 'y':
                    year = int(groups[0])
                    extracted_date = datetime(year, 6, 30)
                    days_range = 365
                else:
                    continue

                start_date = extracted_date - timedelta(days=days_range)
                end_date = extracted_date + timedelta(days=days_range)

                logger.info(f"提取时间 [{date_type}]: {extracted_date.strftime('%Y-%m-%d')}")
                logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} (共{days_range*2}天)")

                return {
                    "extracted_date": extracted_date,
                    "start_timestamp": int(start_date.timestamp()),
                    "end_timestamp": int(end_date.timestamp()),
                    "start_date_str": start_date.strftime("%Y-%m-%d"),
                    "end_date_str": end_date.strftime("%Y-%m-%d"),
                    "has_specific_date": True,
                    "date_type": date_type,
                    "range_days": days_range * 2
                }

            except Exception as e:
                logger.warning(f"时间解析失败: {e}")
                continue

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
            logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

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
# 重排序器类（保持不变）
# ----------------------------
class EnhancedReranker:
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
        if not documents or self.reranker is None:
            return documents[:top_k]

        pairs = []
        for doc in documents:
            doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"[:500]
            pairs.append([question, doc_text])

        logger.info(f"重排序 {len(documents)} 个文档...")
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]

        for doc, score in zip(documents, rerank_scores):
            doc['rerank_score'] = float(score)
            if time_decay and current_timestamp:
                publish_time = doc.get('publish_time', 0)
                if publish_time > 0:
                    days_diff = (current_timestamp - publish_time) / 86400
                    time_factor = max(0.5, 1.0 - (days_diff / 30) * 0.05)
                    doc['final_score'] = score * time_factor
                else:
                    doc['final_score'] = score
            else:
                doc['final_score'] = score

        ranked_docs = sorted(documents, key=lambda x: x['final_score'], reverse=True)

        logger.info(f"重排序完成，返回top-{top_k}")
        for i, doc in enumerate(ranked_docs[:top_k]):
            logger.info(f"  Rank {i+1}: {doc.get('title', '')[:50]}... "
                       f"(分数: {doc['final_score']:.4f})")

        return ranked_docs[:top_k]

# ----------------------------
# 改进的混合检索管理器（稠密+稀疏）
# ----------------------------
class ImprovedHybridRetrievalManager:
    def __init__(self, host: str, port: str, use_reranker: bool = True):
        self.host = host
        self.port = port
        self.connection = None
        self.collection = None
        self.embedding_model = None
        self.reranker = None
        self.use_reranker = use_reranker

    def initialize(self):
        """初始化Milvus连接、BGE-M3模型和重排序器"""
        try:
            logger.info(f"连接 Milvus ({self.host}:{self.port})...")
            self.connection = connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )

            self.collection = Collection(MILVUS_COLLECTION_NAME)
            logger.info(f"已连接到集合: {MILVUS_COLLECTION_NAME}")

            logger.info("加载 BGE-M3 嵌入模型...")
            self.embedding_model = BGEM3EmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                device='cpu',
                use_fp16=False,
                normalize_embeddings=True,
            )

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

    def get_query_embeddings(self, question: str):
        """生成查询的稠密和稀疏向量"""
        if self.embedding_model is None:
            raise ValueError("嵌入模型未初始化")
        embeddings = self.embedding_model.encode_queries([question])
        return embeddings['dense'][0], embeddings['sparse']

    def dense_search(
        self,
        question: str,
        time_range: Optional[Dict[str, Any]] = None,
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """稠密向量检索"""
        if self.collection is None or self.embedding_model is None:
            raise ValueError("检索管理器未初始化")

        try:
            dense_vec, _ = self.get_query_embeddings(question)

            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10, "ef": 50}
            }

            expr = None
            if time_range and time_range.get("has_specific_date", False):
                start_ts = time_range.get("start_timestamp")
                end_ts = time_range.get("end_timestamp")
                if start_ts and end_ts:
                    expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
                    logger.info(f"稠密检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")

            results = self.collection.search(
                data=[dense_vec.tolist()],
                anns_field="dense_embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["doc_id", "title", "content", "publish_time", "publish_date_str", "region"],
                consistency_level="Eventually"
            )

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
                            "retrieval_type": "dense"
                        }
                        formatted_results.append(doc_info)

            logger.info(f"稠密检索完成: {len(formatted_results)} 个文档")
            return formatted_results

        except Exception as e:
            logger.error(f"稠密检索失败: {e}")
            return []

    def sparse_search(
        self,
        question: str,
        time_range: Optional[Dict[str, Any]] = None,
        top_k: int = TOP_K_RETRIEVAL
    ) -> List[Dict]:
        """稀疏向量检索"""
        if self.collection is None or self.embedding_model is None:
            raise ValueError("检索管理器未初始化")

        try:
            _, sparse_vec = self.get_query_embeddings(question)

            search_params = {
                "metric_type": "IP",
                "params": {}
            }

            expr = None
            if time_range and time_range.get("has_specific_date", False):
                start_ts = time_range.get("start_timestamp")
                end_ts = time_range.get("end_timestamp")
                if start_ts and end_ts:
                    expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
                    logger.info(f"稀疏检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")

            results = self.collection.search(
                data=[sparse_vec],
                anns_field="sparse_embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["doc_id", "title", "content", "publish_time", "publish_date_str", "region"],
                consistency_level="Eventually"
            )

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
                            "retrieval_type": "sparse"
                        }
                        formatted_results.append(doc_info)

            logger.info(f"稀疏检索完成: {len(formatted_results)} 个文档")
            return formatted_results

        except Exception as e:
            logger.error(f"稀疏检索失败: {e}")
            return []

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """倒数排名融合（RRF）"""
        logger.info(f"开始RRF融合，k={k}")

        doc_id_to_results = {}

        for rank, result in enumerate(dense_results, 1):
            doc_id = result.get("doc_id", "")
            if doc_id:
                if doc_id not in doc_id_to_results:
                    doc_id_to_results[doc_id] = {
                        "doc_info": result,
                        "dense_rank": rank,
                        "sparse_rank": float('inf')
                    }
                else:
                    doc_id_to_results[doc_id]["dense_rank"] = rank

        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.get("doc_id", "")
            if doc_id:
                if doc_id not in doc_id_to_results:
                    doc_id_to_results[doc_id] = {
                        "doc_info": result,
                        "dense_rank": float('inf'),
                        "sparse_rank": rank
                    }
                else:
                    doc_id_to_results[doc_id]["sparse_rank"] = rank

        rrf_results = []
        for doc_id, info in doc_id_to_results.items():
            dense_rrf = 1 / (k + info["dense_rank"]) if info["dense_rank"] != float('inf') else 0
            sparse_rrf = 1 / (k + info["sparse_rank"]) if info["sparse_rank"] != float('inf') else 0
            total_rrf_score = dense_rrf + sparse_rrf

            rrf_result = {
                "doc_info": info["doc_info"],
                "dense_rank": info["dense_rank"],
                "sparse_rank": info["sparse_rank"],
                "rrf_score": total_rrf_score
            }
            rrf_results.append(rrf_result)

        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        final_results = []
        for rrf_result in rrf_results:
            doc_info = rrf_result["doc_info"].copy()
            doc_info["score"] = rrf_result["rrf_score"]
            doc_info["rrf_score"] = rrf_result["rrf_score"]
            doc_info["dense_rank"] = rrf_result["dense_rank"]
            doc_info["sparse_rank"] = rrf_result["sparse_rank"]
            final_results.append(doc_info)

        logger.info(f"RRF融合完成: {len(final_results)} 个文档")
        return final_results

    def hybrid_search_v2(
        self,
        question: str,
        time_range: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """改进的混合检索流程（稠密+稀疏）"""
        logger.info(f"开始混合检索: {question}")

        dense_results = self.dense_search(question, time_range, TOP_K_RETRIEVAL)
        sparse_results = self.sparse_search(question, time_range, TOP_K_RETRIEVAL)

        logger.info(f"初始检索: 稠密{len(dense_results)}, 稀疏{len(sparse_results)}")

        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)

        logger.info(f"RRF融合后: {len(fused_results)} 个文档")

        if self.reranker and fused_results:
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

        filtered_results = [doc for doc in final_results if doc.get('final_score', 0) > 0.1]

        logger.info(f"重排序后top-{TOP_K_FINAL}: {len(final_results)} 个文档")
        logger.info(f"分数过滤后(>0.1): {len(filtered_results)} 个文档")

        return {
            "all_retrieved": {
                "dense_results": dense_results[:10],
                "sparse_results": sparse_results[:10],
                "fused_results": fused_results[:10],
                "reranked_results": final_results
            },
            "final_results": final_results,
            "filtered_results": filtered_results
        }

# 全局检索管理器
retrieval_manager = None

# ----------------------------
# Prompt构建函数（保持不变）
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

def build_answer_messages(question):
    messages = [
        {
            "role": "system",
            "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
        },
        {
            "role": "user",
            "content": f"请回答下面的问题：\n\n问题：{question}"
        }
    ]
    return messages

def build_rag_answer_messages_v2(question: str, contexts: List[Dict]) -> List[Dict]:
    if not contexts:
        logger.warning("无检索结果，使用基础prompt")
        return [
            {
                "role": "system",
                "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答“基于现有资料，我无法回答这个问题”。"
            },
            {
                "role": "user",
                "content": f"问题: {question}"
            }
        ]

    context_parts = []
    for i, doc in enumerate(contexts, 1):
        title = doc.get("title", f"文档{i}")
        content = doc.get("content", "")[:800]
        pub_date = doc.get("publish_date_str", "未知")
        region = doc.get("region", "未知")

        context_parts.append(
            f"【文档{i}】\n"
            f"标题: {title}\n"
            f"时间: {pub_date}\n"
            f"地区: {region}\n"
            f"内容: {content}\n"
        )

    context_text = "\n".join(context_parts)

    system_prompt = """你是专业的新闻问答助手，严格基于参考资料回答问题。"""

    user_prompt = f"""# 参考资料
{context_text}

# 需要回答的问题
{question}

# 回答要求
请按照以下步骤思考并回答：

## 步骤1：分析问题要点
- 理解问题的核心是什么
- 识别关键信息：人物、事件、时间、地点、数据等

## 步骤2：定位相关证据
- 在参考资料中找出与问题唯一或最相关的所有片段
- 注意信息的时效性和相关性， 若资料存在多篇内容冲突，优先选择发布时间最新的资料内容作答

## 步骤3：基于证据回答
- **严格只使用参考资料中的信息**
- 将找到的证据自然整合到回答中
- 明确引用信息来源（例如："根据文档3的报道..."）
- 如果不同来源有冲突，如实说明

## 步骤4：最终检查
- 确认每个事实都有参考资料支持
- 检查是否直接回答了问题
- 确保回答简洁清晰

# 回答格式
- 如果根据参考资料能回答：直接给出答案
- 如果完全无法回答："基于现有资料，我无法回答这个问题"
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# ----------------------------
# 辅助函数（保持不变）
# ----------------------------
def parse_self_confidence(model_answer):
    try:
        lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
        self_conf = None
        for line in lines:
            if "置信度" in line:
                val_str = line.split(':')[-1].strip()
                try:
                    self_conf = float(val_str)
                except ValueError:
                    logger.warning(f"置信度值无法转为浮点数: '{val_str}'")
                    self_conf = None
                return self_conf
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
# 模型加载
# ----------------------------
def load_model(model_path):
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
    global retrieval_manager

    logger.info("初始化混合检索系统...")
    try:
        retrieval_manager = ImprovedHybridRetrievalManager(
            MILVUS_HOST,
            MILVUS_PORT,
            use_reranker=True
        )
        retrieval_manager.initialize()
        logger.info("混合检索系统初始化成功")
        return True
    except Exception as e:
        logger.error(f"混合检索系统初始化失败: {e}")
        return False

# ----------------------------
# 生成函数（添加时间统计参数）
# ----------------------------
def generate_confidence(model, tokenizer, question, start_time=None):
    messages = build_confidence_messages(question)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs.sequences[0, prompt_len:]
    confidence_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    self_conf = parse_self_confidence(confidence_output)

    logger.info(f"生成置信度: {self_conf}, 原始输出: {confidence_output}")

    if start_time:
        generation_time = time.perf_counter() - start_time
        logger.info(f"置信度生成耗时: {generation_time:.3f}秒")
        return self_conf, confidence_output, generation_time
    else:
        return self_conf, confidence_output

def generate_answer_with_context(model, tokenizer, question, contexts=None, start_time=None):
    if contexts:
        logger.info(f"使用RAG模式生成答案，上下文数量: {len(contexts)}")
        messages = build_rag_answer_messages_v2(question, contexts)
    else:
        logger.info("使用普通模式生成答案")
        messages = build_answer_messages(question)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)

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

    generated_tokens = outputs.sequences[0, prompt_len:]
    raw_model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    ppl, min_token_prob = float('inf'), 0.0
    if outputs.scores:
        generate_logits = torch.stack(outputs.scores, dim=0)
        ppl, min_token_prob = compute_ppl_minprob(generated_tokens, generate_logits)

    logger.info(f"生成答案完成，ppl: {ppl:.4f}, min_token_prob: {min_token_prob:.4f}")

    if start_time:
        generation_time = time.perf_counter() - start_time
        logger.info(f"答案生成耗时: {generation_time:.3f}秒")
        return raw_model_output, ppl, min_token_prob, generation_time
    else:
        return raw_model_output, ppl, min_token_prob

# ----------------------------
# 主函数（移除关键词相关代码）
# ----------------------------
def main():
    # 不再需要 jieba 初始化
    # 加载模型
    model, tokenizer = load_model(MODEL_BASE_PATH)

    # 初始化检索系统
    retrieval_enabled = initialize_retrieval()
    if not retrieval_enabled:
        logger.warning("检索系统初始化失败，将只使用模型直接推理")

    # 加载输入数据
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    total_samples = len(input_data)
    total_processing_time = 0.0
    total_confidence_time = 0.0
    total_answer_time = 0.0
    total_retrieval_time = 0.0
    total_rag_answer_time = 0.0
    sample_times = []

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
                sample_start_time = time.perf_counter()

                # 1. 提取时间信息
                logger.info("提取时间信息...")
                time_extract_start = time.perf_counter()
                time_range = extract_time_from_question(question)
                time_extract_time = time.perf_counter() - time_extract_start

                # 2. 生成置信度
                logger.info("生成置信度...")
                confidence_start_time = time.perf_counter()
                self_conf, confidence_output, confidence_time = generate_confidence(
                    model, tokenizer, question, start_time=confidence_start_time
                )
                total_confidence_time += confidence_time

                # 3. 生成原始答案
                logger.info("生成原始答案...")
                answer_start_time = time.perf_counter()
                raw_model_output, ppl, min_token_prob, answer_time = generate_answer_with_context(
                    model, tokenizer, question, contexts=None, start_time=answer_start_time
                )
                total_answer_time += answer_time

                # 4. 决策
                logger.info("进行决策判断...")
                should_answer_directly = (
                    ppl <= PPL_THRESHOLD and
                    min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
                    (self_conf is not None and self_conf >= SELF_CONF_THRESHOLD)
                )

                logger.info(f"阈值判断: ppl={ppl:.4f}<={PPL_THRESHOLD}, "
                           f"min_prob={min_token_prob:.4f}>={MIN_TOKEN_PROB_THRESHOLD}, "
                           f"conf={self_conf}>={SELF_CONF_THRESHOLD if self_conf else 'None'}")

                final_answer = ""
                use_retrieval = False
                retrieved_docs = []
                has_supporting_info = False
                retrieval_time = 0.0
                rag_answer_time = 0.0
                total_retrieved_docs = 0
                filtered_docs_count = 0

                if should_answer_directly:
                    final_answer = raw_model_output
                    decision_reason = "满足阈值条件，直接使用模型答案"
                    logger.info("决策: 直接回答")
                else:
                    use_retrieval = True
                    decision_reason = "未满足阈值条件，启用混合检索"
                    logger.info("决策: 使用RAG")

                    if retrieval_enabled and retrieval_manager:
                        # 执行混合检索（不再需要关键词提取）
                        logger.info("开始混合检索...")
                        retrieval_start_time = time.perf_counter()
                        hybrid_results = retrieval_manager.hybrid_search_v2(question, time_range)
                        retrieval_end_time = time.perf_counter()
                        retrieval_time = retrieval_end_time - retrieval_start_time
                        total_retrieval_time += retrieval_time

                        filtered_docs = hybrid_results.get("filtered_results", [])
                        all_docs = hybrid_results.get("final_results", [])

                        retrieved_docs = filtered_docs
                        total_retrieved_docs = len(all_docs)
                        filtered_docs_count = len(filtered_docs)

                        has_supporting_info = len(retrieved_docs) > 0
                        logger.info(f"检索到 {len(all_docs)} 个文档，其中 {len(filtered_docs)} 个分数大于0.1")
                        logger.info(f"检索耗时: {retrieval_time:.3f}秒")

                        if has_supporting_info:
                            logger.info("基于检索结果生成答案...")
                            rag_answer_start_time = time.perf_counter()
                            rag_answer, rag_ppl, rag_min_prob, rag_answer_time = generate_answer_with_context(
                                model, tokenizer, question, contexts=retrieved_docs, start_time=rag_answer_start_time
                            )
                            total_rag_answer_time += rag_answer_time
                            final_answer = rag_answer
                            decision_reason += "，检索到相关信息，基于检索结果生成答案"
                            ppl, min_token_prob = rag_ppl, rag_min_prob
                        else:
                            final_answer = random.choice(REJECTION_RESPONSES)
                            decision_reason += "，未检索到相关信息，拒绝回答"
                    else:
                        final_answer = random.choice(REJECTION_RESPONSES)
                        decision_reason += "，检索系统不可用，拒绝回答"

                sample_end_time = time.perf_counter()
                sample_processing_time = sample_end_time - sample_start_time
                total_processing_time += sample_processing_time

                # 构建结果（移除关键词相关字段）
                retrieval_decision = {
                    "ppl": float(round(ppl, 4)) if np.isfinite(ppl) else None,
                    "min_token_prob": float(round(min_token_prob, 4)),
                    "self_reported_confidence": float(round(self_conf, 4)) if isinstance(self_conf, (int, float)) else None,
                    "confidence_output": confidence_output,
                    "should_answer_directly": bool(should_answer_directly),
                    "use_retrieval": use_retrieval,
                    "decision_reason": decision_reason,
                    "final_model_answer": final_answer,
                    "filtered_docs_count": filtered_docs_count,
                }

                retrieval_decision["time_stats"] = {
                    "total_processing_time": round(sample_processing_time, 3),
                    "confidence_generation_time": round(confidence_time, 3),
                    "answer_generation_time": round(answer_time, 3),
                    "time_extraction_time": round(time_extract_time, 3),
                    "retrieval_time": round(retrieval_time, 3) if use_retrieval else 0,
                    "rag_answer_generation_time": round(rag_answer_time, 3) if use_retrieval and has_supporting_info else 0,
                }

                if use_retrieval:
                    retrieval_decision["filtered_docs_count"] = filtered_docs_count

                # 保存过滤后的文档预览（不再包含 keywords）
                retrieval_decision["retrieved_docs_preview"] = [
                    {
                        "doc_id": doc.get("doc_id", ""),
                        "title": doc.get("title", ""),
                        "content_preview": doc.get("content", ""),
                        "publish_date": doc.get("publish_date_str", ""),
                        "retrieval_type": doc.get("retrieval_type", "unknown"),
                        "final_score": float(round(doc.get("final_score", 0), 4)),
                        "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
                    }
                    for doc in retrieved_docs
                ]

                if time_range:
                    retrieval_decision["time_extraction"] = {
                        "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d"),
                        "date_type": time_range.get("date_type", "relative"),
                        "range_days": time_range.get("range_days", "N/A"),
                        "start_date": time_range.get("start_date_str"),
                        "end_date": time_range.get("end_date_str")
                    }

                output_item = {
                    "source_id": item.get("source_id", idx),
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "model_answer": final_answer,
                    "raw_model_output": raw_model_output,
                    "retrieval_decision": retrieval_decision
                }

                sample_times.append({
                    "sample_id": idx,
                    "question": question[:50] + "..." if len(question) > 50 else question,
                    "total_time": sample_processing_time,
                    "confidence_time": confidence_time,
                    "answer_time": answer_time,
                    "retrieval_time": retrieval_time,
                    "rag_answer_time": rag_answer_time,
                    "use_retrieval": use_retrieval,
                    "retrieved_docs_total": total_retrieved_docs,
                    "filtered_docs_count": filtered_docs_count,
                    "final_answer_length": len(final_answer)
                })

                if idx > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
                f_out.flush()

                logger.info(f"第 {idx+1} 个问题处理完成，总耗时: {sample_processing_time:.3f}秒")

            except Exception as e:
                logger.error(f"第{idx}条数据处理失败: {e}", exc_info=True)
                continue

        f_out.write("\n]")

    # 时间统计（保持不变）
    if total_samples > 0:
        avg_processing_time = total_processing_time / total_samples
        avg_confidence_time = total_confidence_time / total_samples
        avg_answer_time = total_answer_time / total_samples
        avg_retrieval_time = total_retrieval_time / total_samples if total_retrieval_time > 0 else 0
        avg_rag_answer_time = total_rag_answer_time / total_samples if total_rag_answer_time > 0 else 0

        rag_samples = sum(1 for sample in sample_times if sample["use_retrieval"])
        direct_answer_samples = total_samples - rag_samples

        time_stats = {
            "total_samples": total_samples,
            "total_processing_time_seconds": round(total_processing_time, 3),
            "average_processing_time_seconds": round(avg_processing_time, 3),
            "breakdown": {
                "confidence_generation": {
                    "total_seconds": round(total_confidence_time, 3),
                    "average_seconds": round(avg_confidence_time, 3),
                    "percentage": round((total_confidence_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                },
                "answer_generation": {
                    "total_seconds": round(total_answer_time, 3),
                    "average_seconds": round(avg_answer_time, 3),
                    "percentage": round((total_answer_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                },
                "retrieval": {
                    "total_seconds": round(total_retrieval_time, 3),
                    "average_seconds": round(avg_retrieval_time, 3) if rag_samples > 0 else 0,
                    "percentage": round((total_retrieval_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0,
                    "rag_samples_count": rag_samples
                },
                "rag_answer_generation": {
                    "total_seconds": round(total_rag_answer_time, 3),
                    "average_seconds": round(avg_rag_answer_time, 3) if rag_samples > 0 else 0,
                    "percentage": round((total_rag_answer_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                }
            },
            "decision_statistics": {
                "direct_answer_samples": direct_answer_samples,
                "rag_answer_samples": rag_samples,
                "direct_answer_percentage": round((direct_answer_samples / total_samples) * 100, 2) if total_samples > 0 else 0,
                "rag_answer_percentage": round((rag_samples / total_samples) * 100, 2) if total_samples > 0 else 0
            },
            "sample_details": sample_times,
            "performance_summary": {
                "samples_per_hour": round(3600 / avg_processing_time, 2) if avg_processing_time > 0 else 0,
                "direct_answer_speed": f"{avg_confidence_time + avg_answer_time:.3f}s/answer",
                "rag_speed": f"{avg_confidence_time + avg_answer_time + avg_retrieval_time + avg_rag_answer_time:.3f}s/answer" if rag_samples > 0 else "N/A",
                "retrieval_efficiency": f"{avg_retrieval_time:.3f}s/query" if rag_samples > 0 else "N/A"
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info("\n" + "="*60)
        logger.info("时间统计摘要:")
        logger.info("="*60)
        logger.info(f"处理样本总数: {total_samples}")
        logger.info(f"决策统计: 直接回答 {direct_answer_samples} 个，RAG回答 {rag_samples} 个")
        logger.info(f"总处理时间: {total_processing_time:.2f}秒")
        logger.info(f"平均处理时间: {avg_processing_time:.3f}秒/样本")
        logger.info(f"  - 置信度生成: {avg_confidence_time:.3f}秒 ({time_stats['breakdown']['confidence_generation']['percentage']}%)")
        logger.info(f"  - 答案生成: {avg_answer_time:.3f}秒 ({time_stats['breakdown']['answer_generation']['percentage']}%)")
        if rag_samples > 0:
            logger.info(f"  - 检索时间: {avg_retrieval_time:.3f}秒 ({time_stats['breakdown']['retrieval']['percentage']}%)")
            logger.info(f"  - RAG答案生成: {avg_rag_answer_time:.3f}秒 ({time_stats['breakdown']['rag_answer_generation']['percentage']}%)")
        logger.info(f"处理速度: {time_stats['performance_summary']['samples_per_hour']} 样本/小时")

    logger.info(f"✅ 推理完成，结果保存至 {OUTPUT_PATH}")
    logger.info(f"日志文件: generate_train_optimized.log")

if __name__ == "__main__":
    main()