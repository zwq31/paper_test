# # ==========================================
# # 纯RAG问答系统完整代码
# # generate_rag_pure.py
# # ==========================================

# import os 
# import json
# import torch
# import logging
# import numpy as np
# import time  # 新增：导入time模块
# from tqdm import tqdm
# from typing import List, Dict, Optional, Any, Tuple  # 新增：导入Tuple
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import re
# from datetime import datetime, timedelta
# import jieba
# import jieba.posseg as pseg

# # RAG相关库
# from pymilvus import connections, Collection
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # ----------------------------
# # 日志配置
# # ----------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='generate_rag_pure.log',
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
# MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk/test_nodecide_basemodel.json"
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
# # 时间提取
# # ----------------------------
# def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
#     """
#     从问题中提取时间信息
#     """
    
#     # 1. 处理绝对时间
#     date_patterns = [
#         (r'(\d{4})年(\d{1,2})月(\d{1,2})日', 'ymd'),   # 年月日
#         (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ymd'),       # 2023-01-15
#         (r'(\d{4})/(\d{1,2})/(\d{1,2})', 'ymd'),       # 2023/01/15
#         (r'(\d{4})年(\d{1,2})月', 'ym'),               # 年月
#         (r'(\d{4})-(\d{1,2})', 'ym'),                  # 2023-01
#         (r'(\d{4})/(\d{1,2})', 'ym'),                  # 2023/01
#         (r'(\d{4})年', 'y'),                           # 只有年份
#     ]
    
#     for pattern, date_type in date_patterns:
#         match = re.search(pattern, question)
#         if match:
#             try:
#                 groups = match.groups()
                
#                 if date_type == 'ymd':  # 年月日 - 前后1个月
#                     year, month, day = map(int, groups)
#                     extracted_date = datetime(year, month, day)
#                     days_range = 30
#                     need_filter = True
                    
#                 elif date_type == 'ym':  # 年月 - 前后2个月
#                     year, month = map(int, groups)
#                     extracted_date = datetime(year, month, 1)
#                     days_range = 60
#                     need_filter = True
                    
#                 elif date_type == 'y':  # 只有年份 - 前后1年
#                     year = int(groups[0])
#                     extracted_date = datetime(year, 6, 30)  # 年中
#                     days_range = 365
#                     need_filter = True
                
#                 if need_filter:
#                     start_date = extracted_date - timedelta(days=days_range)
#                     end_date = extracted_date + timedelta(days=days_range)
                    
#                     start_timestamp = int(start_date.timestamp())
#                     end_timestamp = int(end_date.timestamp())
                    
#                     logger.info(f"提取时间 [{date_type}]: {extracted_date.strftime('%Y-%m-%d')}")
                    
#                     return {
#                         "extracted_date": extracted_date,
#                         "start_timestamp": start_timestamp,
#                         "end_timestamp": end_timestamp,
#                         "start_date_str": start_date.strftime("%Y-%m-%d"),
#                         "end_date_str": end_date.strftime("%Y-%m-%d"),
#                         "has_specific_date": True,
#                         "date_type": date_type,
#                         "range_days": days_range * 2
#                     }
                
#             except Exception as e:
#                 logger.warning(f"时间解析失败: {e}")
#                 continue
    
#     # 2. 处理相对时间
#     relative_time_patterns = {
#         r'今天|当天': (0, 1),
#         r'昨天|昨日': (-1, 1),
#         r'前天': (-2, 1),
#         r'大前天': (-3, 1),
#         r'上周|上星期': (-7, 7),
#         r'本周|这周': (-3, 3),
#         r'上个月|上月': (-30, 30),
#         r'这个月|本月': (-15, 15),
#         r'去年|上一年': (-365, 365),
#         r'今年|本年': (-180, 180),
#         r'最近|近期': (-30, 0),
#     }
    
#     current_date = datetime.now()
#     for pattern, (days_offset, days_range) in relative_time_patterns.items():
#         if re.search(pattern, question):
#             center_date = current_date + timedelta(days=days_offset)
#             start_date = center_date - timedelta(days=abs(days_range))
#             end_date = center_date + timedelta(days=abs(days_range))
            
#             logger.info(f"相对时间提取: {pattern}")
            
#             return {
#                 "extracted_date": center_date,
#                 "start_timestamp": int(start_date.timestamp()),
#                 "end_timestamp": int(end_date.timestamp()),
#                 "start_date_str": start_date.strftime("%Y-%m-%d"),
#                 "end_date_str": end_date.strftime("%Y-%m-%d"),
#                 "has_specific_date": True,
#                 "is_relative": True,
#                 "relative_pattern": pattern
#             }
    
#     logger.info("未提取到时间信息，不进行时间过滤")
#     return None

# # ----------------------------
# # 关键词提取
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
# # 重排序器类
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
#         current_timestamp: Optional[int] = None
#     ) -> List[Dict]:
#         """
#         重排序文档
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
            
#             # 时间衰减
#             if time_decay and current_timestamp:
#                 publish_time = doc.get('publish_time', 0)
#                 if publish_time > 0:
#                     days_diff = (current_timestamp - publish_time) / 86400
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
# # 混合检索管理器
# # ----------------------------
# class HybridRetrievalManager:
#     """混合检索管理器"""
    
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
#         """获取文本的嵌入向量"""
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
#                 # 对关键词进行简单的转义处理
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
    
#     def hybrid_search(
#         self, 
#         question: str, 
#         time_range: Optional[Dict] = None
#     ) -> Dict[str, Any]:
#         """
#         混合检索流程
#         """
#         logger.info(f"开始混合检索: {question}")
        
#         # 1. 分别检索
#         vector_results = self.vector_search(question, time_range, TOP_K_RETRIEVAL)
#         keyword_results = self.keyword_search(question, time_range, TOP_K_RETRIEVAL)
        
#         logger.info(f"初始检索: 向量{len(vector_results)}, 关键词{len(keyword_results)}")
        
#         # 2. RRF融合
#         fused_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
#         logger.info(f"RRF融合后: {len(fused_results)} 个文档")
        
#         # 3. 重排序
#         if self.reranker and fused_results:
#             current_ts = int(time.time())
#             final_results = self.reranker.rerank(
#                 question, 
#                 fused_results, 
#                 top_k=TOP_K_FINAL,
#                 time_decay=True,
#                 current_timestamp=current_ts
#             )
#         else:
#             final_results = fused_results[:TOP_K_FINAL]
        
#         # 新增：过滤出分数大于0.1的文档
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
#             "filtered_results": filtered_results  # 新增：过滤后的结果
#         }

# # 全局检索管理器
# retrieval_manager = None

# # ----------------------------
# # RAG Prompt构建函数
# # ----------------------------
# def build_rag_prompt(question: str, contexts: List[Dict]) -> List[Dict]:
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
#             f"内容: {content}\n"
#         )
    
#     context_text = "\n".join(context_parts)
    
# #     # 保持简洁的系统提示，专注于角色和核心原则
# #     system_prompt = """你是专业的新闻问答助手，严格基于参考资料回答问题。"""

# #     # 增强的用户提示，引导逐步推理
# #     user_prompt = f"""# 参考资料
# # {context_text}

# # # 需要回答的问题
# # {question}

# # # 回答要求
# # 请按照以下步骤思考并回答：

# # ## 步骤1：分析问题要点
# # - 理解问题的核心是什么
# # - 识别关键信息：人物、事件、时间、地点、数据等

# # ## 步骤2：定位相关证据
# # - 在参考资料中找出与问题唯一或最相关的所有片段
# # - 注意信息的时效性和相关性， 若资料存在多篇内容冲突，优先选择发布时间最新的资料内容作答

# # ## 步骤3：基于证据回答
# # - **严格只使用参考资料中的信息**
# # - 将找到的证据自然整合到回答中
# # - 明确引用信息来源（例如："根据文档3的报道..."）
# # - 如果不同来源有冲突，如实说明

# # ## 步骤4：最终检查
# # - 确认每个事实都有参考资料支持
# # - 检查是否直接回答了问题
# # - 确保回答简洁清晰

# # # 回答格式
# # - 如果根据参考资料能回答：直接给出答案
# # - 如果根据参考资料无法回答："基于现有资料，我无法回答这个问题"
# # """
    
# #     return [
# #         {"role": "system", "content": system_prompt},
# #         {"role": "user", "content": user_prompt}
# #     ]
#     system_prompt = """你是一个新闻问答助手。你的任务是基于提供的参考资料回答用户问题。

#     核心原则:
#     请按照以下步骤思考并回答：

#     ## 步骤1：分析问题要点
#     - 理解问题的核心是什么
#     - 识别关键信息：人物、事件、时间、地点、数据等

#     ## 步骤2：定位相关证据
#     - 在参考资料中找出与问题唯一或最相关的所有片段
#     - 注意信息的时效性和相关性， 若资料存在多篇内容冲突，优先选择发布时间最新的资料内容作答

#     ## 步骤3：基于证据回答
#     - **严格只使用参考资料中的信息**
#     - 将找到的证据自然整合到回答中
#     - 明确引用信息来源（例如："根据文档3的报道..."）
#     - 如果不同来源有冲突，如实说明

#     ## 步骤4：最终检查
#     - 确认每个事实都有参考资料支持
#     - 检查是否直接回答了问题
#     - 确保回答简洁清晰
#     回答格式:
#     - 如果根据参考资料能回答: 直接给出答案
#     - 如果根据参考资料不能回答: "根据现有资料，我无法回答这个问题。"，即便你知道这个问题答案，如果参考资料没有相关信息，也要回答"根据现有资料，我无法回答这个问题。"
#     """
        
#     user_prompt = f"""参考资料:
# {context_text}

# 问题: {question}

# 请基于上述资料回答问题。"""
        
#     return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]

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
#         retrieval_manager = HybridRetrievalManager(
#             MILVUS_HOST, 
#             MILVUS_PORT,
#             use_reranker=True  # 启用重排序
#         )
#         retrieval_manager.initialize()
#         logger.info("混合检索系统初始化成功")
#         return True
#     except Exception as e:
#         logger.error(f"混合检索系统初始化失败: {e}")
#         return False

# # ----------------------------
# # 生成函数（新增时间统计参数）
# # ----------------------------
# def generate_answer_with_context(model, tokenizer, question, contexts=None, start_time=None):
#     """生成答案，可选传入开始时间"""
#     if contexts:
#         logger.info(f"使用RAG模式生成答案，上下文数量: {len(contexts)}")
#         messages = build_rag_prompt(question, contexts)
#     else:
#         logger.info("无检索结果，使用基础prompt")
#         messages = [
#             {
#                 "role": "system",
#                 "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答“基于现有资料，我无法回答这个问题。"
#             },
#             {
#                 "role": "user", 
#                 "content": f"问题: {question}"
#             }
#         ]
    
#     prompt_text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             return_dict_in_generate=True,
#             output_scores=False,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     generated_tokens = outputs.sequences[0, inputs['input_ids'].size(1):]
#     answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
#     logger.info(f"生成答案完成")
    
#     # 如果传入了开始时间，计算生成时间
#     if start_time:
#         generation_time = time.perf_counter() - start_time
#         logger.info(f"模型生成耗时: {generation_time:.3f}秒")
#         return answer, generation_time
#     else:
#         return answer

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
#         logger.error("检索系统初始化失败，无法进行RAG问答")
#         return
    
#     # 加载输入数据
#     with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
#         input_data = json.load(f)
    
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
#     # 新增：时间统计变量
#     total_samples = len(input_data)
#     total_processing_time = 0.0
#     total_retrieval_time = 0.0
#     total_generation_time = 0.0
#     sample_times = []  # 存储每个样本的详细时间
    
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
                
#                 # 1. 提取时间信息
#                 logger.info("提取时间信息...")
#                 time_range = extract_time_from_question(question)
                
#                 # 2. 提取关键词
#                 extracted_keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
                
#                 # 3. 执行混合检索
#                 logger.info("开始混合检索...")
#                 retrieval_start_time = time.perf_counter()
#                 hybrid_results = retrieval_manager.hybrid_search(question, time_range)
#                 retrieval_end_time = time.perf_counter()
#                 retrieval_time = retrieval_end_time - retrieval_start_time
#                 total_retrieval_time += retrieval_time
                
#                 # 获取过滤后的结果（分数大于0.1的）
#                 filtered_docs = hybrid_results.get("filtered_results", [])
#                 all_docs = hybrid_results.get("final_results", [])
                
#                 # 使用过滤后的文档作为上下文
#                 retrieved_docs = filtered_docs
                
#                 has_supporting_info = len(retrieved_docs) > 0
#                 logger.info(f"检索到 {len(all_docs)} 个文档，其中 {len(filtered_docs)} 个分数大于0.1")
#                 logger.info(f"检索耗时: {retrieval_time:.3f}秒")
                
#                 # 4. 基于检索结果生成答案
#                 logger.info("基于检索结果生成答案...")
#                 generation_start_time = time.perf_counter()
#                 answer, generation_time = generate_answer_with_context(
#                     model, tokenizer, question, contexts=retrieved_docs, start_time=generation_start_time
#                 )
#                 total_generation_time += generation_time
                
#                 # 计算样本总处理时间
#                 sample_end_time = time.perf_counter()
#                 sample_processing_time = sample_end_time - sample_start_time
#                 total_processing_time += sample_processing_time
                
#                 # 构建结果
#                 retrieval_info = {
#                     "has_supporting_info": has_supporting_info,
#                     "extracted_keywords": extracted_keywords,
#                     "filtered_docs_count": len(filtered_docs),  # 新增：过滤后的文档数
#                 }
                
#                 # 只保存过滤后的文档预览
#                 retrieval_info["retrieved_docs_preview"] = [
#                     {
#                         "doc_id": doc.get("doc_id", ""),
#                         "title": doc.get("title", ""),
#                         "content_preview": doc.get("content", ""),
#                         "publish_date": doc.get("publish_date_str", ""),
#                         "retrieval_type": doc.get("retrieval_type", "unknown"),
#                         "final_score": float(round(doc.get("final_score", 0), 4)),
#                         "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
#                     }
#                     for doc in retrieved_docs  # 使用过滤后的文档
#                 ]
                
#                 # 添加时间信息
#                 if time_range:
#                     retrieval_info["time_extraction"] = {
#                         "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d"),
#                         "date_type": time_range.get("date_type", "relative"),
#                         "range_days": time_range.get("range_days", "N/A"),
#                     }
                
#                 # 新增：添加时间统计字段
#                 output_item = {
#                     "source_id": item.get("source_id", idx),
#                     "question": question,
#                     "standard_answer": item.get("answer", ""),
#                     "model_answer": answer,
#                     "retrieval_info": retrieval_info,
#                     "processing_times": {  # 新增：详细时间统计
#                         "total_processing_time": round(sample_processing_time, 3),  # 样本总处理时间（秒）
#                         "retrieval_time": round(retrieval_time, 3),  # 检索时间（秒）
#                         "generation_time": round(generation_time, 3),  # 生成时间（秒）
#                     }
#                 }
                
#                 # 存储样本时间详情
#                 sample_times.append({
#                     "sample_id": idx,
#                     "question": question[:50] + "..." if len(question) > 50 else question,
#                     "total_time": sample_processing_time,
#                     "retrieval_time": retrieval_time,
#                     "generation_time": generation_time,
#                     "retrieved_docs_total": len(all_docs),
#                     "filtered_docs_count": len(filtered_docs)
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
    
#     # 新增：计算并保存时间统计
#     if total_samples > 0:
#         avg_processing_time = total_processing_time / total_samples
#         avg_retrieval_time = total_retrieval_time / total_samples
#         avg_generation_time = total_generation_time / total_samples
        
#         time_stats = {
#             "total_samples": total_samples,
#             "total_processing_time_seconds": round(total_processing_time, 3),
#             "average_processing_time_seconds": round(avg_processing_time, 3),
#             "breakdown": {
#                 "retrieval": {
#                     "total_seconds": round(total_retrieval_time, 3),
#                     "average_seconds": round(avg_retrieval_time, 3),
#                     "percentage": round((total_retrieval_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
#                 },
#                 "generation": {
#                     "total_seconds": round(total_generation_time, 3),
#                     "average_seconds": round(avg_generation_time, 3),
#                     "percentage": round((total_generation_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
#                 }
#             },
#             "sample_details": sample_times,
#             "performance_summary": {
#                 "samples_per_hour": round(3600 / avg_processing_time, 2) if avg_processing_time > 0 else 0,
#                 "retrieval_efficiency": f"{avg_retrieval_time:.3f}s/doc",
#                 "generation_speed": f"{avg_generation_time:.3f}s/answer"
#             },
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
        
        
#         # 输出时间统计到日志
#         logger.info("\n" + "="*60)
#         logger.info("时间统计摘要:")
#         logger.info("="*60)
#         logger.info(f"处理样本总数: {total_samples}")
#         logger.info(f"总处理时间: {total_processing_time:.2f}秒")
#         logger.info(f"平均处理时间: {avg_processing_time:.3f}秒/样本")
#         logger.info(f"  - 平均检索时间: {avg_retrieval_time:.3f}秒 ({time_stats['breakdown']['retrieval']['percentage']}%)")
#         logger.info(f"  - 平均生成时间: {avg_generation_time:.3f}秒 ({time_stats['breakdown']['generation']['percentage']}%)")
#         logger.info(f"处理速度: {time_stats['performance_summary']['samples_per_hour']} 样本/小时")
    
#     logger.info(f"✅ RAG问答完成，结果保存至 {OUTPUT_PATH}")
#     logger.info(f"日志文件: generate_rag_pure.log")


# if __name__ == "__main__":
#     main()



# ==========================================================大模型接口================================================================


# # -*- coding: utf-8 -*-
# """
# RAG问答系统 - 精简输出版
# 功能：批量处理输入问题，仅输出最终JSON结果，不保存任何中间文件。
# """

# import os
# import json
# import time
# import sys
# from typing import List, Dict, Optional, Any
# from tqdm import tqdm
# import re
# from datetime import datetime, timedelta
# import jieba
# import jieba.posseg as pseg

# # RAG相关库
# from pymilvus import connections, Collection
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # 大模型接口
# from openai import OpenAI

# # ----------------------------
# # 配置路径和参数
# # ----------------------------
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk_api/ds_128.json"
# STOPWORDS_PATH = "/mnt/project/milvus/data/hit_stopwords.txt"

# # 大模型配置
# API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# API_KEY = "__SET_VIA_ENVIRONMENT__"
# MODEL_NAME = "deepseek-v3"
# MAX_TOKENS = 4096
# TEMPERATURE = 0.1

# # 批量处理配置
# BATCH_SIZE = 100

# # Milvus配置
# MILVUS_HOST = "122.226.69.234"
# MILVUS_PORT = "3042"
# MILVUS_COLLECTION_NAME = "news_collection_128"
# EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
# RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# # 检索参数
# TOP_K_RETRIEVAL = 20
# TOP_K_FINAL = 5

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
#         print(f"已加载 {len(stopwords)} 个停用词")
#     else:
#         stopwords = base_stopwords
#         print("使用内置停用词")
#     return stopwords

# STOPWORDS = load_stopwords(STOPWORDS_PATH)

# # ----------------------------
# # 时间提取
# # ----------------------------
# def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
#     """从问题中提取时间信息"""
    
#     date_patterns = [
#         (r'(\d{4})年(\d{1,2})月(\d{1,2})日', 'ymd'),
#         (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ymd'),
#         (r'(\d{4})/(\d{1,2})/(\d{1,2})', 'ymd'),
#         (r'(\d{4})年(\d{1,2})月', 'ym'),
#         (r'(\d{4})-(\d{1,2})', 'ym'),
#         (r'(\d{4})/(\d{1,2})', 'ym'),
#         (r'(\d{4})年', 'y'),
#     ]
    
#     for pattern, date_type in date_patterns:
#         match = re.search(pattern, question)
#         if match:
#             try:
#                 groups = match.groups()
                
#                 if date_type == 'ymd':
#                     year, month, day = map(int, groups)
#                     extracted_date = datetime(year, month, day)
#                     days_range = 30
#                     need_filter = True
                    
#                 elif date_type == 'ym':
#                     year, month = map(int, groups)
#                     extracted_date = datetime(year, month, 1)
#                     days_range = 60
#                     need_filter = True
                    
#                 elif date_type == 'y':
#                     year = int(groups[0])
#                     extracted_date = datetime(year, 6, 30)
#                     days_range = 365
#                     need_filter = True
                
#                 if need_filter:
#                     start_date = extracted_date - timedelta(days=days_range)
#                     end_date = extracted_date + timedelta(days=days_range)
                    
#                     start_timestamp = int(start_date.timestamp())
#                     end_timestamp = int(end_date.timestamp())
                    
#                     print(f"提取时间 [{date_type}]: {extracted_date.strftime('%Y-%m-%d')}")
                    
#                     return {
#                         "extracted_date": extracted_date,
#                         "start_timestamp": start_timestamp,
#                         "end_timestamp": end_timestamp,
#                         "start_date_str": start_date.strftime("%Y-%m-%d"),
#                         "end_date_str": end_date.strftime("%Y-%m-%d"),
#                         "has_specific_date": True,
#                         "date_type": date_type,
#                         "range_days": days_range * 2
#                     }
                
#             except Exception as e:
#                 print(f"时间解析失败: {e}")
#                 continue
    
#     relative_time_patterns = {
#         r'今天|当天': (0, 1),
#         r'昨天|昨日': (-1, 1),
#         r'前天': (-2, 1),
#         r'大前天': (-3, 1),
#         r'上周|上星期': (-7, 7),
#         r'本周|这周': (-3, 3),
#         r'上个月|上月': (-30, 30),
#         r'这个月|本月': (-15, 15),
#         r'去年|上一年': (-365, 365),
#         r'今年|本年': (-180, 180),
#         r'最近|近期': (-30, 0),
#     }
    
#     current_date = datetime.now()
#     for pattern, (days_offset, days_range) in relative_time_patterns.items():
#         if re.search(pattern, question):
#             center_date = current_date + timedelta(days=days_offset)
#             start_date = center_date - timedelta(days=abs(days_range))
#             end_date = center_date + timedelta(days=abs(days_range))
            
#             print(f"相对时间提取: {pattern}")
            
#             return {
#                 "extracted_date": center_date,
#                 "start_timestamp": int(start_date.timestamp()),
#                 "end_timestamp": int(end_date.timestamp()),
#                 "start_date_str": start_date.strftime("%Y-%m-%d"),
#                 "end_date_str": end_date.strftime("%Y-%m-%d"),
#                 "has_specific_date": True,
#                 "is_relative": True,
#                 "relative_pattern": pattern
#             }
    
#     print("未提取到时间信息，不进行时间过滤")
#     return None

# # ----------------------------
# # 关键词提取
# # ----------------------------
# def extract_keywords_enhanced(question: str, stopwords: set, top_k: int = 15) -> List[str]:
#     """增强的关键词提取 - 使用词性加权"""
#     if not question or len(question.strip()) == 0:
#         return []

#     POS_WEIGHTS = {
#         'n': 2.0,
#         'nr': 3.0,
#         'ns': 3.0,
#         'nt': 3.0,
#         'nz': 2.5,
#         'v': 1.8,
#         'vn': 2.0,
#         'a': 1.3,
#         'an': 1.5,
#         'i': 1.5,
#     }
    
#     words_with_pos = pseg.cut(question)
    
#     word_scores = {}
#     for word, pos in words_with_pos:
#         w = word.strip()
        
#         if len(w) <= 1:
#             continue
#         if w in stopwords:
#             continue
#         if w.isdigit():
#             continue
#         if not any('\u4e00' <= c <= '\u9fff' or c.isalpha() for c in w):
#             continue
        
#         pos_prefix = pos[0] if pos else 'x'
#         weight = POS_WEIGHTS.get(pos_prefix, 0.5)
        
#         if len(w) >= 3 and pos_prefix in ['nr', 'ns', 'nt', 'nz']:
#             weight *= 1.2
        
#         if w in word_scores:
#             word_scores[w] += weight
#         else:
#             word_scores[w] = weight
    
#     sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
#     keywords = [word for word, score in sorted_words[:top_k]]
    
#     print(f"增强关键词提取: {keywords}")
#     return keywords

# # ----------------------------
# # 文档清理函数
# # ----------------------------
# def clean_retrieved_documents_for_context(docs: List[Dict]) -> List[Dict]:
#     """清理检索到的文档，只保留构建RAG prompt所需的字段"""
#     if not docs:
#         return []
    
#     cleaned_docs = []
#     for doc in docs:
#         cleaned_doc = {
#             "title": str(doc.get("title", "")),
#             "content": str(doc.get("content", "")),
#             "publish_date_str": str(doc.get("publish_date_str", "")),
#             "region": str(doc.get("region", "")),
#         }
#         cleaned_docs.append(cleaned_doc)
    
#     return cleaned_docs

# def clean_retrieved_documents_for_preview(docs: List[Dict]) -> List[Dict]:
#     """清理检索到的文档，用于保存预览信息"""
#     if not docs:
#         return []
    
#     cleaned_docs = []
#     for doc in docs:
#         cleaned_doc = {
#             "doc_id": str(doc.get("doc_id", "")),
#             "title": str(doc.get("title", "")),
#             "content_preview": str(doc.get("content", ""))[:800],
#             "publish_date": str(doc.get("publish_date_str", "")),
#             "retrieval_type": str(doc.get("retrieval_type", "unknown")),
#             "final_score": float(round(doc.get("final_score", 0), 4)),
#             "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
#         }
#         cleaned_docs.append(cleaned_doc)
    
#     return cleaned_docs

# # ----------------------------
# # RAG Prompt构建函数
# # ----------------------------
# def build_rag_prompt(question: str, contexts: List[Dict]) -> List[Dict]:
#     """构建RAG Prompt"""
#     if not contexts:
#         return [
#             {
#                 "role": "system",
#                 "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答'基于现有资料，我无法回答这个问题'。"
#             },
#             {
#                 "role": "user", 
#                 "content": f"问题: {question}"
#             }
#         ]
    
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
#             f"内容: {content}\n"
#         )
    
#     context_text = "\n".join(context_parts)

#     system_prompt = """你是一个新闻问答助手。你的任务是基于提供的参考资料回答用户问题。

#     核心原则:
#     请按照以下步骤思考并回答：

#     ## 步骤1：分析问题要点
#     - 理解问题的核心是什么
#     - 识别关键信息：人物、事件、时间、地点、数据等

#     ## 步骤2：定位相关证据
#     - 在参考资料中找出与问题唯一或最相关的所有片段
#     - 注意信息的时效性和相关性， 若资料存在多篇内容冲突，优先选择发布时间最新的资料内容作答

#     ## 步骤3：基于证据回答
#     - **严格只使用参考资料中的信息**
#     - 将找到的证据自然整合到回答中
#     - 明确引用信息来源（例如："根据文档3的报道..."）
#     - 如果不同来源有冲突，如实说明

#     ## 步骤4：最终检查
#     - 确认每个事实都有参考资料支持
#     - 检查是否直接回答了问题
#     - 确保回答简洁清晰
#     回答格式:
#     - 如果能回答: 直接给出答案
#     - 如果不能回答: "根据现有资料，我无法回答这个问题。"
#     """
        
#     user_prompt = f"""参考资料:
# {context_text}

# 问题: {question}

# 请基于上述资料回答问题。"""
        
#     return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
# #     system_prompt = """你是一个新闻问答助手。你的任务是基于提供的参考资料回答用户问题。

# # 核心原则:
# # 1. **严格依据参考资料** - 只使用资料中提到的信息
# # 2. **保持简洁** - 直接回答问题，不要冗余说明

# # 回答格式:
# # - 如果能回答: 直接给出答案
# # - 如果不能回答: "根据现有资料，我无法回答这个问题。"
# # """
        
# #     user_prompt = f"""参考资料:
# # {context_text}

# # 问题: {question}

# # 请基于上述资料回答问题。"""
        
# #     return [
# #             {"role": "system", "content": system_prompt},
# #             {"role": "user", "content": user_prompt}
# #         ]

# # ----------------------------
# # 重排序器类
# # ----------------------------
# class EnhancedReranker:
#     """增强的重排序器"""
    
#     def __init__(self, model_name: str = RERANKER_MODEL):
#         print(f"加载重排序模型: {model_name}")
#         try:
#             self.reranker = CrossEncoder(model_name, max_length=512)
#             print("重排序模型加载完成")
#         except Exception as e:
#             print(f"重排序模型加载失败: {e}")
#             self.reranker = None
    
#     def rerank(
#         self, 
#         question: str, 
#         documents: List[Dict], 
#         top_k: int = 5,
#         time_decay: bool = True,
#         current_timestamp: Optional[int] = None
#     ) -> List[Dict]:
#         """重排序文档"""
#         if not documents or self.reranker is None:
#             return documents[:top_k]
        
#         pairs = []
#         for doc in documents:
#             doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"[:500]
#             pairs.append([question, doc_text])
        
#         print(f"重排序 {len(documents)} 个文档...")
#         try:
#             rerank_scores = self.reranker.predict(pairs)
#         except Exception as e:
#             print(f"重排序失败: {e}")
#             return documents[:top_k]
        
#         for doc, score in zip(documents, rerank_scores):
#             doc['rerank_score'] = float(score)
            
#             if time_decay and current_timestamp:
#                 publish_time = doc.get('publish_time', 0)
#                 if publish_time > 0:
#                     days_diff = (current_timestamp - publish_time) / 86400
#                     time_factor = max(0.5, 1.0 - (days_diff / 30) * 0.05)
#                     doc['final_score'] = score * time_factor
#                 else:
#                     doc['final_score'] = score
#             else:
#                 doc['final_score'] = score
        
#         ranked_docs = sorted(documents, key=lambda x: x['final_score'], reverse=True)
        
#         print(f"重排序完成，返回top-{top_k}")
#         for i, doc in enumerate(ranked_docs[:top_k]):
#             print(f"  Rank {i+1}: {doc.get('title', '')[:50]}... (分数: {doc['final_score']:.4f})")
        
#         return ranked_docs[:top_k]

# # ----------------------------
# # 混合检索管理器
# # ----------------------------
# class HybridRetrievalManager:
#     """混合检索管理器"""
    
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
#             print(f"连接 Milvus ({self.host}:{self.port})...")
#             self.connection = connections.connect(
#                 alias="default",
#                 host=self.host,
#                 port=self.port
#             )
            
#             self.collection = Collection(MILVUS_COLLECTION_NAME)
#             print(f"已连接到集合: {MILVUS_COLLECTION_NAME}")
            
#             print("加载embedding模型...")
#             self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
#             if self.use_reranker:
#                 try:
#                     self.reranker = EnhancedReranker(RERANKER_MODEL)
#                 except Exception as e:
#                     print(f"重排序器加载失败: {e}, 将不使用重排序")
#                     self.reranker = None
            
#             print("检索系统初始化完成")
            
#         except Exception as e:
#             print(f"检索系统初始化失败: {e}")
#             raise
    
#     def get_embedding(self, text: str) -> List[float]:
#         """获取文本的嵌入向量"""
#         if self.embedding_model is None:
#             raise ValueError("嵌入模型未初始化")
        
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
#             question_embedding = self.get_embedding(question)
            
#             search_params = {
#                 "metric_type": "IP",
#                 "params": {"nprobe": 10, "ef": 50}
#             }
            
#             expr = None
#             if time_range and time_range.get("has_specific_date", False):
#                 start_ts = time_range.get("start_timestamp")
#                 end_ts = time_range.get("end_timestamp")
#                 if start_ts and end_ts:
#                     expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
#                     print(f"向量检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")
            
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
            
#             print(f"向量检索完成: {len(formatted_results)} 个文档")
#             return formatted_results
            
#         except Exception as e:
#             print(f"向量检索失败: {e}")
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
#             keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
#             if not keywords:
#                 print("未提取到关键词")
#                 return []
            
#             keyword_expr_parts = []
#             for keyword in keywords[:5]:
#                 safe_keyword = keyword.replace("'", "''").replace('"', '""')
#                 keyword_expr_parts.append(f"array_contains(keywords, '{safe_keyword}')")
            
#             if not keyword_expr_parts:
#                 return []
            
#             keyword_expr = " or ".join(keyword_expr_parts)
            
#             time_expr = ""
#             if time_range and time_range.get("has_specific_date", False):
#                 start_ts = time_range.get("start_timestamp")
#                 end_ts = time_range.get("end_timestamp")
#                 if start_ts and end_ts:
#                     time_expr = f"publish_time >= {start_ts} and publish_time <= {end_ts}"
#                     print(f"关键字检索时间过滤: {time_range.get('start_date_str')} 到 {time_range.get('end_date_str')}")
            
#             if time_expr:
#                 expr = f"({keyword_expr}) and ({time_expr})"
#             else:
#                 expr = keyword_expr
            
#             query_results = self.collection.query(
#                 expr=expr,
#                 output_fields=["doc_id", "title", "content", "publish_time", 
#                               "publish_date_str", "region", "keywords"],
#                 limit=top_k
#             )
            
#             print(f"关键字检索完成: {len(query_results)} 个文档")
            
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
#             print(f"关键字检索失败: {e}")
#             return []
    
#     def reciprocal_rank_fusion(
#         self, 
#         vector_results: List[Dict], 
#         keyword_results: List[Dict], 
#         k: int = 60
#     ) -> List[Dict]:
#         """倒数排名融合（RRF）"""
#         print(f"开始RRF融合，k={k}")
        
#         doc_id_to_results = {}
        
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
        
#         rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
#         final_results = []
#         for rrf_result in rrf_results:
#             doc_info = rrf_result["doc_info"].copy()
#             doc_info["score"] = rrf_result["rrf_score"]
#             doc_info["rrf_score"] = rrf_result["rrf_score"]
#             doc_info["vector_rank"] = rrf_result["vector_rank"]
#             doc_info["keyword_rank"] = rrf_result["keyword_rank"]
#             final_results.append(doc_info)
        
#         print(f"RRF融合完成: {len(final_results)} 个文档")
#         return final_results
    
#     def hybrid_search(
#         self, 
#         question: str, 
#         time_range: Optional[Dict] = None
#     ) -> Dict[str, Any]:
#         """混合检索流程"""
#         print(f"开始混合检索: {question}")
        
#         vector_results = self.vector_search(question, time_range, TOP_K_RETRIEVAL)
#         keyword_results = self.keyword_search(question, time_range, TOP_K_RETRIEVAL)
        
#         print(f"初始检索: 向量{len(vector_results)}, 关键词{len(keyword_results)}")
        
#         fused_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
#         print(f"RRF融合后: {len(fused_results)} 个文档")
        
#         if self.reranker and fused_results:
#             current_ts = int(time.time())
#             final_results = self.reranker.rerank(
#                 question, 
#                 fused_results, 
#                 top_k=TOP_K_FINAL,
#                 time_decay=True,
#                 current_timestamp=current_ts
#             )
#         else:
#             final_results = fused_results[:TOP_K_FINAL]
        
#         filtered_results = [doc for doc in final_results if doc.get('final_score', 0) > 0.1]
        
#         print(f"重排序后top-{TOP_K_FINAL}: {len(final_results)} 个文档")
#         print(f"分数过滤后(>0.1): {len(filtered_results)} 个文档")
        
#         return {
#             "final_results": final_results,
#             "filtered_results": filtered_results
#         }

# # 全局检索管理器
# retrieval_manager = None

# # ----------------------------
# # 大模型接口管理
# # ----------------------------
# class LLMInterfaceManager:
#     """大模型接口管理器"""
    
#     def __init__(self, api_key: str, base_url: str, model_name: str = "deepseek-v3"):
#         """初始化大模型接口"""
#         print(f"初始化大模型接口，模型: {model_name}")
#         try:
#             self.client = OpenAI(
#                 api_key=api_key,
#                 base_url=base_url
#             )
#             self.model_name = model_name
#             print("大模型接口初始化完成")
#         except Exception as e:
#             print(f"大模型接口初始化失败: {e}")
#             raise
    
#     def create_batch_file(self, batch_questions: List[Dict], batch_num: int, output_dir: str) -> str:
#         """创建批量处理文件"""
#         os.makedirs(output_dir, exist_ok=True)
#         batch_file_path = os.path.join(output_dir, f"batch_input_{batch_num}.jsonl")
        
#         print(f"创建批次 {batch_num} 输入文件，包含 {len(batch_questions)} 个问题")
        
#         with open(batch_file_path, 'w', encoding='utf-8') as f:
#             for idx, item in enumerate(batch_questions):
#                 prompt = build_rag_prompt(item["question"], item["contexts"])
                
#                 custom_id = f"batch_{batch_num}_item_{idx}_sid_{item.get('source_id', idx)}"
                
#                 batch_request = {
#                     "custom_id": custom_id,
#                     "method": "POST",
#                     "url": "/v1/chat/completions",
#                     "body": {
#                         "model": self.model_name,
#                         "messages": prompt,
#                         "max_tokens": MAX_TOKENS,
#                         "temperature": TEMPERATURE
#                     }
#                 }
                
#                 f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
        
#         print(f"批次文件已创建: {batch_file_path}")
#         return batch_file_path
    
#     def submit_batch_job(self, batch_file_path: str, batch_num: int) -> Optional[str]:
#         """提交批量任务"""
#         try:
#             print(f"提交批次 {batch_num} 批量任务...")
            
#             with open(batch_file_path, 'rb') as file:
#                 uploaded_file = self.client.files.create(file=file, purpose="batch")
            
#             batch_job = self.client.batches.create(
#                 input_file_id=uploaded_file.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window="72h"
#             )
            
#             print(f"✅ 批次 {batch_num} 提交成功，任务ID: {batch_job.id}")
#             return batch_job.id
            
#         except Exception as e:
#             print(f"❌ 批次 {batch_num} 提交失败: {e}")
#             return None
    
#     def monitor_batch_progress(self, batch_id: str, batch_num: int) -> Optional[str]:
#         """监控批量任务进度"""
#         print(f"🔍 开始监控批次 {batch_num} 进度: {batch_id}")
        
#         start_time = time.time()
#         last_status = None
        
#         while True:
#             try:
#                 batch_status = self.client.batches.retrieve(batch_id)
#                 current_status = batch_status.status
                
#                 if current_status != last_status:
#                     elapsed_hours = (time.time() - start_time) / 3600
#                     print(f"🔄 批次 {batch_num} 状态: {current_status} (已等待 {elapsed_hours:.1f} 小时)")
#                     last_status = current_status
                
#                 if current_status == "completed":
#                     print(f"✅ 批次 {batch_num} 处理完成!")
#                     return batch_status.output_file_id
                    
#                 elif current_status in ["failed", "expired", "cancelled"]:
#                     print(f"❌ 批次 {batch_num} 失败: {current_status}")
#                     if hasattr(batch_status, 'errors'):
#                         print(f"错误详情: {batch_status.errors}")
#                     return None
                    
#                 else:
#                     time.sleep(300)
                    
#             except Exception as e:
#                 print(f"⚠️ 检查批次 {batch_num} 状态失败: {e}")
#                 time.sleep(300)
    
#     def download_batch_results(self, output_file_id: str, batch_num: int, original_batch: List[Dict]) -> List[Dict]:
#         """下载批量任务结果"""
#         try:
#             print(f"下载批次 {batch_num} 结果...")
            
#             result_content = self.client.files.content(output_file_id)
#             results = result_content.text.split('\n')
            
#             batch_results = []
#             success_count = 0
#             error_count = 0
            
#             original_indices = {}
#             for idx, item in enumerate(original_batch):
#                 source_id = item.get('source_id', idx)
#                 custom_id = f"batch_{batch_num}_item_{idx}_sid_{source_id}"
#                 original_indices[custom_id] = idx
            
#             for line_num, line in enumerate(results, 1):
#                 if not line.strip():
#                     continue
                    
#                 try:
#                     result_obj = json.loads(line)
#                     custom_id = result_obj.get("custom_id")
#                     response = result_obj.get("response", {})
                    
#                     if response.get("status_code") == 200:
#                         body = response.get("body", {})
#                         choices = body.get("choices", [])
                        
#                         if choices:
#                             answer = choices[0].get("message", {}).get("content", "")
                            
#                             if custom_id in original_indices:
#                                 item_idx = original_indices[custom_id]
                                
#                                 batch_results.append({
#                                     "item_idx": item_idx,
#                                     "custom_id": custom_id,
#                                     "answer": answer,
#                                     "success": True
#                                 })
#                                 success_count += 1
                                
#                                 if success_count % 20 == 0:
#                                     print(f"✅ 批次 {batch_num}: 已解析 {success_count} 个结果")
#                             else:
#                                 print(f"未找到custom_id对应的原始数据: {custom_id}")
#                                 error_count += 1
#                         else:
#                             error_count += 1
#                     else:
#                         error_count += 1
                        
#                 except Exception as e:
#                     error_count += 1
#                     print(f"解析第 {line_num} 行失败: {e}")
            
#             print(f"📊 批次 {batch_num} 解析完成: 成功 {success_count}, 失败 {error_count}")
#             return batch_results
            
#         except Exception as e:
#             print(f"❌ 下载批次 {batch_num} 结果失败: {e}")
#             return []

# # 全局大模型管理器
# llm_manager = None

# # ----------------------------
# # 主函数（精简版 - 仅输出最终JSON）
# # ----------------------------
# def main():
#     """主函数 - 使用大模型接口批量处理（仅输出最终JSON）"""
#     # 初始化
#     jieba.initialize()
    
#     print("\n" + "="*70)
#     print("🚀 RAG问答系统（精简输出版 - 仅生成最终JSON）")
#     print("="*70)
#     print(f"模型: {MODEL_NAME}")
#     print(f"批量大小: {BATCH_SIZE}")
#     print(f"输入: {INPUT_DATA_PATH}")
#     print(f"输出: {OUTPUT_PATH}")
#     print("="*70)
    
#     start_time = time.time()
    
#     try:
#         # 1. 初始化检索系统
#         print("初始化混合检索系统...")
#         global retrieval_manager
#         retrieval_manager = HybridRetrievalManager(
#             MILVUS_HOST, 
#             MILVUS_PORT,
#             use_reranker=True
#         )
#         retrieval_manager.initialize()
#         print("✅ 混合检索系统初始化成功")
        
#         # 2. 初始化大模型接口
#         print("初始化大模型接口...")
#         global llm_manager
#         llm_manager = LLMInterfaceManager(
#             api_key=API_KEY,
#             base_url=API_BASE_URL,
#             model_name=MODEL_NAME
#         )
#         print("✅ 大模型接口初始化成功")
        
#         # 3. 加载输入数据
#         print(f"加载输入数据: {INPUT_DATA_PATH}")
#         with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
#             input_data = json.load(f)
        
#         total_samples = len(input_data)
#         print(f"加载完成: {total_samples} 条数据")
        
#         # 4. 创建输出目录
#         output_dir = os.path.dirname(OUTPUT_PATH)
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 5. 分批处理（无中间状态保存）
#         total_batches = (len(input_data) + BATCH_SIZE - 1) // BATCH_SIZE
#         all_results = []
        
#         print(f"数据分 {total_batches} 批处理")
        
#         for batch_num in range(1, total_batches + 1):
#             batch_start = (batch_num - 1) * BATCH_SIZE
#             batch_end = batch_num * BATCH_SIZE
#             batch_data = input_data[batch_start:batch_end]
            
#             print(f"\n🔄 开始处理批次 {batch_num}/{total_batches}")
#             print(f"📊 批次数据: {len(batch_data)} 条")
            
#             # 5.1 检索批次内所有问题
#             batch_questions = []
            
#             for idx, item in enumerate(tqdm(batch_data, desc=f"检索批次 {batch_num}")):
#                 try:
#                     question = item.get("question", "").strip()
#                     if not question:
#                         continue
                    
#                     source_id = item.get("source_id", idx + batch_start)
                    
#                     # 1. 提取时间信息
#                     time_range = extract_time_from_question(question)
                    
#                     # 2. 提取关键词
#                     extracted_keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
                    
#                     # 3. 执行混合检索
#                     hybrid_results = retrieval_manager.hybrid_search(question, time_range)
                    
#                     # 获取过滤后的结果
#                     filtered_docs = hybrid_results.get("filtered_results", [])
                    
#                     # 使用过滤后的文档作为上下文
#                     retrieved_docs = filtered_docs
                    
#                     has_supporting_info = len(retrieved_docs) > 0
                    
#                     # 构建检索信息
#                     retrieval_info = {
#                         "has_supporting_info": has_supporting_info,
#                         "extracted_keywords": extracted_keywords,
#                         "filtered_docs_count": len(filtered_docs),
#                     }
                    
#                     # 只保存过滤后的文档预览
#                     retrieval_info["retrieved_docs_preview"] = [
#                         {
#                             "doc_id": doc.get("doc_id", ""),
#                             "title": doc.get("title", ""),
#                             "content_preview": str(doc.get("content", ""))[:800],
#                             "publish_date": doc.get("publish_date_str", ""),
#                             "retrieval_type": doc.get("retrieval_type", "unknown"),
#                             "final_score": float(round(doc.get("final_score", 0), 4)),
#                             "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
#                         }
#                         for doc in retrieved_docs
#                     ]
                    
#                     # 添加时间信息
#                     if time_range:
#                         retrieval_info["time_extraction"] = {
#                             "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d"),
#                             "date_type": time_range.get("date_type", "relative"),
#                             "range_days": time_range.get("range_days", "N/A"),
#                         }
                    
#                     # 清理文档用于上下文构建
#                     contexts = clean_retrieved_documents_for_context(retrieved_docs)
                    
#                     # 构建问题信息
#                     question_info = {
#                         "source_id": source_id,
#                         "question": question,
#                         "standard_answer": item.get("answer", ""),
#                         "contexts": contexts,
#                         "retrieval_info": retrieval_info,
#                         "filtered_docs": filtered_docs,
#                         "item_idx_in_batch": idx
#                     }
                    
#                     batch_questions.append(question_info)
                    
#                 except Exception as e:
#                     print(f"处理批次 {batch_num} 中第 {idx} 条数据失败: {e}")
#                     continue
            
#             print(f"✅ 批次 {batch_num} 检索完成")
            
#             # 5.2 创建批量输入文件
#             batch_file_path = llm_manager.create_batch_file(
#                 batch_questions, 
#                 batch_num, 
#                 output_dir
#             )
            
#             # 5.3 提交批量任务
#             batch_id = llm_manager.submit_batch_job(batch_file_path, batch_num)
#             if not batch_id:
#                 print(f"❌ 批次 {batch_num} 提交失败，跳过")
#                 continue
            
#             # 5.4 监控批量任务进度
#             output_file_id = llm_manager.monitor_batch_progress(batch_id, batch_num)
#             if not output_file_id:
#                 print(f"❌ 批次 {batch_num} 处理失败，跳过")
#                 continue
            
#             # 5.5 下载和处理结果
#             batch_results = llm_manager.download_batch_results(
#                 output_file_id, 
#                 batch_num,
#                 batch_questions
#             )
            
#             if batch_results:
#                 # 5.6 合并检索结果和生成结果
#                 batch_results_map = {}
#                 for result in batch_results:
#                     if result.get("success"):
#                         batch_results_map[result.get("item_idx")] = result
                
#                 batch_output_items = []
#                 for idx, question_info in enumerate(batch_questions):
#                     matching_result = batch_results_map.get(idx)
                    
#                     if matching_result:
#                         answer = matching_result["answer"]
#                         generation_success = True
#                     else:
#                         answer = "生成失败"
#                         generation_success = False
                    
#                     # 构建输出项
#                     output_item = {
#                         "source_id": question_info.get("source_id", idx + batch_start),
#                         "question": question_info["question"],
#                         "standard_answer": question_info["standard_answer"],
#                         "model_answer": answer,
#                         "retrieval_info": question_info["retrieval_info"],
#                         "processing_times": {
#                             "total_processing_time": 0.0,
#                             "retrieval_time": 0.0,
#                             "generation_time": 0.0,
#                         }
#                     }
                    
#                     batch_output_items.append(output_item)
                
#                 # 5.7 添加到总结果
#                 all_results.extend(batch_output_items)
                
#                 print(f"✅ 批次 {batch_num} 完成!")
#                 print(f"📊 当前进度: {len(all_results)}/{total_samples} 条数据")
                
#             else:
#                 print(f"❌ 批次 {batch_num} 未生成有效结果")
        
#         # 6. 一次性保存最终结果（覆盖写入）
#         print(f"\n💾 保存最终结果到: {OUTPUT_PATH}")
#         with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
#             json.dump(all_results, f, ensure_ascii=False, indent=2)
        
#         # 7. 最终统计
#         total_time = time.time() - start_time
#         days = int(total_time // 86400)
#         hours = int((total_time % 86400) // 3600)
#         minutes = int((total_time % 3600) // 60)
        
#         print("\n" + "="*70)
#         print("🎉 RAG问答批量处理完成!")
#         print("="*70)
#         print(f"总样本数: {len(all_results)}")
#         print(f"总处理时间: {days}天{hours}小时{minutes}分钟")
#         print("="*70)
#         print(f"输出文件: {OUTPUT_PATH}")
#         print("="*70)
        
#     except Exception as e:
#         print(f"❌ 处理过程中发生错误: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


# ===============================================大模型接口无时间============================================
# # -*- coding: utf-8 -*-
# """
# RAG问答系统 - 精简输出版 (无时间过滤)
# 功能：批量处理输入问题，仅输出最终JSON结果，不保存任何中间文件。
# 修改：移除了所有时间提取和时间过滤逻辑，对所有问题使用全量数据集检索。
# """

# import os
# import json
# import time
# import sys
# from typing import List, Dict, Optional, Any
# from tqdm import tqdm
# import re
# from datetime import datetime, timedelta
# import jieba
# import jieba.posseg as pseg

# # RAG相关库
# from pymilvus import connections, Collection
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # 大模型接口
# from openai import OpenAI

# # ----------------------------
# # 配置路径和参数
# # ----------------------------
# INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
# OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk_api/ds_450_notime"
# STOPWORDS_PATH = "/mnt/project/milvus/data/hit_stopwords.txt"

# # 大模型配置
# API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# API_KEY = "__SET_VIA_ENVIRONMENT__"
# MODEL_NAME = "deepseek-v3"
# MAX_TOKENS = 4096
# TEMPERATURE = 0.1

# # 批量处理配置
# BATCH_SIZE = 100

# # Milvus配置
# MILVUS_HOST = "122.226.69.234"
# MILVUS_PORT = "3042"
# MILVUS_COLLECTION_NAME = "news_collection_450"
# EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
# RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# # 检索参数
# TOP_K_RETRIEVAL = 20
# TOP_K_FINAL = 5

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
#         print(f"已加载 {len(stopwords)} 个停用词")
#     else:
#         stopwords = base_stopwords
#         print("使用内置停用词")
#     return stopwords

# STOPWORDS = load_stopwords(STOPWORDS_PATH)

# # ----------------------------
# # 关键词提取 (保持不变)
# # ----------------------------
# def extract_keywords_enhanced(question: str, stopwords: set, top_k: int = 15) -> List[str]:
#     """增强的关键词提取 - 使用词性加权"""
#     if not question or len(question.strip()) == 0:
#         return []

#     POS_WEIGHTS = {
#         'n': 2.0,
#         'nr': 3.0,
#         'ns': 3.0,
#         'nt': 3.0,
#         'nz': 2.5,
#         'v': 1.8,
#         'vn': 2.0,
#         'a': 1.3,
#         'an': 1.5,
#         'i': 1.5,
#     }
    
#     words_with_pos = pseg.cut(question)
    
#     word_scores = {}
#     for word, pos in words_with_pos:
#         w = word.strip()
        
#         if len(w) <= 1:
#             continue
#         if w in stopwords:
#             continue
#         if w.isdigit():
#             continue
#         if not any('\u4e00' <= c <= '\u9fff' or c.isalpha() for c in w):
#             continue
        
#         pos_prefix = pos[0] if pos else 'x'
#         weight = POS_WEIGHTS.get(pos_prefix, 0.5)
        
#         if len(w) >= 3 and pos_prefix in ['nr', 'ns', 'nt', 'nz']:
#             weight *= 1.2
        
#         if w in word_scores:
#             word_scores[w] += weight
#         else:
#             word_scores[w] = weight
    
#     sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
#     keywords = [word for word, score in sorted_words[:top_k]]
    
#     print(f"增强关键词提取: {keywords}")
#     return keywords

# # ----------------------------
# # 文档清理函数
# # ----------------------------
# def clean_retrieved_documents_for_context(docs: List[Dict]) -> List[Dict]:
#     """清理检索到的文档，只保留构建RAG prompt所需的字段"""
#     if not docs:
#         return []
    
#     cleaned_docs = []
#     for doc in docs:
#         cleaned_doc = {
#             "title": str(doc.get("title", "")),
#             "content": str(doc.get("content", "")),
#             "publish_date_str": str(doc.get("publish_date_str", "")),
#             "region": str(doc.get("region", "")),
#         }
#         cleaned_docs.append(cleaned_doc)
    
#     return cleaned_docs

# # ----------------------------
# # RAG Prompt构建函数
# # ----------------------------
# def build_rag_prompt(question: str, contexts: List[Dict]) -> List[Dict]:
#     """构建RAG Prompt"""
#     if not contexts:
#         return [
#             {
#                 "role": "system",
#                 "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答'基于现有资料，我无法回答这个问题'。"
#             },
#             {
#                 "role": "user", 
#                 "content": f"问题: {question}"
#             }
#         ]
    
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
#             f"内容: {content}\n"
#         )
    
#     context_text = "\n".join(context_parts)

#     system_prompt = """你是一个新闻问答助手。你的任务是基于提供的参考资料回答用户问题。

#     核心原则:
#     请按照以下步骤思考并回答：

#     ## 步骤1：分析问题要点
#     - 理解问题的核心是什么
#     - 识别关键信息：人物、事件、时间、地点、数据等

#     ## 步骤2：定位相关证据
#     - 在参考资料中找出与问题唯一或最相关的所有片段
#     - 注意信息的时效性和相关性， 若资料存在多篇内容冲突，优先选择发布时间最新的资料内容作答

#     ## 步骤3：基于证据回答
#     - **严格只使用参考资料中的信息**
#     - 将找到的证据自然整合到回答中
#     - 明确引用信息来源（例如："根据文档3的报道..."）
#     - 如果不同来源有冲突，如实说明

#     ## 步骤4：最终检查
#     - 确认每个事实都有参考资料支持
#     - 检查是否直接回答了问题
#     - 确保回答简洁清晰
#     回答格式:
#     - 如果能回答: 直接给出答案
#     - 如果不能回答: "根据现有资料，我无法回答这个问题。"
#     """
        
#     user_prompt = f"""参考资料:
# {context_text}

# 问题: {question}

# 请基于上述资料回答问题。"""
        
#     return [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]

# # ----------------------------
# # 重排序器类
# # ----------------------------
# class EnhancedReranker:
#     """增强的重排序器"""
    
#     def __init__(self, model_name: str = RERANKER_MODEL):
#         print(f"加载重排序模型: {model_name}")
#         try:
#             self.reranker = CrossEncoder(model_name, max_length=512)
#             print("重排序模型加载完成")
#         except Exception as e:
#             print(f"重排序模型加载失败: {e}")
#             self.reranker = None
    
#     def rerank(
#         self, 
#         question: str, 
#         documents: List[Dict], 
#         top_k: int = 5,
#         time_decay: bool = True,
#         current_timestamp: Optional[int] = None
#     ) -> List[Dict]:
#         """重排序文档"""
#         if not documents or self.reranker is None:
#             return documents[:top_k]
        
#         pairs = []
#         for doc in documents:
#             doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"[:500]
#             pairs.append([question, doc_text])
        
#         print(f"重排序 {len(documents)} 个文档...")
#         try:
#             rerank_scores = self.reranker.predict(pairs)
#         except Exception as e:
#             print(f"重排序失败: {e}")
#             return documents[:top_k]
        
#         for doc, score in zip(documents, rerank_scores):
#             doc['rerank_score'] = float(score)
            
#             # 即使没有时间过滤，这里仍然可以保留时间衰减逻辑（如果需要），
#             # 或者完全禁用时间衰减。这里保留原逻辑，因为current_timestamp总是存在的。
#             if time_decay and current_timestamp:
#                 publish_time = doc.get('publish_time', 0)
#                 if publish_time > 0:
#                     days_diff = (current_timestamp - publish_time) / 86400
#                     time_factor = max(0.5, 1.0 - (days_diff / 30) * 0.05)
#                     doc['final_score'] = score * time_factor
#                 else:
#                     doc['final_score'] = score
#             else:
#                 doc['final_score'] = score
        
#         ranked_docs = sorted(documents, key=lambda x: x['final_score'], reverse=True)
        
#         print(f"重排序完成，返回top-{top_k}")
#         for i, doc in enumerate(ranked_docs[:top_k]):
#             print(f"  Rank {i+1}: {doc.get('title', '')[:50]}... (分数: {doc['final_score']:.4f})")
        
#         return ranked_docs[:top_k]

# # ----------------------------
# # 混合检索管理器 (已移除时间过滤逻辑)
# # ----------------------------
# class HybridRetrievalManager:
#     """混合检索管理器 (无时间过滤)"""
    
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
#             print(f"连接 Milvus ({self.host}:{self.port})...")
#             self.connection = connections.connect(
#                 alias="default",
#                 host=self.host,
#                 port=self.port
#             )
            
#             self.collection = Collection(MILVUS_COLLECTION_NAME)
#             print(f"已连接到集合: {MILVUS_COLLECTION_NAME}")
            
#             print("加载embedding模型...")
#             self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
#             if self.use_reranker:
#                 try:
#                     self.reranker = EnhancedReranker(RERANKER_MODEL)
#                 except Exception as e:
#                     print(f"重排序器加载失败: {e}, 将不使用重排序")
#                     self.reranker = None
            
#             print("检索系统初始化完成")
            
#         except Exception as e:
#             print(f"检索系统初始化失败: {e}")
#             raise
    
#     def get_embedding(self, text: str) -> List[float]:
#         """获取文本的嵌入向量"""
#         if self.embedding_model is None:
#             raise ValueError("嵌入模型未初始化")
        
#         query_prompt = "为这个问题生成搜索表示: "
#         embedding = self.embedding_model.encode(query_prompt + text)
#         return embedding.tolist()
    
#     def vector_search(
#         self, 
#         question: str, 
#         top_k: int = TOP_K_RETRIEVAL
#     ) -> List[Dict]:
#         """向量检索 (已移除时间过滤)"""
#         if self.collection is None or self.embedding_model is None:
#             raise ValueError("检索管理器未初始化")
        
#         try:
#             question_embedding = self.get_embedding(question)
            
#             search_params = {
#                 "metric_type": "IP",
#                 "params": {"nprobe": 10, "ef": 50}
#             }
            
#             # 不再构建 expr，始终为 None (全库检索)
#             expr = None
            
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
            
#             print(f"向量检索完成: {len(formatted_results)} 个文档")
#             return formatted_results
            
#         except Exception as e:
#             print(f"向量检索失败: {e}")
#             return []
    
#     def keyword_search(
#         self, 
#         question: str, 
#         top_k: int = TOP_K_RETRIEVAL
#     ) -> List[Dict]:
#         """关键字检索 (已移除时间过滤)"""
#         if self.collection is None:
#             raise ValueError("检索管理器未初始化")
        
#         try:
#             keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
#             if not keywords:
#                 print("未提取到关键词")
#                 return []
            
#             keyword_expr_parts = []
#             for keyword in keywords[:5]:
#                 safe_keyword = keyword.replace("'", "''").replace('"', '""')
#                 keyword_expr_parts.append(f"array_contains(keywords, '{safe_keyword}')")
            
#             if not keyword_expr_parts:
#                 return []
            
#             # 只使用关键词表达式，不再拼接时间表达式
#             expr = " or ".join(keyword_expr_parts)
            
#             query_results = self.collection.query(
#                 expr=expr,
#                 output_fields=["doc_id", "title", "content", "publish_time", 
#                               "publish_date_str", "region", "keywords"],
#                 limit=top_k
#             )
            
#             print(f"关键字检索完成: {len(query_results)} 个文档")
            
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
#             print(f"关键字检索失败: {e}")
#             return []
    
#     def reciprocal_rank_fusion(
#         self, 
#         vector_results: List[Dict], 
#         keyword_results: List[Dict], 
#         k: int = 60
#     ) -> List[Dict]:
#         """倒数排名融合（RRF）"""
#         print(f"开始RRF融合，k={k}")
        
#         doc_id_to_results = {}
        
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
        
#         rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
#         final_results = []
#         for rrf_result in rrf_results:
#             doc_info = rrf_result["doc_info"].copy()
#             doc_info["score"] = rrf_result["rrf_score"]
#             doc_info["rrf_score"] = rrf_result["rrf_score"]
#             doc_info["vector_rank"] = rrf_result["vector_rank"]
#             doc_info["keyword_rank"] = rrf_result["keyword_rank"]
#             final_results.append(doc_info)
        
#         print(f"RRF融合完成: {len(final_results)} 个文档")
#         return final_results
    
#     def hybrid_search(
#         self, 
#         question: str
#     ) -> Dict[str, Any]:
#         """混合检索流程 (已移除时间参数)"""
#         print(f"开始混合检索: {question}")
        
#         # 不再传递 time_range
#         vector_results = self.vector_search(question, TOP_K_RETRIEVAL)
#         keyword_results = self.keyword_search(question, TOP_K_RETRIEVAL)
        
#         print(f"初始检索: 向量{len(vector_results)}, 关键词{len(keyword_results)}")
        
#         fused_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
#         print(f"RRF融合后: {len(fused_results)} 个文档")
        
#         if self.reranker and fused_results:
#             current_ts = int(time.time())
#             final_results = self.reranker.rerank(
#                 question, 
#                 fused_results, 
#                 top_k=TOP_K_FINAL,
#                 time_decay=True,
#                 current_timestamp=current_ts
#             )
#         else:
#             final_results = fused_results[:TOP_K_FINAL]
        
#         filtered_results = [doc for doc in final_results if doc.get('final_score', 0) > 0.1]
        
#         print(f"重排序后top-{TOP_K_FINAL}: {len(final_results)} 个文档")
#         print(f"分数过滤后(>0.1): {len(filtered_results)} 个文档")
        
#         return {
#             "final_results": final_results,
#             "filtered_results": filtered_results
#         }

# # 全局检索管理器
# retrieval_manager = None

# # ----------------------------
# # 大模型接口管理
# # ----------------------------
# class LLMInterfaceManager:
#     """大模型接口管理器"""
    
#     def __init__(self, api_key: str, base_url: str, model_name: str = "deepseek-v3"):
#         """初始化大模型接口"""
#         print(f"初始化大模型接口，模型: {model_name}")
#         try:
#             self.client = OpenAI(
#                 api_key=api_key,
#                 base_url=base_url
#             )
#             self.model_name = model_name
#             print("大模型接口初始化完成")
#         except Exception as e:
#             print(f"大模型接口初始化失败: {e}")
#             raise
    
#     def create_batch_file(self, batch_questions: List[Dict], batch_num: int, output_dir: str) -> str:
#         """创建批量处理文件"""
#         os.makedirs(output_dir, exist_ok=True)
#         batch_file_path = os.path.join(output_dir, f"batch_input_{batch_num}.jsonl")
        
#         print(f"创建批次 {batch_num} 输入文件，包含 {len(batch_questions)} 个问题")
        
#         with open(batch_file_path, 'w', encoding='utf-8') as f:
#             for idx, item in enumerate(batch_questions):
#                 prompt = build_rag_prompt(item["question"], item["contexts"])
                
#                 custom_id = f"batch_{batch_num}_item_{idx}_sid_{item.get('source_id', idx)}"
                
#                 batch_request = {
#                     "custom_id": custom_id,
#                     "method": "POST",
#                     "url": "/v1/chat/completions",
#                     "body": {
#                         "model": self.model_name,
#                         "messages": prompt,
#                         "max_tokens": MAX_TOKENS,
#                         "temperature": TEMPERATURE
#                     }
#                 }
                
#                 f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
        
#         print(f"批次文件已创建: {batch_file_path}")
#         return batch_file_path
    
#     def submit_batch_job(self, batch_file_path: str, batch_num: int) -> Optional[str]:
#         """提交批量任务"""
#         try:
#             print(f"提交批次 {batch_num} 批量任务...")
            
#             with open(batch_file_path, 'rb') as file:
#                 uploaded_file = self.client.files.create(file=file, purpose="batch")
            
#             batch_job = self.client.batches.create(
#                 input_file_id=uploaded_file.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window="72h"
#             )
            
#             print(f"✅ 批次 {batch_num} 提交成功，任务ID: {batch_job.id}")
#             return batch_job.id
            
#         except Exception as e:
#             print(f"❌ 批次 {batch_num} 提交失败: {e}")
#             return None
    
#     def monitor_batch_progress(self, batch_id: str, batch_num: int) -> Optional[str]:
#         """监控批量任务进度"""
#         print(f"🔍 开始监控批次 {batch_num} 进度: {batch_id}")
        
#         start_time = time.time()
#         last_status = None
        
#         while True:
#             try:
#                 batch_status = self.client.batches.retrieve(batch_id)
#                 current_status = batch_status.status
                
#                 if current_status != last_status:
#                     elapsed_hours = (time.time() - start_time) / 3600
#                     print(f"🔄 批次 {batch_num} 状态: {current_status} (已等待 {elapsed_hours:.1f} 小时)")
#                     last_status = current_status
                
#                 if current_status == "completed":
#                     print(f"✅ 批次 {batch_num} 处理完成!")
#                     return batch_status.output_file_id
                    
#                 elif current_status in ["failed", "expired", "cancelled"]:
#                     print(f"❌ 批次 {batch_num} 失败: {current_status}")
#                     if hasattr(batch_status, 'errors'):
#                         print(f"错误详情: {batch_status.errors}")
#                     return None
                    
#                 else:
#                     time.sleep(300)
                    
#             except Exception as e:
#                 print(f"⚠️ 检查批次 {batch_num} 状态失败: {e}")
#                 time.sleep(300)
    
#     def download_batch_results(self, output_file_id: str, batch_num: int, original_batch: List[Dict]) -> List[Dict]:
#         """下载批量任务结果"""
#         try:
#             print(f"下载批次 {batch_num} 结果...")
            
#             result_content = self.client.files.content(output_file_id)
#             results = result_content.text.split('\n')
            
#             batch_results = []
#             success_count = 0
#             error_count = 0
            
#             original_indices = {}
#             for idx, item in enumerate(original_batch):
#                 source_id = item.get('source_id', idx)
#                 custom_id = f"batch_{batch_num}_item_{idx}_sid_{source_id}"
#                 original_indices[custom_id] = idx
            
#             for line_num, line in enumerate(results, 1):
#                 if not line.strip():
#                     continue
                    
#                 try:
#                     result_obj = json.loads(line)
#                     custom_id = result_obj.get("custom_id")
#                     response = result_obj.get("response", {})
                    
#                     if response.get("status_code") == 200:
#                         body = response.get("body", {})
#                         choices = body.get("choices", [])
                        
#                         if choices:
#                             answer = choices[0].get("message", {}).get("content", "")
                            
#                             if custom_id in original_indices:
#                                 item_idx = original_indices[custom_id]
                                
#                                 batch_results.append({
#                                     "item_idx": item_idx,
#                                     "custom_id": custom_id,
#                                     "answer": answer,
#                                     "success": True
#                                 })
#                                 success_count += 1
                                
#                                 if success_count % 20 == 0:
#                                     print(f"✅ 批次 {batch_num}: 已解析 {success_count} 个结果")
#                             else:
#                                 print(f"未找到custom_id对应的原始数据: {custom_id}")
#                                 error_count += 1
#                         else:
#                             error_count += 1
#                     else:
#                         error_count += 1
                        
#                 except Exception as e:
#                     error_count += 1
#                     print(f"解析第 {line_num} 行失败: {e}")
            
#             print(f"📊 批次 {batch_num} 解析完成: 成功 {success_count}, 失败 {error_count}")
#             return batch_results
            
#         except Exception as e:
#             print(f"❌ 下载批次 {batch_num} 结果失败: {e}")
#             return []

# # 全局大模型管理器
# llm_manager = None

# # ----------------------------
# # 主函数
# # ----------------------------
# def main():
#     """主函数 - 使用大模型接口批量处理（无时间过滤）"""
#     # 初始化
#     jieba.initialize()
    
#     print("\n" + "="*70)
#     print("🚀 RAG问答系统（无时间过滤版 - 全量检索）")
#     print("="*70)
#     print(f"模型: {MODEL_NAME}")
#     print(f"批量大小: {BATCH_SIZE}")
#     print(f"输入: {INPUT_DATA_PATH}")
#     print(f"输出: {OUTPUT_PATH}")
#     print("="*70)
    
#     start_time = time.time()
    
#     try:
#         # 1. 初始化检索系统
#         print("初始化混合检索系统...")
#         global retrieval_manager
#         retrieval_manager = HybridRetrievalManager(
#             MILVUS_HOST, 
#             MILVUS_PORT,
#             use_reranker=True
#         )
#         retrieval_manager.initialize()
#         print("✅ 混合检索系统初始化成功")
        
#         # 2. 初始化大模型接口
#         print("初始化大模型接口...")
#         global llm_manager
#         llm_manager = LLMInterfaceManager(
#             api_key=API_KEY,
#             base_url=API_BASE_URL,
#             model_name=MODEL_NAME
#         )
#         print("✅ 大模型接口初始化成功")
        
#         # 3. 加载输入数据
#         print(f"加载输入数据: {INPUT_DATA_PATH}")
#         with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
#             input_data = json.load(f)
        
#         total_samples = len(input_data)
#         print(f"加载完成: {total_samples} 条数据")
        
#         # 4. 创建输出目录
#         output_dir = os.path.dirname(OUTPUT_PATH)
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 5. 分批处理
#         total_batches = (len(input_data) + BATCH_SIZE - 1) // BATCH_SIZE
#         all_results = []
        
#         print(f"数据分 {total_batches} 批处理")
        
#         for batch_num in range(1, total_batches + 1):
#             batch_start = (batch_num - 1) * BATCH_SIZE
#             batch_end = batch_num * BATCH_SIZE
#             batch_data = input_data[batch_start:batch_end]
            
#             print(f"\n🔄 开始处理批次 {batch_num}/{total_batches}")
#             print(f"📊 批次数据: {len(batch_data)} 条")
            
#             # 5.1 检索批次内所有问题
#             batch_questions = []
            
#             for idx, item in enumerate(tqdm(batch_data, desc=f"检索批次 {batch_num}")):
#                 try:
#                     question = item.get("question", "").strip()
#                     if not question:
#                         continue
                    
#                     source_id = item.get("source_id", idx + batch_start)
                    
#                     # 1. 提取关键词 (保留)
#                     extracted_keywords = extract_keywords_enhanced(question, STOPWORDS, top_k=15)
                    
#                     # 2. 执行混合检索 (不再传递 time_range)
#                     hybrid_results = retrieval_manager.hybrid_search(question)
                    
#                     # 获取过滤后的结果
#                     filtered_docs = hybrid_results.get("filtered_results", [])
                    
#                     # 使用过滤后的文档作为上下文
#                     retrieved_docs = filtered_docs
                    
#                     has_supporting_info = len(retrieved_docs) > 0
                    
#                     # 构建检索信息
#                     retrieval_info = {
#                         "has_supporting_info": has_supporting_info,
#                         "extracted_keywords": extracted_keywords,
#                         "filtered_docs_count": len(filtered_docs),
#                     }
                    
#                     # 只保存过滤后的文档预览
#                     retrieval_info["retrieved_docs_preview"] = [
#                         {
#                             "doc_id": doc.get("doc_id", ""),
#                             "title": doc.get("title", ""),
#                             "content_preview": str(doc.get("content", ""))[:800],
#                             "publish_date": doc.get("publish_date_str", ""),
#                             "retrieval_type": doc.get("retrieval_type", "unknown"),
#                             "final_score": float(round(doc.get("final_score", 0), 4)),
#                             "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
#                         }
#                         for doc in retrieved_docs
#                     ]
                    
#                     # 【移除】不再添加 time_extraction 字段
                    
#                     # 清理文档用于上下文构建
#                     contexts = clean_retrieved_documents_for_context(retrieved_docs)
                    
#                     # 构建问题信息
#                     question_info = {
#                         "source_id": source_id,
#                         "question": question,
#                         "standard_answer": item.get("answer", ""),
#                         "contexts": contexts,
#                         "retrieval_info": retrieval_info,
#                         "filtered_docs": filtered_docs,
#                         "item_idx_in_batch": idx
#                     }
                    
#                     batch_questions.append(question_info)
                    
#                 except Exception as e:
#                     print(f"处理批次 {batch_num} 中第 {idx} 条数据失败: {e}")
#                     continue
            
#             print(f"✅ 批次 {batch_num} 检索完成")
            
#             # 5.2 创建批量输入文件
#             batch_file_path = llm_manager.create_batch_file(
#                 batch_questions, 
#                 batch_num, 
#                 output_dir
#             )
            
#             # 5.3 提交批量任务
#             batch_id = llm_manager.submit_batch_job(batch_file_path, batch_num)
#             if not batch_id:
#                 print(f"❌ 批次 {batch_num} 提交失败，跳过")
#                 continue
            
#             # 5.4 监控批量任务进度
#             output_file_id = llm_manager.monitor_batch_progress(batch_id, batch_num)
#             if not output_file_id:
#                 print(f"❌ 批次 {batch_num} 处理失败，跳过")
#                 continue
            
#             # 5.5 下载和处理结果
#             batch_results = llm_manager.download_batch_results(
#                 output_file_id, 
#                 batch_num,
#                 batch_questions
#             )
            
#             if batch_results:
#                 # 5.6 合并检索结果和生成结果
#                 batch_results_map = {}
#                 for result in batch_results:
#                     if result.get("success"):
#                         batch_results_map[result.get("item_idx")] = result
                
#                 batch_output_items = []
#                 for idx, question_info in enumerate(batch_questions):
#                     matching_result = batch_results_map.get(idx)
                    
#                     if matching_result:
#                         answer = matching_result["answer"]
#                         generation_success = True
#                     else:
#                         answer = "生成失败"
#                         generation_success = False
                    
#                     # 构建输出项
#                     output_item = {
#                         "source_id": question_info.get("source_id", idx + batch_start),
#                         "question": question_info["question"],
#                         "standard_answer": question_info["standard_answer"],
#                         "model_answer": answer,
#                         "retrieval_info": question_info["retrieval_info"],
#                         "processing_times": {
#                             "total_processing_time": 0.0,
#                             "retrieval_time": 0.0,
#                             "generation_time": 0.0,
#                         }
#                     }
                    
#                     batch_output_items.append(output_item)
                
#                 # 5.7 添加到总结果
#                 all_results.extend(batch_output_items)
                
#                 print(f"✅ 批次 {batch_num} 完成!")
#                 print(f"📊 当前进度: {len(all_results)}/{total_samples} 条数据")
                
#             else:
#                 print(f"❌ 批次 {batch_num} 未生成有效结果")
        
#         # 6. 一次性保存最终结果
#         print(f"\n💾 保存最终结果到: {OUTPUT_PATH}")
#         with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
#             json.dump(all_results, f, ensure_ascii=False, indent=2)
        
#         # 7. 最终统计
#         total_time = time.time() - start_time
#         days = int(total_time // 86400)
#         hours = int((total_time % 86400) // 3600)
#         minutes = int((total_time % 3600) // 60)
        
#         print("\n" + "="*70)
#         print("🎉 RAG问答批量处理完成!")
#         print("="*70)
#         print(f"总样本数: {len(all_results)}")
#         print(f"总处理时间: {days}天{hours}小时{minutes}分钟")
#         print("="*70)
#         print(f"输出文件: {OUTPUT_PATH}")
#         print("="*70)
        
#     except Exception as e:
#         print(f"❌ 处理过程中发生错误: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


# ===========================================================bge-m3检索大模型接口=====================================================
# -*- coding: utf-8 -*-
"""
RAG问答系统 - 精简输出版（支持BGE-M3稠密+稀疏向量检索）
功能：批量处理输入问题，仅输出最终JSON结果，不保存任何中间文件。
"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import re
from datetime import datetime, timedelta

# RAG相关库
from pymilvus import connections, Collection
from sentence_transformers import CrossEncoder
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 大模型接口
from openai import OpenAI

# ----------------------------
# 配置路径和参数
# ----------------------------
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_m3/test_qwen2.5.json"

# 大模型配置
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "__SET_VIA_ENVIRONMENT__"
MODEL_NAME = "deepseek-v3"
MAX_TOKENS = 4096
TEMPERATURE = 0.1

# 批量处理配置
BATCH_SIZE = 100

# Milvus配置
MILVUS_HOST = "122.226.69.234"
MILVUS_PORT = "3042"
MILVUS_COLLECTION_NAME = "news_collection_m3_450"      # 使用新建的稠密+稀疏集合
EMBEDDING_MODEL = "/mnt/project/milvus/models/bge-m3"                     # BGE-M3模型名称
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 检索参数
TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 5

# ----------------------------
# 时间提取
# ----------------------------
def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
    """从问题中提取时间信息"""
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
                
                print(f"提取时间 [{date_type}]: {extracted_date.strftime('%Y-%m-%d')}")
                return {
                    "extracted_date": extracted_date,
                    "start_timestamp": int(start_date.timestamp()),
                    "end_timestamp": int(end_date.timestamp()),
                    "start_date_str": start_date.strftime("%Y-%m-%d"),
                    "end_date_str": end_date.strftime("%Y-%m-%d"),
                    "has_specific_date": True,
                    "date_type": date_type,
                }
            except Exception as e:
                print(f"时间解析失败: {e}")
                continue
    
    relative_time_patterns = {
        r'今天|当天': (0, 1),
        r'昨天|昨日': (-1, 1),
        r'前天': (-2, 1),
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
            
            print(f"相对时间提取: {pattern}")
            return {
                "extracted_date": center_date,
                "start_timestamp": int(start_date.timestamp()),
                "end_timestamp": int(end_date.timestamp()),
                "start_date_str": start_date.strftime("%Y-%m-%d"),
                "end_date_str": end_date.strftime("%Y-%m-%d"),
                "has_specific_date": True,
                "is_relative": True,
            }
    
    print("未提取到时间信息，不进行时间过滤")
    return None

# ----------------------------
# 文档清理函数
# ----------------------------
def clean_retrieved_documents_for_context(docs: List[Dict]) -> List[Dict]:
    """清理检索到的文档，只保留构建RAG prompt所需的字段"""
    if not docs:
        return []
    cleaned = []
    for doc in docs:
        cleaned.append({
            "title": str(doc.get("title", "")),
            "content": str(doc.get("content", "")),
            "publish_date_str": str(doc.get("publish_date_str", "")),
            "region": str(doc.get("region", "")),
        })
    return cleaned

def clean_retrieved_documents_for_preview(docs: List[Dict]) -> List[Dict]:
    """清理检索到的文档，用于保存预览信息"""
    if not docs:
        return []
    cleaned = []
    for doc in docs:
        cleaned.append({
            "doc_id": str(doc.get("doc_id", "")),
            "title": str(doc.get("title", "")),
            "content_preview": str(doc.get("content", ""))[:800],
            "publish_date": str(doc.get("publish_date_str", "")),
            "retrieval_type": str(doc.get("retrieval_type", "unknown")),
            "final_score": float(round(doc.get("final_score", 0), 4)),
            "rerank_score": float(round(doc.get("rerank_score", 0), 4)) if doc.get("rerank_score") is not None else None,
        })
    return cleaned

# ----------------------------
# RAG Prompt构建函数
# ----------------------------
def build_rag_prompt(question: str, contexts: List[Dict]) -> List[Dict]:
    """构建RAG Prompt"""
    if not contexts:
        return [
            {"role": "system", "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答'基于现有资料，我无法回答这个问题'。"},
            {"role": "user", "content": f"问题: {question}"}
        ]
    
    context_parts = []
    for i, doc in enumerate(contexts, 1):
        context_parts.append(
            f"【文档{i}】\n"
            f"标题: {doc.get('title', f'文档{i}')}\n"
            f"时间: {doc.get('publish_date_str', '未知')}\n"
            f"内容: {doc.get('content', '')[:800]}\n"
        )
    
    system_prompt = """你是一个新闻问答助手。你的任务是基于提供的参考资料回答用户问题。

核心原则:
1. 严格依据参考资料中的信息回答。
2. 如果不同来源有冲突，优先选择发布时间最新的资料。
3. 回答要简洁清晰，直接给出答案。
4. 如果无法从参考资料中找到答案，回答："根据现有资料，我无法回答这个问题。"
"""
    user_prompt = f"参考资料:\n{''.join(context_parts)}\n问题: {question}\n请基于上述资料回答问题。"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# ----------------------------
# 重排序器类
# ----------------------------
class EnhancedReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"加载重排序模型: {model_name}")
        try:
            self.reranker = CrossEncoder(model_name, max_length=512)
            print("重排序模型加载完成")
        except Exception as e:
            print(f"重排序模型加载失败: {e}")
            self.reranker = None
    
    def rerank(self, question: str, documents: List[Dict], top_k: int = 5,
               time_decay: bool = True, current_timestamp: Optional[int] = None) -> List[Dict]:
        if not documents or self.reranker is None:
            return documents[:top_k]
        
        pairs = []
        for doc in documents:
            doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"[:500]
            pairs.append([question, doc_text])
        
        try:
            rerank_scores = self.reranker.predict(pairs)
        except Exception as e:
            print(f"重排序失败: {e}")
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
        
        ranked = sorted(documents, key=lambda x: x['final_score'], reverse=True)
        return ranked[:top_k]

# ----------------------------
# 混合检索管理器（稠密+稀疏）
# ----------------------------
class HybridRetrievalManager:
    def __init__(self, host: str, port: str, use_reranker: bool = True):
        self.host = host
        self.port = port
        self.collection = None
        self.embedding_model = None
        self.reranker = None
        self.use_reranker = use_reranker

    def initialize(self):
        """初始化Milvus连接、BGE-M3模型和重排序器"""
        try:
            print(f"连接 Milvus ({self.host}:{self.port})...")
            connections.connect(alias="default", host=self.host, port=self.port)
            self.collection = Collection(MILVUS_COLLECTION_NAME)
            print(f"已连接到集合: {MILVUS_COLLECTION_NAME}")

            print("加载 BGE-M3 嵌入模型...")
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
                    print(f"重排序器加载失败: {e}, 将不使用重排序")
                    self.reranker = None

            print("检索系统初始化完成")
        except Exception as e:
            print(f"检索系统初始化失败: {e}")
            raise

    def get_query_embeddings(self, question: str):
        """生成查询的稠密和稀疏向量"""
        if self.embedding_model is None:
            raise ValueError("嵌入模型未初始化")
        embeddings = self.embedding_model.encode_queries([question])
        return embeddings['dense'][0], embeddings['sparse']

    def dense_search(self, question: str, time_range: Optional[Dict] = None, top_k: int = TOP_K_RETRIEVAL):
        """稠密向量检索"""
        dense_vec, _ = self.get_query_embeddings(question)
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        expr = self._build_time_expr(time_range)

        results = self.collection.search(
            data=[dense_vec.tolist()],
            anns_field="dense_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id", "title", "content", "publish_time", "publish_date_str", "region"],
            consistency_level="Eventually"
        )

        formatted = []
        if results:
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    formatted.append({
                        "id": hit.id,
                        "doc_id": entity.get("doc_id", ""),
                        "score": hit.score,
                        "title": entity.get("title", ""),
                        "content": entity.get("content", ""),
                        "publish_time": entity.get("publish_time", 0),
                        "publish_date_str": entity.get("publish_date_str", ""),
                        "region": entity.get("region", ""),
                        "retrieval_type": "dense"
                    })
        print(f"稠密检索完成: {len(formatted)} 个文档")
        return formatted

    def sparse_search(self, question: str, time_range: Optional[Dict] = None, top_k: int = TOP_K_RETRIEVAL):
        """稀疏向量检索"""
        _, sparse_vec = self.get_query_embeddings(question)
        search_params = {"metric_type": "IP", "params": {}}
        expr = self._build_time_expr(time_range)

        results = self.collection.search(
            data=[sparse_vec],
            anns_field="sparse_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id", "title", "content", "publish_time", "publish_date_str", "region"],
            consistency_level="Eventually"
        )

        formatted = []
        if results:
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    formatted.append({
                        "id": hit.id,
                        "doc_id": entity.get("doc_id", ""),
                        "score": hit.score,
                        "title": entity.get("title", ""),
                        "content": entity.get("content", ""),
                        "publish_time": entity.get("publish_time", 0),
                        "publish_date_str": entity.get("publish_date_str", ""),
                        "region": entity.get("region", ""),
                        "retrieval_type": "sparse"
                    })
        print(f"稀疏检索完成: {len(formatted)} 个文档")
        return formatted

    def _build_time_expr(self, time_range: Optional[Dict]) -> Optional[str]:
        """构建时间过滤表达式"""
        if time_range and time_range.get("has_specific_date", False):
            start_ts = time_range.get("start_timestamp")
            end_ts = time_range.get("end_timestamp")
            if start_ts and end_ts:
                return f"publish_time >= {start_ts} and publish_time <= {end_ts}"
        return None

    def reciprocal_rank_fusion(self, results_a: List[Dict], results_b: List[Dict], k: int = 60) -> List[Dict]:
        """倒数排名融合（RRF）"""
        doc_map = {}
        for rank, doc in enumerate(results_a, 1):
            doc_id = doc.get("doc_id")
            if doc_id:
                doc_map.setdefault(doc_id, {"doc_info": doc, "rank_a": rank, "rank_b": float('inf')})
        for rank, doc in enumerate(results_b, 1):
            doc_id = doc.get("doc_id")
            if doc_id:
                if doc_id in doc_map:
                    doc_map[doc_id]["rank_b"] = rank
                else:
                    doc_map[doc_id] = {"doc_info": doc, "rank_a": float('inf'), "rank_b": rank}

        fused = []
        for doc_id, info in doc_map.items():
            score_a = 1 / (k + info["rank_a"]) if info["rank_a"] != float('inf') else 0
            score_b = 1 / (k + info["rank_b"]) if info["rank_b"] != float('inf') else 0
            doc = info["doc_info"].copy()
            doc["score"] = score_a + score_b
            doc["rrf_score"] = score_a + score_b
            doc["dense_rank"] = info["rank_a"] if info["rank_a"] != float('inf') else None
            doc["sparse_rank"] = info["rank_b"] if info["rank_b"] != float('inf') else None
            fused.append(doc)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        print(f"RRF融合完成: {len(fused)} 个文档")
        return fused

    def hybrid_search(self, question: str, time_range: Optional[Dict] = None) -> Dict[str, Any]:
        """混合检索：稠密 + 稀疏 + RRF + 重排序"""
        print(f"开始混合检索: {question}")
        dense_results = self.dense_search(question, time_range, TOP_K_RETRIEVAL)
        sparse_results = self.sparse_search(question, time_range, TOP_K_RETRIEVAL)

        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)

        if self.reranker and fused_results:
            current_ts = int(time.time())
            final_results = self.reranker.rerank(
                question, fused_results, top_k=TOP_K_FINAL,
                time_decay=True, current_timestamp=current_ts
            )
        else:
            final_results = fused_results[:TOP_K_FINAL]

        filtered = [doc for doc in final_results if doc.get('final_score', 0) > 0.1]
        return {"final_results": final_results, "filtered_results": filtered}

# 全局检索管理器
retrieval_manager = None

# ----------------------------
# 大模型接口管理
# ----------------------------
class LLMInterfaceManager:
    def __init__(self, api_key: str, base_url: str, model_name: str = "deepseek-v3"):
        print(f"初始化大模型接口，模型: {model_name}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def create_batch_file(self, batch_questions: List[Dict], batch_num: int, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        batch_file = os.path.join(output_dir, f"batch_input_{batch_num}.jsonl")
        with open(batch_file, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(batch_questions):
                prompt = build_rag_prompt(item["question"], item["contexts"])
                custom_id = f"batch_{batch_num}_item_{idx}_sid_{item.get('source_id', idx)}"
                req = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": prompt,
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMPERATURE
                    }
                }
                f.write(json.dumps(req, ensure_ascii=False) + '\n')
        return batch_file

    def submit_batch_job(self, batch_file: str, batch_num: int) -> Optional[str]:
        try:
            with open(batch_file, 'rb') as f:
                uploaded = self.client.files.create(file=f, purpose="batch")
            job = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="72h"
            )
            print(f"✅ 批次 {batch_num} 提交成功，任务ID: {job.id}")
            return job.id
        except Exception as e:
            print(f"❌ 批次 {batch_num} 提交失败: {e}")
            return None

    def monitor_batch_progress(self, batch_id: str, batch_num: int) -> Optional[str]:
        print(f"🔍 监控批次 {batch_num} 进度: {batch_id}")
        start = time.time()
        while True:
            try:
                status = self.client.batches.retrieve(batch_id)
                if status.status == "completed":
                    print(f"✅ 批次 {batch_num} 完成")
                    return status.output_file_id
                elif status.status in ["failed", "expired", "cancelled"]:
                    print(f"❌ 批次 {batch_num} 失败: {status.status}")
                    return None
                time.sleep(300)
            except Exception as e:
                print(f"⚠️ 检查状态失败: {e}")
                time.sleep(300)

    def download_batch_results(self, output_file_id: str, batch_num: int, original_batch: List[Dict]) -> List[Dict]:
        try:
            content = self.client.files.content(output_file_id).text
            lines = content.split('\n')
            results = []
            idx_map = {f"batch_{batch_num}_item_{i}_sid_{item.get('source_id', i)}": i
                       for i, item in enumerate(original_batch)}
            for line in lines:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("custom_id")
                    if cid in idx_map and obj.get("response", {}).get("status_code") == 200:
                        answer = obj["response"]["body"]["choices"][0]["message"]["content"]
                        results.append({"item_idx": idx_map[cid], "answer": answer, "success": True})
                except:
                    pass
            print(f"批次 {batch_num} 解析完成: 成功 {len(results)} 条")
            return results
        except Exception as e:
            print(f"下载批次 {batch_num} 结果失败: {e}")
            return []

# 全局大模型管理器
llm_manager = None

# ----------------------------
# 主函数
# ----------------------------
def main():
    print("\n" + "="*70)
    print("🚀 RAG问答系统（精简输出版 - 稠密+稀疏混合检索）")
    print("="*70)
    print(f"模型: {MODEL_NAME}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"输入: {INPUT_DATA_PATH}")
    print(f"输出: {OUTPUT_PATH}")
    print("="*70)

    start_time = time.time()

    try:
        # 1. 初始化检索系统
        global retrieval_manager, llm_manager
        retrieval_manager = HybridRetrievalManager(MILVUS_HOST, MILVUS_PORT, use_reranker=True)
        retrieval_manager.initialize()

        # 2. 初始化大模型接口
        llm_manager = LLMInterfaceManager(API_KEY, API_BASE_URL, MODEL_NAME)

        # 3. 加载输入数据
        with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        total = len(input_data)
        print(f"加载完成: {total} 条数据")

        output_dir = os.path.dirname(OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)

        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        all_results = []

        for batch_num in range(1, total_batches + 1):
            batch_start = (batch_num - 1) * BATCH_SIZE
            batch_end = batch_num * BATCH_SIZE
            batch_data = input_data[batch_start:batch_end]

            print(f"\n🔄 处理批次 {batch_num}/{total_batches} (数据 {len(batch_data)} 条)")

            batch_questions = []
            for idx, item in enumerate(tqdm(batch_data, desc=f"检索批次 {batch_num}")):
                question = item.get("question", "").strip()
                if not question:
                    continue

                source_id = item.get("source_id", idx + batch_start)
                time_range = extract_time_from_question(question)
                hybrid_results = retrieval_manager.hybrid_search(question, time_range)
                filtered_docs = hybrid_results.get("filtered_results", [])

                retrieval_info = {
                    "has_supporting_info": len(filtered_docs) > 0,
                    "filtered_docs_count": len(filtered_docs),
                    "retrieved_docs_preview": clean_retrieved_documents_for_preview(filtered_docs)
                }
                if time_range:
                    retrieval_info["time_extraction"] = {
                        "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d"),
                        "date_type": time_range.get("date_type", "relative"),
                    }

                contexts = clean_retrieved_documents_for_context(filtered_docs)
                batch_questions.append({
                    "source_id": source_id,
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "contexts": contexts,
                    "retrieval_info": retrieval_info,
                    "filtered_docs": filtered_docs,
                    "item_idx_in_batch": idx
                })

            # 创建批量任务
            batch_file = llm_manager.create_batch_file(batch_questions, batch_num, output_dir)
            batch_id = llm_manager.submit_batch_job(batch_file, batch_num)
            if not batch_id:
                continue

            output_file_id = llm_manager.monitor_batch_progress(batch_id, batch_num)
            if not output_file_id:
                continue

            batch_results = llm_manager.download_batch_results(output_file_id, batch_num, batch_questions)

            if batch_results:
                result_map = {r["item_idx"]: r for r in batch_results if r.get("success")}
                for idx, q_info in enumerate(batch_questions):
                    match = result_map.get(idx)
                    answer = match["answer"] if match else "生成失败"
                    all_results.append({
                        "source_id": q_info["source_id"],
                        "question": q_info["question"],
                        "standard_answer": q_info["standard_answer"],
                        "model_answer": answer,
                        "retrieval_info": q_info["retrieval_info"],
                        "processing_times": {"total_processing_time": 0.0}
                    })
                print(f"✅ 批次 {batch_num} 完成，累计 {len(all_results)}/{total} 条")

        # 保存最终结果
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        print(f"\n🎉 全部完成！耗时 {int(elapsed//3600)}小时{int((elapsed%3600)//60)}分钟")
        print(f"输出文件: {OUTPUT_PATH}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# ==============================================bge-m3训练模型接口==============================================
# ==========================================
# 纯RAG问答系统完整代码（使用BGE-M3稠密+稀疏向量）
# generate_rag_pure.py
# ==========================================

import os
import json
import torch
import logging
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    filename='generate_rag_pure.log',
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
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test_chunk/test_nodecide_basemodel.json"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# Milvus配置
# ----------------------------
MILVUS_HOST = "122.226.69.234"
MILVUS_PORT = "3042"
MILVUS_COLLECTION_NAME = "news_collection_m3"          # 使用新建的稠密+稀疏集合
EMBEDDING_MODEL = "BAAI/bge-m3"                         # BGE-M3模型名称
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 检索参数
TOP_K_RETRIEVAL = 20  # 初始检索数量
TOP_K_FINAL = 5       # 最终返回数量

# ----------------------------
# 时间提取（保持不变）
# ----------------------------
def extract_time_from_question(question: str) -> Optional[Dict[str, Any]]:
    """从问题中提取时间信息"""
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
# 混合检索管理器（稠密+稀疏）
# ----------------------------
class HybridRetrievalManager:
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

        # 处理稠密检索结果
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

        # 处理稀疏检索结果
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

        # 计算RRF分数
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

    def hybrid_search(
        self,
        question: str,
        time_range: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """混合检索流程"""
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
# RAG Prompt构建函数（保持不变）
# ----------------------------
def build_rag_prompt(question: str, contexts: List[Dict]) -> List[Dict]:
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
            f"内容: {content}\n"
        )

    context_text = "\n".join(context_parts)

    system_prompt = """你是一个新闻问答助手。你的任务是基于提供的参考资料回答用户问题。

核心原则:
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
回答格式:
- 如果根据参考资料能回答: 直接给出答案
- 如果根据参考资料不能回答: "根据现有资料，我无法回答这个问题。"，即便你知道这个问题答案，如果参考资料没有相关信息，也要回答"根据现有资料，我无法回答这个问题。"
"""

    user_prompt = f"""参考资料:
{context_text}

问题: {question}

请基于上述资料回答问题。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

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
        retrieval_manager = HybridRetrievalManager(
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
# 生成函数
# ----------------------------
def generate_answer_with_context(model, tokenizer, question, contexts=None, start_time=None):
    if contexts:
        logger.info(f"使用RAG模式生成答案，上下文数量: {len(contexts)}")
        messages = build_rag_prompt(question, contexts)
    else:
        logger.info("无检索结果，使用基础prompt")
        messages = [
            {
                "role": "system",
                "content": "你是一个新闻问答助手。请基于参考资料回答问题，如果没有参考资料，就诚实地回答“基于现有资料，我无法回答这个问题。”"
            },
            {
                "role": "user",
                "content": f"问题: {question}"
            }
        ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs.sequences[0, inputs['input_ids'].size(1):]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    logger.info(f"生成答案完成")

    if start_time:
        generation_time = time.perf_counter() - start_time
        logger.info(f"模型生成耗时: {generation_time:.3f}秒")
        return answer, generation_time
    else:
        return answer

# ----------------------------
# 主函数
# ----------------------------
def main():
    # 加载模型
    model, tokenizer = load_model(MODEL_BASE_PATH)

    # 初始化检索系统
    retrieval_enabled = initialize_retrieval()
    if not retrieval_enabled:
        logger.error("检索系统初始化失败，无法进行RAG问答")
        return

    # 加载输入数据
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    total_samples = len(input_data)
    total_processing_time = 0.0
    total_retrieval_time = 0.0
    total_generation_time = 0.0
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
                time_range = extract_time_from_question(question)

                # 2. 执行混合检索（无需关键词提取）
                retrieval_start_time = time.perf_counter()
                hybrid_results = retrieval_manager.hybrid_search(question, time_range)
                retrieval_end_time = time.perf_counter()
                retrieval_time = retrieval_end_time - retrieval_start_time
                total_retrieval_time += retrieval_time

                filtered_docs = hybrid_results.get("filtered_results", [])
                all_docs = hybrid_results.get("final_results", [])

                retrieved_docs = filtered_docs
                has_supporting_info = len(retrieved_docs) > 0

                logger.info(f"检索到 {len(all_docs)} 个文档，其中 {len(filtered_docs)} 个分数大于0.1")
                logger.info(f"检索耗时: {retrieval_time:.3f}秒")

                # 3. 生成答案
                generation_start_time = time.perf_counter()
                answer, generation_time = generate_answer_with_context(
                    model, tokenizer, question, contexts=retrieved_docs, start_time=generation_start_time
                )
                total_generation_time += generation_time

                sample_end_time = time.perf_counter()
                sample_processing_time = sample_end_time - sample_start_time
                total_processing_time += sample_processing_time

                # 构建检索信息（不再包含关键词）
                retrieval_info = {
                    "has_supporting_info": has_supporting_info,
                    "filtered_docs_count": len(filtered_docs),
                }

                retrieval_info["retrieved_docs_preview"] = [
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
                    retrieval_info["time_extraction"] = {
                        "extracted_date": time_range.get("extracted_date").strftime("%Y-%m-%d"),
                        "date_type": time_range.get("date_type", "relative"),
                        "range_days": time_range.get("range_days", "N/A"),
                    }

                output_item = {
                    "source_id": item.get("source_id", idx),
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "model_answer": answer,
                    "retrieval_info": retrieval_info,
                    "processing_times": {
                        "total_processing_time": round(sample_processing_time, 3),
                        "retrieval_time": round(retrieval_time, 3),
                        "generation_time": round(generation_time, 3),
                    }
                }

                sample_times.append({
                    "sample_id": idx,
                    "question": question[:50] + "..." if len(question) > 50 else question,
                    "total_time": sample_processing_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "retrieved_docs_total": len(all_docs),
                    "filtered_docs_count": len(filtered_docs)
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

    # 输出时间统计
    if total_samples > 0:
        avg_processing_time = total_processing_time / total_samples
        avg_retrieval_time = total_retrieval_time / total_samples
        avg_generation_time = total_generation_time / total_samples

        time_stats = {
            "total_samples": total_samples,
            "total_processing_time_seconds": round(total_processing_time, 3),
            "average_processing_time_seconds": round(avg_processing_time, 3),
            "breakdown": {
                "retrieval": {
                    "total_seconds": round(total_retrieval_time, 3),
                    "average_seconds": round(avg_retrieval_time, 3),
                    "percentage": round((total_retrieval_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                },
                "generation": {
                    "total_seconds": round(total_generation_time, 3),
                    "average_seconds": round(avg_generation_time, 3),
                    "percentage": round((total_generation_time / total_processing_time) * 100, 2) if total_processing_time > 0 else 0
                }
            },
            "sample_details": sample_times,
            "performance_summary": {
                "samples_per_hour": round(3600 / avg_processing_time, 2) if avg_processing_time > 0 else 0,
                "retrieval_efficiency": f"{avg_retrieval_time:.3f}s/doc",
                "generation_speed": f"{avg_generation_time:.3f}s/answer"
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info("\n" + "="*60)
        logger.info("时间统计摘要:")
        logger.info("="*60)
        logger.info(f"处理样本总数: {total_samples}")
        logger.info(f"总处理时间: {total_processing_time:.2f}秒")
        logger.info(f"平均处理时间: {avg_processing_time:.3f}秒/样本")
        logger.info(f"  - 平均检索时间: {avg_retrieval_time:.3f}秒 ({time_stats['breakdown']['retrieval']['percentage']}%)")
        logger.info(f"  - 平均生成时间: {avg_generation_time:.3f}秒 ({time_stats['breakdown']['generation']['percentage']}%)")
        logger.info(f"处理速度: {time_stats['performance_summary']['samples_per_hour']} 样本/小时")

    logger.info(f"✅ RAG问答完成，结果保存至 {OUTPUT_PATH}")
    logger.info(f"日志文件: generate_rag_pure.log")

if __name__ == "__main__":
    main()