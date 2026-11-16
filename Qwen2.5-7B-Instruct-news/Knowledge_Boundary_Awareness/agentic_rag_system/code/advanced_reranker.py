"""
高级文档重排序模块
实现多种重排序策略以提升检索质量
"""

import logging
import re
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)


class AdvancedReranker:
    """高级重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
        """
        try:
            self.reranker = CrossEncoder(model_name)
            logger.info(f"成功加载重排序模型: {model_name}")
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {e}, 将使用基础排序")
            self.reranker = None
    
    def rerank_by_cross_encoder(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        使用CrossEncoder重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top-k文档
            
        Returns:
            重排序后的文档列表
        """
        if not documents or not self.reranker:
            return documents[:top_k]
        
        try:
            # 构建查询-文档对
            pairs = []
            for doc in documents:
                # 使用标题+内容前200字
                doc_text = doc.get("title", "") + " " + doc.get("content", "")[:200]
                pairs.append([query, doc_text])
            
            # 计算重排序分数
            scores = self.reranker.predict(pairs)
            
            # 将分数添加到文档中
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # 按重排序分数排序
            reranked = sorted(
                documents,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )
            
            logger.info(f"CrossEncoder重排序完成: {len(reranked)} -> {top_k}")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents[:top_k]
    
    def rerank_by_diversity(
        self,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        diversity_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        基于多样性的重排序(MMR)
        
        Args:
            documents: 文档列表
            top_k: 返回top-k文档
            diversity_weight: 多样性权重(0-1)
            
        Returns:
            重排序后的文档列表
        """
        if len(documents) <= top_k:
            return documents
        
        selected = []
        remaining = documents.copy()
        
        # 第一个文档选择分数最高的
        selected.append(remaining.pop(0))
        
        # 迭代选择剩余文档
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for idx, doc in enumerate(remaining):
                # 相关性分数
                relevance = doc.get("rrf_score", 0) or doc.get("rerank_score", 0)
                
                # 多样性分数(与已选文档的相似度)
                diversity = self._compute_diversity(doc, selected)
                
                # MMR分数
                mmr_score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        logger.info(f"多样性重排序完成: 选择了 {len(selected)} 个文档")
        return selected
    
    def _compute_diversity(
        self,
        doc: Dict[str, Any],
        selected_docs: List[Dict[str, Any]]
    ) -> float:
        """
        计算文档与已选文档集合的多样性
        
        Returns:
            多样性分数(0-1),越高越多样
        """
        if not selected_docs:
            return 1.0
        
        # 简单的基于关键词重叠的多样性度量
        doc_keywords = set(doc.get("keywords", []))
        
        similarities = []
        for sel_doc in selected_docs:
            sel_keywords = set(sel_doc.get("keywords", []))
            if doc_keywords and sel_keywords:
                intersection = len(doc_keywords & sel_keywords)
                union = len(doc_keywords | sel_keywords)
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        # 返回与已选文档的平均相似度的补数
        avg_similarity = np.mean(similarities) if similarities else 0
        return 1 - avg_similarity
    
    def rerank_by_temporal_relevance(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        time_weight: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        基于时间相关性的重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            time_weight: 时间权重(0-1)
            
        Returns:
            重排序后的文档列表
        """
        import re
        from datetime import datetime
        
        # 检查查询中是否包含时间信息
        has_time_query = bool(re.search(r'\d{4}年|\d{4}-\d{2}|去年|今年|最近', query))
        
        if not has_time_query:
            return documents  # 查询无时间信息,不需要时间重排序
        
        try:
            current_time = datetime.now().timestamp()
            
            for doc in documents:
                base_score = doc.get("rrf_score", 0) or doc.get("rerank_score", 0)
                
                # 计算时间分数
                publish_time = doc.get("publish_time", 0)
                if publish_time > 0:
                    # 归一化时间分数(越新分数越高)
                    time_diff = abs(current_time - publish_time)
                    max_time_diff = 365 * 24 * 3600  # 1年
                    time_score = max(0, 1 - time_diff / max_time_diff)
                else:
                    time_score = 0.5  # 无时间信息给中等分数
                
                # 综合分数
                doc["temporal_score"] = (1 - time_weight) * base_score + time_weight * time_score
            
            # 按时间加权分数排序
            reranked = sorted(
                documents,
                key=lambda x: x.get("temporal_score", 0),
                reverse=True
            )
            
            logger.info("时间相关性重排序完成")
            return reranked
            
        except Exception as e:
            logger.error(f"时间重排序失败: {e}")
            return documents
    
    def rerank_comprehensive(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        use_cross_encoder: bool = True,
        use_diversity: bool = True,
        use_temporal: bool = True
    ) -> List[Dict[str, Any]]:
        """
        综合重排序策略
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top-k文档
            use_cross_encoder: 是否使用CrossEncoder
            use_diversity: 是否使用多样性
            use_temporal: 是否使用时间相关性
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        logger.info(f"开始综合重排序: {len(documents)} 个文档")
        
        # 第1步: CrossEncoder重排序
        if use_cross_encoder and self.reranker:
            documents = self.rerank_by_cross_encoder(query, documents, top_k=min(50, len(documents)))
        
        # 第2步: 时间相关性重排序
        if use_temporal:
            documents = self.rerank_by_temporal_relevance(query, documents)
        
        # 第3步: 多样性重排序
        if use_diversity:
            documents = self.rerank_by_diversity(documents, top_k=top_k)
        
        logger.info(f"综合重排序完成: 返回 {len(documents[:top_k])} 个文档")
        return documents[:top_k]


# 全局单例
_reranker_instance = None

def get_reranker(model_name: str = "BAAI/bge-reranker-large") -> AdvancedReranker:
    """获取重排序器单例"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = AdvancedReranker(model_name)
    return _reranker_instance