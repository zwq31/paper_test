# keywords_extractor.py
import os
import re
import jieba
import jieba.posseg as pseg
import jieba.analyse
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class NewsKeywordExtractor:
    """专门针对新闻优化的关键词提取器 - 增加关键词数量"""
    
    def __init__(self, stopwords):
        self.stopwords = stopwords
        
    def extract_named_entities(self, text):
        """提取命名实体（地名、人名、机构名等）"""
        entities = []
        words = pseg.cut(text)
        
        for word, flag in words:
            if len(word) <= 1:
                continue
            if word in self.stopwords:
                continue
            
            # 识别命名实体 - 新闻中重要的实体类型
            if flag in ['nr', 'ns', 'nt', 'nz']:  # 人名、地名、机构名、其他专名
                entities.append((word, 3.0))  # 实体权重更高
            elif flag.startswith('n'):  # 普通名词
                entities.append((word, 1.5))
            elif flag.startswith('v'):  # 动词
                entities.append((word, 1.2))
            elif flag.startswith('a'):  # 形容词
                entities.append((word, 1.0))
        
        return entities
    
    def extract_ngram_phrases(self, text, n=2):
        """提取N-gram短语，保留重要短语"""
        words = jieba.lcut(text)
        phrases = []
        
        for i in range(len(words) - n + 1):
            phrase = ''.join(words[i:i+n])
            # 过滤条件（放松限制）
            if len(phrase) <= 1:
                continue
            if any(w in self.stopwords for w in words[i:i+n]):
                continue
            if re.fullmatch(r'[^\w\u4e00-\u9fff]+', phrase):
                continue
            if phrase.isdigit():
                continue
            
            # 检查短语中是否包含重要成分（名词或动词）
            pos_tags = []
            for w in words[i:i+n]:
                for flag in pseg.cut(w):
                    pos_tags.append(flag.flag)
            
            # 放宽条件：只要有意义的内容
            if any(tag.startswith('n') or tag.startswith('v') or tag.startswith('a') for tag in pos_tags):
                phrases.append(phrase)
        
        return phrases
    
    def extract_by_textrank(self, text, top_k=60):
        """使用TextRank算法 - 增加关键词数量"""
        try:
            # 设置允许的词性，专注新闻相关内容
            allow_pos = ('n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v', 'a', 'ad')
            keywords = jieba.analyse.textrank(
                text, 
                topK=top_k,
                withWeight=True,
                allowPOS=allow_pos
            )
            return keywords
        except:
            return []
    
    def extract_by_tfidf(self, title, content, top_k=60):
        """使用TF-IDF，标题权重更高 - 增加关键词数量"""
        try:
            # 标题重复3次以增加权重
            weighted_text = (title + " ") * 3 + content
            
            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: [w for w in jieba.lcut(x) 
                                   if w not in self.stopwords and len(w) > 0],
                ngram_range=(1, 3),  # 1-3元短语，增加3元短语
                max_features=300  # 增加特征数量
            )
            
            tfidf_matrix = vectorizer.fit_transform([weighted_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # 获取top_k关键词
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        except:
            return []
    
    def extract_keywords(self, text, top_k=20):  # 针对单个文本的提取
        """主提取函数 - 优化集成方法，增加关键词数量"""
        # 将文本同时作为标题和内容
        title = text[:100]  # 取前100字作为"标题"
        content = text
        
        # 短文本处理
        if len(text) < 100:
            return self.extract_for_short_text(text, top_k)
        
        # 1. 提取命名实体和词性关键词
        pos_keywords = self.extract_named_entities(text)
        
        # 2. TextRank关键词
        textrank_kws = self.extract_by_textrank(text, top_k=top_k*2)
        
        # 3. TF-IDF关键词（加强标题）
        tfidf_kws = self.extract_by_tfidf(title, content, top_k=top_k*2)
        
        # 4. 提取短语（2-3元）
        phrases_2 = self.extract_ngram_phrases(text, n=2)
        phrases_3 = self.extract_ngram_phrases(text, n=3)
        phrase_counter = Counter(phrases_2 + phrases_3)
        
        # 5. 合并和加权评分
        keyword_scores = {}
        
        # 词性权重：实体权重最高
        for word, weight in pos_keywords:
            keyword_scores[word] = keyword_scores.get(word, 0) + weight * 2.0
        
        # TextRank权重
        for word, score in textrank_kws:
            keyword_scores[word] = keyword_scores.get(word, 0) + score * 1.5
        
        # TF-IDF权重
        for word, score in tfidf_kws:
            keyword_scores[word] = keyword_scores.get(word, 0) + score * 1.2
        
        # 短语权重
        for phrase, freq in phrase_counter.most_common(top_k):
            weight = freq * 0.9  # 稍微提高权重
            if phrase in keyword_scores:
                keyword_scores[phrase] += weight
            else:
                keyword_scores[phrase] = weight
        
        # 6. 标题词汇额外加分
        title_words = [w for w in jieba.lcut(title) if len(w) > 0 and w not in self.stopwords]
        for word in title_words:
            if word in keyword_scores:
                keyword_scores[word] *= 1.4  # 标题词重要性提升40%
        
        # 7. 排序和过滤
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_keywords = []
        seen_keywords = set()
        
        for word, score in sorted_keywords:
            # 放松过滤条件
            if len(word) <= 1 and word not in ['AI', 'VR', '5G']:  # 保留一些常用缩写
                continue
            if word in self.stopwords:
                continue
            if word.isdigit() and len(word) < 3:
                continue
            if re.fullmatch(r'[^\w\u4e00-\u9fff]+', word):
                continue
            if word in seen_keywords:
                continue
            
            # 长度限制
            if len(word) > 30:
                continue
                
            seen_keywords.add(word)
            final_keywords.append(word)
            
            if len(final_keywords) >= top_k:
                break
        
        return final_keywords
    
    def extract_for_short_text(self, text, top_k=20):
        """短文本专用提取"""
        try:
            allow_pos = ('n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v', 'a', 'ad', 'an')
            keywords = jieba.analyse.textrank(
                text, 
                topK=top_k,
                withWeight=False,
                allowPOS=allow_pos
            )
            
            # 如果TextRank提取不够，补充词频统计
            if len(keywords) < top_k:
                words = [w for w in jieba.lcut(text) 
                        if w not in self.stopwords and len(w) > 0]
                word_freq = Counter(words)
                additional_keywords = [word for word, _ in word_freq.most_common(top_k * 2)]
                # 合并并去重
                for word in additional_keywords:
                    if word not in keywords:
                        keywords.append(word)
                    if len(keywords) >= top_k:
                        break
            
            return keywords[:top_k]
        except:
            # 降级方案：简单的词频统计
            words = [w for w in jieba.lcut(text) 
                    if w not in self.stopwords and len(w) > 0]
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(top_k)]


def load_stopwords(stopwords_path: str) -> set:
    """加载停用词"""
    stopwords = set()
    
    # 基础停用词
    base_stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', '那', '但', '中', '从', '对', '而', '且', '或', '之', '为', '以',
        '及', '将', '并', '等', '个', '年', '月', '日', '时', '与', '于', '就', '又',
        '还', '没', '不', '都', '好', '很', '得', '多', '最', '更', '不', '没', '没'
    }
    
    if os.path.exists(stopwords_path):
        with open(stopwords_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and len(word) > 1:
                    stopwords.add(word)
        stopwords.update(base_stopwords)
    else:
        stopwords = base_stopwords
    
    return stopwords


def extract_keywords_from_question(question: str, keyword_extractor: NewsKeywordExtractor) -> list:
    """
    从问题中提取关键词 - 使用高级关键词提取器
    
    Args:
        question: 问题文本
        keyword_extractor: 关键词提取器实例
        
    Returns:
        关键词列表
    """
    return keyword_extractor.extract_keywords(question, top_k=15)