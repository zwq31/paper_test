"""
数据处理模块 - 严格按照论文方法处理数据
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from sklearn.model_selection import train_test_split
import pickle
import os
import logging

from configs.config import data_config, training_config

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理类 - 论文方法实现"""
    
    @staticmethod
    def load_paper_dataset(faithful_path: str = None, hallucination_path: str = None, 
                          max_samples: int = None) -> List[Dict]:
        """
        加载论文使用的数据集
        假设数据已经是多次采样筛选过的
        
        参数:
            faithful_path: 忠实数据路径
            hallucination_path: 幻觉数据路径
            max_samples: 每类最大样本数
        
        返回:
            List[Dict]: 合并后的数据列表
        """
        if faithful_path is None:
            faithful_path = data_config.faithful_data_path
        if hallucination_path is None:
            hallucination_path = data_config.hallucination_data_path
        
        logger.info(f"加载论文数据集: {faithful_path}, {hallucination_path}")
        
        # 加载忠实数据
        with open(faithful_path, 'r', encoding='utf-8') as f:
            faithful_data = json.load(f)
        
        # 加载幻觉数据
        with open(hallucination_path, 'r', encoding='utf-8') as f:
            hallucination_data = json.load(f)
        
        logger.info(f"忠实样本数: {len(faithful_data)}")
        logger.info(f"幻觉样本数: {len(hallucination_data)}")
        
        # 为数据添加标签
        for item in faithful_data:
            item['label'] = 0  # 0表示忠实（论文中faithful）
            # 确保有standard_answer字段
            if 'standard_answer' not in item:
                item['standard_answer'] = item.get('answer', '')
        
        for item in hallucination_data:
            item['label'] = 1  # 1表示幻觉（论文中hallucinatory）
            if 'standard_answer' not in item:
                item['standard_answer'] = item.get('answer', '')
        
        # 限制样本数量
        if max_samples is None:
            max_samples = data_config.max_samples_per_class
        
        if max_samples:
            faithful_data = faithful_data[:max_samples]
            hallucination_data = hallucination_data[:max_samples]
            logger.info(f"每类限制为 {max_samples} 个样本")
        
        # 合并数据
        all_data = faithful_data + hallucination_data
        
        # 打乱数据
        random.seed(training_config.random_seed)
        random.shuffle(all_data)
        
        logger.info(f"总样本数: {len(all_data)} (忠实: {len(faithful_data)}, 幻觉: {len(hallucination_data)})")
        
        # 验证数据格式
        sample_keys = ['question', 'model_answer', 'standard_answer', 'label']
        for i, sample in enumerate(all_data[:5]):
            for key in sample_keys:
                if key not in sample:
                    logger.warning(f"样本 {i} 缺少字段: {key}")
        
        return all_data
    
    @staticmethod
    def prepare_paper_training_data(features: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """
        按照论文方法准备训练数据
        
        论文方法:
        - 正样本(y=1): 幻觉样本的答案词元特征
        - 负样本(y=0): 1) 忠实样本的答案词元特征 2) 所有样本的非答案词元特征
        
        参数:
            features: 提取的特征列表
        
        返回:
            X: 特征矩阵
            y: 标签数组
            neuron_names: 神经元名称列表
            metadata: 样本元数据
        """
        logger.info("按照论文方法准备训练数据...")
        
        if not features:
            raise ValueError("特征列表为空")
        
        # 获取所有神经元名称（确保顺序一致）
        neuron_names = sorted(list(features[0]['features_answer'].keys()))
        logger.info(f"神经元总数: {len(neuron_names)}")
        
        X = []
        y = []
        metadata = []  # 跟踪每个训练样本的来源
        
        for feat_idx, feat in enumerate(features):
            label = feat.get('label', 0)
            
            # 构建答案词元特征向量
            answer_feat_vector = np.zeros(len(neuron_names))
            for i, neuron in enumerate(neuron_names):
                answer_feat_vector[i] = feat['features_answer'].get(neuron, 0.0)
            
            # 构建非答案词元特征向量
            other_feat_vector = np.zeros(len(neuron_names))
            for i, neuron in enumerate(neuron_names):
                other_feat_vector[i] = feat['features_other'].get(neuron, 0.0)
            
            # 按照论文的标签策略
            if label == 1:  # 幻觉样本
                # 答案词元特征作为正样本
                X.append(answer_feat_vector)
                y.append(1)
                metadata.append({
                    'type': 'hallucination_answer',
                    'original_idx': feat_idx,
                    'question': feat.get('question', '')[:50]
                })
                
                # 非答案词元特征作为负样本
                X.append(other_feat_vector)
                y.append(0)
                metadata.append({
                    'type': 'hallucination_other',
                    'original_idx': feat_idx,
                    'question': feat.get('question', '')[:50]
                })
            
            else:  # 忠实样本 (label == 0)
                # 答案词元特征作为负样本
                X.append(answer_feat_vector)
                y.append(0)
                metadata.append({
                    'type': 'faithful_answer',
                    'original_idx': feat_idx,
                    'question': feat.get('question', '')[:50]
                })
                
                # 非答案词元特征作为负样本
                X.append(other_feat_vector)
                y.append(0)
                metadata.append({
                    'type': 'faithful_other',
                    'original_idx': feat_idx,
                    'question': feat.get('question', '')[:50]
                })
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"训练数据形状: X={X.shape}, y={y.shape}")
        
        # 统计类别分布
        positive_count = sum(y == 1)
        negative_count = sum(y == 0)
        logger.info(f"类别分布: 正样本={positive_count}, 负样本={negative_count}")
        logger.info(f"正样本比例: {positive_count/len(y):.2%}")
        
        # 统计来源
        hallucination_answer = sum(1 for m in metadata if m['type'] == 'hallucination_answer')
        hallucination_other = sum(1 for m in metadata if m['type'] == 'hallucination_other')
        faithful_answer = sum(1 for m in metadata if m['type'] == 'faithful_answer')
        faithful_other = sum(1 for m in metadata if m['type'] == 'faithful_other')
        
        logger.info(f"样本来源分布:")
        logger.info(f"  幻觉-答案: {hallucination_answer}")
        logger.info(f"  幻觉-其他: {hallucination_other}")
        logger.info(f"  忠实-答案: {faithful_answer}")
        logger.info(f"  忠实-其他: {faithful_other}")
        
        return X, y, neuron_names, metadata
    
    @staticmethod
    def split_data_paper_method(X: np.ndarray, y: np.ndarray, metadata: List[Dict], 
                               test_size: float = None) -> Tuple:
        """
        按照论文方法划分训练集和测试集
        确保同一个原始样本的不同特征不会同时出现在训练集和测试集
        """
        if test_size is None:
            test_size = 1 - data_config.train_ratio
        
        # 获取原始样本ID
        original_indices = [m['original_idx'] for m in metadata]
        
        # 获取唯一的原始样本
        unique_original_indices = list(set(original_indices))
        
        # 划分原始样本
        train_original, test_original = train_test_split(
            unique_original_indices, 
            test_size=test_size, 
            random_state=training_config.random_seed
        )
        
        # 根据原始样本划分特征
        train_indices = [i for i, idx in enumerate(original_indices) if idx in train_original]
        test_indices = [i for i, idx in enumerate(original_indices) if idx in test_original]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        train_metadata = [metadata[i] for i in train_indices]
        test_metadata = [metadata[i] for i in test_indices]
        
        logger.info(f"训练集: {X_train.shape[0]} 样本 ({len(train_original)} 个原始样本)")
        logger.info(f"测试集: {X_test.shape[0]} 样本 ({len(test_original)} 个原始样本)")
        
        return X_train, X_test, y_train, y_test, train_metadata, test_metadata
    
    @staticmethod
    def load_features(features_path: str) -> List[Dict]:
        """加载已提取的特征"""
        logger.info(f"加载特征文件: {features_path}")
        
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        
        logger.info(f"加载 {len(features)} 个特征")
        return features
    
    @staticmethod
    def save_features(features: List[Dict], features_path: str):
        """保存特征到文件"""
        logger.info(f"保存特征到: {features_path}")
        
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
    
    @staticmethod
    def analyze_features(features: List[Dict]):
        """分析特征统计信息"""
        if not features:
            logger.warning("没有特征可分析")
            return None
        
        # 收集所有神经元的激活值
        answer_activations = []
        other_activations = []
        
        for feat in features:
            answer_activations.extend(list(feat['features_answer'].values()))
            other_activations.extend(list(feat['features_other'].values()))
        
        # 计算统计信息
        stats = {
            'total_samples': len(features),
            'total_neurons': len(features[0]['features_answer']) if features else 0,
            'answer_mean': float(np.mean(answer_activations)),
            'answer_std': float(np.std(answer_activations)),
            'other_mean': float(np.mean(other_activations)),
            'other_std': float(np.std(other_activations)),
            'answer_min': float(np.min(answer_activations)),
            'answer_max': float(np.max(answer_activations)),
            'other_min': float(np.min(other_activations)),
            'other_max': float(np.max(other_activations)),
            'answer_median': float(np.median(answer_activations)),
            'other_median': float(np.median(other_activations))
        }
        
        logger.info("CETT特征统计信息:")
        logger.info(f"  总样本数: {stats['total_samples']}")
        logger.info(f"  神经元总数: {stats['total_neurons']}")
        logger.info(f"  答案词元CETT均值: {stats['answer_mean']:.6f} ± {stats['answer_std']:.6f}")
        logger.info(f"  非答案词元CETT均值: {stats['other_mean']:.6f} ± {stats['other_std']:.6f}")
        logger.info(f"  答案词元CETT范围: [{stats['answer_min']:.6f}, {stats['answer_max']:.6f}]")
        logger.info(f"  非答案词元CETT范围: [{stats['other_min']:.6f}, {stats['other_max']:.6f}]")
        
        return stats
    
    @staticmethod
    def filter_low_activation_features(features: List[Dict], threshold: float = 1e-6) -> List[Dict]:
        """过滤掉激活度过低的特征"""
        filtered = []
        
        for feat in features:
            # 检查答案词元特征的平均激活度
            answer_values = list(feat['features_answer'].values())
            avg_activation = np.mean(answer_values) if answer_values else 0.0
            
            if avg_activation >= threshold:
                filtered.append(feat)
            else:
                logger.debug(f"过滤低激活样本: {feat.get('question', '')[:50]}..., 平均激活度: {avg_activation:.6f}")
        
        logger.info(f"过滤特征: {len(features)} -> {len(filtered)} (阈值: {threshold})")
        
        return filtered