"""
H-Neurons完整训练管道 - 梯度提升版本（修复报告生成错误）
完全按照参考代码的特征筛选方式（VarianceThreshold + SelectPercentile 10%）
第二阶段使用LightGBM进行训练
修复报告生成中的数组真值判断错误
"""

import os
import json
import pickle
import logging
import time
import gc
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from tqdm import tqdm

# ==================== 配置 ====================
@dataclass
class Config:
    # 数据路径
    features_file: str = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
    output_dir: str = "/mnt/project/H-Neurous/output/gradient_boosting_reference"
    
    # 训练参数
    test_size: float = 0.15
    random_seed: int = 42
    
    # 特征选择参数（完全按照参考代码）
    feature_selection_percentile: int = 10  # 保留前10%特征
    feature_selection_method: str = "f_classif"  # 与参考代码一致
    
    # LightGBM参数
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1正则化
    reg_lambda: float = 0.1  # L2正则化
    
    # 早停参数
    early_stopping_rounds: int = 50
    
    # 日志和实验
    experiment_name: str = "h_neurons_gradient_boosting_reference"
    
    def __post_init__(self):
        # 创建目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)

# ==================== 日志配置 ====================
class LoggerSetup:
    @staticmethod
    def setup():
        log_dir = "/mnt/project/H-Neurous/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"h_neurons_gradient_boosting_reference_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

# ==================== 流式特征加载器（完全按照参考代码） ====================
class StreamingFeatureLoader:
    """流式加载CETT特征文件 - 完全按照参考代码"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_features_streaming(self, features_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        流式加载特征文件并构建训练数据 - 完全按照参考代码
        """
        self.logger.info(f"流式加载特征文件: {features_file}")
        
        # 获取文件信息
        file_size = os.path.getsize(features_file)
        self.logger.info(f"特征文件大小: {file_size / (1024**3):.2f} GB")
        
        # 1. 首先读取第一行获取特征维度
        feature_dim = None
        neuron_names = None
        
        with open(features_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                first_sample = json.loads(first_line)
                if 'features_answer' in first_sample:
                    feature_dim = len(first_sample['features_answer'])
        
        if feature_dim is None:
            raise ValueError("特征文件为空或格式错误")
        
        self.logger.info(f"特征维度: {feature_dim}")
        
        # 2. 基于特征维度生成神经元名称（完全按照参考代码）
        num_layers = 28
        neurons_per_layer = feature_dim // num_layers
        
        if feature_dim % num_layers == 0:
            neuron_names = [f"layer_{l}_neuron_{n}" 
                          for l in range(num_layers) 
                          for n in range(neurons_per_layer)]
        else:
            # 如果不匹配，使用通用命名
            neuron_names = [f"neuron_{i}" for i in range(feature_dim)]
        
        # 3. 流式处理文件
        X_list = []
        y_list = []
        
        # 统计
        stats = {
            'pos_hallucination_answer': 0,
            'neg_faithful_answer': 0,
            'neg_hallucination_other': 0,
            'neg_faithful_other': 0
        }
        
        # 统计总行数
        total_lines = 0
        with open(features_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        self.logger.info(f"总样本数: {total_lines}")
        
        # 重新打开文件进行流式处理
        with open(features_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="流式处理")):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    label = sample['label']
                    
                    # 处理答案令牌特征
                    if 'features_answer' in sample:
                        feat_answer_list = sample['features_answer']
                        x_vector = np.array(feat_answer_list, dtype=np.float32)
                        
                        # 检查并清理特征
                        if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
                            x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 按照论文标记策略
                        if label == 1:  # 幻觉样本的答案令牌 -> 正样本
                            X_list.append(x_vector)
                            y_list.append(1)
                            stats['pos_hallucination_answer'] += 1
                        else:  # 忠实样本的答案令牌 -> 负样本
                            X_list.append(x_vector)
                            y_list.append(0)
                            stats['neg_faithful_answer'] += 1
                    
                    # 处理非答案令牌特征
                    if 'features_other' in sample:
                        feat_other_list = sample['features_other']
                        x_vector = np.array(feat_other_list, dtype=np.float32)
                        
                        # 检查并清理特征
                        if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
                            x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 所有非答案令牌都是负样本
                        X_list.append(x_vector)
                        y_list.append(0)
                        
                        if label == 1:
                            stats['neg_hallucination_other'] += 1
                        else:
                            stats['neg_faithful_other'] += 1
                    
                    # 定期清理内存
                    if (line_num + 1) % 100 == 0:
                        gc.collect()
                        
                except Exception as e:
                    self.logger.warning(f"行 {line_num} 处理错误: {e}")
                    continue
        
        # 转换为numpy数组
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # 最终清理
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.logger.info(f"论文标记策略统计:")
        self.logger.info(f"  正样本(幻觉答案令牌): {stats['pos_hallucination_answer']}")
        self.logger.info(f"  负样本(忠实答案令牌): {stats['neg_faithful_answer']}")
        self.logger.info(f"  负样本(幻觉非答案令牌): {stats['neg_hallucination_other']}")
        self.logger.info(f"  负样本(忠实非答案令牌): {stats['neg_faithful_other']}")
        self.logger.info(f"  总训练样本数: {len(X)}")
        self.logger.info(f"  正样本比例: {np.mean(y):.3f}")
        self.logger.info(f"  特征矩阵形状: {X.shape}")
        
        return X, y, neuron_names

# ==================== 梯度提升分类器 ====================
class GradientBoostingHNeuronsClassifier:
    """基于梯度提升决策树的H-Neurons分类器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型组件（完全按照参考代码的特征筛选）
        self.model = None
        self.var_selector = VarianceThreshold(threshold=0)  # 与参考代码完全一致
        self.percentile_selector = SelectPercentile(f_classif, percentile=config.feature_selection_percentile)  # 完全一致
        self.scaler = StandardScaler()
        
        # H-Neurons信息
        self.neuron_names = None
        self.h_neuron_indices = None
        self.h_neuron_names = None
        self.feature_importance = None
        
        # 训练历史
        self.evals_result = {}
        self.best_iteration = 0
        
    def apply_reference_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用参考代码的特征筛选方法"""
        self.logger.info("应用参考代码的特征筛选...")
        
        # 阶段1: 剔除常数特征（与参考代码完全一致）
        self.logger.info("剔除死神经元...")
        X_var_selected = self.var_selector.fit_transform(X)
        var_selected_indices = np.where(self.var_selector.get_support())[0]
        self.logger.info(f"方差筛选后特征数: {X_var_selected.shape[1]}")
        
        # 阶段2: 统计学特征筛选（与参考代码完全一致）
        self.logger.info("进行统计学特征筛选...")
        X_percentile_selected = self.percentile_selector.fit_transform(X_var_selected, y)
        percentile_selected_indices = np.where(self.percentile_selector.get_support())[0]
        
        # 映射回原始索引
        selected_indices = var_selected_indices[percentile_selected_indices]
        
        self.logger.info(f"筛选后特征数: {X_percentile_selected.shape[1]}")
        self.logger.info(f"原始特征数: {X.shape[1]}")
        self.logger.info(f"筛选比例: {X_percentile_selected.shape[1] / X.shape[1] * 100:.2f}%")
        
        return X_percentile_selected, selected_indices
    
    def train_with_cv(self, X: np.ndarray, y: np.ndarray, neuron_names: List[str]) -> 'GradientBoostingHNeuronsClassifier':
        """使用交叉验证训练梯度提升模型"""
        self.logger.info("开始训练梯度提升分类器...")
        self.neuron_names = neuron_names
        
        # 1. 应用参考代码的特征筛选
        X_selected, selected_indices = self.apply_reference_feature_selection(X, y)
        
        # 2. 标准化
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 3. 划分训练验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=self.config.random_seed
        )
        
        self.logger.info(f"训练集: {X_train.shape[0]} 样本")
        self.logger.info(f"验证集: {X_val.shape[0]} 样本")
        
        # 4. 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 5. 设置LightGBM参数
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'n_estimators': self.config.n_estimators,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,  # L1正则化
            'reg_lambda': self.config.reg_lambda,  # L2正则化
            'min_child_samples': self.config.min_child_samples,
            'random_state': self.config.random_seed,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.logger.info("LightGBM参数:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        
        # 6. 训练模型
        self.logger.info("开始训练LightGBM模型...")
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=self.config.n_estimators,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(50),
                lgb.record_evaluation(self.evals_result)
            ]
        )
        
        self.best_iteration = self.model.best_iteration
        self.logger.info(f"训练完成，最佳迭代次数: {self.best_iteration}")
        
        # 7. 获取特征重要性
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        # 归一化重要性
        if self.feature_importance.sum() > 0:
            self.feature_importance = self.feature_importance / self.feature_importance.sum()
        
        self.logger.info(f"特征重要性统计:")
        self.logger.info(f"  最大重要性: {self.feature_importance.max():.4f}")
        self.logger.info(f"  最小重要性: {self.feature_importance.min():.4f}")
        self.logger.info(f"  平均重要性: {self.feature_importance.mean():.4f}")
        self.logger.info(f"  零重要性特征数: {np.sum(self.feature_importance == 0)}")
        
        # 8. 识别H-Neurons（重要性高的特征）
        importance_threshold = np.percentile(self.feature_importance, 90)  # 前10%
        h_neuron_mask = self.feature_importance > importance_threshold
        h_indices_selected = np.where(h_neuron_mask)[0]
        
        # 映射回原始索引
        self.h_neuron_indices = selected_indices[h_indices_selected]
        self.h_neuron_names = [neuron_names[i] for i in self.h_neuron_indices]
        
        self.logger.info(f"识别出 {len(self.h_neuron_indices)} 个 H-Neurons")
        self.logger.info(f"占总神经元的 {len(self.h_neuron_indices)/len(self.neuron_names)*100:.4f}%")
        
        return self
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型性能"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 应用相同的特征筛选
        X_test_var_selected = self.var_selector.transform(X_test)
        X_test_selected = self.percentile_selector.transform(X_test_var_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # 预测
        y_prob = self.model.predict(X_test_scaled)
        y_pred = (y_prob > 0.5).astype(int)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # 详细分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        return {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'classification_report': report,
            'confusion_matrix': cm_dict,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist()
            }
        }
    
    def analyze_feature_importance(self) -> Dict:
        """分析特征重要性"""
        if self.feature_importance is None:
            return {}
        
        # 按重要性排序
        sorted_indices = np.argsort(self.feature_importance)[::-1]
        sorted_importance = self.feature_importance[sorted_indices]
        
        # 统计信息
        total_importance = sorted_importance.sum()
        cumulative_importance = np.cumsum(sorted_importance)
        
        # 找到达到不同累积重要性所需的特征数
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        threshold_stats = {}
        
        for threshold in thresholds:
            n_features = np.argmax(cumulative_importance >= threshold) + 1
            threshold_stats[f'cumulative_{int(threshold*100)}%'] = {
                'n_features': int(n_features),
                'percentage': n_features / len(sorted_importance) * 100
            }
        
        # 按层统计重要性
        layer_importance = {}
        if self.h_neuron_names:
            for i, name in enumerate(self.h_neuron_names):
                if name.startswith('layer_'):
                    try:
                        parts = name.split('_')
                        layer_num = int(parts[1])
                        if layer_num not in layer_importance:
                            layer_importance[layer_num] = 0
                        
                        # 获取该特征的重要性
                        if i < len(self.feature_importance):
                            layer_importance[layer_num] += self.feature_importance[i]
                    except:
                        continue
        
        # 总体统计
        analysis = {
            'total_features': len(self.feature_importance),
            'total_importance': float(total_importance),
            'max_importance': float(sorted_importance.max()),
            'min_importance': float(sorted_importance.min()),
            'mean_importance': float(sorted_importance.mean()),
            'median_importance': float(np.median(sorted_importance)),
            'zero_importance_features': int(np.sum(self.feature_importance == 0)),
            'threshold_statistics': threshold_stats,
            'layer_importance_distribution': layer_importance,
            'top_features': []
        }
        
        # 保存最重要的特征
        top_n = min(100, len(sorted_indices))
        for i in range(top_n):
            idx = sorted_indices[i]
            importance = sorted_importance[i]
            
            # 映射回原始名称
            if idx < len(self.h_neuron_indices):
                original_idx = self.h_neuron_indices[idx]
                name = self.neuron_names[original_idx]
                analysis['top_features'].append({
                    'rank': i + 1,
                    'name': name,
                    'importance': float(importance),
                    'cumulative_importance': float(cumulative_importance[i])
                })
        
        self.logger.info("特征重要性分析:")
        self.logger.info(f"  总特征数: {analysis['total_features']}")
        self.logger.info(f"  零重要性特征: {analysis['zero_importance_features']}")
        self.logger.info(f"  最大重要性: {analysis['max_importance']:.4f}")
        
        for threshold, stats in threshold_stats.items():
            self.logger.info(f"  {threshold}: {stats['n_features']}个特征 ({stats['percentage']:.1f}%)")
        
        return analysis
    
    def analyze_h_neurons(self) -> Dict:
        """分析H-Neurons的统计特性"""
        if self.h_neuron_indices is None or len(self.h_neuron_indices) == 0:
            self.logger.warning("未发现H-Neurons")
            return {}
        
        # 按层统计
        layer_dist = {}
        for name in self.h_neuron_names:
            if name.startswith('layer_'):
                try:
                    parts = name.split('_')
                    layer_num = int(parts[1])
                    layer_dist[layer_num] = layer_dist.get(layer_num, 0) + 1
                except:
                    continue
        
        # 按重要性统计
        if self.feature_importance is not None and len(self.feature_importance) == len(self.h_neuron_indices):
            h_neurons_importance = self.feature_importance
            
            importance_stats = {
                'min': float(np.min(h_neurons_importance)),
                'max': float(np.max(h_neurons_importance)),
                'mean': float(np.mean(h_neurons_importance)),
                'std': float(np.std(h_neurons_importance)),
                'median': float(np.median(h_neurons_importance))
            }
        else:
            importance_stats = {}
        
        # 总体统计
        stats = {
            'total_neurons': len(self.neuron_names),
            'h_neuron_count': len(self.h_neuron_indices),
            'h_neuron_ratio': len(self.h_neuron_indices) / len(self.neuron_names),
            'layer_distribution': layer_dist,
            'importance_statistics': importance_stats,
            'lightgbm_params': {
                'n_estimators': self.config.n_estimators,
                'learning_rate': self.config.learning_rate,
                'max_depth': self.config.max_depth,
                'best_iteration': self.best_iteration
            },
            'feature_selection': {
                'variance_threshold': 0,
                'percentile': self.config.feature_selection_percentile,
                'method': self.config.feature_selection_method
            }
        }
        
        # 打印统计
        self.logger.info("H-Neurons统计分析:")
        self.logger.info(f"  总数: {stats['h_neuron_count']}")
        self.logger.info(f"  比例: {stats['h_neuron_ratio']*100:.4f}%")
        self.logger.info(f"  是否<0.1%: {'✓ 是' if stats['h_neuron_ratio'] < 0.001 else '⚠ 否'}")
        
        if layer_dist:
            self.logger.info("  按层分布:")
            for layer in sorted(layer_dist.keys()):
                self.logger.info(f"    层{layer}: {layer_dist[layer]}个神经元")
        
        return stats
    
    def save(self, output_dir: str) -> str:
        """保存模型和结果"""
        self.logger.info(f"保存模型到: {output_dir}")
        
        # 保存完整模型
        model_data = {
            'model': self.model,
            'var_selector': self.var_selector,
            'percentile_selector': self.percentile_selector,
            'scaler': self.scaler,
            'neuron_names': self.neuron_names,
            'h_neuron_indices': self.h_neuron_indices,
            'h_neuron_names': self.h_neuron_names,
            'feature_importance': self.feature_importance,
            'evals_result': self.evals_result,
            'best_iteration': self.best_iteration,
            'config': self.config
        }
        
        model_path = os.path.join(output_dir, "gradient_boosting_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 保存H-Neurons列表
        h_neurons_list = []
        if self.feature_importance is not None and self.h_neuron_names:
            for i, name in enumerate(self.h_neuron_names):
                importance = self.feature_importance[i] if i < len(self.feature_importance) else 0
                h_neurons_list.append({
                    'index': int(self.h_neuron_indices[i]),
                    'name': name,
                    'importance': float(importance)
                })
        
        # 按重要性排序
        h_neurons_list.sort(key=lambda x: x['importance'], reverse=True)
        
        json_path = os.path.join(output_dir, "h_neurons_gradient_boosting.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(h_neurons_list, f, indent=2, ensure_ascii=False)
        
        # 保存特征重要性分析
        importance_analysis = self.analyze_feature_importance()
        importance_path = os.path.join(output_dir, "feature_importance_analysis.json")
        with open(importance_path, 'w', encoding='utf-8') as f:
            json.dump(importance_analysis, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats = self.analyze_h_neurons()
        stats_path = os.path.join(output_dir, "h_neurons_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 保存LightGBM模型（二进制格式）
        lgb_model_path = os.path.join(output_dir, "lightgbm_model.txt")
        self.model.save_model(lgb_model_path)
        
        self.logger.info(f"模型保存完成: {model_path}")
        self.logger.info(f"H-Neurons列表: {json_path}")
        self.logger.info(f"特征重要性分析: {importance_path}")
        self.logger.info(f"LightGBM原生模型: {lgb_model_path}")
        
        return output_dir

# ==================== 梯度提升训练器 ====================
class GradientBoostingTrainer:
    """梯度提升训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件
        self.feature_loader = StreamingFeatureLoader(config)
        self.classifier = GradientBoostingHNeuronsClassifier(config)
    
    def run(self) -> Tuple[GradientBoostingHNeuronsClassifier, str]:
        """运行训练流程"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons训练管道 - 梯度提升版本（参考代码特征筛选）")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 阶段1: 流式加载特征
            self.logger.info("\n[阶段1/4] 流式加载特征")
            
            if not os.path.exists(self.config.features_file):
                raise FileNotFoundError(f"特征文件不存在: {self.config.features_file}")
            
            X, y, neuron_names = self.feature_loader.load_features_streaming(self.config.features_file)
            
            # 阶段2: 划分数据集
            self.logger.info("\n[阶段2/4] 划分数据集")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                stratify=y,
                random_state=self.config.random_seed
            )
            
            self.logger.info(f"训练集: {X_train.shape[0]} 样本")
            self.logger.info(f"测试集: {X_test.shape[0]} 样本")
            self.logger.info(f"特征维度: {X_train.shape[1]}")
            self.logger.info(f"正样本比例 - 训练集: {np.mean(y_train):.3f}, 测试集: {np.mean(y_test):.3f}")
            
            # 阶段3: 训练梯度提升分类器
            self.logger.info("\n[阶段3/4] 训练梯度提升分类器")
            self.classifier.train_with_cv(X_train, y_train, neuron_names)
            
            # 阶段4: 评估模型
            self.logger.info("\n[阶段4/4] 评估模型性能")
            eval_results = self.classifier.evaluate(X_test, y_test)
            
            # 阶段5: 分析结果
            importance_analysis = self.classifier.analyze_feature_importance()
            h_stats = self.classifier.analyze_h_neurons()
            
            # 阶段6: 保存结果
            output_dir = self.classifier.save(self.config.experiment_dir)
            
            # 保存完整的实验结果
            self._save_experiment_results(eval_results, h_stats, importance_analysis, start_time, output_dir)
            
            # 生成报告 - 修复了数组真值判断错误
            self._generate_final_report(eval_results, h_stats, importance_analysis, 
                                       time.time() - start_time, output_dir)
            
            return self.classifier, output_dir
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _save_experiment_results(self, eval_results: Dict, h_stats: Dict, 
                                importance_analysis: Dict, start_time: float, output_dir: str):
        """保存完整的实验结果"""
        results = {
            'experiment_info': {
                'name': self.config.experiment_name,
                'method': 'gradient_boosting_with_lightgbm',
                'feature_selection': 'reference_method',
                'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_time_seconds': time.time() - start_time
            },
            'config': {
                'features_file': self.config.features_file,
                'test_size': self.config.test_size,
                'random_seed': self.config.random_seed,
                'n_estimators': self.config.n_estimators,
                'learning_rate': self.config.learning_rate,
                'max_depth': self.config.max_depth,
                'num_leaves': self.config.num_leaves,
                'reg_alpha': self.config.reg_alpha,
                'reg_lambda': self.config.reg_lambda,
                'feature_selection_percentile': self.config.feature_selection_percentile,
                'feature_selection_method': self.config.feature_selection_method,
                'early_stopping_rounds': self.config.early_stopping_rounds
            },
            'evaluation': eval_results,
            'h_neurons_statistics': h_stats,
            'feature_importance_analysis': importance_analysis,
            'training_info': {
                'best_iteration': self.classifier.best_iteration
            }
        }
        
        results_path = os.path.join(output_dir, "experiment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验结果保存到: {results_path}")
    
    def _generate_final_report(self, eval_results: Dict, h_stats: Dict, 
                              importance_analysis: Dict, total_time: float, output_dir: str):
        """生成最终训练报告 - 修复了数组真值判断错误"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        threshold_stats = importance_analysis.get('threshold_statistics', {})
        
        # 安全地获取长度，避免数组真值判断错误
        feature_importance_len = 0
        if self.classifier.feature_importance is not None:
            feature_importance_len = len(self.classifier.feature_importance)
        
        neuron_names_len = 0
        if self.classifier.neuron_names is not None:
            neuron_names_len = len(self.classifier.neuron_names)
        
        report = f"""
{'='*60}
H-Neurons训练完成报告 - 梯度提升版本（参考代码特征筛选）
{'='*60}

特征筛选方法（完全按照参考代码）:
  - 第一阶段: VarianceThreshold(threshold=0) - 剔除常数特征
  - 第二阶段: SelectPercentile(f_classif, percentile={self.config.feature_selection_percentile}) - 选择前{self.config.feature_selection_percentile}%特征

梯度提升方法:
  - 算法: LightGBM梯度提升决策树
  - 树数量: {self.config.n_estimators}棵树
  - 树深度: {self.config.max_depth}
  - 学习率: {self.config.learning_rate}
  - 正则化: L1={self.config.reg_alpha}, L2={self.config.reg_lambda}
  - 早停策略: {self.config.early_stopping_rounds}轮无提升则停止

特征重要性分析:
  - 总特征数: {importance_analysis.get('total_features', 'N/A')}
  - 零重要性特征: {importance_analysis.get('zero_importance_features', 'N/A')}
  - 累积重要性:
"""
        
        for threshold, stats in threshold_stats.items():
            report += f"    {threshold}: {stats['n_features']}个特征 ({stats['percentage']:.1f}%)\n"
        
        report += f"""
数据统计:
  - 原始特征维度: {neuron_names_len}
  - 筛选后特征维度: {feature_importance_len}
  - 筛选比例: {feature_importance_len/neuron_names_len*100 if neuron_names_len>0 else 0:.2f}%
  - 正样本比例: {np.mean(eval_results.get('predictions', {}).get('y_true', [0])) if eval_results.get('predictions') and len(eval_results['predictions'].get('y_true', [])) > 0 else 'N/A':.3f}

H-Neurons发现:
  - H-Neurons数量: {h_stats.get('h_neuron_count', 0)}
  - H-Neurons比例: {h_stats.get('h_neuron_ratio', 0)*100:.4f}%
  - 是否<0.1%: {'✓ 是' if h_stats.get('h_neuron_ratio', 1) < 0.001 else '⚠ 否'}
  - 最佳迭代次数: {self.classifier.best_iteration}

模型性能:
  - 准确率: {eval_results.get('accuracy', 0):.4f}
  - AUC: {eval_results.get('auc', 0):.4f}

输出文件:
  - {output_dir}/gradient_boosting_model.pkl - 完整的模型对象
  - {output_dir}/h_neurons_gradient_boosting.json - H-Neurons详细信息(按重要性排序)
  - {output_dir}/feature_importance_analysis.json - 特征重要性分析
  - {output_dir}/lightgbm_model.txt - LightGBM原生模型文件
  - {output_dir}/h_neurons_statistics.json - 统计信息
  - {output_dir}/experiment_results.json - 完整实验结果
  - {output_dir}/training_report.txt - 本报告

使用说明:
  1. 加载模型:
     with open('{output_dir}/gradient_boosting_model.pkl', 'rb') as f:
         model_data = pickle.load(f)
  
  2. 获取H-Neurons:
     h_neurons = model_data['h_neuron_names']
     importance = model_data['feature_importance']
  
  3. 进行预测:
     # 先应用相同的特征筛选
     X_test_var = model_data['var_selector'].transform(new_X)
     X_test_percentile = model_data['percentile_selector'].transform(X_test_var)
     X_test_scaled = model_data['scaler'].transform(X_test_percentile)
     predictions = model_data['model'].predict(X_test_scaled)

{'='*60}
"""
        
        self.logger.info(report)
        
        # 保存报告到文件
        report_path = os.path.join(output_dir, "training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

# ==================== 主函数 ====================
def main_gradient_boosting():
    """梯度提升主函数"""
    # 设置配置
    config = Config()
    
    # 设置日志
    logger = LoggerSetup.setup()
    
    # 创建训练器
    trainer = GradientBoostingTrainer(config)
    
    # 运行训练
    logger.info("开始梯度提升训练...")
    classifier, output_dir = trainer.run()
    
    return classifier, output_dir

if __name__ == "__main__":
    try:
        print("="*60)
        print("开始H-Neurons训练 - 梯度提升版本（参考代码特征筛选）...")
        print("="*60)
        
        classifier, output_dir = main_gradient_boosting()
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"输出目录: {output_dir}")
        print()
        print("主要文件:")
        print(f"  1. {output_dir}/gradient_boosting_model.pkl - 完整模型")
        print(f"  2. {output_dir}/h_neurons_gradient_boosting.json - H-Neurons列表")
        print(f"  3. {output_dir}/feature_importance_analysis.json - 特征重要性分析")
        print(f"  4. {output_dir}/lightgbm_model.txt - LightGBM原生模型")
        print("="*60)
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

