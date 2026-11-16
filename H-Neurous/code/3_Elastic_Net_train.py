"""
H-Neurons完整训练管道 - 弹性网络版本
完全按照参考代码的特征筛选方式（VarianceThreshold + SelectPercentile 10%）
第二阶段使用弹性网络进行训练
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from tqdm import tqdm

# ==================== 配置 ====================
@dataclass
class Config:
    # 数据路径
    features_file: str = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
    output_dir: str = "/mnt/project/H-Neurous/output/elastic_net_reference"
    
    # 训练参数
    test_size: float = 0.15
    cv_folds: int = 5
    random_seed: int = 42
    
    # 弹性网络参数
    C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    l1_ratios: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    
    # 特征选择参数（完全按照参考代码）
    feature_selection_percentile: int = 10  # 保留前10%特征
    feature_selection_method: str = "f_classif"  # 与参考代码一致
    
    # 稳定性选择参数
    stability_iterations: int = 50
    stability_threshold: float = 0.8
    
    # 日志和实验
    experiment_name: str = "h_neurons_elastic_net_reference"
    
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
        log_file = os.path.join(log_dir, f"h_neurons_elastic_net_reference_{timestamp}.log")
        
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

# ==================== 弹性网络分类器 ====================
class ElasticNetHNeuronsClassifier:
    """基于弹性网络的H-Neurons分类器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型组件（完全按照参考代码的特征筛选）
        self.clf = None
        self.var_selector = VarianceThreshold(threshold=0)  # 与参考代码完全一致
        self.percentile_selector = SelectPercentile(f_classif, percentile=config.feature_selection_percentile)  # 完全一致
        self.scaler = StandardScaler()
        
        # H-Neurons信息
        self.neuron_names = None
        self.h_neuron_indices = None
        self.h_neuron_names = None
        self.h_neuron_weights = None
        self.feature_importance = None
        
        # 训练结果
        self.best_params = {}
        self.cv_results = {}
    
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
    
    def grid_search_cv(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """弹性网络参数网格搜索"""
        self.logger.info("开始弹性网络参数网格搜索...")
        
        # 参数网格
        param_grid = {
            'C': self.config.C_values,
            'l1_ratio': self.config.l1_ratios
        }
        
        # 创建弹性网络分类器
        base_clf = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.random_seed,
            tol=1e-4
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_seed),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.logger.info("网格搜索完成:")
        self.logger.info(f"  最佳参数: {grid_search.best_params_}")
        self.logger.info(f"  最佳交叉验证AUC: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_with_stability_selection(self, X: np.ndarray, y: np.ndarray, neuron_names: List[str]) -> 'ElasticNetHNeuronsClassifier':
        """使用稳定性选择训练弹性网络"""
        self.logger.info("开始训练弹性网络分类器...")
        self.neuron_names = neuron_names
        
        # 1. 应用参考代码的特征筛选
        X_selected, selected_indices = self.apply_reference_feature_selection(X, y)
        
        # 2. 标准化
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 3. 参数网格搜索
        grid_results = self.grid_search_cv(X_scaled, y)
        self.best_params = grid_results['best_params']
        self.cv_results = grid_results['cv_results']
        
        # 4. 使用最佳参数训练最终模型
        self.clf = LogisticRegression(
            C=self.best_params['C'],
            penalty='elasticnet',
            solver='saga',
            l1_ratio=self.best_params['l1_ratio'],
            class_weight='balanced',
            max_iter=2000,
            random_state=self.config.random_seed,
            tol=1e-4
        )
        
        self.clf.fit(X_scaled, y)
        
        # 5. 应用稳定性选择
        self.logger.info("应用稳定性选择...")
        stable_indices = self._stability_selection(X_scaled, y)
        
        # 6. 映射回原始特征空间
        self._map_to_original_features(selected_indices, stable_indices)
        
        return self
    
    def _stability_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """稳定性选择：多次重采样，选择稳定特征"""
        self.logger.info(f"稳定性选择: {self.config.stability_iterations} 次迭代")
        
        n_features = X.shape[1]
        selection_counts = np.zeros(n_features)
        
        for i in tqdm(range(self.config.stability_iterations), desc="稳定性选择"):
            # 自助采样
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 训练弹性网络
            clf_boot = LogisticRegression(
                C=self.best_params['C'],
                penalty='elasticnet',
                solver='saga',
                l1_ratio=self.best_params['l1_ratio'],
                class_weight='balanced',
                max_iter=1000,
                random_state=i,
                tol=1e-3
            )
            
            clf_boot.fit(X_boot, y_boot)
            
            # 记录非零权重
            weights = np.abs(clf_boot.coef_[0])
            selected = weights > 1e-4
            selection_counts[selected] += 1
        
        # 计算选择概率
        selection_probs = selection_counts / self.config.stability_iterations
        
        # 选择稳定特征（选择概率 > 阈值）
        stable_indices = np.where(selection_probs > self.config.stability_threshold)[0]
        
        self.logger.info(f"稳定性选择结果: {len(stable_indices)} 个稳定特征 (阈值={self.config.stability_threshold})")
        self.logger.info(f"  最小选择概率: {selection_probs.min():.3f}")
        self.logger.info(f"  最大选择概率: {selection_probs.max():.3f}")
        self.logger.info(f"  平均选择概率: {selection_probs.mean():.3f}")
        
        return stable_indices
    
    def _map_to_original_features(self, selected_indices: np.ndarray, stable_indices: np.ndarray):
        """将稳定特征索引映射回原始特征索引"""
        self.logger.info("映射特征到原始空间...")
        
        # 映射到筛选后的特征索引
        self.h_neuron_indices = selected_indices[stable_indices]
        self.h_neuron_names = [self.neuron_names[i] for i in self.h_neuron_indices]
        
        # 获取权重
        if hasattr(self.clf, 'coef_'):
            weights_selected = self.clf.coef_[0]
            stable_weights = weights_selected[stable_indices]
            
            # 创建完整的权重向量
            weights_full = np.zeros(len(self.neuron_names))
            for idx, weight in zip(self.h_neuron_indices, stable_weights):
                weights_full[idx] = weight
            
            self.h_neuron_weights = stable_weights
            self.feature_importance = weights_full
        
        self.logger.info(f"识别出 {len(self.h_neuron_indices)} 个 H-Neurons")
        self.logger.info(f"占总神经元的 {len(self.h_neuron_indices)/len(self.neuron_names)*100:.4f}%")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型性能"""
        # 应用相同的特征筛选
        X_test_var_selected = self.var_selector.transform(X_test)
        X_test_selected = self.percentile_selector.transform(X_test_var_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # 预测
        y_pred = self.clf.predict(X_test_scaled)
        y_prob = self.clf.predict_proba(X_test_scaled)[:, 1]
        
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
        
        # 按权重统计
        if self.h_neuron_weights is not None:
            weights_abs = np.abs(self.h_neuron_weights)
            weight_stats = {
                'min': float(np.min(weights_abs)),
                'max': float(np.max(weights_abs)),
                'mean': float(np.mean(weights_abs)),
                'std': float(np.std(weights_abs)),
                'median': float(np.median(weights_abs))
            }
        else:
            weight_stats = {}
        
        # 总体统计
        stats = {
            'total_neurons': len(self.neuron_names),
            'h_neuron_count': len(self.h_neuron_indices),
            'h_neuron_ratio': len(self.h_neuron_indices) / len(self.neuron_names),
            'layer_distribution': layer_dist,
            'weight_statistics': weight_stats,
            'elastic_net_params': self.best_params,
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
        self.logger.info(f"  L1比例: {self.best_params.get('l1_ratio', 'N/A')}")
        self.logger.info(f"  正则化强度C: {self.best_params.get('C', 'N/A')}")
        
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
            'classifier': self.clf,
            'var_selector': self.var_selector,
            'percentile_selector': self.percentile_selector,
            'scaler': self.scaler,
            'neuron_names': self.neuron_names,
            'h_neuron_indices': self.h_neuron_indices,
            'h_neuron_names': self.h_neuron_names,
            'h_neuron_weights': self.h_neuron_weights,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'cv_results': self.cv_results,
            'config': self.config
        }
        
        model_path = os.path.join(output_dir, "elastic_net_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 保存H-Neurons列表为JSON
        h_neurons_list = []
        if self.h_neuron_weights is not None:
            for i, (name, weight) in enumerate(zip(self.h_neuron_names, self.h_neuron_weights)):
                h_neurons_list.append({
                    'index': int(self.h_neuron_indices[i]),
                    'name': name,
                    'weight': float(weight),
                    'abs_weight': float(abs(weight))
                })
        
        # 按权重排序
        h_neurons_list.sort(key=lambda x: x['abs_weight'], reverse=True)
        
        json_path = os.path.join(output_dir, "h_neurons_elastic_net.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(h_neurons_list, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats = self.analyze_h_neurons()
        stats_path = os.path.join(output_dir, "h_neurons_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"模型保存完成: {model_path}")
        self.logger.info(f"H-Neurons列表: {json_path}")
        
        return output_dir

# ==================== 弹性网络训练器 ====================
class ElasticNetTrainer:
    """弹性网络训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件
        self.feature_loader = StreamingFeatureLoader(config)
        self.classifier = ElasticNetHNeuronsClassifier(config)
    
    def run(self) -> Tuple[ElasticNetHNeuronsClassifier, str]:
        """运行训练流程"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons训练管道 - 弹性网络版本（参考代码特征筛选）")
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
            
            # 阶段3: 训练弹性网络分类器
            self.logger.info("\n[阶段3/4] 训练弹性网络分类器")
            self.classifier.train_with_stability_selection(X_train, y_train, neuron_names)
            
            # 阶段4: 评估模型
            self.logger.info("\n[阶段4/4] 评估模型性能")
            eval_results = self.classifier.evaluate(X_test, y_test)
            
            # 阶段5: 分析H-Neurons
            h_stats = self.classifier.analyze_h_neurons()
            
            # 阶段6: 保存结果
            output_dir = self.classifier.save(self.config.experiment_dir)
            
            # 保存完整的实验结果
            self._save_experiment_results(eval_results, h_stats, start_time, output_dir)
            
            # 生成报告
            self._generate_final_report(eval_results, h_stats, time.time() - start_time, output_dir)
            
            return self.classifier, output_dir
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _save_experiment_results(self, eval_results: Dict, h_stats: Dict, start_time: float, output_dir: str):
        """保存完整的实验结果"""
        results = {
            'experiment_info': {
                'name': self.config.experiment_name,
                'method': 'elastic_net_with_stability_selection',
                'feature_selection': 'reference_method',
                'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_time_seconds': time.time() - start_time
            },
            'config': {
                'features_file': self.config.features_file,
                'test_size': self.config.test_size,
                'cv_folds': self.config.cv_folds,
                'random_seed': self.config.random_seed,
                'C_values': self.config.C_values,
                'l1_ratios': self.config.l1_ratios,
                'feature_selection_percentile': self.config.feature_selection_percentile,
                'feature_selection_method': self.config.feature_selection_method,
                'stability_iterations': self.config.stability_iterations,
                'stability_threshold': self.config.stability_threshold
            },
            'evaluation': eval_results,
            'h_neurons_statistics': h_stats,
            'best_parameters': self.classifier.best_params
        }
        
        results_path = os.path.join(output_dir, "experiment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验结果保存到: {results_path}")
    
    def _generate_final_report(self, eval_results: Dict, h_stats: Dict, total_time: float, output_dir: str):
        """生成最终训练报告"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        report = f"""
{'='*60}
H-Neurons训练完成报告 - 弹性网络版本（参考代码特征筛选）
{'='*60}

特征筛选方法（完全按照参考代码）:
  - 第一阶段: VarianceThreshold(threshold=0) - 剔除常数特征
  - 第二阶段: SelectPercentile(f_classif, percentile={self.config.feature_selection_percentile}) - 选择前{self.config.feature_selection_percentile}%特征

弹性网络方法:
  - 正则化类型: 弹性网络(L1+L2)
  - 网格搜索: 自动寻找最佳L1比例和正则化强度
  - L1比例范围: {self.config.l1_ratios}
  - 正则化强度C范围: {self.config.C_values}
  - 稳定性选择: {self.config.stability_iterations}次自助采样，阈值{self.config.stability_threshold}

数据统计:
  - 原始特征维度: {self.classifier.neuron_names and len(self.classifier.neuron_names) or 'N/A'}
  - 筛选后特征维度: {self.classifier.clf and self.classifier.clf.coef_.shape[1] or 'N/A'}
  - 筛选比例: {self.classifier.clf and self.classifier.clf.coef_.shape[1]/len(self.classifier.neuron_names)*100 or 'N/A':.2f}%
  - 正样本比例: {np.mean(eval_results.get('predictions', {}).get('y_true', [0])) if eval_results.get('predictions') else 'N/A':.3f}

H-Neurons发现:
  - H-Neurons数量: {h_stats.get('h_neuron_count', 0)}
  - H-Neurons比例: {h_stats.get('h_neuron_ratio', 0)*100:.4f}%
  - 是否<0.1%: {'✓ 是' if h_stats.get('h_neuron_ratio', 1) < 0.001 else '⚠ 否'}
  - 最佳L1比例: {self.classifier.best_params.get('l1_ratio', 'N/A')}
  - 最佳正则化强度C: {self.classifier.best_params.get('C', 'N/A')}

模型性能:
  - 准确率: {eval_results.get('accuracy', 0):.4f}
  - AUC: {eval_results.get('auc', 0):.4f}

输出文件:
  - {output_dir}/elastic_net_model.pkl - 完整的模型对象
  - {output_dir}/h_neurons_elastic_net.json - H-Neurons详细信息(按权重排序)
  - {output_dir}/h_neurons_statistics.json - 统计信息
  - {output_dir}/experiment_results.json - 完整实验结果
  - {output_dir}/training_report.txt - 本报告

使用说明:
  1. 加载模型:
     with open('{output_dir}/elastic_net_model.pkl', 'rb') as f:
         model_data = pickle.load(f)
  
  2. 获取H-Neurons:
     h_neurons = model_data['h_neuron_names']
     h_weights = model_data['h_neuron_weights']
  
  3. 进行预测:
     # 先应用相同的特征筛选
     X_test_var = model_data['var_selector'].transform(new_X)
     X_test_percentile = model_data['percentile_selector'].transform(X_test_var)
     X_test_scaled = model_data['scaler'].transform(X_test_percentile)
     predictions = model_data['classifier'].predict(X_test_scaled)

{'='*60}
"""
        
        self.logger.info(report)
        
        # 保存报告到文件
        report_path = os.path.join(output_dir, "training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

# ==================== 主函数 ====================
def main_elastic_net():
    """弹性网络主函数"""
    # 设置配置
    config = Config()
    
    # 设置日志
    logger = LoggerSetup.setup()
    
    # 创建训练器
    trainer = ElasticNetTrainer(config)
    
    # 运行训练
    logger.info("开始弹性网络训练...")
    classifier, output_dir = trainer.run()
    
    return classifier, output_dir

if __name__ == "__main__":
    try:
        print("="*60)
        print("开始H-Neurons训练 - 弹性网络版本（参考代码特征筛选）...")
        print("="*60)
        
        classifier, output_dir = main_elastic_net()
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"输出目录: {output_dir}")
        print()
        print("主要文件:")
        print(f"  1. {output_dir}/elastic_net_model.pkl - 完整模型")
        print(f"  2. {output_dir}/h_neurons_elastic_net.json - H-Neurons列表")
        print(f"  3. {output_dir}/experiment_results.json - 实验结果")
        print("="*60)
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()