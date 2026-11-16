
"""
H-Neurons完整训练管道 - 迭代剪枝版本
完全按照参考代码的特征筛选方式（VarianceThreshold + SelectPercentile 10%）
第二阶段使用迭代剪枝进行训练
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
    output_dir: str = "/mnt/project/H-Neurous/output/iterative_pruning_reference"
    
    # 训练参数
    test_size: float = 0.15
    random_seed: int = 42
    
    # 特征选择参数（完全按照参考代码）
    feature_selection_percentile: int = 10  # 保留前10%特征
    feature_selection_method: str = "f_classif"  # 与参考代码一致
    
    # 迭代剪枝参数
    pruning_steps: int = 10  # 剪枝步数
    keep_ratio: float = 0.1  # 每步保留的比例
    min_features: int = 100  # 最小特征数
    retrain_iterations: int = 3  # 每次剪枝后重新训练的迭代次数
    
    # 模型参数
    C: float = 0.01  # 逻辑回归正则化强度
    max_iter: int = 1000
    
    # 日志和实验
    experiment_name: str = "h_neurons_iterative_pruning_reference"
    
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
        log_file = os.path.join(log_dir, f"h_neurons_iterative_pruning_reference_{timestamp}.log")
        
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

# ==================== 迭代剪枝分类器 ====================
class IterativePruningHNeuronsClassifier:
    """基于迭代剪枝的H-Neurons分类器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型组件（完全按照参考代码的特征筛选）
        self.classifier = None
        self.var_selector = VarianceThreshold(threshold=0)  # 与参考代码完全一致
        self.percentile_selector = SelectPercentile(f_classif, percentile=config.feature_selection_percentile)  # 完全一致
        self.scaler = StandardScaler()
        
        # H-Neurons信息
        self.neuron_names = None
        self.h_neuron_indices = None
        self.h_neuron_names = None
        self.h_neuron_weights = None
        
        # 训练历史
        self.pruning_history = []
        self.performance_history = []
        
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
    
    def iterative_pruning(self, X: np.ndarray, y: np.ndarray, neuron_names: List[str]) -> 'IterativePruningHNeuronsClassifier':
        """迭代剪枝算法"""
        self.logger.info("开始迭代剪枝训练...")
        self.neuron_names = neuron_names
        
        # 1. 应用参考代码的特征筛选
        X_selected, selected_indices = self.apply_reference_feature_selection(X, y)
        
        # 2. 标准化
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 当前特征集
        current_X = X_scaled.copy()
        current_indices = selected_indices.copy()
        current_features = current_X.shape[1]
        
        self.logger.info(f"初始特征数: {current_features}")
        
        # 3. 迭代剪枝循环
        for step in range(self.config.pruning_steps):
            self.logger.info(f"\n剪枝步骤 {step + 1}/{self.config.pruning_steps}")
            
            # 训练逻辑回归模型
            self.logger.info("  训练逻辑回归模型...")
            clf = LogisticRegression(
                C=self.config.C,
                penalty='l2',  # 使用L2正则化
                solver='saga',
                class_weight='balanced',
                max_iter=self.config.max_iter,
                random_state=self.config.random_seed + step
            )
            
            # 交叉验证评估
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_seed)
            cv_scores = []
            
            for train_idx, val_idx in skf.split(current_X, y):
                X_train_fold = current_X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = current_X[val_idx]
                y_val_fold = y[val_idx]
                
                clf.fit(X_train_fold, y_train_fold)
                y_pred_prob = clf.predict_proba(X_val_fold)[:, 1]
                auc = roc_auc_score(y_val_fold, y_pred_prob)
                cv_scores.append(auc)
            
            mean_auc = np.mean(cv_scores)
            
            # 在整个训练集上训练
            clf.fit(current_X, y)
            
            # 获取特征重要性（权重绝对值）
            feature_importance = np.abs(clf.coef_[0])
            
            # 记录历史
            self.pruning_history.append({
                'step': step,
                'n_features': current_features,
                'cv_auc': mean_auc,
                'feature_importance': feature_importance.tolist()
            })
            
            # 检查是否达到最小特征数
            if current_features <= self.config.min_features:
                self.logger.info(f"达到最小特征数 {self.config.min_features}，停止剪枝")
                self.classifier = clf
                final_indices = current_indices
                break
            
            # 计算要保留的特征数
            keep_n = max(
                self.config.min_features,
                int(current_features * self.config.keep_ratio)
            )
            
            # 选择最重要的特征
            top_indices = np.argsort(feature_importance)[-keep_n:]
            
            self.logger.info(f"  当前特征数: {current_features}")
            self.logger.info(f"  保留特征数: {keep_n}")
            self.logger.info(f"  交叉验证AUC: {mean_auc:.4f}")
            
            # 更新特征集
            current_features = keep_n
            current_X = current_X[:, top_indices]
            current_indices = current_indices[top_indices]
            
            # 重新训练（可选）
            if self.config.retrain_iterations > 0:
                for retrain_step in range(self.config.retrain_iterations):
                    clf_retrain = LogisticRegression(
                        C=self.config.C,
                        penalty='l2',
                        solver='saga',
                        class_weight='balanced',
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_seed + retrain_step
                    )
                    clf_retrain.fit(current_X, y)
                    
                    # 更新特征重要性
                    feature_importance = np.abs(clf_retrain.coef_[0])
                    top_indices = np.argsort(feature_importance)[-keep_n:]
                    current_X = current_X[:, top_indices]
                    current_indices = current_indices[top_indices]
            
            # 保存最终分类器
            self.classifier = clf
        
        # 4. 映射回原始特征空间
        self.logger.info("\n映射特征到原始空间...")
        self.h_neuron_indices = current_indices
        self.h_neuron_names = [self.neuron_names[i] for i in self.h_neuron_indices]
        
        # 获取权重
        if hasattr(self.classifier, 'coef_'):
            self.h_neuron_weights = self.classifier.coef_[0]
        
        self.logger.info(f"迭代剪枝完成，识别出 {len(self.h_neuron_indices)} 个 H-Neurons")
        self.logger.info(f"占总神经元的 {len(self.h_neuron_indices)/len(self.neuron_names)*100:.4f}%")
        
        return self
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型性能"""
        if self.classifier is None:
            raise ValueError("模型未训练")
        
        # 应用相同的特征筛选
        X_test_var_selected = self.var_selector.transform(X_test)
        X_test_selected = self.percentile_selector.transform(X_test_var_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # 只保留选中的特征（根据迭代剪枝的结果）
        if self.h_neuron_indices is not None:
            # 需要从筛选后的特征中进一步选择
            # 首先获取在percentile筛选后的特征索引
            var_selected_indices = np.where(self.var_selector.get_support())[0]
            percentile_selected_indices = np.where(self.percentile_selector.get_support())[0]
            final_selected_indices = var_selected_indices[percentile_selected_indices]
            
            # 找到这些索引中哪些是H-Neurons
            h_mask = np.isin(final_selected_indices, self.h_neuron_indices)
            X_test_final = X_test_scaled[:, h_mask]
        else:
            X_test_final = X_test_scaled
        
        # 预测
        y_pred = self.classifier.predict(X_test_final)
        y_prob = self.classifier.predict_proba(X_test_final)[:, 1]
        
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
    
    def analyze_pruning_history(self) -> Dict:
        """分析剪枝历史"""
        if not self.pruning_history:
            return {}
        
        steps = [h['step'] for h in self.pruning_history]
        n_features = [h['n_features'] for h in self.pruning_history]
        cv_auc = [h['cv_auc'] for h in self.pruning_history]
        
        # 计算特征减少比例
        feature_reduction = [(n_features[0] - n) / n_features[0] * 100 for n in n_features]
        
        # 找到性能峰值
        peak_idx = np.argmax(cv_auc)
        
        analysis = {
            'initial_features': n_features[0],
            'final_features': n_features[-1],
            'total_reduction': feature_reduction[-1],
            'peak_performance': {
                'step': steps[peak_idx],
                'features': n_features[peak_idx],
                'auc': cv_auc[peak_idx],
                'reduction': feature_reduction[peak_idx]
            },
            'pruning_history': self.pruning_history
        }
        
        self.logger.info("剪枝历史分析:")
        self.logger.info(f"  初始特征数: {analysis['initial_features']}")
        self.logger.info(f"  最终特征数: {analysis['final_features']}")
        self.logger.info(f"  总减少比例: {analysis['total_reduction']:.1f}%")
        self.logger.info(f"  性能峰值 - 步骤{analysis['peak_performance']['step']}:")
        self.logger.info(f"    特征数: {analysis['peak_performance']['features']}")
        self.logger.info(f"    AUC: {analysis['peak_performance']['auc']:.4f}")
        self.logger.info(f"    减少比例: {analysis['peak_performance']['reduction']:.1f}%")
        
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
            'pruning_steps': self.config.pruning_steps,
            'keep_ratio': self.config.keep_ratio,
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
        self.logger.info(f"  剪枝步数: {stats['pruning_steps']}")
        self.logger.info(f"  每步保留比例: {stats['keep_ratio']}")
        
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
            'classifier': self.classifier,
            'var_selector': self.var_selector,
            'percentile_selector': self.percentile_selector,
            'scaler': self.scaler,
            'neuron_names': self.neuron_names,
            'h_neuron_indices': self.h_neuron_indices,
            'h_neuron_names': self.h_neuron_names,
            'h_neuron_weights': self.h_neuron_weights,
            'pruning_history': self.pruning_history,
            'config': self.config
        }
        
        model_path = os.path.join(output_dir, "iterative_pruning_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 保存H-Neurons列表
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
        
        json_path = os.path.join(output_dir, "h_neurons_iterative_pruning.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(h_neurons_list, f, indent=2, ensure_ascii=False)
        
        # 保存剪枝历史
        pruning_analysis = self.analyze_pruning_history()
        history_path = os.path.join(output_dir, "pruning_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(pruning_analysis, f, indent=2, ensure_ascii=False)
        
        # 保存统计信息
        stats = self.analyze_h_neurons()
        stats_path = os.path.join(output_dir, "h_neurons_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"模型保存完成: {model_path}")
        self.logger.info(f"H-Neurons列表: {json_path}")
        self.logger.info(f"剪枝历史: {history_path}")
        
        return output_dir

# ==================== 迭代剪枝训练器 ====================
class IterativePruningTrainer:
    """迭代剪枝训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件
        self.feature_loader = StreamingFeatureLoader(config)
        self.classifier = IterativePruningHNeuronsClassifier(config)
    
    def run(self) -> Tuple[IterativePruningHNeuronsClassifier, str]:
        """运行训练流程"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons训练管道 - 迭代剪枝版本（参考代码特征筛选）")
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
            
            # 阶段3: 迭代剪枝训练
            self.logger.info("\n[阶段3/4] 迭代剪枝训练")
            self.classifier.iterative_pruning(X_train, y_train, neuron_names)
            
            # 阶段4: 评估模型
            self.logger.info("\n[阶段4/4] 评估模型性能")
            eval_results = self.classifier.evaluate(X_test, y_test)
            
            # 阶段5: 分析结果
            pruning_analysis = self.classifier.analyze_pruning_history()
            h_stats = self.classifier.analyze_h_neurons()
            
            # 阶段6: 保存结果
            output_dir = self.classifier.save(self.config.experiment_dir)
            
            # 保存完整的实验结果
            self._save_experiment_results(eval_results, h_stats, pruning_analysis, start_time, output_dir)
            
            # 生成报告
            self._generate_final_report(eval_results, h_stats, pruning_analysis, time.time() - start_time, output_dir)
            
            return self.classifier, output_dir
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _save_experiment_results(self, eval_results: Dict, h_stats: Dict, 
                                pruning_analysis: Dict, start_time: float, output_dir: str):
        """保存完整的实验结果"""
        results = {
            'experiment_info': {
                'name': self.config.experiment_name,
                'method': 'iterative_pruning',
                'feature_selection': 'reference_method',
                'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_time_seconds': time.time() - start_time
            },
            'config': {
                'features_file': self.config.features_file,
                'test_size': self.config.test_size,
                'random_seed': self.config.random_seed,
                'pruning_steps': self.config.pruning_steps,
                'keep_ratio': self.config.keep_ratio,
                'min_features': self.config.min_features,
                'retrain_iterations': self.config.retrain_iterations,
                'C': self.config.C,
                'max_iter': self.config.max_iter,
                'feature_selection_percentile': self.config.feature_selection_percentile,
                'feature_selection_method': self.config.feature_selection_method
            },
            'evaluation': eval_results,
            'h_neurons_statistics': h_stats,
            'pruning_analysis': pruning_analysis
        }
        
        results_path = os.path.join(output_dir, "experiment_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验结果保存到: {results_path}")
    
    def _generate_final_report(self, eval_results: Dict, h_stats: Dict, 
                              pruning_analysis: Dict, total_time: float, output_dir: str):
        """生成最终训练报告"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        peak_info = pruning_analysis.get('peak_performance', {})
        
        report = f"""
{'='*60}
H-Neurons训练完成报告 - 迭代剪枝版本（参考代码特征筛选）
{'='*60}

特征筛选方法（完全按照参考代码）:
  - 第一阶段: VarianceThreshold(threshold=0) - 剔除常数特征
  - 第二阶段: SelectPercentile(f_classif, percentile={self.config.feature_selection_percentile}) - 选择前{self.config.feature_selection_percentile}%特征

迭代剪枝方法:
  - 剪枝策略: {self.config.pruning_steps}步剪枝，每步保留{self.config.keep_ratio*100}%特征
  - 停止条件: 最小特征数={self.config.min_features}
  - 重训练: {self.config.retrain_iterations}次重训练优化权重
  - 正则化: L2正则化，C={self.config.C}

剪枝效果:
  - 初始特征数: {pruning_analysis.get('initial_features', 'N/A')}
  - 最终特征数: {pruning_analysis.get('final_features', 'N/A')}
  - 总减少比例: {pruning_analysis.get('total_reduction', 0):.1f}%
  - 性能峰值 - 步骤{peak_info.get('step', 'N/A')}:
      AUC: {peak_info.get('auc', 0):.4f}
      特征数: {peak_info.get('features', 'N/A')}
      减少比例: {peak_info.get('reduction', 0):.1f}%

数据统计:
  - 原始特征维度: {self.classifier.neuron_names and len(self.classifier.neuron_names) or 'N/A'}
  - 筛选后特征维度: {pruning_analysis.get('initial_features', 'N/A')}
  - H-Neurons特征数: {h_stats.get('h_neuron_count', 0)}
  - 筛选比例: {h_stats.get('h_neuron_count', 0)/(self.classifier.neuron_names and len(self.classifier.neuron_names) or 1)*100:.4f}%
  - 正样本比例: {np.mean(eval_results.get('predictions', {}).get('y_true', [0])) if eval_results.get('predictions') else 'N/A':.3f}

H-Neurons发现:
  - H-Neurons数量: {h_stats.get('h_neuron_count', 0)}
  - H-Neurons比例: {h_stats.get('h_neuron_ratio', 0)*100:.4f}%
  - 是否<0.1%: {'✓ 是' if h_stats.get('h_neuron_ratio', 1) < 0.001 else '⚠ 否'}

模型性能:
  - 准确率: {eval_results.get('accuracy', 0):.4f}
  - AUC: {eval_results.get('auc', 0):.4f}

输出文件:
  - {output_dir}/iterative_pruning_model.pkl - 完整的模型对象
  - {output_dir}/h_neurons_iterative_pruning.json - H-Neurons详细信息
  - {output_dir}/pruning_history.json - 剪枝历史分析
  - {output_dir}/h_neurons_statistics.json - 统计信息
  - {output_dir}/experiment_results.json - 完整实验结果
  - {output_dir}/training_report.txt - 本报告

使用说明:
  1. 加载模型:
     with open('{output_dir}/iterative_pruning_model.pkl', 'rb') as f:
         model_data = pickle.load(f)
  
  2. 获取H-Neurons:
     h_neurons = model_data['h_neuron_names']
     h_weights = model_data['h_neuron_weights']
  
  3. 查看剪枝历史:
     pruning_history = model_data['pruning_history']

{'='*60}
"""
        
        self.logger.info(report)
        
        # 保存报告到文件
        report_path = os.path.join(output_dir, "training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

# ==================== 主函数 ====================
def main_iterative_pruning():
    """迭代剪枝主函数"""
    # 设置配置
    config = Config()
    
    # 设置日志
    logger = LoggerSetup.setup()
    
    # 创建训练器
    trainer = IterativePruningTrainer(config)
    
    # 运行训练
    logger.info("开始迭代剪枝训练...")
    classifier, output_dir = trainer.run()
    
    return classifier, output_dir

if __name__ == "__main__":
    try:
        print("="*60)
        print("开始H-Neurons训练 - 迭代剪枝版本（参考代码特征筛选）...")
        print("="*60)
        
        classifier, output_dir = main_iterative_pruning()
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"输出目录: {output_dir}")
        print()
        print("主要文件:")
        print(f"  1. {output_dir}/iterative_pruning_model.pkl - 完整模型")
        print(f"  2. {output_dir}/h_neurons_iterative_pruning.json - H-Neurons列表")
        print(f"  3. {output_dir}/pruning_history.json - 剪枝历史分析")
        print("="*60)
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()