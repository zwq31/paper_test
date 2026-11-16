"""
H-Neurons PCA降维训练管道
在原有代码基础上修改，使用PCA降维替换特征选择
保持流式加载特征，只修改训练部分
"""

import os
import json
import pickle
import logging
import time
import gc
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==================== 配置 ====================
@dataclass
class PCATrainConfig:
    # 数据路径
    features_file: str = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
    output_dir: str = "/mnt/project/H-Neurous/output/pca_results"
    
    # PCA参数
    pca_components: int = 1000  # 降到1000维
    random_seed: int = 42
    test_size: float = 0.15
    cv_folds: int = 5
    
    # 分类器参数
    C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    
    # 特征处理参数
    use_scaling: bool = True
    max_samples: Optional[int] = None  # 可选：限制样本数量
    
    # 实验名称
    experiment_name: str = "h_neurons_pca"
    
    def __post_init__(self):
        # 创建目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

# ==================== PCA特征处理器（流式加载） ====================
class PCAFeatureProcessor:
    """
    PCA特征处理器 - 流式加载版本
    保持与原有PaperFeatureProcessor相同的接口
    """
    
    def __init__(self, config: PCATrainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # PCA相关组件
        self.scaler = StandardScaler() if config.use_scaling else None
        self.pca = PCA(n_components=config.pca_components, random_state=config.random_seed)
        
        # 缓存
        self.all_features = []
        self.all_labels = []
        self.feature_dim = None
        
    def load_features_streaming(self, features_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        流式加载特征文件并构建训练数据 - 按照论文标记策略
        返回PCA降维后的特征和标签
        """
        self.logger.info(f"流式加载特征文件: {features_file}")
        
        # 获取文件信息
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"特征文件不存在: {features_file}")
        
        file_size = os.path.getsize(features_file)
        self.logger.info(f"特征文件大小: {file_size / (1024**3):.2f} GB")
        
        # 1. 首先读取第一行获取特征维度
        with open(features_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                first_sample = json.loads(first_line)
                if 'features_answer' in first_sample:
                    self.feature_dim = len(first_sample['features_answer'])
        
        if self.feature_dim is None:
            raise ValueError("特征文件为空或格式错误")
        
        self.logger.info(f"原始特征维度: {self.feature_dim}")
        self.logger.info(f"PCA目标维度: {self.config.pca_components}")
        
        # 2. 流式处理文件，收集特征
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
        processed_count = 0
        with open(features_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="流式处理特征")):
                if self.config.max_samples and processed_count >= self.config.max_samples:
                    self.logger.info(f"达到最大样本限制: {self.config.max_samples}")
                    break
                    
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
                            self.all_features.append(x_vector)
                            self.all_labels.append(1)
                            stats['pos_hallucination_answer'] += 1
                        else:  # 忠实样本的答案令牌 -> 负样本
                            self.all_features.append(x_vector)
                            self.all_labels.append(0)
                            stats['neg_faithful_answer'] += 1
                    
                    # 处理非答案令牌特征
                    if 'features_other' in sample:
                        feat_other_list = sample['features_other']
                        x_vector = np.array(feat_other_list, dtype=np.float32)
                        
                        # 检查并清理特征
                        if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
                            x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        # 所有非答案令牌都是负样本
                        self.all_features.append(x_vector)
                        self.all_labels.append(0)
                        
                        if label == 1:
                            stats['neg_hallucination_other'] += 1
                        else:
                            stats['neg_faithful_other'] += 1
                    
                    processed_count += 1
                    
                    # 定期清理内存
                    if processed_count % 1000 == 0:
                        gc.collect()
                        
                except Exception as e:
                    self.logger.warning(f"行 {line_num} 处理错误: {e}")
                    continue
        
        # 转换为numpy数组
        X_original = np.array(self.all_features, dtype=np.float32)
        y = np.array(self.all_labels, dtype=np.int32)
        
        # 最终清理
        X_original = np.nan_to_num(X_original, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.logger.info(f"论文标记策略统计:")
        self.logger.info(f"  正样本(幻觉答案令牌): {stats['pos_hallucination_answer']}")
        self.logger.info(f"  负样本(忠实答案令牌): {stats['neg_faithful_answer']}")
        self.logger.info(f"  负样本(幻觉非答案令牌): {stats['neg_hallucination_other']}")
        self.logger.info(f"  负样本(忠实非答案令牌): {stats['neg_faithful_other']}")
        self.logger.info(f"  总训练样本数: {len(X_original)}")
        self.logger.info(f"  正样本比例: {np.mean(y):.3f}")
        self.logger.info(f"  原始特征矩阵形状: {X_original.shape}")
        
        # 3. 应用PCA降维
        self.logger.info(f"\n应用PCA降维: {X_original.shape[1]} → {self.config.pca_components}")
        
        # 标准化（可选）
        if self.scaler is not None:
            self.logger.info("标准化特征...")
            X_scaled = self.scaler.fit_transform(X_original)
        else:
            X_scaled = X_original
        
        # PCA降维
        X_pca = self.pca.fit_transform(X_scaled)
        
        # 解释方差
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        self.logger.info(f"PCA解释方差比例: {explained_variance:.4f}")
        self.logger.info(f"降维后特征矩阵形状: {X_pca.shape}")
        
        # 保存PCA信息
        self._save_pca_info()
        
        return X_pca, y
    
    def _save_pca_info(self):
        """保存PCA相关信息"""
        pca_info = {
            'n_components': self.config.pca_components,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'total_explained_variance': float(np.sum(self.pca.explained_variance_ratio_)),
            'singular_values': self.pca.singular_values_.tolist()
        }
        
        info_path = os.path.join(self.config.experiment_dir, "pca_info.json")
        with open(info_path, 'w') as f:
            json.dump(pca_info, f, indent=2)
        
        self.logger.info(f"PCA信息保存到: {info_path}")

# ==================== PCA分类器 ====================
class PCAClassifier:
    """基于PCA降维特征的分类器"""
    
    def __init__(self, config: PCATrainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型
        self.classifier = None
        self.feature_processor = None
        
        # 结果
        self.results = {}
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """训练PCA降维后的分类器"""
        self.logger.info(f"训练数据形状: {X.shape}")
        self.logger.info(f"正样本比例: {np.mean(y):.3f}")
        
        # 1. 划分数据集
        self.logger.info("划分数据集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_seed
        )
        
        self.logger.info(f"训练集: {X_train.shape[0]} 样本")
        self.logger.info(f"测试集: {X_test.shape[0]} 样本")
        
        # 2. 交叉验证寻找最佳C
        self.logger.info("交叉验证寻找最佳C值...")
        best_score = -1
        best_C = self.config.C_values[0]
        skf = StratifiedKFold(n_splits=min(self.config.cv_folds, len(y_train)), 
                             shuffle=True, 
                             random_state=self.config.random_seed)
        
        for C in tqdm(self.config.C_values, desc="交叉验证"):
            try:
                clf = LogisticRegression(
                    C=C,
                    penalty='l1',
                    solver='saga',
                    class_weight='balanced',
                    max_iter=500,
                    random_state=self.config.random_seed,
                    tol=1e-3
                )
                cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='roc_auc')
                mean_score = np.mean(cv_scores)
                self.logger.info(f"C={C}: ROC_AUC={mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_C = C
            except Exception as e:
                self.logger.warning(f"C={C} 训练失败: {e}")
        
        # 3. 使用最佳C训练最终模型
        self.logger.info(f"使用最佳 C={best_C} 训练最终模型...")
        self.classifier = LogisticRegression(
            C=best_C,
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.random_seed,
            tol=1e-4
        )
        self.classifier.fit(X_train, y_train)
        
        # 4. 评估模型
        self.logger.info("评估模型...")
        y_pred = self.classifier.predict(X_test)
        y_prob = self.classifier.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        self.logger.info(f"测试集准确率: {accuracy:.4f}")
        self.logger.info(f"测试集AUC: {auc:.4f}")
        self.logger.info("\n分类报告:")
        self.logger.info(classification_report(y_test, y_pred))
        
        # 保存结果
        self.results = {
            'best_C': best_C,
            'best_cv_score': float(best_score),
            'test_accuracy': float(accuracy),
            'test_auc': float(auc),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'data_statistics': {
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'feature_dim_original': self.feature_processor.feature_dim if self.feature_processor else None,
                'feature_dim_pca': X.shape[1],
                'positive_ratio': float(np.mean(y))
            }
        }
        
        # 分析PCA成分的重要性
        self._analyze_pca_components(X_train)
        
        return self
    
    def _analyze_pca_components(self, X_train: np.ndarray):
        """分析PCA成分的重要性"""
        if self.classifier is None:
            return
        
        weights = self.classifier.coef_[0]
        
        # 找出最重要的PCA成分
        top_indices = np.argsort(np.abs(weights))[-20:][::-1]
        
        self.logger.info(f"\n最重要的PCA成分 (前20个):")
        for i, idx in enumerate(top_indices[:20]):
            weight = weights[idx]
            if hasattr(self.feature_processor.pca, 'explained_variance_ratio_'):
                var_ratio = self.feature_processor.pca.explained_variance_ratio_[idx]
                self.logger.info(f"  成分 {idx:3d}: 权重={weight:8.4f}, 解释方差={var_ratio:.6f}")
            else:
                self.logger.info(f"  成分 {idx:3d}: 权重={weight:8.4f}")
        
        self.results['top_pca_components'] = [
            {
                'component': int(idx), 
                'weight': float(weights[idx]), 
                'variance_ratio': float(self.feature_processor.pca.explained_variance_ratio_[idx]) 
                if hasattr(self.feature_processor.pca, 'explained_variance_ratio_') else None
            }
            for idx in top_indices[:20]
        ]
    
    def save(self):
        """保存模型"""
        if self.classifier is None or self.feature_processor is None:
            raise ValueError("请先训练模型")
        
        # 保存完整模型
        model_data = {
            'classifier': self.classifier,
            'scaler': self.feature_processor.scaler,
            'pca': self.feature_processor.pca,
            'results': self.results,
            'config': {
                'features_file': self.config.features_file,
                'pca_components': self.config.pca_components,
                'test_size': self.config.test_size,
                'random_seed': self.config.random_seed,
                'cv_folds': self.config.cv_folds,
                'use_scaling': self.config.use_scaling,
                'max_samples': self.config.max_samples
            }
        }
        
        model_path = os.path.join(self.config.experiment_dir, "pca_classifier.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 保存结果为JSON
        results_path = os.path.join(self.config.experiment_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"模型保存到: {model_path}")
        self.logger.info(f"结果保存到: {results_path}")
        
        return model_path

# ==================== 主训练流程 ====================
class PCATrainer:
    """PCA训练器 - 流式加载版本"""
    
    def __init__(self, config: PCATrainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件
        self.feature_processor = PCAFeatureProcessor(config)
        self.classifier = PCAClassifier(config)
        
    def run(self):
        """运行PCA训练流程"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons PCA降维训练管道")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 阶段1: 流式加载特征并PCA降维
            self.logger.info("\n[阶段1/3] 流式加载特征并PCA降维")
            
            # 设置feature_processor引用
            self.classifier.feature_processor = self.feature_processor
            
            # 加载特征并降维
            X_pca, y = self.feature_processor.load_features_streaming(self.config.features_file)
            
            # 阶段2: 训练分类器
            self.logger.info("\n[阶段2/3] 训练分类器")
            self.classifier.train(X_pca, y)
            
            # 阶段3: 保存模型
            self.logger.info("\n[阶段3/3] 保存模型")
            model_path = self.classifier.save()
            
            # 生成报告
            total_time = time.time() - start_time
            self._generate_report(total_time)
            
            return self.classifier, model_path
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _generate_report(self, total_time: float):
        """生成训练报告"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        results = self.classifier.results
        
        report = f"""
{'='*60}
H-Neurons PCA训练完成报告
{'='*60}

方法说明:
  - 特征加载: 流式处理，避免内存溢出
  - 降维方法: PCA线性降维
  - 原始维度: {results['data_statistics']['feature_dim_original']}
  - PCA维度: {self.config.pca_components}
  - 解释方差: {self.feature_processor.pca.explained_variance_ratio_.sum():.4f}
  - 标准化: {'是' if self.config.use_scaling else '否'}

数据统计:
  总样本数: {results['data_statistics']['train_samples'] + results['data_statistics']['test_samples']}
  训练集样本: {results['data_statistics']['train_samples']}
  测试集样本: {results['data_statistics']['test_samples']}
  正样本比例: {results['data_statistics']['positive_ratio']:.3f}

模型性能:
  最佳C值: {results['best_C']}
  交叉验证AUC: {results['best_cv_score']:.4f}
  测试集准确率: {results['test_accuracy']:.4f}
  测试集AUC: {results['test_auc']:.4f}

训练信息:
  开始时间: {datetime.fromtimestamp(time.time() - total_time).strftime('%Y-%m-%d %H:%M:%S')}
  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  总耗时: {hours}小时{minutes}分钟{seconds}秒

输出目录: {self.config.experiment_dir}
  1. pca_classifier.pkl - 训练好的PCA分类器
  2. results.json - 完整实验结果
  3. pca_info.json - PCA相关信息
  4. training_report.txt - 本报告

{'='*60}
"""
        
        self.logger.info(report)
        
        # 保存报告到文件
        report_path = os.path.join(self.config.experiment_dir, "training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

# ==================== 主函数 ====================
def main():
    """主函数"""
    # 设置配置
    config = PCATrainConfig(
        features_file="/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl",
        pca_components=1000,  # 降到1000维
        max_samples=5000,     # 可选：限制样本数量
        experiment_name="h_neurons_pca_demo"
    )
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.experiment_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    
    # 创建训练器
    trainer = PCATrainer(config)
    
    # 运行训练
    classifier, model_path = trainer.run()
    
    print("\n" + "="*60)
    print("PCA训练完成!")
    print("="*60)
    print(f"输出目录: {config.experiment_dir}")
    print(f"模型文件: {model_path}")
    
    return classifier, model_path

if __name__ == "__main__":
    try:
        print("="*60)
        print("开始H-Neurons PCA训练...")
        print("="*60)
        
        classifier, model_path = main()
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()