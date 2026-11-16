"""
H-Neurons 稀疏自编码器训练管道
在原有代码基础上修改，使用稀疏自编码器降维
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==================== 稀疏自编码器模型 ====================
class SparseAutoencoder(nn.Module):
    """
    稀疏自编码器 - 严格按照论文需求设计
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_target: float = 0.01):
        super(SparseAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        
        self.hidden_dim = hidden_dim
        self.sparsity_target = sparsity_target
        
    def forward(self, x):
        # 编码
        hidden = self.encoder(x)
        
        # 计算稀疏损失（KL散度）
        if self.training:
            self.sparsity_loss = self._sparsity_penalty(hidden)
        else:
            self.sparsity_loss = 0
        
        # 解码
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden
    
    def _sparsity_penalty(self, hidden):
        """基于KL散度的稀疏惩罚"""
        batch_size = hidden.size(0)
        
        # 计算每个隐层神经元的平均激活
        rho_hat = torch.mean(hidden, dim=0)
        
        # KL散度
        rho = torch.tensor([self.sparsity_target], device=hidden.device)
        
        # 避免log(0)
        epsilon = 1e-8
        rho_hat = torch.clamp(rho_hat, epsilon, 1-epsilon)
        
        kl = rho * torch.log(rho / rho_hat) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        
        return torch.sum(kl)
    
    def encode(self, x):
        """仅编码，用于特征提取"""
        with torch.no_grad():
            return self.encoder(x)

# ==================== 配置 ====================
@dataclass
class SAETrainConfig:
    # 数据路径
    features_file: str = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
    output_dir: str = "/mnt/project/H-Neurous/output/sae_results"
    
    # 自编码器参数
    hidden_dim: int = 1000  # 隐藏层维度
    sparsity_target: float = 0.01  # 稀疏目标（1%激活）
    ae_epochs: int = 30
    ae_batch_size: int = 1
    ae_learning_rate: float = 0.001
    sparsity_weight: float = 0.2  # 稀疏惩罚权重
    
    # 分类器参数
    random_seed: int = 42
    test_size: float = 0.15
    cv_folds: int = 5
    C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    
    # 特征处理参数
    use_scaling: bool = True
    max_samples: Optional[int] = None  # 可选：限制样本数量
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 实验名称
    experiment_name: str = "h_neurons_sae"
    
    def __post_init__(self):
        # 创建目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

# ==================== SAE特征处理器（流式加载） ====================
class SAEFeatureProcessor:
    """
    SAE特征处理器 - 流式加载版本
    保持与原有PaperFeatureProcessor相同的接口
    """
    
    def __init__(self, config: SAETrainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件
        self.scaler = StandardScaler() if config.use_scaling else None
        self.autoencoder = None
        
        # 缓存
        self.all_features = []
        self.all_labels = []
        self.feature_dim = None
        
    def load_features_streaming(self, features_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        流式加载特征文件并构建训练数据 - 按照论文标记策略
        先收集所有特征，用于训练自编码器
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
        self.logger.info(f"SAE隐藏层维度: {self.config.hidden_dim}")
        self.logger.info(f"压缩比: {self.feature_dim/self.config.hidden_dim:.1f}x")
        
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
        
        return X_original, y
    
    def train_autoencoder(self, X_original: np.ndarray):
        """训练稀疏自编码器"""
        self.logger.info(f"\n训练稀疏自编码器...")
        self.logger.info(f"输入维度: {X_original.shape[1]}")
        self.logger.info(f"隐藏层维度: {self.config.hidden_dim}")
        self.logger.info(f"稀疏目标: {self.config.sparsity_target}")
        self.logger.info(f"训练轮次: {self.config.ae_epochs}")
        self.logger.info(f"设备: {self.config.device}")
        
        # 1. 标准化（可选）
        if self.scaler is not None:
            self.logger.info("标准化特征...")
            X_scaled = self.scaler.fit_transform(X_original)
        else:
            X_scaled = X_original
        
        # 2. 创建自编码器
        self.autoencoder = SparseAutoencoder(
            input_dim=X_scaled.shape[1],
            hidden_dim=self.config.hidden_dim,
            sparsity_target=self.config.sparsity_target
        ).to(self.config.device)
        
        # 3. 准备数据加载器
        train_tensor = torch.FloatTensor(X_scaled).to(self.config.device)
        train_dataset = TensorDataset(train_tensor, train_tensor)  # 自编码器用输入作为目标
        train_loader = DataLoader(train_dataset, 
                                 batch_size=self.config.ae_batch_size, 
                                 shuffle=True)
        
        # 4. 优化器和损失函数
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.config.ae_learning_rate)
        reconstruction_criterion = nn.MSELoss()
        
        # 5. 训练循环
        train_losses = []
        
        for epoch in range(self.config.ae_epochs):
            self.autoencoder.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 前向传播
                reconstructed, hidden = self.autoencoder(data)
                
                # 计算损失
                recon_loss = reconstruction_criterion(reconstructed, target)
                
                # 总损失 = 重建损失 + 稀疏惩罚
                sparsity_loss = self.autoencoder.sparsity_loss
                total_loss = recon_loss + self.config.sparsity_weight * sparsity_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"轮次 {epoch+1}/{self.config.ae_epochs}, 平均损失: {avg_loss:.6f}")
        
        # 6. 分析稀疏性
        self._analyze_sparsity(X_scaled)
        
        # 7. 保存训练损失
        self.ae_training_losses = train_losses
        
        return self
    
    def _analyze_sparsity(self, X_scaled: np.ndarray):
        """分析自编码器隐层的稀疏性"""
        if self.autoencoder is None:
            return
        
        # 计算隐层激活的稀疏度
        self.autoencoder.eval()
        with torch.no_grad():
            sample_tensor = torch.FloatTensor(X_scaled[:100]).to(self.config.device)
            hidden = self.autoencoder.encode(sample_tensor).cpu().numpy()
        
        # 计算激活率（激活值>阈值的比例）
        threshold = 0.01
        activation_rate = np.mean(hidden > threshold)
        
        self.logger.info(f"\n自编码器隐层稀疏性分析:")
        self.logger.info(f"  激活阈值: {threshold}")
        self.logger.info(f"  平均激活率: {activation_rate:.4f}")
        self.logger.info(f"  稀疏目标: {self.config.sparsity_target}")
        
        self.activation_rate = activation_rate
    
    def extract_features(self, X_original: np.ndarray) -> np.ndarray:
        """使用训练好的自编码器提取特征"""
        if self.autoencoder is None:
            raise ValueError("请先训练自编码器")
        
        # 标准化（如果使用了标准化）
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_original)
        else:
            X_scaled = X_original
        
        # 提取特征
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.config.device)
            X_sae = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        self.logger.info(f"提取特征完成: {X_original.shape[1]} → {X_sae.shape[1]}")
        
        return X_sae
    
    def save_ae_info(self, output_dir: str):
        """保存自编码器相关信息"""
        ae_info = {
            'input_dim': self.feature_dim,
            'hidden_dim': self.config.hidden_dim,
            'sparsity_target': self.config.sparsity_target,
            'activation_rate': self.activation_rate if hasattr(self, 'activation_rate') else None,
            'training_losses': self.ae_training_losses if hasattr(self, 'ae_training_losses') else None
        }
        
        info_path = os.path.join(output_dir, "sae_info.json")
        with open(info_path, 'w') as f:
            json.dump(ae_info, f, indent=2)
        
        # 保存模型
        model_path = os.path.join(output_dir, "sparse_autoencoder.pth")
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'config': {
                'input_dim': self.feature_dim,
                'hidden_dim': self.config.hidden_dim,
                'sparsity_target': self.config.sparsity_target
            }
        }, model_path)
        
        self.logger.info(f"自编码器信息保存到: {info_path}")
        self.logger.info(f"自编码器模型保存到: {model_path}")

# ==================== SAE分类器 ====================
class SAEClassifier:
    """基于SAE降维特征的分类器"""
    
    def __init__(self, config: SAETrainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 模型
        self.classifier = None
        self.feature_processor = None
        
        # 结果
        self.results = {}
        
    def train(self, X_sae: np.ndarray, y: np.ndarray):
        """训练SAE降维后的分类器"""
        self.logger.info(f"训练数据形状: {X_sae.shape}")
        self.logger.info(f"正样本比例: {np.mean(y):.3f}")
        
        # 1. 划分数据集
        self.logger.info("划分数据集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_sae, y,
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
                'feature_dim_sae': X_sae.shape[1],
                'positive_ratio': float(np.mean(y)),
                'sae_activation_rate': self.feature_processor.activation_rate if self.feature_processor else None
            }
        }
        
        return self
    
    def save(self):
        """保存模型"""
        if self.classifier is None or self.feature_processor is None:
            raise ValueError("请先训练模型")
        
        # 保存完整模型
        model_data = {
            'classifier': self.classifier,
            'scaler': self.feature_processor.scaler,
            'autoencoder_state_dict': self.feature_processor.autoencoder.state_dict() if self.feature_processor.autoencoder else None,
            'results': self.results,
            'config': {
                'input_dim': self.feature_processor.feature_dim,  # 添加输入维度
                'features_file': self.config.features_file,
                'hidden_dim': self.config.hidden_dim,
                'sparsity_target': self.config.sparsity_target,
                'test_size': self.config.test_size,
                'random_seed': self.config.random_seed,
                'cv_folds': self.config.cv_folds,
                'use_scaling': self.config.use_scaling,
                'max_samples': self.config.max_samples,
                'device': self.config.device
            }
        }
        
        model_path = os.path.join(self.config.experiment_dir, "sae_classifier.pkl")
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
class SAETrainer:
    """SAE训练器 - 流式加载版本"""
    
    def __init__(self, config: SAETrainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件
        self.feature_processor = SAEFeatureProcessor(config)
        self.classifier = SAEClassifier(config)
        
    def run(self):
        """运行SAE训练流程"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons 稀疏自编码器训练管道")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 阶段1: 流式加载特征
            self.logger.info("\n[阶段1/4] 流式加载特征")
            
            # 设置feature_processor引用
            self.classifier.feature_processor = self.feature_processor
            
            # 加载特征
            X_original, y = self.feature_processor.load_features_streaming(self.config.features_file)
            
            # 阶段2: 训练稀疏自编码器
            self.logger.info("\n[阶段2/4] 训练稀疏自编码器")
            self.feature_processor.train_autoencoder(X_original)
            
            # 阶段3: 提取降维特征并训练分类器
            self.logger.info("\n[阶段3/4] 提取降维特征并训练分类器")
            X_sae = self.feature_processor.extract_features(X_original)
            self.classifier.train(X_sae, y)
            
            # 阶段4: 保存模型
            self.logger.info("\n[阶段4/4] 保存模型")
            model_path = self.classifier.save()
            self.feature_processor.save_ae_info(self.config.experiment_dir)
            
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
H-Neurons SAE训练完成报告
{'='*60}

方法说明:
  - 特征加载: 流式处理，避免内存溢出
  - 降维方法: 稀疏自编码器（非线性）
  - 原始维度: {results['data_statistics']['feature_dim_original']}
  - SAE隐藏层维度: {self.config.hidden_dim}
  - 压缩比: {results['data_statistics']['feature_dim_original']/self.config.hidden_dim:.1f}x
  - 稀疏目标: {self.config.sparsity_target}
  - 实际激活率: {results['data_statistics']['sae_activation_rate']:.4f}
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
  1. sae_classifier.pkl - 训练好的SAE分类器
  2. sparse_autoencoder.pth - 稀疏自编码器模型
  3. results.json - 完整实验结果
  4. sae_info.json - SAE相关信息
  5. training_report.txt - 本报告

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
    config = SAETrainConfig(
        features_file="/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl",
        hidden_dim=500,  # 降到1000维
        sparsity_target=0.01,  # 1%稀疏度
        ae_epochs=30,
        max_samples=5000,  # 可选：限制样本数量
        experiment_name="h_neurons_sae_demo"
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
    trainer = SAETrainer(config)
    
    # 运行训练
    classifier, model_path = trainer.run()
    
    print("\n" + "="*60)
    print("SAE训练完成!")
    print("="*60)
    print(f"输出目录: {config.experiment_dir}")
    print(f"模型文件: {model_path}")
    
    return classifier, model_path

if __name__ == "__main__":
    try:
        print("="*60)
        print("开始H-Neurons SAE训练...")
        print("="*60)
        
        classifier, model_path = main()
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()