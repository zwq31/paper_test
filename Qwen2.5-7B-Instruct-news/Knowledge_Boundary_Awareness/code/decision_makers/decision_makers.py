import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from typing import List, Tuple, Dict
import pickle
import json

# ==================== 数据加载工具 ====================

def load_json_data(file_path: str, score_threshold: int = 4):
    """
    加载 JSON 数据并提取特征和标签
    
    参数:
        file_path: JSON 文件路径
        score_threshold: 分数阈值，大于等于该值视为正确回答
        
    返回:
        X: [N, 3] 特征矩阵 (PPL, Min_Prob, Confidence)
        y: [N] 标签向量 (1=正确, 0=错误)
        data: 原始数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果数据是单个字典对象，转换为列表
    if isinstance(data, dict):
        data = [data]
    
    X = []
    y = []
    
    for item in data:
        # 提取特征
        refusal = item.get('refusal_decision', {})
        ppl = refusal.get('ppl', 0.0)
        min_prob = refusal.get('min_token_prob', 0.0)
        confidence = refusal.get('self_reported_confidence', 0.0)
        
        # 提取标签：根据 correctness.score 判断是否正确
        score = item.get('evaluation_results', {}).get('correctness', {}).get('score', 0)
        label = 1 if score >= score_threshold else 0
        
        X.append([ppl, min_prob, confidence])
        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"从 {file_path} 加载了 {len(y)} 条样本")
    print(f"  - 正确 (label=1): {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"  - 错误 (label=0): {len(y)-np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")
    
    return X, y, data


def create_enhanced_features(X: np.ndarray) -> np.ndarray:
    """
    创建增强特征（包括原始特征、对数变换、交互项、平方项等）
    
    参数:
        X: [N, 3] 基础特征 (PPL, Min_Prob, Confidence)
        
    返回:
        X_enhanced: [N, 13] 增强后的特征矩阵
    """
    ppl = X[:, 0]
    min_prob = X[:, 1]
    confidence = X[:, 2]
    
    features = [
        ppl,                                    # 0: PPL（困惑度）
        min_prob,                               # 1: 最小 token 概率
        confidence,                             # 2: 自报告置信度
        np.log(ppl + 1e-6),                    # 3: PPL 的对数（加小常数防零）
        ppl * min_prob,                         # 4: PPL 与最小概率的乘积
        min_prob * confidence,                  # 5: 最小概率与置信度的乘积
        ppl * confidence,                       # 6: PPL 与置信度的乘积
        confidence / (ppl + 1e-6),             # 7: 置信度 / PPL
        min_prob / (ppl + 1e-6),               # 8: 最小概率 / PPL
        ppl ** 2,                               # 9: PPL 的平方
        min_prob ** 2,                          # 10: 最小概率的平方
        confidence ** 2,                        # 11: 置信度的平方
        ppl * min_prob * confidence,            # 12: 三者交互项
    ]
    
    X_enhanced = np.column_stack(features)
    return X_enhanced.astype(np.float32)


# ==================== 方案1: MLP 决策器 ====================

class MLPDecisionMaker(nn.Module):
    """基于多层感知机（MLP）的决策器"""
    
    def __init__(self, input_dim: int = 13, hidden_dims: List[int] = [64, 32, 16], 
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层：线性层 + BatchNorm + ReLU + Dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层：线性层 + Sigmoid（用于二分类概率输出）
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class MLPDetector:
    """MLP 决策器的封装类，提供训练、预测、保存/加载接口"""
    
    def __init__(self, input_dim: int = 13, hidden_dims: List[int] = [64, 32, 16],
                 dropout: float = 0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MLPDecisionMaker(input_dim, hidden_dims, dropout).to(self.device)
        self.scaler = StandardScaler()  # 用于特征标准化
        self.threshold = 0.5  # 分类阈值
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 100, batch_size: int = 32, lr: float = 0.001,
            verbose: bool = True):
        """训练模型"""
        
        # 对输入特征进行标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 转换为 PyTorch 张量
        X_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss()  # 二分类交叉熵
        
        # 开始训练
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 每10轮打印一次损失
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"第 {epoch+1}/{epochs} 轮, 损失: {avg_loss:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测为正类（正确）的概率"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy().flatten()
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别（0 或 1）"""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
    
    def save(self, path: str):
        """保存模型（包括网络权重、标准化器、阈值）"""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold
        }, path)
    
    def load(self, path: str):
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']


# ==================== 方案2: 注意力机制决策器 ====================

class AttentionDecisionMaker(nn.Module):
    """基于注意力机制的决策器"""
    
    def __init__(self, feature_dim: int = 13, hidden_dim: int = 64):
        super().__init__()
        
        # 特征编码器：将原始特征映射到高维空间
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多头自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 分类头：将注意力输出映射为概率
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 编码输入特征
        encoded = self.feature_encoder(x)  # [B, hidden_dim]
        
        # 添加虚拟序列维度（因为注意力机制需要序列输入）
        encoded = encoded.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 应用自注意力
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # 移除序列维度（池化）
        pooled = attended.squeeze(1)  # [B, hidden_dim]
        
        # 分类输出
        output = self.classifier(pooled)
        
        return output


class AttentionDetector:
    """注意力机制决策器的封装类"""
    
    def __init__(self, feature_dim: int = 13, hidden_dim: int = 64):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AttentionDecisionMaker(feature_dim, hidden_dim).to(self.device)
        self.scaler = StandardScaler()
        self.threshold = 0.5
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            epochs: int = 100, batch_size: int = 32, lr: float = 0.001,
            verbose: bool = True):
        """训练模型"""
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 转为张量
        X_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"第 {epoch+1}/{epochs} 轮, 损失: {avg_loss:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测正类概率"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy().flatten()
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device,weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint['threshold']


# ==================== 方案3: 集成学习决策器 ====================

class EnsembleDetector:
    """集成学习决策器（结合随机森林、XGBoost、逻辑回归、梯度提升）"""
    
    def __init__(self):
        # 定义多个基础分类器
        self.classifiers = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'lr': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        # 元学习器：用于融合基础分类器的预测结果
        self.meta_learner = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.threshold = 0.5
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        """训练集成模型（两阶段训练）"""
        
        # 标准化输入特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 第一阶段：训练所有基础分类器
        if verbose:
            print("\n正在训练基础分类器...")
        
        base_predictions = []
        for name, clf in self.classifiers.items():
            if verbose:
                print(f"  正在训练 {name}...")
            clf.fit(X_train_scaled, y_train)
            pred_proba = clf.predict_proba(X_train_scaled)[:, 1]  # 获取正类概率
            base_predictions.append(pred_proba)
        
        # 第二阶段：用基础分类器的预测结果作为新特征，训练元学习器
        if verbose:
            print("  正在训练元学习器...")
        X_meta = np.column_stack(base_predictions)  # [N, 4]
        self.meta_learner.fit(X_meta, y_train)
        
        if verbose:
            print("训练完成！")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """使用集成模型预测正类概率"""
        X_scaled = self.scaler.transform(X)
        
        # 获取每个基础分类器的预测概率
        base_predictions = []
        for clf in self.classifiers.values():
            pred_proba = clf.predict_proba(X_scaled)[:, 1]
            base_predictions.append(pred_proba)
        
        # 用元学习器融合预测
        X_meta = np.column_stack(base_predictions)
        final_proba = self.meta_learner.predict_proba(X_meta)[:, 1]
        
        return final_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别（0 或 1）"""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """获取部分模型的特征重要性（如 RF 和 XGBoost）"""
        importance_dict = {}
        
        if hasattr(self.classifiers['rf'], 'feature_importances_'):
            importance_dict['rf'] = self.classifiers['rf'].feature_importances_
        
        if hasattr(self.classifiers['xgb'], 'feature_importances_'):
            importance_dict['xgb'] = self.classifiers['xgb'].feature_importances_
        
        return importance_dict
    
    def save(self, path: str):
        """保存整个集成模型（使用 pickle）"""
        with open(path, 'wb') as f:
            pickle.dump({
                'classifiers': self.classifiers,
                'meta_learner': self.meta_learner,
                'scaler': self.scaler,
                'threshold': self.threshold
            }, f)
    
    def load(self, path: str):
        """从文件加载集成模型"""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
            self.classifiers = checkpoint['classifiers']
            self.meta_learner = checkpoint['meta_learner']
            self.scaler = checkpoint['scaler']
            self.threshold = checkpoint['threshold']