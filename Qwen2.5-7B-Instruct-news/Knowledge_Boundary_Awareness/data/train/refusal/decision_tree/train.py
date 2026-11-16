"""
知识边界识别模块 - 基于决策树的完整训练流程
根据score字段构建标签：score >= 3为正确（应该回答），score < 3为错误（应该拒绝）
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
import pickle

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class BoundaryDetectionDataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = None
        self.df = None
        self.feature_names = None
        
    def load_data(self) -> List[Dict]:
        """加载JSON数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ 数据加载成功，共 {len(self.data)} 条样本")
        return self.data
    
    def extract_features_and_labels(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        提取特征和标签
        标签定义：score >= 3为正确（应该回答），score < 3为错误（应该拒绝）
        """
        if self.data is None:
            self.load_data()
        
        features = []
        labels = []
        sample_ids = []
        
        for idx, item in enumerate(self.data):
            try:
                # 提取基础特征
                rd = item.get('refusal_decision', {})
                
                # 三个核心特征
                ppl = rd.get('ppl', 0.0)
                min_token_prob = rd.get('min_token_prob', 0.0)
                confidence = rd.get('self_reported_confidence', 0.0)
                
                # 获取score字段
                score = item.get('evaluation_results', {}).get('correctness', {}).get('score', 0)
                
                # 构建标签：score >= 3为1（正确/应该回答），否则为0（错误/应该拒绝）
                if score == 3:
                    continue  # 或跳过该样本
                label = 1 if score > 3 else 0
                
                # 基本特征向量
                feature_vector = [
                    ppl,
                    min_token_prob,
                    confidence
                ]
                
                # 添加衍生特征（可选）
                # 1. 困惑度的对数变换
                log_ppl = np.log(ppl + 1e-6) if ppl > 0 else -10
                
                # 2. 最小词元概率的对数变换
                log_min_prob = np.log(min_token_prob + 1e-6) if min_token_prob > 0 else -10
                
                # 3. 困惑度与置信度的比率
                ppl_confidence_ratio = ppl / (confidence + 1e-6) if confidence > 0 else 1000
                
                # 4. 置信度与最小概率的乘积
                conf_min_prob = confidence * min_token_prob
                
                # 5. 困惑度与最小概率的比值
                ppl_min_prob_ratio = ppl / (min_token_prob + 1e-6) if min_token_prob > 0 else 1000
                
                # 6. 三个特征的加权组合
                weighted_score = 0.3 * ppl + 0.5 * min_token_prob + 0.2 * confidence
                
                # 组合所有特征
                full_feature_vector = feature_vector + [
                    log_ppl,
                    log_min_prob,
                    ppl_confidence_ratio,
                    conf_min_prob,
                    ppl_min_prob_ratio,
                    weighted_score
                ]
                
                features.append(full_feature_vector)
                labels.append(label)
                sample_ids.append(item.get('source_id', idx))
                
            except Exception as e:
                print(f"⚠️ 处理第{idx}条数据时出错: {e}")
                continue
        
        # 创建特征名称列表
        self.feature_names = [
            'ppl', 'min_token_prob', 'confidence',
            'log_ppl', 'log_min_prob', 'ppl_confidence_ratio',
            'conf_min_prob', 'ppl_min_prob_ratio', 'weighted_score'
        ]
        
        self.df = pd.DataFrame(features, columns=self.feature_names)
        self.df['label'] = labels
        self.df['sample_id'] = sample_ids
        self.df['original_score'] = [item.get('evaluation_results', {}).get('correctness', {}).get('score', 0) 
                                     for item in self.data[:len(self.df)]]
        
        print(f"✅ 特征提取完成，共 {len(self.df)} 条有效样本")
        print(f"📊 类别分布:")
        print(f"   正样本（score >= 3，应该回答）: {self.df['label'].sum()} ({self.df['label'].mean()*100:.1f}%)")
        print(f"   负样本（score < 3，应该拒绝）: {len(self.df) - self.df['label'].sum()} ({100 - self.df['label'].mean()*100:.1f}%)")
        
        print(f"\n📈 Score分布统计:")
        print(self.df['original_score'].describe())
        print(f"\nScore分布:")
        for score in range(0, 6):
            count = (self.df['original_score'] == score).sum()
            print(f"  Score {score}: {count}条 ({count/len(self.df)*100:.1f}%)")
        
        return self.df[self.feature_names], self.df['label']
    
    def explore_data(self):
        """数据探索性分析"""
        if self.df is None:
            print("请先执行extract_features_and_labels")
            return
        
        # 基本统计信息
        print("\n📈 特征统计信息:")
        print(self.df.describe())
        
        # 特征相关性
        print("\n📊 特征相关性（与标签）:")
        correlations = self.df.corr()['label'].sort_values(ascending=False)
        print(correlations)
        
        # 可视化特征分布
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names[:9]):
            ax = axes[i]
            for label in [0, 1]:
                subset = self.df[self.df['label'] == label]
                ax.hist(subset[feature], alpha=0.5, label=f'Label {label}', bins=30)
            ax.set_xlabel(feature)
            ax.set_ylabel('频数')
            ax.legend()
            ax.set_title(f'{feature} 分布')
        
        plt.tight_layout()
        plt.savefig('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/image1/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 相关系数热力图
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df[self.feature_names + ['label']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/image1/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df

class DecisionTreeBoundaryDetector:
    """决策树边界检测器"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.model = None
        self.scaler = None
        self.best_params = None
        self.cv_results = None
        
    def create_pipeline(self, model_type='decision_tree', **kwargs):
        """创建预处理和模型管道"""
        if model_type == 'decision_tree':
            model = DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',  # 处理类别不平衡
                **kwargs
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                **kwargs
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 创建管道：标准化 + 模型
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        return pipeline
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   model_type='decision_tree', use_grid_search=True):
        """训练模型"""
        
        if use_grid_search:
            # 网格搜索优化超参数
            if model_type == 'decision_tree':
                param_grid = {
                    'model__max_depth': [3, 5, 7, 10, 15, None],
                    'model__min_samples_split': [2, 5, 10, 20],
                    'model__min_samples_leaf': [1, 2, 5, 10],
                    'model__criterion': ['gini', 'entropy'],
                    'model__max_features': ['sqrt', 'log2', None]
                }
            elif model_type == 'random_forest':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [5, 10, 15, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt', 'log2']
                }
            elif model_type == 'gradient_boosting':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            
            # 创建基础管道
            pipeline = self.create_pipeline(model_type)
            
            # 使用F1分数作为评估指标（适合不平衡数据）
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=5,  # 5折交叉验证
                scoring='f1_macro',  # 使用宏平均F1
                n_jobs=-1,
                verbose=1
            )
            
            print(f"🔍 开始网格搜索优化 {model_type}...")
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_results = grid_search.cv_results_
            
            print(f"✅ 最佳参数: {grid_search.best_params_}")
            print(f"✅ 最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
            
        else:
            # 使用默认参数
            self.model = self.create_pipeline(model_type)
            self.model.fit(X_train, y_train)
        
        # 如果提供验证集，计算验证集性能
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred, average='macro')
            print(f"✅ 验证集F1分数: {val_f1:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, save_plots=True):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算各项指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print("\n" + "="*60)
        print("📊 模型评估结果")
        print("="*60)
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率 (宏平均): {precision:.4f}")
        print(f"召回率 (宏平均): {recall:.4f}")
        print(f"F1分数 (宏平均): {f1:.4f}")
        
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=['应该拒绝', '应该回答']))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)
        
        if save_plots:
            # 可视化评估结果
            self.plot_evaluation_results(X_test, y_test, y_pred, y_pred_proba)
        
        # 特征重要性（如果是树模型）
        if hasattr(self.model.named_steps['model'], 'feature_importances_'):
            self.plot_feature_importance()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_evaluation_results(self, X_test, y_test, y_pred, y_pred_proba):
        """可视化评估结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['应该拒绝', '应该回答'])
        disp.plot(ax=ax1, cmap='Blues')
        ax1.set_title('混淆矩阵')
        
        # 2. ROC曲线
        ax2 = axes[0, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('假正率')
        ax2.set_ylabel('真正率')
        ax2.set_title('ROC曲线')
        ax2.legend(loc="lower right")
        
        # 3. 精确率-召回率曲线
        ax3 = axes[1, 0]
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        ax3.plot(recall_vals, precision_vals, color='green', lw=2)
        ax3.set_xlabel('召回率')
        ax3.set_ylabel('精确率')
        ax3.set_title('精确率-召回率曲线')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        
        # 4. 特征重要性（如果是决策树）
        ax4 = axes[1, 1]
        if hasattr(self.model.named_steps['model'], 'feature_importances_'):
            importances = self.model.named_steps['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax4.bar(range(len(importances)), importances[indices])
            ax4.set_xlabel('特征')
            ax4.set_ylabel('重要性')
            ax4.set_title('特征重要性')
            ax4.set_xticks(range(len(importances)))
            ax4.set_xticklabels([self.feature_names[i] for i in indices], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/image1/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. 决策边界可视化（仅使用前两个特征）
        if len(self.feature_names) >= 2:
            self.plot_decision_boundary(X_test, y_test, self.feature_names[0], self.feature_names[1])
    
    def plot_feature_importance(self):
        """绘制特征重要性图"""
        if hasattr(self.model.named_steps['model'], 'feature_importances_'):
            importances = self.model.named_steps['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("特征重要性")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlim([-1, len(importances)])
            plt.tight_layout()
            plt.savefig('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/image1/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 打印特征重要性
            print("\n📊 特征重要性排序:")
            for i in indices:
                print(f"  {self.feature_names[i]}: {importances[i]:.4f}")
    
    def plot_decision_boundary(self, X, y, feature1_idx, feature2_idx):
        """绘制决策边界（仅使用两个特征）"""
        if isinstance(feature1_idx, str):
            feature1_idx = self.feature_names.index(feature1_idx)
        if isinstance(feature2_idx, str):
            feature2_idx = self.feature_names.index(feature2_idx)
        
        # 提取两个特征
        X_2d = X[:, [feature1_idx, feature2_idx]]
        
        # 训练一个简化模型用于可视化
        simple_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        simple_model.fit(X_2d, y)
        
        # 创建网格
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # 预测网格点
        Z = simple_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        
        # 绘制数据点
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors='k', 
                            cmap='RdYlBu', s=50)
        
        plt.xlabel(self.feature_names[feature1_idx])
        plt.ylabel(self.feature_names[feature2_idx])
        plt.title(f"决策边界: {self.feature_names[feature1_idx]} vs {self.feature_names[feature2_idx]}")
        plt.colorbar(scatter, label='类别')
        plt.tight_layout()
        plt.savefig('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/image1/decision_boundary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_decision_tree(self, max_depth=3):
        """可视化决策树"""
        if self.model is None or not isinstance(self.model.named_steps['model'], DecisionTreeClassifier):
            print("无法可视化非决策树模型")
            return
        
        # 创建一个简化版本用于可视化
        plt.figure(figsize=(20, 10))
        
        plot_tree(
            self.model.named_steps['model'],
            feature_names=self.feature_names,
            class_names=['应该拒绝', '应该回答'],
            filled=True,
            rounded=True,
            max_depth=max_depth,
            fontsize=10
        )
        
        plt.title("决策树结构")
        plt.tight_layout()
        plt.savefig('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/image1/decision_tree.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 导出文本表示
        tree_text = export_text(
            self.model.named_steps['model'],
            feature_names=self.feature_names,
            max_depth=max_depth
        )
        print("\n决策树规则（文本表示）:")
        print(tree_text)
    
    def save_model(self, model_path='boundary_detector.pkl'):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有训练好的模型可以保存")
        
        # 保存整个对象
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'best_params': self.best_params
            }, f)
        
        print(f"✅ 模型已保存到 {model_path}")
    
    def load_model(self, model_path='boundary_detector.pkl'):
        """加载模型"""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.model = saved_data['model']
        self.feature_names = saved_data['feature_names']
        self.best_params = saved_data.get('best_params')
        
        print(f"✅ 模型已从 {model_path} 加载")
        return self
    
    def predict(self, features):
        """预测单个样本"""
        if self.model is None:
            raise ValueError("请先加载或训练模型")
        
        # 确保特征顺序正确
        if isinstance(features, dict):
            # 如果是字典，转换为正确顺序的数组
            feature_array = [features.get(name, 0) for name in self.feature_names]
        elif isinstance(features, list):
            # 如果是列表，直接使用
            feature_array = features
        else:
            raise ValueError("特征必须是字典或列表")
        
        # 转换为二维数组
        X = np.array([feature_array])
        
        # 预测
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        return {
            'prediction': prediction,  # 0: 应该拒绝, 1: 应该回答
            'probability': proba[1],   # 属于"应该回答"的概率
            'class': '应该回答' if prediction == 1 else '应该拒绝',
            'probabilities': proba.tolist()
        }

def main():
    """主函数：完整的训练流程"""
    
    # ========== 1. 加载和准备数据 ==========
    print("="*60)
    print("第1步：加载和准备数据")
    print("="*60)
    
    data_loader = BoundaryDetectionDataLoader('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/filtered_evaluation.json')  # 替换为你的文件路径
    X, y = data_loader.extract_features_and_labels()
    
    # 探索数据
    data_loader.explore_data()
    
    # ========== 2. 划分训练集和测试集 ==========
    print("\n" + "="*60)
    print("第2步：划分训练集和测试集")
    print("="*60)
    
    # 保持类别分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"测试集类别分布: {np.bincount(y_test)}")
    
    # ========== 3. 训练决策树模型 ==========
    print("\n" + "="*60)
    print("第3步：训练决策树模型")
    print("="*60)
    
    detector = DecisionTreeBoundaryDetector(data_loader.feature_names)
    
    # 训练模型（使用网格搜索优化）
    model = detector.train_model(
        X_train.values, y_train.values,
        model_type='decision_tree',  # 也可以尝试 'random_forest' 或 'gradient_boosting'
        use_grid_search=True
    )
    
    # ========== 4. 评估模型 ==========
    print("\n" + "="*60)
    print("第4步：评估模型")
    print("="*60)
    
    evaluation_results = detector.evaluate_model(X_test.values, y_test.values, save_plots=True)
    
    # ========== 5. 可视化决策树 ==========
    print("\n" + "="*60)
    print("第5步：可视化决策树")
    print("="*60)
    
    detector.plot_decision_tree(max_depth=3)
    
    # ========== 6. 保存模型 ==========
    print("\n" + "="*60)
    print("第6步：保存模型")
    print("="*60)
    
    detector.save_model('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/news_qa_boundary_detector_2400.pkl')
    
   

if __name__ == "__main__":
    main()