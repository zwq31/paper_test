import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

class DecisionModelTrainer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """加载预处理后的数据"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 提取特征和标签
        features = []
        labels = []
        
        for sample in self.data:
            features.append([sample['ppl'], sample['min_token_prob']])
            labels.append(sample['label'])
        
        self.X = np.array(features)
        self.y = np.array(labels)
        
        print(f"📥 Data loading completed:")
        print(f"   Total samples: {len(self.data)}")
        print(f"   Feature dimension: {self.X.shape}")
        print(f"   Label distribution - Answer: {sum(self.y == 0)}, Refuse: {sum(self.y == 1)}")
        
        return self.X, self.y
    
    def prepare_training_data(self, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data preparation completed:")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """训练多个分类模型"""
        # 定义模型和参数网格
        models_config = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }
        }
        
        print("🤖 Starting model training...")
        
        best_f1 = 0
        
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            # 使用网格搜索寻找最佳参数
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # 最佳模型
            best_model = grid_search.best_estimator_
            
            # 在测试集上评估
            y_pred = best_model.predict(self.X_test_scaled)
            y_prob = best_model.predict_proba(self.X_test_scaled)[:, 1]
            
            # 计算指标
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            
            # 计算业务指标
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            refusal_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            refusal_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # 存储模型和结果
            self.models[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'refusal_precision': refusal_precision,
                'refusal_recall': refusal_recall,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            print(f"  ✅ {name} training completed")
            print(f"     Best parameters: {grid_search.best_params_}")
            print(f"     Accuracy: {accuracy:.4f}")
            print(f"     F1 Score: {f1:.4f}")
            print(f"     Refusal Precision: {refusal_precision:.4f}")
            print(f"     Refusal Recall: {refusal_recall:.4f}")
            
            # 更新最佳模型
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name
                self.best_model = best_model
        
        print(f"\n🏆 Best Model: {self.best_model_name} (F1: {best_f1:.4f})")
        
        return self.models
    
    def evaluate_models(self):
        """详细评估所有模型"""
        print("=" * 60)
        print("Model Evaluation Report")
        print("=" * 60)
        
        for name, results in self.models.items():
            print(f"\n📊 {name} Detailed Results:")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   F1 Score: {results['f1_score']:.4f}")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall: {results['recall']:.4f}")
            print(f"   Refusal Precision: {results['refusal_precision']:.4f}")
            print(f"   Refusal Recall: {results['refusal_recall']:.4f}")
            
            # 混淆矩阵
            cm = results['confusion_matrix']
            print(f"   Confusion Matrix:")
            print(f"              Predict Answer   Predict Refuse")
            print(f"   Actual Answer    {cm[0,0]:>6}           {cm[0,1]:>6}")
            print(f"   Actual Refuse    {cm[1,0]:>6}           {cm[1,1]:>6}")
    
    def plot_model_comparison(self):
        """绘制模型比较图"""
        # 使用agg backend，避免显示问题
        plt.switch_backend('Agg')
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        model_names = list(self.models.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.models[name][metric] for name in model_names]
            bars = axes[i].bar(model_names, values, alpha=0.7)
            
            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("💾 Model comparison chart saved as 'model_comparison.png'")
        
    def plot_feature_importance(self):
        """绘制特征重要性（针对树模型）"""
        if self.best_model_name in ['RandomForest', 'GradientBoosting']:
            # 使用agg backend
            plt.switch_backend('Agg')
            
            feature_importance = self.best_model.feature_importances_
            feature_names = ['ppl', 'min_token_prob']
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\n🔍 Feature Importance:")
            print(importance_df)
            
            # 绘制特征重要性图
            plt.figure(figsize=(8, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("💾 Feature importance chart saved as 'feature_importance.png'")
            
            # 返回Python原生类型
            return importance_df.to_dict('records')
        
        return None
    
    def save_model(self, output_path='refusal_decision_model.pkl'):
        """保存最佳模型和相关配置"""
        model_info = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_names': ['ppl', 'min_token_prob'],
            'performance': self.models[self.best_model_name],
            'training_data_info': {
                'total_samples': len(self.X),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'class_distribution': {
                    'should_answer': sum(self.y == 0),
                    'should_refuse': sum(self.y == 1)
                }
            },
            'threshold': 0.5  # 默认决策阈值
        }
        
        joblib.dump(model_info, output_path)
        print(f"💾 Model saved to: {output_path}")
        
        # 同时保存所有模型信息 - 使用安全的序列化方法
        all_models_info = {
            'best_model': self.best_model_name,
            'all_models': {},
            'feature_importance': self.plot_feature_importance()
        }
        
        # 处理模型信息，移除不能序列化的对象
        for name, info in self.models.items():
            serializable_info = {}
            for key, value in info.items():
                if key not in ['model', 'predictions', 'probabilities']:  # 移除不能序列化的对象
                    # 使用安全的类型转换方法
                    serializable_info[key] = self.convert_to_serializable(value)
            all_models_info['all_models'][name] = serializable_info
        
        # 保存JSON信息
        with open('all_models_info.json', 'w', encoding='utf-8') as f:
            json.dump(all_models_info, f, ensure_ascii=False, indent=2)
        
        print("💾 Model information saved to: 'all_models_info.json'")
        
        return output_path
    
    def convert_to_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)

# 主训练函数
def train_decision_model():
    """训练决策模型的主函数"""
    # 配置路径
    TRAINING_DATA_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/decision_data/decision_training_data.json"
    MODEL_OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/decision_model/refusal_decision_model.pkl"
    
    # 1. 初始化训练器
    trainer = DecisionModelTrainer(TRAINING_DATA_FILE)
    
    # 2. 加载数据
    X, y = trainer.load_data()
    
    # 3. 准备训练数据
    trainer.prepare_training_data()
    
    # 4. 训练模型
    models = trainer.train_models()
    
    # 5. 评估模型
    trainer.evaluate_models()
    
    # 6. 可视化比较
    trainer.plot_model_comparison()
    
    # 7. 保存模型
    model_path = trainer.save_model(MODEL_OUTPUT_PATH)
    
    print(f"\n🎉 Training completed!")
    print(f"   Best Model: {trainer.best_model_name}")
    print(f"   Test F1 Score: {trainer.models[trainer.best_model_name]['f1_score']:.4f}")
    print(f"   Model File: {model_path}")
    
    return trainer

if __name__ == "__main__":
    trainer = train_decision_model()