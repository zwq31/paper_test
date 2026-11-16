# """
# H-Neurons第二次训练 - 基于全维度特征文件和已识别神经元
# 在1050个已识别神经元上训练，保存时映射回全维度，只保留权重>0的神经元
# """

# import os
# import json
# import pickle
# import logging
# import time
# import gc
# from datetime import datetime
# from typing import List, Dict, Tuple, Optional, Set
# from dataclasses import dataclass, field
# import warnings
# warnings.filterwarnings('ignore')

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler

# # ==================== 配置 ====================
# @dataclass
# class Config:
#     # 数据路径
#     full_dimension_features_path: str = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
    
#     # 第一次训练结果路径
#     first_training_dir: str = "/mnt/project/H-Neurous/output/saga_chat"
#     first_h_neurons_file: str = "h_neurons.json"
#     first_classifier_file: str = "h_neurons_classifier.pkl"
    
#     # 输出目录
#     output_dir: str = "/mnt/project/H-Neurous/output"
    
#     # 训练参数
#     test_size: float = 0.15
#     cv_folds: int = 5
#     random_seed: int = 42
#     C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    
#     # 第二次训练参数
#     weight_threshold: float = 1e-4  # 只保存权重大于这个值的神经元
    
#     # 日志和实验
#     experiment_name: str = "h_neurons_second_training_filtered"
    
#     def __post_init__(self):
#         # 创建目录
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         # 创建实验目录
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
#         os.makedirs(self.experiment_dir, exist_ok=True)
        
#         # 验证文件存在
#         if not os.path.exists(self.full_dimension_features_path):
#             raise FileNotFoundError(f"全维度特征文件不存在: {self.full_dimension_features_path}")
        
#         first_h_neurons_path = os.path.join(self.first_training_dir, self.first_h_neurons_file)
#         if not os.path.exists(first_h_neurons_path):
#             raise FileNotFoundError(f"第一次训练的H-Neurons文件不存在: {first_h_neurons_path}")

# # ==================== 日志配置 ====================
# class LoggerSetup:
#     @staticmethod
#     def setup():
#         log_dir = "/mnt/project/H-Neurous/logs"
#         os.makedirs(log_dir, exist_ok=True)
        
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         log_file = os.path.join(log_dir, f"h_neurons_second_training_{timestamp}.log")
        
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler(log_file),
#                 logging.StreamHandler()
#             ]
#         )
#         return logging.getLogger(__name__)

# # ==================== H-Neurons加载器 ====================
# class HNeuronsLoader:
#     """加载第一次训练得到的H-Neurons，只保留权重>0的"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
    
#     def load_first_training_neurons(self) -> Tuple[List[int], List[str], List[float]]:
#         """
#         加载第一次训练的H-Neurons，只保留权重大于0的
#         返回：(神经元索引列表, 神经元名称列表, 权重列表)
#         """
#         self.logger.info(f"加载第一次训练的H-Neurons从: {self.config.first_training_dir}")
        
#         # 1. 加载H-Neurons列表
#         h_neurons_path = os.path.join(self.config.first_training_dir, self.config.first_h_neurons_file)
#         with open(h_neurons_path, 'r', encoding='utf-8') as f:
#             h_neurons = json.load(f)
        
#         self.logger.info(f"第一次训练共识别 {len(h_neurons)} 个H-Neurons")
        
#         # 2. 只保留权重大于0的神经元
#         positive_neurons = []
#         for neuron in h_neurons:
#             weight = neuron.get('weight', 0.0)
#             if weight > self.config.weight_threshold:
#                 positive_neurons.append(neuron)
        
#         self.logger.info(f"权重大于{self.config.weight_threshold}的神经元: {len(positive_neurons)}")
        
#         # 3. 加载分类器获取全维度神经元名称
#         classifier_path = os.path.join(self.config.first_training_dir, self.config.first_classifier_file)
#         all_neuron_names = []
        
#         if os.path.exists(classifier_path):
#             with open(classifier_path, 'rb') as f:
#                 classifier_data = pickle.load(f)
            
#             all_neuron_names = classifier_data.get('neuron_names', [])
#             self.logger.info(f"从分类器加载 {len(all_neuron_names)} 个全维度神经元名称")
#         else:
#             self.logger.warning("未找到分类器文件，无法获取全维度神经元名称")
        
#         # 4. 解析神经元名称到全维度索引的映射
#         neuron_indices = []
#         neuron_names = []
#         neuron_weights = []
        
#         for neuron in positive_neurons:
#             neuron_name = neuron.get('name', '')
#             neuron_weight = neuron.get('weight', 0.0)
            
#             # 尝试从名称解析索引
#             idx = self._parse_neuron_name_to_index(neuron_name, all_neuron_names)
#             if idx is not None:
#                 neuron_indices.append(idx)
#                 neuron_names.append(neuron_name)
#                 neuron_weights.append(neuron_weight)
        
#         # 去重并排序
#         unique_data = {}
#         for idx, name, weight in zip(neuron_indices, neuron_names, neuron_weights):
#             if idx not in unique_data or weight > unique_data[idx][1]:
#                 unique_data[idx] = (name, weight)
        
#         # 重新整理数据
#         neuron_indices = sorted(unique_data.keys())
#         neuron_names = [unique_data[idx][0] for idx in neuron_indices]
#         neuron_weights = [unique_data[idx][1] for idx in neuron_indices]
        
#         self.logger.info(f"成功映射 {len(neuron_indices)} 个正权重神经元到全维度索引")
        
#         return neuron_indices, neuron_names, neuron_weights
    
#     def _parse_neuron_name_to_index(self, neuron_name: str, all_neuron_names: List[str]) -> Optional[int]:
#         """将神经元名称解析为全维度索引"""
#         # 方法1: 如果在全维度名称列表中，直接使用索引
#         if neuron_name in all_neuron_names:
#             return all_neuron_names.index(neuron_name)
        
#         # 方法2: 尝试解析名称格式 layer_{layer}_neuron_{neuron}
#         try:
#             if neuron_name.startswith('layer_') and '_neuron_' in neuron_name:
#                 parts = neuron_name.split('_')
#                 layer_idx = int(parts[1])
#                 neuron_idx = int(parts[3])
                
#                 # 根据模型结构计算全维度索引
#                 # Qwen2.5-7B: 28层，每层18944个神经元
#                 neurons_per_layer = 18944
#                 calculated_index = layer_idx * neurons_per_layer + neuron_idx
                
#                 # 验证索引在合理范围内
#                 if calculated_index < len(all_neuron_names) or len(all_neuron_names) == 0:
#                     return calculated_index
#                 else:
#                     self.logger.warning(f"计算索引 {calculated_index} 超出范围 {len(all_neuron_names)}")
#         except Exception as e:
#             self.logger.debug(f"无法解析神经元名称 {neuron_name}: {e}")
        
#         return None

# # ==================== 全维度特征处理器 ====================
# class FullDimensionFeatureProcessor:
#     """从全维度特征文件中筛选指定神经元的特征"""
    
#     def __init__(self, config: Config, selected_neuron_indices: List[int]):
#         self.config = config
#         self.selected_indices = sorted(set(selected_neuron_indices))
#         self.logger = logging.getLogger(__name__)
        
#         # 从第一行获取原始特征维度
#         self.original_feature_dim = self._get_original_feature_dim()
#         self.logger.info(f"原始特征维度: {self.original_feature_dim}")
    
#     def _get_original_feature_dim(self) -> int:
#         """从特征文件获取原始特征维度"""
#         try:
#             with open(self.config.full_dimension_features_path, 'r', encoding='utf-8') as f:
#                 first_line = f.readline()
#                 if not first_line:
#                     raise ValueError("特征文件为空")
                
#                 first_sample = json.loads(first_line)
#                 if 'features_answer' not in first_sample:
#                     raise ValueError("特征文件中没有features_answer字段")
                
#                 return len(first_sample['features_answer'])
#         except Exception as e:
#             self.logger.error(f"获取原始特征维度失败: {e}")
#             return 0
    
#     def validate_indices(self) -> List[int]:
#         """验证并筛选有效的神经元索引"""
#         valid_indices = [idx for idx in self.selected_indices if idx < self.original_feature_dim]
        
#         if len(valid_indices) != len(self.selected_indices):
#             invalid_count = len(self.selected_indices) - len(valid_indices)
#             self.logger.warning(f"{invalid_count} 个索引超出范围(0-{self.original_feature_dim-1})，已过滤")
        
#         self.logger.info(f"有效神经元索引: {len(valid_indices)}")
#         return valid_indices
    
#     def load_and_filter_features(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
#         """
#         流式加载全维度特征文件，筛选选中的神经元特征
#         返回：(特征矩阵X, 标签y, 神经元名称列表)
#         """
#         # 验证索引
#         self.selected_indices = self.validate_indices()
#         if not self.selected_indices:
#             raise ValueError("没有有效的神经元索引")
        
#         self.logger.info(f"开始处理特征文件，筛选 {len(self.selected_indices)} 个神经元")
        
#         # 获取总样本数
#         total_samples = 0
#         with open(self.config.full_dimension_features_path, 'r', encoding='utf-8') as f:
#             total_samples = sum(1 for _ in f)
        
#         self.logger.info(f"总样本行数: {total_samples}")
        
#         # 准备收集数据
#         X_list = []
#         y_list = []
        
#         stats = {
#             'total_samples': 0,
#             'valid_samples': 0,
#             'pos_samples': 0,
#             'neg_samples': 0,
#             'answer_tokens': 0,
#             'other_tokens': 0
#         }
        
#         # 预分配索引数组用于快速筛选
#         selected_indices_array = np.array(self.selected_indices, dtype=np.int32)
        
#         with open(self.config.full_dimension_features_path, 'r', encoding='utf-8') as f:
#             for line_num, line in enumerate(tqdm(f, total=total_samples, desc="处理特征")):
#                 if not line.strip():
#                     continue
                
#                 try:
#                     sample = json.loads(line)
#                     label = sample['label']
                    
#                     # 处理答案令牌特征
#                     if 'features_answer' in sample:
#                         feat_answer_full = np.array(sample['features_answer'], dtype=np.float32)
                        
#                         if len(feat_answer_full) != self.original_feature_dim:
#                             self.logger.warning(f"行 {line_num} 特征维度不匹配")
#                             continue
                        
#                         # 快速筛选选中神经元的特征
#                         x_vector = feat_answer_full[selected_indices_array]
                        
#                         # 清理异常值
#                         if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
#                             x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
#                         X_list.append(x_vector)
#                         y_list.append(1 if label == 1 else 0)
                        
#                         stats['answer_tokens'] += 1
#                         stats['pos_samples' if label == 1 else 'neg_samples'] += 1
#                         stats['valid_samples'] += 1
                    
#                     # 处理非答案令牌特征（可选）
#                     if 'features_other' in sample:
#                         feat_other_full = np.array(sample['features_other'], dtype=np.float32)
                        
#                         if len(feat_other_full) == self.original_feature_dim:
#                             x_vector_other = feat_other_full[selected_indices_array]
                            
#                             if np.any(np.isnan(x_vector_other)) or np.any(np.isinf(x_vector_other)):
#                                 x_vector_other = np.nan_to_num(x_vector_other, nan=0.0, posinf=1e6, neginf=-1e6)
                            
#                             X_list.append(x_vector_other)
#                             y_list.append(0)  # 所有非答案令牌都是负样本
                            
#                             stats['other_tokens'] += 1
#                             stats['valid_samples'] += 1
                    
#                     stats['total_samples'] += 1
                    
#                     # 定期清理内存
#                     if (line_num + 1) % 100 == 0:
#                         gc.collect()
                        
#                 except Exception as e:
#                     self.logger.debug(f"行 {line_num} 处理错误: {e}")
#                     continue
        
#         # 转换为numpy数组
#         if len(X_list) == 0:
#             raise ValueError("没有成功处理任何样本")
        
#         X = np.array(X_list, dtype=np.float32)
#         y = np.array(y_list, dtype=np.int32)
        
#         # 最终清理
#         X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
#         self.logger.info(f"处理统计:")
#         self.logger.info(f"  总读取样本: {stats['total_samples']}")
#         self.logger.info(f"  有效特征样本: {stats['valid_samples']}")
#         self.logger.info(f"  答案令牌: {stats['answer_tokens']}")
#         self.logger.info(f"  非答案令牌: {stats['other_tokens']}")
#         self.logger.info(f"  正样本(幻觉): {stats['pos_samples']}")
#         self.logger.info(f"  负样本(忠实): {stats['neg_samples']}")
#         self.logger.info(f"  特征矩阵形状: {X.shape}")
#         self.logger.info(f"  标签分布 - 正样本比例: {np.mean(y):.3f}")
        
#         # 生成简化的神经元名称
#         neuron_names = [f"neuron_{i}" for i in range(len(self.selected_indices))]
        
#         return X, y, neuron_names

# # ==================== 第二次训练分类器 ====================
# class SecondTrainingClassifier:
#     """第二次训练分类器 - 在选中的神经元上训练，只保存权重>0的神经元"""
    
#     def __init__(self, config: Config, selected_indices: List[int], 
#                  original_neuron_names: List[str], first_training_weights: List[float]):
#         self.config = config
#         self.selected_indices = selected_indices  # 全维度索引
#         self.original_neuron_names = original_neuron_names
#         self.first_training_weights = first_training_weights
#         self.logger = logging.getLogger(__name__)
        
#         self.clf = None
#         self.scaler = StandardScaler()
#         self.second_training_weights = None  # 第二次训练的权重（1050维）
#         self.final_h_neuron_indices = []  # 最终H-Neurons的全维度索引
#         self.final_h_neuron_weights = []  # 最终H-Neurons的权重
    
#     def validate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """验证和清理数据"""
#         if np.any(np.isnan(X)):
#             X = np.nan_to_num(X, nan=0.0)
            
#         if np.any(np.isinf(X)):
#             X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)
        
#         if np.any(np.isnan(y)):
#             y = np.nan_to_num(y, nan=0).astype(np.int32)
            
#         return X, y
    
#     def train(self, X_train: np.ndarray, y_train: np.ndarray):
#         """训练稀疏逻辑回归分类器"""
#         self.logger.info(f"第二次训练开始 - 特征维度: {X_train.shape[1]}")
#         self.logger.info(f"在 {len(self.selected_indices)} 个选中的神经元上训练")
        
#         # 1. 标准化特征
#         self.logger.info("特征标准化...")
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_train_scaled, y_train = self.validate_data(X_train_scaled, y_train)
        
#         # 2. 交叉验证选择最佳C
#         best_score = -1
#         best_C = self.config.C_values[0]
        
#         if len(np.unique(y_train)) > 1:  # 确保有多个类别
#             skf = StratifiedKFold(n_splits=min(self.config.cv_folds, 3), 
#                                  shuffle=True, 
#                                  random_state=self.config.random_seed)
            
#             self.logger.info("交叉验证选择最佳正则化参数...")
#             for C in tqdm(self.config.C_values, desc="交叉验证"):
#                 try:
#                     clf = LogisticRegression(
#                         C=C,
#                         penalty='l1',
#                         solver='saga',
#                         class_weight='balanced',
#                         max_iter=500,
#                         random_state=self.config.random_seed,
#                         tol=1e-3
#                     )
#                     cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
#                     mean_score = np.mean(cv_scores)
#                     self.logger.info(f"C={C}: ROC_AUC={mean_score:.4f}")
                    
#                     if mean_score > best_score:
#                         best_score = mean_score
#                         best_C = C
#                 except Exception as e:
#                     self.logger.warning(f"C={C} 训练失败: {e}")
        
#         # 3. 最终训练
#         self.logger.info(f"使用最佳 C={best_C} 训练最终模型...")
#         self.clf = LogisticRegression(
#             C=best_C,
#             penalty='l1',
#             solver='saga',
#             class_weight='balanced',
#             max_iter=1000,
#             random_state=self.config.random_seed,
#             tol=1e-4
#         )
#         self.clf.fit(X_train_scaled, y_train)
        
#         # 4. 获取第二次训练的权重（1050维）
#         self.second_training_weights = self.clf.coef_[0]
        
#         # 5. 筛选权重大于阈值的神经元
#         significant_mask = self.second_training_weights > self.config.weight_threshold
#         significant_indices_1050 = np.where(significant_mask)[0]
#         significant_weights_1050 = self.second_training_weights[significant_indices_1050]
        
#         self.logger.info(f"第二次训练结果:")
#         self.logger.info(f"  总特征数: {len(self.second_training_weights)}")
#         self.logger.info(f"  权重大于{self.config.weight_threshold}的神经元: {len(significant_indices_1050)}")
        
#         # 6. 映射回全维度索引
#         self.final_h_neuron_indices = []
#         self.final_h_neuron_weights = []
        
#         for idx_1050, weight in zip(significant_indices_1050, significant_weights_1050):
#             if idx_1050 < len(self.selected_indices):
#                 full_dim_idx = self.selected_indices[idx_1050]
#                 self.final_h_neuron_indices.append(full_dim_idx)
#                 self.final_h_neuron_weights.append(float(weight))
        
#         self.logger.info(f"映射到全维度的H-Neurons: {len(self.final_h_neuron_indices)}")
        
#         return self
    
#     def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
#         """评估模型性能"""
#         X_test_scaled = self.scaler.transform(X_test)
        
#         y_prob = self.clf.predict_proba(X_test_scaled)[:, 1]
#         y_pred = self.clf.predict(X_test_scaled)
        
#         accuracy = accuracy_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_prob)
        
#         return {
#             'accuracy': float(accuracy),
#             'auc': float(auc),
#             'report': classification_report(y_test, y_pred, output_dict=True)
#         }
    
#     def analyze_neurons(self) -> Dict:
#         """分析神经元统计"""
#         if self.second_training_weights is None:
#             return {}
        
#         stats = {
#             'total_trained_neurons': len(self.selected_indices),
#             'significant_neurons_second': len(self.final_h_neuron_indices),
#             'weight_threshold': self.config.weight_threshold,
#             'weight_stats': {
#                 'mean': float(np.mean(self.second_training_weights)),
#                 'std': float(np.std(self.second_training_weights)),
#                 'min': float(np.min(self.second_training_weights)),
#                 'max': float(np.max(self.second_training_weights)),
#                 'positive_count': int(np.sum(self.second_training_weights > 0)),
#                 'negative_count': int(np.sum(self.second_training_weights < 0)),
#             }
#         }
        
#         return stats
    
#     def save(self, output_dir: str) -> str:
#         """保存模型和结果"""
#         if self.clf is None:
#             raise ValueError("模型未训练")
        
#         # 1. 保存第二次训练的分类器（包含权重映射）
#         classifier_data = {
#             'classifier': self.clf,
#             'scaler': self.scaler,
#             'selected_indices': self.selected_indices,  # 1050个全维度索引
#             'second_training_weights': self.second_training_weights.tolist() if self.second_training_weights is not None else [],
#             'final_h_neuron_indices': self.final_h_neuron_indices,
#             'final_h_neuron_weights': self.final_h_neuron_weights,
#             'config': {
#                 'weight_threshold': self.config.weight_threshold,
#                 'first_training_dir': self.config.first_training_dir
#             }
#         }
        
#         classifier_path = os.path.join(output_dir, "second_training_classifier.pkl")
#         with open(classifier_path, 'wb') as f:
#             pickle.dump(classifier_data, f)
        
#         # 2. 保存第二次训练的H-Neurons（全维度，只包含权重>0的）
#         h_neurons_list = []
        
#         for full_idx, weight in zip(self.final_h_neuron_indices, self.final_h_neuron_weights):
#             # 获取神经元名称
#             neuron_name = ""
#             if full_idx < len(self.original_neuron_names):
#                 neuron_name = self.original_neuron_names[full_idx]
#             else:
#                 # 推断名称
#                 neurons_per_layer = 18944
#                 layer_idx = full_idx // neurons_per_layer
#                 neuron_idx = full_idx % neurons_per_layer
#                 neuron_name = f"layer_{layer_idx}_neuron_{neuron_idx}"
            
#             # 获取第一次训练的权重
#             first_weight = 0.0
#             if full_idx in self.selected_indices:
#                 idx_in_selected = self.selected_indices.index(full_idx)
#                 if idx_in_selected < len(self.first_training_weights):
#                     first_weight = self.first_training_weights[idx_in_selected]
            
#             h_neurons_list.append({
#                 'original_index': int(full_idx),
#                 'name': neuron_name,
#                 'first_training_weight': float(first_weight),
#                 'second_training_weight': float(weight),
#                 'weight_change': float(weight - first_weight)
#             })
        
#         # 按权重降序排序
#         h_neurons_list.sort(key=lambda x: abs(x['second_training_weight']), reverse=True)
        
#         h_neurons_path = os.path.join(output_dir, "h_neurons_second_training.json")
#         with open(h_neurons_path, 'w', encoding='utf-8') as f:
#             json.dump(h_neurons_list, f, indent=2, ensure_ascii=False)
        
#         self.logger.info(f"保存 {len(h_neurons_list)} 个H-Neurons到 {h_neurons_path}")
        
#         return output_dir

# # ==================== 主训练流程 ====================
# class SecondTrainingTrainer:
#     """第二次训练器 - 在1050个神经元上训练，保存权重>0的神经元"""
    
#     def __init__(self, config: Config):
#         self.config = config
#         self.logger = logging.getLogger(__name__)
        
#         # 加载第一次训练的H-Neurons
#         self.neuron_loader = HNeuronsLoader(config)
#         self.selected_indices = None
#         self.original_neuron_names = None
#         self.first_training_weights = None
        
#         # 处理器和分类器
#         self.feature_processor = None
#         self.classifier = None
    
#     def load_and_prepare_neurons(self):
#         """加载并准备第一次训练的神经元"""
#         self.logger.info("加载第一次训练的H-Neurons...")
        
#         # 加载第一次训练的神经元（只保留权重>0的）
#         selected_indices, original_neuron_names, first_training_weights = \
#             self.neuron_loader.load_first_training_neurons()
        
#         if not selected_indices:
#             raise ValueError("第一次训练没有找到权重大于0的神经元")
        
#         self.selected_indices = selected_indices
#         self.original_neuron_names = original_neuron_names
#         self.first_training_weights = first_training_weights
        
#         self.logger.info(f"准备在 {len(self.selected_indices)} 个神经元上进行第二次训练")
        
#         # 创建特征处理器
#         self.feature_processor = FullDimensionFeatureProcessor(
#             self.config, self.selected_indices
#         )
        
#         # 创建分类器
#         self.classifier = SecondTrainingClassifier(
#             self.config, self.selected_indices,
#             self.original_neuron_names, self.first_training_weights
#         )
        
#         return self
    
#     def train(self) -> Tuple[SecondTrainingClassifier, str]:
#         """执行第二次训练"""
#         self.logger.info("="*60)
#         self.logger.info("H-Neurons第二次训练启动")
#         self.logger.info("="*60)
        
#         start_time = time.time()
        
#         try:
#             # 阶段1: 加载神经元
#             self.logger.info("\n[阶段1/4] 加载第一次训练的H-Neurons")
#             self.load_and_prepare_neurons()
            
#             # 阶段2: 加载并筛选特征
#             self.logger.info("\n[阶段2/4] 从全维度特征文件中筛选神经元特征")
#             X, y, _ = self.feature_processor.load_and_filter_features()
            
#             # 阶段3: 划分数据集
#             self.logger.info("\n[阶段3/4] 划分训练集和测试集")
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y,
#                 test_size=self.config.test_size,
#                 stratify=y,
#                 random_state=self.config.random_seed
#             )
            
#             self.logger.info(f"数据集统计:")
#             self.logger.info(f"  总样本数: {X.shape[0]}")
#             self.logger.info(f"  训练集: {X_train.shape[0]}")
#             self.logger.info(f"  测试集: {X_test.shape[0]}")
#             self.logger.info(f"  特征维度: {X_train.shape[1]}")
#             self.logger.info(f"  正样本比例: {np.mean(y_train):.3f} (训练集)")
            
#             # 阶段4: 训练分类器
#             self.logger.info("\n[阶段4/4] 训练第二次分类器")
#             self.classifier.train(X_train, y_train)
            
#             # 评估
#             eval_results = self.classifier.evaluate(X_test, y_test)
            
#             # 分析
#             neuron_stats = self.classifier.analyze_neurons()
            
#             # 保存结果
#             output_dir = self.classifier.save(self.config.experiment_dir)
            
#             # 保存完整的实验结果
#             results = {
#                 'training_config': {
#                     'first_training_dir': self.config.first_training_dir,
#                     'full_dimension_features': self.config.full_dimension_features_path,
#                     'selected_neurons_count': len(self.selected_indices),
#                     'test_size': self.config.test_size,
#                     'weight_threshold': self.config.weight_threshold,
#                     'method': 'second_training_on_selected_neurons'
#                 },
#                 'data_statistics': {
#                     'total_samples': X.shape[0],
#                     'train_samples': X_train.shape[0],
#                     'test_samples': X_test.shape[0],
#                     'feature_dimension': X.shape[1],
#                     'positive_ratio': float(np.mean(y))
#                 },
#                 'first_training_info': {
#                     'initial_neurons': len(self.selected_indices),
#                     'initial_positive_weights': len([w for w in self.first_training_weights if w > 0])
#                 },
#                 'second_training_results': {
#                     'evaluation': eval_results,
#                     'neuron_statistics': neuron_stats,
#                     'final_h_neurons_count': len(self.classifier.final_h_neuron_indices)
#                 },
#                 'training_info': {
#                     'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
#                     'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                     'total_time_seconds': time.time() - start_time
#                 }
#             }
            
#             results_path = os.path.join(output_dir, "second_training_results.json")
#             with open(results_path, 'w', encoding='utf-8') as f:
#                 json.dump(results, f, indent=2, ensure_ascii=False)
            
#             # 生成报告
#             self._generate_final_report(results, output_dir, time.time() - start_time)
            
#             return self.classifier, output_dir
            
#         except Exception as e:
#             self.logger.error(f"第二次训练失败: {e}")
#             import traceback
#             self.logger.error(traceback.format_exc())
#             raise
    
#     def _generate_final_report(self, results: Dict, output_dir: str, total_time: float):
#         """生成最终报告"""
#         hours = int(total_time // 3600)
#         minutes = int((total_time % 3600) // 60)
#         seconds = int(total_time % 60)
        
#         # 加载第一次训练的结果
#         first_results_path = os.path.join(self.config.first_training_dir, "experiment_results.json")
#         first_h_neurons_path = os.path.join(self.config.first_training_dir, "h_neurons.json")
        
#         first_accuracy = "N/A"
#         first_auc = "N/A"
#         first_neuron_count = 0
        
#         if os.path.exists(first_results_path):
#             with open(first_results_path, 'r') as f:
#                 first_results = json.load(f)
#                 first_accuracy = first_results.get('evaluation', {}).get('accuracy', 'N/A')
#                 first_auc = first_results.get('evaluation', {}).get('auc', 'N/A')
        
#         if os.path.exists(first_h_neurons_path):
#             with open(first_h_neurons_path, 'r') as f:
#                 first_h_neurons = json.load(f)
#                 first_neuron_count = len(first_h_neurons)
        
#         report = f"""
# {'='*80}
# H-Neurons第二次训练完成报告
# {'='*80}

# 训练策略:
#   - 基础: 基于第一次训练的 {first_neuron_count} 个H-Neurons
#   - 筛选: 使用其中 {results['first_training_info']['initial_neurons']} 个权重大于0的神经元
#   - 训练: 在这{results['data_statistics']['feature_dimension']}个神经元上重新训练L1逻辑回归
#   - 保存: 只保留第二次训练后权重大于{results['training_config']['weight_threshold']}的神经元
#   - 映射: 结果映射回全维度索引，便于后续使用

# 性能对比:
#               第一次训练      第二次训练      变化
#   准确率:    {first_accuracy if isinstance(first_accuracy, str) else f'{first_accuracy:.4f}':<10}  {results['second_training_results']['evaluation']['accuracy']:.4f}      {f"+{results['second_training_results']['evaluation']['accuracy'] - (first_accuracy if isinstance(first_accuracy, float) else 0):.4f}" if isinstance(first_accuracy, float) else "N/A"}
#   AUC:       {first_auc if isinstance(first_auc, str) else f'{first_auc:.4f}':<10}  {results['second_training_results']['evaluation']['auc']:.4f}      {f"+{results['second_training_results']['evaluation']['auc'] - (first_auc if isinstance(first_auc, float) else 0):.4f}" if isinstance(first_auc, float) else "N/A"}

# 神经元分析:
#   第一次训练H-Neurons总数: {first_neuron_count}
#   用于第二次训练的神经元: {results['first_training_info']['initial_neurons']}
#   第二次训练后H-Neurons数: {results['second_training_results']['final_h_neurons_count']}
#   神经元减少比例: {(1 - results['second_training_results']['final_h_neurons_count']/results['first_training_info']['initial_neurons'])*100:.1f}%

# 权重统计(第二次训练):
#   平均权重: {results['second_training_results']['neuron_statistics']['weight_stats']['mean']:.6f}
#   权重标准差: {results['second_training_results']['neuron_statistics']['weight_stats']['std']:.6f}
#   最小权重: {results['second_training_results']['neuron_statistics']['weight_stats']['min']:.6f}
#   最大权重: {results['second_training_results']['neuron_statistics']['weight_stats']['max']:.6f}
#   正权重神经元: {results['second_training_results']['neuron_statistics']['weight_stats']['positive_count']}
#   负权重神经元: {results['second_training_results']['neuron_statistics']['weight_stats']['negative_count']}

# 数据统计:
#   总样本数: {results['data_statistics']['total_samples']}
#   训练集样本: {results['data_statistics']['train_samples']}
#   测试集样本: {results['data_statistics']['test_samples']}
#   特征维度: {results['data_statistics']['feature_dimension']}
#   正样本比例: {results['data_statistics']['positive_ratio']:.3f}

# 训练信息:
#   开始时间: {results['training_info']['start_time']}
#   结束时间: {results['training_info']['end_time']}
#   总耗时: {hours}小时{minutes}分钟{seconds}秒

# 输出文件 ({output_dir}):
#   1. second_training_classifier.pkl - 第二次训练的分类器(包含权重映射)
#   2. h_neurons_second_training.json - 第二次训练的H-Neurons(全维度索引，权重>0)
#   3. second_training_results.json - 完整实验结果
#   4. second_training_report.txt - 本报告

# 结果解读:
#   1. 第二次训练进一步精炼了H-Neurons，保留了最重要的神经元
#   2. 神经元数量减少，但可能提高了模型的解释性和稳定性
#   3. 权重变化反映了神经元在第二次训练中的重要性调整
#   4. 映射到全维度索引便于后续的模型编辑和分析

# {'='*80}
# """
        
#         self.logger.info(report)
        
#         # 保存报告
#         report_path = os.path.join(output_dir, "second_training_report.txt")
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write(report)

# # ==================== 主函数 ====================
# def main():
#     """主函数"""
#     # 设置配置
#     config = Config()
    
#     # 设置日志
#     logger = LoggerSetup.setup()
    
#     # 创建训练器
#     trainer = SecondTrainingTrainer(config)
    
#     # 运行训练
#     logger.info("开始第二次训练...")
#     classifier, output_dir = trainer.train()
    
#     return classifier, output_dir

# # ==================== 比较工具 ====================
# def compare_neurons(first_training_dir: str, second_training_dir: str):
#     """比较两次训练的神经元"""
#     # 加载第一次训练的H-Neurons
#     with open(os.path.join(first_training_dir, "h_neurons.json"), 'r') as f:
#         first_neurons = json.load(f)
    
#     # 加载第二次训练的H-Neurons
#     with open(os.path.join(second_training_dir, "h_neurons_second_training.json"), 'r') as f:
#         second_neurons = json.load(f)
    
#     # 提取索引和权重
#     first_indices = set()
#     first_weight_map = {}
    
#     for neuron in first_neurons:
#         idx = neuron.get('original_index')
#         if idx is None:
#             # 尝试从名称推断
#             name = neuron.get('name', '')
#             if name.startswith('layer_') and '_neuron_' in name:
#                 parts = name.split('_')
#                 layer_idx = int(parts[1])
#                 neuron_idx = int(parts[3])
#                 neurons_per_layer = 18944
#                 idx = layer_idx * neurons_per_layer + neuron_idx
        
#         if idx is not None:
#             first_indices.add(idx)
#             first_weight_map[idx] = neuron['weight']
    
#     second_indices = set([neuron['original_index'] for neuron in second_neurons])
#     second_weight_map = {neuron['original_index']: neuron['second_training_weight'] for neuron in second_neurons}
    
#     # 分析
#     intersection = first_indices.intersection(second_indices)
#     only_first = first_indices - second_indices
#     only_second = second_indices - first_indices
    
#     print("="*80)
#     print("第一次 vs 第二次训练神经元比较")
#     print("="*80)
#     print(f"第一次训练神经元数: {len(first_neurons)}")
#     print(f"第二次训练神经元数: {len(second_neurons)}")
#     print(f"交集神经元数: {len(intersection)} ({len(intersection)/len(first_indices)*100:.1f}%)")
#     print(f"仅第一次训练出现: {len(only_first)}")
#     print(f"仅第二次训练出现: {len(only_second)}")
#     print()
    
#     # 权重变化分析
#     if intersection:
#         weight_changes = []
#         for idx in intersection:
#             if idx in first_weight_map and idx in second_weight_map:
#                 change = second_weight_map[idx] - first_weight_map[idx]
#                 weight_changes.append(abs(change))
        
#         print("权重变化分析:")
#         print(f"  平均绝对变化: {np.mean(weight_changes):.6f}")
#         print(f"  最大绝对变化: {np.max(weight_changes):.6f}")
#         print(f"  最小绝对变化: {np.min(weight_changes):.6f}")
    
#     return {
#         'first_count': len(first_neurons),
#         'second_count': len(second_neurons),
#         'intersection': len(intersection),
#         'intersection_ratio': len(intersection)/len(first_indices) if first_indices else 0
#     }

# if __name__ == "__main__":
#     try:
#         print("="*80)
#         print("H-Neurons第二次训练启动")
#         print("="*80)
#         print("训练策略:")
#         print("  1. 使用第一次训练得到的1050个H-Neurons")
#         print("  2. 从全维度特征文件中提取这些神经元的特征")
#         print("  3. 训练L1正则化逻辑回归")
#         print("  4. 只保存权重>0的神经元，映射回全维度索引")
#         print("="*80)
        
#         classifier, output_dir = main()
        
#         print("\n" + "="*80)
#         print("第二次训练完成!")
#         print("="*80)
#         print(f"输出目录: {output_dir}")
#         print()
#         print("重要文件:")
#         print(f"  1. h_neurons_second_training.json - 第二次训练的H-Neurons（权重>0，全维度索引）")
#         print(f"  2. second_training_classifier.pkl - 完整的第二次训练模型")
#         print(f"  3. second_training_report.txt - 详细训练报告")
#         print()
#         print("使用示例:")
#         print(f"  # 加载第二次训练的H-Neurons")
#         print(f"  with open('{output_dir}/h_neurons_second_training.json', 'r') as f:")
#         print(f"      h_neurons_second = json.load(f)")
#         print()
#         print(f"  # 比较两次训练")
#         print(f"  compare_neurons('第一次训练目录', '{output_dir}')")
#         print("="*80)
        
#     except Exception as e:
#         print(f"训练失败: {e}")
#         import traceback
#         traceback.print_exc()


"""
H-Neurons第二次训练 - 基于全维度特征文件和已识别神经元
训练完成后保存格式与第一次训练完全一致
"""

import os
import json
import pickle
import logging
import time
import gc
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif

# ==================== 配置 ====================
@dataclass
class Config:
    # 数据路径
    full_dimension_features_path: str = "/mnt/project/H-Neurous/output/saga_chat/cett_features_vectorized.jsonl"
    
    # 第一次训练结果路径
    first_training_dir: str = "/mnt/project/H-Neurous/output/saga_chat"
    first_h_neurons_file: str = "h_neurons.json"
    first_classifier_file: str = "h_neurons_classifier.pkl"
    
    # 输出目录
    output_dir: str = "/mnt/project/H-Neurous/output"
    
    # 训练参数
    test_size: float = 0.15
    cv_folds: int = 5
    random_seed: int = 42
    C_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    
    # 第二次训练参数
    weight_threshold: float = 1e-4  # 只保存权重大于这个值的神经元
    
    # 日志和实验
    experiment_name: str = "h_neurons_second_training"
    
    def __post_init__(self):
        # 创建目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(self.output_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 验证文件存在
        if not os.path.exists(self.full_dimension_features_path):
            raise FileNotFoundError(f"全维度特征文件不存在: {self.full_dimension_features_path}")
        
        first_h_neurons_path = os.path.join(self.first_training_dir, self.first_h_neurons_file)
        if not os.path.exists(first_h_neurons_path):
            raise FileNotFoundError(f"第一次训练的H-Neurons文件不存在: {first_h_neurons_path}")

# ==================== 日志配置 ====================
class LoggerSetup:
    @staticmethod
    def setup():
        log_dir = "/mnt/project/H-Neurous/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"h_neurons_second_training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

# ==================== H-Neurons加载器 ====================
class HNeuronsLoader:
    """加载第一次训练得到的H-Neurons"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_first_training_data(self) -> Tuple[List[int], List[str], Dict]:
        """
        加载第一次训练的所有数据
        返回：(选中的神经元索引列表, 全维度神经元名称列表, 第一次训练的分类器数据)
        """
        self.logger.info(f"加载第一次训练数据从: {self.config.first_training_dir}")
        
        # 1. 加载H-Neurons列表
        h_neurons_path = os.path.join(self.config.first_training_dir, self.config.first_h_neurons_file)
        with open(h_neurons_path, 'r', encoding='utf-8') as f:
            h_neurons = json.load(f)
        
        self.logger.info(f"第一次训练共识别 {len(h_neurons)} 个H-Neurons")
        
        # 2. 加载分类器数据
        classifier_path = os.path.join(self.config.first_training_dir, self.config.first_classifier_file)
        with open(classifier_path, 'rb') as f:
            classifier_data = pickle.load(f)
        
        # 3. 获取全维度神经元名称
        all_neuron_names = classifier_data.get('neuron_names', [])
        self.logger.info(f"从分类器加载 {len(all_neuron_names)} 个全维度神经元名称")
        
        # 4. 只保留权重大于0的神经元
        selected_indices = []
        selected_names = []
        
        for neuron in h_neurons:
            weight = neuron.get('weight', 0.0)
            name = neuron.get('name', '')
            
            if weight > self.config.weight_threshold:
                # 尝试从名称解析索引
                idx = self._parse_neuron_name_to_index(name, all_neuron_names)
                if idx is not None:
                    selected_indices.append(idx)
                    selected_names.append(name)
        
        # 去重并排序
        unique_data = {}
        for idx, name in zip(selected_indices, selected_names):
            if idx not in unique_data:
                unique_data[idx] = name
        
        selected_indices = sorted(unique_data.keys())
        selected_names = [unique_data[idx] for idx in selected_indices]
        
        self.logger.info(f"成功选中 {len(selected_indices)} 个神经元用于第二次训练")
        
        return selected_indices, all_neuron_names, classifier_data
    
    def _parse_neuron_name_to_index(self, neuron_name: str, all_neuron_names: List[str]) -> Optional[int]:
        """将神经元名称解析为全维度索引"""
        # 方法1: 如果在全维度名称列表中，直接使用索引
        if neuron_name in all_neuron_names:
            return all_neuron_names.index(neuron_name)
        
        # 方法2: 尝试解析名称格式 layer_{layer}_neuron_{neuron}
        try:
            if neuron_name.startswith('layer_') and '_neuron_' in neuron_name:
                parts = neuron_name.split('_')
                layer_idx = int(parts[1])
                neuron_idx = int(parts[3])
                
                # 计算全维度索引
                neurons_per_layer = 18944
                calculated_index = layer_idx * neurons_per_layer + neuron_idx
                
                # 验证索引在合理范围内
                if calculated_index < len(all_neuron_names):
                    return calculated_index
        except Exception as e:
            self.logger.debug(f"无法解析神经元名称 {neuron_name}: {e}")
        
        return None

# ==================== 全维度特征处理器 ====================
class FullDimensionFeatureProcessor:
    """从全维度特征文件中筛选指定神经元的特征"""
    
    def __init__(self, config: Config, selected_neuron_indices: List[int]):
        self.config = config
        self.selected_indices = sorted(set(selected_neuron_indices))
        self.logger = logging.getLogger(__name__)
        
        # 从第一行获取原始特征维度
        self.original_feature_dim = self._get_original_feature_dim()
        self.logger.info(f"原始特征维度: {self.original_feature_dim}")
    
    def _get_original_feature_dim(self) -> int:
        """从特征文件获取原始特征维度"""
        try:
            with open(self.config.full_dimension_features_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line:
                    raise ValueError("特征文件为空")
                
                first_sample = json.loads(first_line)
                if 'features_answer' not in first_sample:
                    raise ValueError("特征文件中没有features_answer字段")
                
                return len(first_sample['features_answer'])
        except Exception as e:
            self.logger.error(f"获取原始特征维度失败: {e}")
            return 0
    
    def validate_indices(self) -> List[int]:
        """验证并筛选有效的神经元索引"""
        valid_indices = [idx for idx in self.selected_indices if idx < self.original_feature_dim]
        
        if len(valid_indices) != len(self.selected_indices):
            invalid_count = len(self.selected_indices) - len(valid_indices)
            self.logger.warning(f"{invalid_count} 个索引超出范围(0-{self.original_feature_dim-1})，已过滤")
        
        self.logger.info(f"有效神经元索引: {len(valid_indices)}")
        return valid_indices
    
    def load_and_filter_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        流式加载全维度特征文件，筛选选中的神经元特征
        返回：(特征矩阵X, 标签y)
        """
        # 验证索引
        self.selected_indices = self.validate_indices()
        if not self.selected_indices:
            raise ValueError("没有有效的神经元索引")
        
        self.logger.info(f"开始处理特征文件，筛选 {len(self.selected_indices)} 个神经元")
        
        # 获取总样本数
        total_samples = 0
        with open(self.config.full_dimension_features_path, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for _ in f)
        
        self.logger.info(f"总样本行数: {total_samples}")
        
        # 准备收集数据
        X_list = []
        y_list = []
        
        stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'pos_samples': 0,
            'neg_samples': 0
        }
        
        # 预分配索引数组用于快速筛选
        selected_indices_array = np.array(self.selected_indices, dtype=np.int32)
        
        with open(self.config.full_dimension_features_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_samples, desc="处理特征")):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    label = sample['label']
                    
                    # 处理答案令牌特征
                    if 'features_answer' in sample:
                        feat_answer_full = np.array(sample['features_answer'], dtype=np.float32)
                        
                        if len(feat_answer_full) != self.original_feature_dim:
                            self.logger.warning(f"行 {line_num} 特征维度不匹配")
                            continue
                        
                        # 快速筛选选中神经元的特征
                        x_vector = feat_answer_full[selected_indices_array]
                        
                        # 清理异常值
                        if np.any(np.isnan(x_vector)) or np.any(np.isinf(x_vector)):
                            x_vector = np.nan_to_num(x_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        X_list.append(x_vector)
                        y_list.append(1 if label == 1 else 0)
                        
                        stats['pos_samples' if label == 1 else 'neg_samples'] += 1
                        stats['valid_samples'] += 1
                    
                    # 处理非答案令牌特征
                    if 'features_other' in sample:
                        feat_other_full = np.array(sample['features_other'], dtype=np.float32)
                        
                        if len(feat_other_full) == self.original_feature_dim:
                            x_vector_other = feat_other_full[selected_indices_array]
                            
                            if np.any(np.isnan(x_vector_other)) or np.any(np.isinf(x_vector_other)):
                                x_vector_other = np.nan_to_num(x_vector_other, nan=0.0, posinf=1e6, neginf=-1e6)
                            
                            X_list.append(x_vector_other)
                            y_list.append(0)
                            
                            stats['valid_samples'] += 1
                    
                    stats['total_samples'] += 1
                    
                    # 定期清理内存
                    if (line_num + 1) % 100 == 0:
                        gc.collect()
                        
                except Exception as e:
                    self.logger.debug(f"行 {line_num} 处理错误: {e}")
                    continue
        
        # 转换为numpy数组
        if len(X_list) == 0:
            raise ValueError("没有成功处理任何样本")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # 最终清理
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.logger.info(f"处理统计:")
        self.logger.info(f"  总读取样本: {stats['total_samples']}")
        self.logger.info(f"  有效特征样本: {stats['valid_samples']}")
        self.logger.info(f"  正样本(幻觉): {stats['pos_samples']}")
        self.logger.info(f"  负样本(忠实): {stats['neg_samples']}")
        self.logger.info(f"  特征矩阵形状: {X.shape}")
        self.logger.info(f"  正样本比例: {np.mean(y):.3f}")
        
        return X, y

# ==================== 第二次训练分类器 ====================
class SecondTrainingClassifier:
    """第二次训练分类器 - 与第一次训练保持相同格式"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 与第一次训练相同的组件
        self.clf = None
        self.var_selector = VarianceThreshold(threshold=0)
        self.percentile_selector = SelectPercentile(f_classif, percentile=10)
        self.scaler = StandardScaler()
        
        # 其他数据
        self.h_neuron_indices = None
        self.h_neuron_names = None
        self.h_neuron_weights = None
        self.neuron_names = None
        self.selected_indices = None
    
    def set_neuron_names(self, neuron_names: List[str], selected_indices: List[int]):
        """设置神经元名称和选中的索引"""
        self.neuron_names = neuron_names
        self.selected_indices = selected_indices
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练稀疏逻辑回归分类器"""
        self.logger.info(f"第二次训练开始 - 特征维度: {X_train.shape[1]}")
        self.logger.info(f"在 {len(self.selected_indices)} 个选中的神经元上训练")
        
        # 注意：这里跳过特征选择步骤，因为我们已经在选中的神经元上训练
        # 但为了保持格式一致，我们仍然使用这些组件
        
        # 1. 标准化特征
        self.logger.info("特征标准化...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = self._validate_data(X_train_scaled)
        
        # 2. 交叉验证选择最佳C
        best_score = -1
        best_C = self.config.C_values[0]
        
        if len(np.unique(y_train)) > 1:
            skf = StratifiedKFold(n_splits=min(self.config.cv_folds, 3), 
                                 shuffle=True, 
                                 random_state=self.config.random_seed)
            
            self.logger.info("交叉验证选择最佳正则化参数...")
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
                    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
                    mean_score = np.mean(cv_scores)
                    self.logger.info(f"C={C}: ROC_AUC={mean_score:.4f}")
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_C = C
                except Exception as e:
                    self.logger.warning(f"C={C} 训练失败: {e}")
        
        # 3. 最终训练
        self.logger.info(f"使用最佳 C={best_C} 训练最终模型...")
        self.clf = LogisticRegression(
            C=best_C,
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.random_seed,
            tol=1e-4
        )
        self.clf.fit(X_train_scaled, y_train)
        
        # 4. 获取权重并筛选H-Neurons
        weights = self.clf.coef_[0]
        
        # 筛选权重大于阈值的神经元
        significant_mask = weights > self.config.weight_threshold
        significant_indices_1050 = np.where(significant_mask)[0]
        significant_weights_1050 = weights[significant_indices_1050]
        
        self.logger.info(f"第二次训练结果:")
        self.logger.info(f"  总特征数: {len(weights)}")
        self.logger.info(f"  权重大于{self.config.weight_threshold}的神经元: {len(significant_indices_1050)}")
        
        # 5. 映射回全维度索引
        self.h_neuron_indices = []
        self.h_neuron_weights = []
        self.h_neuron_names = []
        
        for idx_1050, weight in zip(significant_indices_1050, significant_weights_1050):
            if idx_1050 < len(self.selected_indices):
                full_dim_idx = self.selected_indices[idx_1050]
                self.h_neuron_indices.append(full_dim_idx)
                self.h_neuron_weights.append(float(weight))
                
                # 获取神经元名称
                if full_dim_idx < len(self.neuron_names):
                    self.h_neuron_names.append(self.neuron_names[full_dim_idx])
                else:
                    # 根据索引计算神经元名称
                    neurons_per_layer = 18944
                    layer_idx = full_dim_idx // neurons_per_layer
                    neuron_idx = full_dim_idx % neurons_per_layer
                    self.h_neuron_names.append(f"layer_{layer_idx}_neuron_{neuron_idx}")
        
        self.logger.info(f"映射到全维度的H-Neurons: {len(self.h_neuron_indices)}")
        
        return self
    
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """验证和清理数据"""
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0)
            
        if np.any(np.isinf(X)):
            X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)
            
        return X
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型性能"""
        X_test_scaled = self.scaler.transform(X_test)
        
        y_prob = self.clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        return {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def analyze_h_neurons(self) -> Dict:
        """分析H-Neurons的统计特性"""
        stats = {
            'total_neurons': len(self.neuron_names) if self.neuron_names else 0,
            'h_neuron_count': 0,
            'h_neuron_ratio': 0.0,
            'mean_weight': 0.0,
            'layer_distribution': {}
        }
        
        if self.h_neuron_indices is None or len(self.h_neuron_indices) == 0:
            self.logger.warning("未发现符合条件的H-Neurons")
            return stats
        
        stats['h_neuron_count'] = len(self.h_neuron_indices)
        stats['h_neuron_ratio'] = stats['h_neuron_count'] / stats['total_neurons']
        stats['mean_weight'] = float(np.mean(self.h_neuron_weights))
        
        # 计算每一层的神经元分布
        layer_dist = {}
        for name in self.h_neuron_names:
            try:
                if name.startswith('layer_'):
                    parts = name.split('_')
                    if len(parts) >= 2:
                        layer_num = int(parts[1])
                        layer_dist[layer_num] = layer_dist.get(layer_num, 0) + 1
            except:
                continue
        
        stats['layer_distribution'] = layer_dist
        
        return stats
    
    def save(self, output_dir: str):
        """保存模型和特征 - 与第一次训练完全相同的格式"""
        if self.clf is None:
            raise ValueError("模型未训练")
        
        # 1. 保存完整模型 - 文件名与第一次训练相同
        classifier_data = {
            'classifier': self.clf,
            'var_selector': self.var_selector,
            'percentile_selector': self.percentile_selector,
            'scaler': self.scaler,
            'h_neuron_indices': self.h_neuron_indices,
            'neuron_names': self.neuron_names,
            'selected_indices': self.selected_indices
        }
        
        classifier_path = os.path.join(output_dir, "h_neurons_classifier.pkl")
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier_data, f)
        
        self.logger.info(f"分类器保存到: {classifier_path}")
        
        # 2. 保存H-Neurons列表 - 文件名与第一次训练相同
        h_list = []
        for i in range(len(self.h_neuron_indices)):
            h_list.append({
                'name': self.h_neuron_names[i],
                'weight': float(self.h_neuron_weights[i])
            })
        
        # 按权重降序排序
        h_list.sort(key=lambda x: x['weight'], reverse=True)
        
        h_neurons_path = os.path.join(output_dir, "h_neurons.json")
        with open(h_neurons_path, 'w', encoding='utf-8') as f:
            json.dump(h_list, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"H-Neurons保存到: {h_neurons_path}")
        
        # 3. 保存训练配置
        config_info = {
            'first_training_dir': self.config.first_training_dir,
            'full_dimension_features_path': self.config.full_dimension_features_path,
            'selected_neurons_count': len(self.selected_indices),
            'h_neuron_count': len(self.h_neuron_indices),
            'weight_threshold': self.config.weight_threshold,
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = os.path.join(output_dir, "second_training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        return output_dir

# ==================== 主训练流程 ====================
class SecondTrainingTrainer:
    """第二次训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 加载器
        self.neuron_loader = HNeuronsLoader(config)
        
        # 处理器和分类器
        self.feature_processor = None
        self.classifier = None
    
    def train(self) -> Tuple[SecondTrainingClassifier, str]:
        """执行第二次训练"""
        self.logger.info("="*60)
        self.logger.info("H-Neurons第二次训练启动")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 阶段1: 加载第一次训练的数据
            self.logger.info("\n[阶段1/4] 加载第一次训练数据")
            selected_indices, all_neuron_names, first_classifier_data = \
                self.neuron_loader.load_first_training_data()
            
            # 阶段2: 加载并筛选特征
            self.logger.info("\n[阶段2/4] 从全维度特征文件中筛选神经元特征")
            self.feature_processor = FullDimensionFeatureProcessor(
                self.config, selected_indices
            )
            
            X, y = self.feature_processor.load_and_filter_features()
            
            # 阶段3: 划分数据集
            self.logger.info("\n[阶段3/4] 划分训练集和测试集")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                stratify=y,
                random_state=self.config.random_seed
            )
            
            self.logger.info(f"数据集统计:")
            self.logger.info(f"  总样本数: {X.shape[0]}")
            self.logger.info(f"  训练集: {X_train.shape[0]}")
            self.logger.info(f"  测试集: {X_test.shape[0]}")
            self.logger.info(f"  特征维度: {X_train.shape[1]}")
            self.logger.info(f"  正样本比例: {np.mean(y_train):.3f} (训练集)")
            
            # 阶段4: 训练分类器
            self.logger.info("\n[阶段4/4] 训练第二次分类器")
            self.classifier = SecondTrainingClassifier(self.config)
            self.classifier.set_neuron_names(all_neuron_names, selected_indices)
            self.classifier.train(X_train, y_train)
            
            # 评估
            eval_results = self.classifier.evaluate(X_test, y_test)
            
            # 分析
            neuron_stats = self.classifier.analyze_h_neurons()
            
            # 保存结果
            output_dir = self.classifier.save(self.config.experiment_dir)
            
            # 保存完整的实验结果
            results = {
                'config': {
                    'first_training_dir': self.config.first_training_dir,
                    'full_dimension_features_path': self.config.full_dimension_features_path,
                    'selected_neurons_count': len(selected_indices),
                    'test_size': self.config.test_size,
                    'cv_folds': self.config.cv_folds,
                    'random_seed': self.config.random_seed,
                    'C_values': self.config.C_values,
                    'methodology': 'second_training_on_selected_neurons'
                },
                'data_statistics': {
                    'processed_samples': X.shape[0],
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'feature_dimension': X.shape[1],
                    'positive_ratio': float(np.mean(y))
                },
                'first_training_info': {
                    'original_h_neuron_count': len(first_classifier_data.get('h_neuron_indices', [])),
                    'selected_for_second_training': len(selected_indices)
                },
                'evaluation': eval_results,
                'h_neurons': neuron_stats,
                'training_info': {
                    'start_time': datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_time_seconds': time.time() - start_time
                }
            }
            
            results_path = os.path.join(output_dir, "experiment_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 生成报告
            self._generate_final_report(results, output_dir, time.time() - start_time)
            
            return self.classifier, output_dir
            
        except Exception as e:
            self.logger.error(f"第二次训练失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _generate_final_report(self, results: Dict, output_dir: str, total_time: float):
        """生成最终报告"""
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        # 加载第一次训练的结果
        first_results_path = os.path.join(self.config.first_training_dir, "experiment_results.json")
        first_h_neurons_path = os.path.join(self.config.first_training_dir, "h_neurons.json")
        
        first_accuracy = "N/A"
        first_auc = "N/A"
        first_neuron_count = 0
        
        if os.path.exists(first_results_path):
            with open(first_results_path, 'r') as f:
                first_results = json.load(f)
                first_accuracy = first_results.get('evaluation', {}).get('accuracy', 'N/A')
                first_auc = first_results.get('evaluation', {}).get('auc', 'N/A')
        
        if os.path.exists(first_h_neurons_path):
            with open(first_h_neurons_path, 'r') as f:
                first_h_neurons = json.load(f)
                first_neuron_count = len(first_h_neurons)
        
        report = f"""
{'='*80}
H-Neurons第二次训练完成报告
{'='*80}

训练概要:
  - 基础: 基于第一次训练的 {first_neuron_count} 个H-Neurons
  - 筛选: 使用其中 {results['first_training_info']['selected_for_second_training']} 个权重大于0的神经元
  - 训练: 在这{results['data_statistics']['feature_dimension']}个神经元上重新训练L1逻辑回归
  - 保存: 只保留第二次训练后权重大于{self.config.weight_threshold}的神经元
  - 格式: 与第一次训练完全相同的文件格式

性能对比:
              第一次训练      第二次训练      变化
  准确率:    {first_accuracy if isinstance(first_accuracy, str) else f'{first_accuracy:.4f}':<10}  {results['evaluation']['accuracy']:.4f}      {f"+{results['evaluation']['accuracy'] - (first_accuracy if isinstance(first_accuracy, float) else 0):.4f}" if isinstance(first_accuracy, float) else "N/A"}
  AUC:       {first_auc if isinstance(first_auc, str) else f'{first_auc:.4f}':<10}  {results['evaluation']['auc']:.4f}      {f"+{results['evaluation']['auc'] - (first_auc if isinstance(first_auc, float) else 0):.4f}" if isinstance(first_auc, float) else "N/A"}

神经元分析:
  第一次训练H-Neurons总数: {first_neuron_count}
  用于第二次训练的神经元: {results['first_training_info']['selected_for_second_training']}
  第二次训练后H-Neurons数: {results['h_neurons']['h_neuron_count']}
  神经元减少比例: {(1 - results['h_neurons']['h_neuron_count']/results['first_training_info']['selected_for_second_training'])*100:.1f}%
  H-Neurons占总神经元比例: {results['h_neurons']['h_neuron_ratio']*100:.4f}%

权重统计:
  平均权重: {results['h_neurons']['mean_weight']:.6f}
  层分布: {json.dumps(results['h_neurons']['layer_distribution'], indent=2)}

数据统计:
  总样本数: {results['data_statistics']['processed_samples']}
  训练集样本: {results['data_statistics']['train_samples']}
  测试集样本: {results['data_statistics']['test_samples']}
  特征维度: {results['data_statistics']['feature_dimension']}
  正样本比例: {results['data_statistics']['positive_ratio']:.3f}

训练信息:
  开始时间: {results['training_info']['start_time']}
  结束时间: {results['training_info']['end_time']}
  总耗时: {hours}小时{minutes}分钟{seconds}秒

输出文件 ({output_dir}):
  1. h_neurons_classifier.pkl - 分类器(与第一次训练格式相同)
  2. h_neurons.json - H-Neurons列表(与第一次训练格式相同)
  3. experiment_results.json - 完整实验结果
  4. second_training_config.json - 第二次训练配置
  5. second_training_report.txt - 本报告

重要提示:
  1. 保存的文件格式与第一次训练完全相同，可以直接用于推理
  2. 推理代码无需任何修改
  3. 第二次训练进一步精炼了H-Neurons，减少了神经元数量

{'='*80}
"""
        
        self.logger.info(report)
        
        # 保存报告
        report_path = os.path.join(output_dir, "second_training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

# ==================== 主函数 ====================
def main():
    """主函数"""
    # 设置配置
    config = Config()
    
    # 设置日志
    logger = LoggerSetup.setup()
    
    # 创建训练器
    trainer = SecondTrainingTrainer(config)
    
    # 运行训练
    logger.info("开始第二次训练...")
    classifier, output_dir = trainer.train()
    
    return classifier, output_dir

if __name__ == "__main__":
    try:
        print("="*80)
        print("H-Neurons第二次训练启动")
        print("="*80)
        print("训练策略:")
        print("  1. 使用第一次训练得到的1050个H-Neurons")
        print("  2. 从全维度特征文件中提取这些神经元的特征")
        print("  3. 训练L1正则化逻辑回归")
        print("  4. 只保存权重>0的神经元，保存格式与第一次训练完全相同")
        print("="*80)
        
        classifier, output_dir = main()
        
        print("\n" + "="*80)
        print("第二次训练完成!")
        print("="*80)
        print(f"输出目录: {output_dir}")
        print()
        print("重要文件:")
        print(f"  1. h_neurons_classifier.pkl - 可以直接用于推理的分类器")
        print(f"  2. h_neurons.json - H-Neurons列表")
        print(f"  3. second_training_report.txt - 详细训练报告")
        print()
        print("使用说明:")
        print(f"  # 推理时直接使用这个目录下的文件，无需修改代码")
        print(f"  classifier_path = '{output_dir}/h_neurons_classifier.pkl'")
        print(f"  h_neurons_path = '{output_dir}/h_neurons.json'")
        print("="*80)
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()