# -*- coding: utf-8 -*-
"""
知识边界决策器：训练与评估主流程 - 修复版
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 添加项目根目录到Python路径
sys.path.append('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/code/decision_makers')

# 从决策器模块导入
try:
    from decision_makers import (
        load_json_data, create_enhanced_features,
        MLPDetector, AttentionDetector, EnsembleDetector
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试从当前目录导入...")
    # 尝试直接导入函数
    import json
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import pickle


def optimize_threshold(y_true: np.ndarray, y_scores: np.ndarray, verbose: bool = True):
    """
    优化分类阈值：通过最大化Youden指数（敏感性 + 特异性 - 1）选择最佳决策阈值
    """
    # 确保y_scores是1维数组
    if len(y_scores.shape) > 1:
        if y_scores.shape[1] > 1:
            y_scores = y_scores[:, 1]
        else:
            y_scores = y_scores.flatten()
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # 确保没有NaN值
    valid_idx = np.isfinite(tpr) & np.isfinite(fpr) & np.isfinite(thresholds)
    fpr = fpr[valid_idx]
    tpr = tpr[valid_idx]
    thresholds = thresholds[valid_idx]
    
    if len(thresholds) == 0:
        print("警告: 无法计算有效阈值，使用默认阈值0.5")
        return 0.5, 0.0
    
    # Youden指数 = TPR - FPR = 敏感性 + 特异性 - 1
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    best_youden = youden_index[best_idx]
    
    if verbose:
        print(f"\n【阈值优化结果】")
        print(f"  最佳阈值: {best_threshold:.4f}")
        print(f"  Youden指数: {best_youden:.4f}")
        print(f"  敏感性 (TPR): {tpr[best_idx]:.4f}")
        print(f"  特异性 (1-FPR): {1 - fpr[best_idx]:.4f}")
    
    return best_threshold, best_youden


def evaluate_model(detector, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model"):
    """
    评估单个模型在测试集上的性能
    """
    print(f"\n{'='*60}")
    print(f"正在评估模型：{model_name}")
    print(f"{'='*60}")
    
    try:
        # 获取预测标签和预测概率
        y_pred = detector.predict(X_test)
        y_scores = detector.predict_proba(X_test)
        
        # 确保y_scores是1维数组
        if len(y_scores.shape) > 1:
            if y_scores.shape[1] > 1:
                y_scores = y_scores[:, 1]
            else:
                y_scores = y_scores.flatten()
        
        # 计算常用分类指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 检查y_scores是否有有效值用于计算AUC
        if len(np.unique(y_test)) < 2:
            print("警告: 测试集只有一个类别，无法计算AUC")
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(y_test, y_scores)
            except:
                print("警告: 无法计算AUC，设置为0.5")
                auc = 0.5
        
        # 从混淆矩阵中提取 TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sensitivity + specificity - 1
        
        # 打印评估结果
        print(f"\n【性能指标】")
        print(f"  准确率 (Accuracy):    {accuracy:.4f}")
        print(f"  精确率 (Precision):   {precision:.4f}")
        print(f"  召回率 (Recall):      {recall:.4f}")
        print(f"  F1分数:               {f1:.4f}")
        print(f"  AUC-ROC:              {auc:.4f}")
        print(f"  敏感性 (Sensitivity): {sensitivity:.4f}")
        print(f"  特异性 (Specificity): {specificity:.4f}")
        print(f"  Youden指数:           {youden:.4f}")
        
        print(f"\n【混淆矩阵】")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        
        print(f"\n【分类报告】")
        print(classification_report(y_test, y_pred, 
                                    target_names=['错误', '正确'],
                                    digits=4))
        
        # 封装结果
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'youden': youden,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        return results
        
    except Exception as e:
        print(f"评估模型时出错: {e}")
        return None


def plot_roc_curves(detectors: dict, X_test: np.ndarray, y_test: np.ndarray, save_path: str = 'roc_curves.png'):
    """
    绘制多个模型的ROC曲线对比图
    """
    plt.figure(figsize=(10, 8))
    
    for name, detector in detectors.items():
        try:
            y_scores = detector.predict_proba(X_test)
            
            # 确保y_scores是1维数组
            if len(y_scores.shape) > 1:
                if y_scores.shape[1] > 1:
                    y_scores = y_scores[:, 1]
                else:
                    y_scores = y_scores.flatten()
            
            # 检查是否有足够的正负样本
            if len(np.unique(y_test)) < 2:
                print(f"警告: {name} 测试集只有一个类别，跳过ROC曲线")
                continue
                
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            auc = roc_auc_score(y_test, y_scores)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        except Exception as e:
            print(f"绘制 {name} 的ROC曲线时出错: {e}")
    
    # 添加随机分类器参考线
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类器 (AUC = 0.500)')
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('ROC曲线对比', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nROC曲线已保存至: {save_path}")
    plt.close()


def plot_confusion_matrices(results_dict: dict, save_path: str = 'confusion_matrices.png'):
    """
    绘制各模型的混淆矩阵热力图对比
    """
    n_models = len(results_dict)
    if n_models == 0:
        print("没有有效的结果可绘制混淆矩阵")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, results) in zip(axes, results_dict.items()):
        if results is None:
            continue
            
        cm = results['confusion_matrix']
        cm_array = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['预测: 错误', '预测: 正确'],
                    yticklabels=['真实: 错误', '真实: 正确'])
        ax.set_title(f'{name}\n准确率: {results["accuracy"]:.3f}', 
                     fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存至: {save_path}")
    plt.close()


def check_data_quality(X, y, dataset_name):
    """
    检查数据质量
    """
    print(f"\n【{dataset_name} 数据检查】")
    print(f"  样本数: {len(y)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  正样本数: {np.sum(y)} ({np.mean(y)*100:.2f}%)")
    print(f"  负样本数: {len(y)-np.sum(y)} ({(1-np.mean(y))*100:.2f}%)")
    
    # 检查NaN和Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  NaN值数量: {nan_count}")
    print(f"  Inf值数量: {inf_count}")
    
    # 检查特征范围
    for i in range(min(3, X.shape[1])):  # 只检查前3个特征
        print(f"  特征{i}: 均值={np.nanmean(X[:, i]):.4f}, "
              f"标准差={np.nanstd(X[:, i]):.4f}, "
              f"最小值={np.nanmin(X[:, i]):.4f}, "
              f"最大值={np.nanmax(X[:, i]):.4f}")
    
    # 处理NaN和Inf
    if nan_count > 0 or inf_count > 0:
        print("  处理NaN和Inf值...")
        # 用均值填充NaN
        for i in range(X.shape[1]):
            col_mean = np.nanmean(X[:, i])
            X[:, i] = np.where(np.isfinite(X[:, i]), X[:, i], col_mean)
    
    return X


def main():
    """主函数：执行完整的训练与评估流程"""
    
    print("="*60)
    print("知识边界决策器 - 训练与评估 (修复版)")
    print("="*60)
    
    # ========== 第1步：加载数据 ==========
    print("\n【第1步】加载数据...")
    try:
        X_train, y_train, train_data = load_json_data('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/train/evaluation_results.json', score_threshold=4)
        X_test, y_test, test_data = load_json_data('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results.json', score_threshold=4)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return
    
    # ========== 第2步：特征工程和数据检查 ==========
    print("\n【第2步】特征工程和数据检查...")
    X_train_enhanced = create_enhanced_features(X_train)
    X_test_enhanced = create_enhanced_features(X_test)
    
    # 检查数据质量
    X_train_enhanced = check_data_quality(X_train_enhanced, y_train, "训练集")
    X_test_enhanced = check_data_quality(X_test_enhanced, y_test, "测试集")
    
    print(f"增强后特征维度: {X_train_enhanced.shape[1]}")
    
    # ========== 第3步：训练三种决策器 ==========
    print("\n【第3步】训练三种决策模型...")
    
    detectors = {}
    results_dict = {}
    
    # 方案1: 多层感知机 (MLP)
    print("\n" + "-"*60)
    print("正在训练 MLP 决策器...")
    print("-"*60)
    try:
        mlp_detector = MLPDetector(input_dim=13, hidden_dims=[64, 32, 16], dropout=0.3)
        # 强制使用CPU以避免CUDA错误
        mlp_detector.device = 'cpu'
        mlp_detector.model = mlp_detector.model.to('cpu')
        
        # 检查标签范围
        unique_labels = np.unique(y_train)
        print(f"训练标签唯一值: {unique_labels}")
        
        mlp_detector.fit(X_train_enhanced, y_train, epochs=50, batch_size=16, lr=0.001, verbose=True)
        
        # 在训练集上优化阈值
        train_scores = mlp_detector.predict_proba(X_train_enhanced)
        mlp_detector.threshold, _ = optimize_threshold(y_train, train_scores, verbose=True)
        detectors['MLP'] = mlp_detector
        
        # 评估MLP
        results = evaluate_model(mlp_detector, X_test_enhanced, y_test, model_name='MLP')
        results_dict['MLP'] = results
        
    except Exception as e:
        print(f"训练MLP时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 方案2: 注意力机制模型
    print("\n" + "-"*60)
    print("正在训练 注意力机制 决策器...")
    print("-"*60)
    try:
        attn_detector = AttentionDetector(feature_dim=13, hidden_dim=64)
        # 强制使用CPU
        attn_detector.device = 'cpu'
        attn_detector.model = attn_detector.model.to('cpu')
        
        attn_detector.fit(X_train_enhanced, y_train, epochs=50, batch_size=16, lr=0.001, verbose=True)
        
        # 在训练集上优化阈值
        train_scores = attn_detector.predict_proba(X_train_enhanced)
        attn_detector.threshold, _ = optimize_threshold(y_train, train_scores, verbose=True)
        detectors['Attention'] = attn_detector
        
        # 评估Attention
        results = evaluate_model(attn_detector, X_test_enhanced, y_test, model_name='Attention')
        results_dict['Attention'] = results
        
    except Exception as e:
        print(f"训练Attention时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 方案3: 集成模型（融合多个基模型）
    print("\n" + "-"*60)
    print("正在训练 集成 决策器...")
    print("-"*60)
    try:
        ensemble_detector = EnsembleDetector()
        ensemble_detector.fit(X_train_enhanced, y_train, verbose=True)
        
        # 在训练集上优化阈值
        train_scores = ensemble_detector.predict_proba(X_train_enhanced)
        ensemble_detector.threshold, _ = optimize_threshold(y_train, train_scores, verbose=True)
        detectors['Ensemble'] = ensemble_detector
        
        # 评估Ensemble
        results = evaluate_model(ensemble_detector, X_test_enhanced, y_test, model_name='Ensemble')
        results_dict['Ensemble'] = results
        
    except Exception as e:
        print(f"训练集成模型时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 第4步：可视化对比 ==========
    print("\n【第4步】生成可视化图表...")
    
    # 确保目录存在
    os.makedirs('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models', exist_ok=True)
    
    # 绘制ROC曲线
    if detectors:
        plot_roc_curves(detectors, X_test_enhanced, y_test, 
                        save_path='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/roc_curves.png')
    
    # 绘制混淆矩阵
    if results_dict:
        plot_confusion_matrices(results_dict, 
                               save_path='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/confusion_matrices.png')
    
    # ========== 第5步：打印性能对比表格 ==========
    print("\n【第5步】模型性能对比表:")
    print("\n" + "="*90)
    print(f"{'模型':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'AUC':<10} {'Youden':<10}")
    print("="*90)
    
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    for name, results in valid_results.items():
        print(f"{name:<15} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
              f"{results['recall']:<10.4f} {results['f1']:<10.4f} "
              f"{results['auc']:<10.4f} {results['youden']:<10.4f}")
    print("="*90)
    
    # ========== 第6步：保存模型 ==========
    print("\n【第6步】保存训练好的模型...")
    try:
        if 'MLP' in detectors:
            mlp_detector.save('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/mlp_detector.pth')
            print("MLP模型已保存")
        
        if 'Attention' in detectors:
            attn_detector.save('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/attention_detector.pth')
            print("Attention模型已保存")
        
        if 'Ensemble' in detectors:
            ensemble_detector.save('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/ensemble_detector.pkl')
            print("集成模型已保存")
            
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # ========== 第7步：输出集成模型的特征重要性 ==========
    print("\n【第7步】集成模型的特征重要性分析:")
    try:
        if 'Ensemble' in detectors:
            # 定义特征名称（与 create_enhanced_features 一致）
            feature_names = [
                '困惑度(PPL)', '最小概率(Min_Prob)', '置信度(Confidence)', '对数困惑度(Log_PPL)',
                'PPL×Min', 'Min×Conf', 'PPL×Conf',
                'Conf/PPL', 'Min/PPL',
                'PPL²', 'Min²', 'Conf²',
                '三元组合(Triple)'
            ]

            importance_dict = ensemble_detector.get_feature_importance()
            for model_name, importances in importance_dict.items():
                print(f"\n{model_name} 的特征重要性:")
                for feat, imp in zip(feature_names, importances):
                    print(f"  {feat:<20} {imp:.4f}")
    except Exception as e:
        print(f"获取特征重要性时出错: {e}")
    
    print("\n" + "="*60)
    print("训练与评估流程已完成！")
    print("="*60)


if __name__ == '__main__':
    # 设置环境变量以避免CUDA错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    main()