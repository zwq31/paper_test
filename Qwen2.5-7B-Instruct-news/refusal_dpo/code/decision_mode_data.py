import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data_for_decision_model(input_file, output_file, score_threshold=4):
    """
    将原始数据预处理为决策模型训练格式
    """
    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    processed_data = []
    
    for sample in raw_data:
        try:
            # 提取特征
            ppl = sample.get("refusal_decision", {}).get("ppl", 0)
            min_prob = sample.get("refusal_decision", {}).get("min_token_prob", 0)
            
            # 提取分数并确定标签
            score = sample.get("evaluation_results", {}).get("correctness", {}).get("score", 0)
            
            # 标签：1表示应该拒答(score<4)，0表示应该回答(score>=4)
            label = 1 if score < score_threshold else 0
            
            # 构建训练样本
            training_sample = {
                "ppl": float(ppl),
                "min_token_prob": float(min_prob),
                "label": int(label),
                "score": int(score),
                "question": sample.get("question", ""),
                "source_id": sample.get("source_id", 0)
            }
            
            processed_data.append(training_sample)
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 数据统计
    df = pd.DataFrame(processed_data)
    print(f"✅ 数据预处理完成!")
    print(f"   总样本数: {len(processed_data)}")
    print(f"   特征维度: 2 (ppl, min_token_prob)")
    print(f"   标签分布:")
    print(f"     - 应该回答 (label=0): {sum(df['label'] == 0)} 条")
    print(f"     - 应该拒答 (label=1): {sum(df['label'] == 1)} 条")
    print(f"   拒答比例: {sum(df['label'] == 1) / len(df):.2%}")
    
    # 特征统计
    print(f"\n📊 特征统计:")
    print(f"   PPL - 均值: {df['ppl'].mean():.4f}, 标准差: {df['ppl'].std():.4f}")
    print(f"   Min Token Prob - 均值: {df['min_token_prob'].mean():.4f}, 标准差: {df['min_token_prob'].std():.4f}")
    
    return processed_data

def analyze_features(processed_data):
    """分析特征分布和相关性"""
    df = pd.DataFrame(processed_data)
    
    print("\n🔍 特征分析:")
    
    # 特征与标签的相关性
    correlation = df[['ppl', 'min_token_prob', 'label']].corr()
    print("特征相关性矩阵:")
    print(correlation)
    
    # 按标签分组统计特征
    print("\n按标签分组的特征统计:")
    grouped = df.groupby('label')[['ppl', 'min_token_prob']].agg(['mean', 'std', 'min', 'max'])
    print(grouped)
    
    return df

# 使用示例
if __name__ == "__main__":
    INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_5epoch/evaluation_results.json"  # 替换为您的原始数据路径
    OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/decision_data/decision_training_data.json"
    
    # 预处理数据
    processed_data = preprocess_data_for_decision_model(INPUT_FILE, OUTPUT_FILE)
    
    # 分析特征
    df = analyze_features(processed_data)