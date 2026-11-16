import json
import numpy as np
from decision_makers import create_enhanced_features, MLPDetector, AttentionDetector, EnsembleDetector

def predict_and_update(input_file: str, output_file: str, 
                       model_path: str, model_type: str = 'mlp',
                       verbose: bool = True):
    """
    简化的预测函数
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        model_path: 模型文件路径
        model_type: 模型类型 ('mlp', 'attention', 'ensemble')
        verbose: 是否打印详细信息
    """
    
    print("="*60)
    print("Knowledge Boundary Prediction - Simple Mode")
    print("="*60)
    
    # 1. 加载数据
    print(f"\n[Step 1] Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    print(f"Loaded {len(data)} samples")
    
    # 2. 加载模型
    print(f"\n[Step 2] Loading {model_type.upper()} model...")
    if model_type == 'mlp':
        detector = MLPDetector()
        detector.load(model_path)
    elif model_type == 'attention':
        detector = AttentionDetector()
        detector.load(model_path)
    elif model_type == 'ensemble':
        detector = EnsembleDetector()
        detector.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Model loaded! Threshold: {detector.threshold:.4f}")
    
    # 3. 批量预测
    print(f"\n[Step 3] Making predictions...")
    
    answer_count = 0
    refuse_count = 0
    
    for i, item in enumerate(data):
        # 提取特征
        refusal = item.get('refusal_decision', {})
        ppl = refusal.get('ppl', 0.0)
        min_prob = refusal.get('min_token_prob', 0.0)
        confidence = refusal.get('self_reported_confidence', 0.0)
        
        features = np.array([[ppl, min_prob, confidence]], dtype=np.float32)
        features_enhanced = create_enhanced_features(features)
        
        # 预测
        prediction = detector.predict(features_enhanced)[0]
        confidence_score = detector.predict_proba(features_enhanced)[0]
        
        # 决策
        should_answer = bool(prediction == 1)
        
        if should_answer:
            final_answer = item.get('model_answer', '')
            answer_count += 1
        else:
            final_answer = "很抱歉，我无法回答这个问题。"
            refuse_count += 1
        
        # 更新字段
        item['refusal_decision']['should_answer'] = should_answer
        item['refusal_decision']['final_model_answer'] = final_answer
        item['refusal_decision']['decision_confidence'] = float(confidence_score)
        item['refusal_decision']['decision_model'] = model_type
        
        # 打印进度
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples...")
    
    # 4. 保存结果
    print(f"\n[Step 4] Saving results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_file}")
    
    # 5. 统计结果
    print(f"\n[Summary]")
    print(f"  Total samples: {len(data)}")
    print(f"  Should answer: {answer_count} ({answer_count/len(data)*100:.1f}%)")
    print(f"  Should refuse: {refuse_count} ({refuse_count/len(data)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Prediction completed!")
    print("="*60)


if __name__ == '__main__':
    # 使用示例
    # predict_and_update(
    #     input_file='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results.json',           # 输入文件
    #     output_file='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results_mlp.json', # 输出文件
    #     model_path='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/mlp_detector.pth',   # 模型路径
    #     model_type='mlp',                       # 模型类型
    #     verbose=True                            # 显示详细信息
    # )
    
    # 如果要使用其他模型，取消注释下面的代码：
    
    # 使用Attention模型
    predict_and_update(
        input_file='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results.json',
        output_file='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results_attention.json',
        model_path='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/attention_detector.pth',
        model_type='attention',
        verbose=True
    )
    
    # 使用Ensemble模型
    predict_and_update(
        input_file='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results.json',
        output_file='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results_ensemble.json',
        model_path='/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/models/decide_models/ensemble_detector.pkl',
        model_type='ensemble',
        verbose=True
    )