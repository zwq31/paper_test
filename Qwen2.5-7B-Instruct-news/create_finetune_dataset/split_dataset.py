import json
import os
import random
from collections import defaultdict
import pandas as pd

def main():
    # 创建datasets目录（如果不存在）
    os.makedirs("datasets", exist_ok=True)
    
    # 读取原始数据集
    with open("datasets/combined_qa.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"原始数据集包含 {len(data)} 条记录")
    
    # 按source_id和type进行分组
    grouped_data = defaultdict(list)
    for item in data:
        # 尝试获取source_id，兼容不同的字段名
        source_id = None
        if "source_id" in item:
            source_id = item["source_id"]
        elif "sourse_id" in item:  # 注意原数据中的拼写错误
            source_id = item["sourse_id"]
        
        # 获取问题类型
        q_type = item.get("type", "unknown")
        
        # 使用元组(source_id, type)作为键
        key = (str(source_id), q_type)
        grouped_data[key].append(item)
    
    print(f"按source_id和type分组后共有 {len(grouped_data)} 个组")
    
    # 为每个组的数据随机打乱顺序
    for key in grouped_data:
        random.shuffle(grouped_data[key])
    
    # 按照8:1:1的比例划分数据
    train_data = []
    val_data = []
    test_data = []
    
    for key, items in grouped_data.items():
        total = len(items)
        
        # 计算划分点
        train_end = int(total * 0.8)
        val_end = train_end + int(total * 0.1)
        
        # 划分数据
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
    
    # 随机打乱各集合中的数据
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 输出数据集统计信息
    def print_stats(dataset, name):
        source_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for item in dataset:
            source_id = item.get("source_id") or item.get("sourse_id")
            source_counts[str(source_id)] += 1
            type_counts[item.get("type", "unknown")] += 1
        
        print(f"\n{name} 统计信息:")
        print(f"按source_id分布: {dict(source_counts)}")
        print(f"按type分布: {dict(type_counts)}")
    
    print_stats(train_data, "训练集")
    print_stats(val_data, "验证集")
    print_stats(test_data, "测试集")
    
    # 保存划分后的数据集
    with open("datasets/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("datasets/val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open("datasets/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("\n数据集已保存到 datasets 目录")
    
    # # 转换为训练格式（可选）
    # convert_to_training_format(train_data, "datasets/train_format.json")
    # convert_to_training_format(val_data, "datasets/val_format.json")
    
    print("训练格式数据集已保存")

# def convert_to_training_format(data, output_file):
#     """将数据转换为适合Qwen2.5模型训练的格式"""
#     formatted_data = []
    
#     for item in data:
#         formatted_item = {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": f"回答以下问题：{item['question']}"
#                 },
#                 {
#                     "role": "assistant",
#                     "content": item.get("answer", "")
#                 }
#             ]
#         }
#         formatted_data.append(formatted_item)
    
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(formatted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 