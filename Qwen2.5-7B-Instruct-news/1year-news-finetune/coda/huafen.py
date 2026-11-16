# import json
# import random
# from collections import defaultdict

# # 设置随机种子以确保可复现（可选）
# random.seed(42)

# # 读取你的原始数据（假设是 JSON 数组格式）
# with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/combined_qa.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 按 (source_id, type) 分组
# groups = defaultdict(list)
# for item in data:
#     key = (item["source_id"], item["type"])
#     groups[key].append(item)

# # 初始化划分结果
# train_set = []
# val_set = []
# test_set = []

# # 对每个组单独划分
# for key, group_data in groups.items():
#     # 打乱当前组的数据
#     shuffled = group_data.copy()
#     random.shuffle(shuffled)
    
#     n = len(shuffled)
#     if n < 10:
#         # 如果少于10条，仍尽量按比例分配（避免空集）
#         train_n = max(1, int(0.8 * n))
#         val_n = max(1, int(0.1 * n))
#         test_n = n - train_n - val_n
#         if test_n <= 0:
#             test_n = 1
#             train_n = n - val_n - test_n
#             if train_n < 0:
#                 train_n = 0
#     else:
#         # 正常 8:1:1
#         train_n = int(0.8 * n)
#         val_n = int(0.1 * n)
#         test_n = n - train_n - val_n  # 确保总数不变

#     train_set.extend(shuffled[:train_n])
#     val_set.extend(shuffled[train_n : train_n + val_n])
#     test_set.extend(shuffled[train_n + val_n:])

# # 保存结果
# with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/train.json', 'w', encoding='utf-8') as f:
#     json.dump(train_set, f, ensure_ascii=False, indent=2)

# with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/val.json', 'w', encoding='utf-8') as f:
#     json.dump(val_set, f, ensure_ascii=False, indent=2)

# with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/test.json', 'w', encoding='utf-8') as f:
#     json.dump(test_set, f, ensure_ascii=False, indent=2)

# print(f"划分完成！")
# print(f"训练集: {len(train_set)} 条")
# print(f"验证集: {len(val_set)} 条")
# print(f"测试集: {len(test_set)} 条")

import json
import os

# 配置输入输出路径
ANSWERED_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected/augmented_qa_full_28000.json"      # 第一个文件（有真实答案）
REFUSED_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected/unmentioned_qa/unmentioned_qa_batch.json"        # 第二个文件（拒答）
OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected/sft_train.json"          # 输出文件

def load_answered_data(filepath):
    """加载第一个文件（answered），转换字段"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted = []
    for item in data:
        converted.append({
            "question": item["question"],
            "answer": item["answer"],
            "source_id": item["news_id"],  # 映射 news_id -> source_id
            "type": "answered"             # 或 item.get("type", "answered")
        })
    return converted

def load_refused_data(filepath):
    """加载第二个文件（refused），补充 type 字段"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted = []
    for item in data:
        converted.append({
            "question": item["question"],
            "answer": item["answer"],
            "source_id": item["source_id"],
            "type": "refused"
        })
    return converted

def main():
    # 加载两部分数据
    answered = load_answered_data(ANSWERED_FILE)
    refused = load_refused_data(REFUSED_FILE)
    
    # 合并
    merged = answered + refused
    
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 合并完成！")
    print(f"- 已回答问答对: {len(answered)} 条")
    print(f"- 拒答问答对: {len(refused)} 条")
    print(f"- 总计: {len(merged)} 条")
    print(f"- 输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()