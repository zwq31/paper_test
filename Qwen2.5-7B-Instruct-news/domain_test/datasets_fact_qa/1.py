# import json

# # 1. 加载两个文件
# QA_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/datasets_fact_qa/domain_test_result.json"
# SOURCE_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/1year_news_filtered.json"

# with open(QA_FILE, "r", encoding="utf-8") as f:
#     qa_data = json.load(f)  # 假设是列表

# with open(SOURCE_FILE, "r", encoding="utf-8") as f:
#     source_docs = json.load(f)

# # 2. 构建 id -> {content, pub_date} 的映射（正确方式）
# source_map = {
#     doc["id"]: {
#         "content": doc["content"],
#         "pub_date": doc["pub_date"]
#     }
#     for doc in source_docs
# }

# # 3. 提取并合并数据
# merged_data = []
# for item in qa_data:
#     source_id = item.get("source_id")
#     if source_id not in source_map:
#         print(f"⚠️ 未找到 source_id={source_id} 的内容")
#         continue

#     # 提取 source 中的字段
#     source_info = source_map[source_id]

#     # 构造合并后的条目
#     merged_item = {
#         "question": item["question"],
#         "answer": item["answer"],
#         "mol_cpt_qlora_model_answer_prompt": item["mol_cpt_qlora_model_answer_prompt"],
#         "source_id": source_id,
#         "pub_date": source_info["pub_date"],      # 从 source 取 pub_date
#         "context": source_info["content"]         # 从 source 取 content
#     }
#     merged_data.append(merged_item)

# # 4. 保存合并后的结果
# OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/datasets_fact_qa/cpt_qlora_result_prompt.json"
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(merged_data, f, ensure_ascii=False, indent=2)

# print(f"✅ 合并完成，共 {len(merged_data)} 条数据，已保存至 {OUTPUT_FILE}")

# ========================================================整理出调用大模型评估失败的样本====================================================
# import json

# # 请根据你的实际文件名修改
# FAILED_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result5_lou/evaluation_results.json"      # 第一个文件：包含评估失败的文件
# ORIGINAL_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result5.json"         # 第二个文件：原始完整数据文件
# OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/lou55.json" # 输出文件

# def extract_failed_samples():
#     # 1. 读取第一个文件（带评估失败标记）
#     with open(FAILED_FILE, 'r', encoding='utf-8') as f:
#         failed_data = json.load(f)

#     # 2. 读取第二个文件（原始完整数据）
#     with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
#         original_data = json.load(f)

#     # 3. 构建查找索引：以 (source_id, question) 为 key
#     # 因为 source_id 可能重复，question 是唯一标识
#     original_index = {}
#     for item in original_data:
#         key = (item['source_id'], item['question'])
#         original_index[key] = item

#     # 4. 遍历第一个文件，找出所有评估失败的样本
#     failed_samples = []
#     for item in failed_data:
#         correctness = item.get("evaluation_results", {}).get("correctness", {})
#         reason = correctness.get("reason", "")
        
#         if "评估失败" in reason or "Expecting value" in reason:
#             key = (item['source_id'], item['question'])
#             if key in original_index:
#                 failed_samples.append(original_index[key])
#             else:
#                 print(f"⚠️ 未在原始文件中找到匹配项: source_id={item['source_id']}, question={item['question'][:50]}...")

#     # 5. 保存结果
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         json.dump(failed_samples, f, ensure_ascii=False, indent=2)

#     print(f"✅ 提取完成！共找到 {len(failed_samples)} 条评估失败的原始数据，已保存至: {OUTPUT_FILE}")

# # 执行
# if __name__ == "__main__":
#     extract_failed_samples()

# ======================================================统计每个id的次数===================================================
import json
from collections import Counter

# 假设你的数据是一个 JSON 文件，比如 data.json
INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result4/evaluation_results.json"   # 替换为你的实际文件名

def find_source_id_duplicates():
    # 1. 读取 JSON 数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 统计每个 source_id 出现的次数
    source_ids = [item['source_id'] for item in data]
    counter = Counter(source_ids)
    
    # 3. 找出出现次数 > 2 的 source_id
    duplicated_ids = {sid: count for sid, count in counter.items() if count > 2}
    
    if not duplicated_ids:
        print("✅ 没有 source_id 出现超过 2 次。")
        return
    
    print("⚠️ 以下 source_id 出现次数超过 2 次：")
    for sid, count in sorted(duplicated_ids.items()):
        print(f"   source_id = {sid}, 出现 {count} 次")
    
# 执行
if __name__ == "__main__":
    find_source_id_duplicates()

# ======================================================过滤评估失败样本========================================================
# import json

# # 文件路径配置
# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result5/evaluation_results.json"           # 输入文件，替换为你的实际文件名
# OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/train_generate_result/result5/evaluation_results.json"  # 输出文件名

# def remove_evaluation_failures():
#     # 1. 读取原始 JSON 数据
#     with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 2. 过滤：保留不是“评估失败”的样本
#     cleaned_data = []
#     removed_count = 0

#     for item in data:
#         reason = item.get("evaluation_results", {}).get("correctness", {}).get("reason", "")
        
#         # 判断是否是你要删除的失败类型
#         if reason == "评估失败: Expecting value: line 1 column 1 (char 0)":
#             removed_count += 1
#             continue  # 跳过，不加入结果
#         else:
#             cleaned_data.append(item)
    
#     # 3. 保存过滤后的数据
#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
#     # 4. 输出统计信息
#     print(f"✅ 过滤完成！")
#     print(f"   原始样本数: {len(data)}")
#     print(f"   删除评估失败样本数: {removed_count}")
#     print(f"   剩余有效样本数: {len(cleaned_data)}")
#     print(f"   已保存至: {OUTPUT_FILE}")

# # 执行
# if __name__ == "__main__":
#     remove_evaluation_failures()