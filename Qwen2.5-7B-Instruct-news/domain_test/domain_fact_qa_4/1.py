# 分割数据集

# import json
# from collections import defaultdict
# import math

# # 读取原始数据
# input_file = '/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/fact_qa_4.json'  # 替换为你的文件路径
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 按 id 分组
# groups = defaultdict(list)
# for item in data:
#     groups[item['id']].append(item)

# # 初始化两个输出文件的数据
# part1 = []
# part2 = []

# # 处理每个 id
# for id_key, items in groups.items():
#     n = len(items)
#     if n == 4:
#         # 各分两个
#         part1.extend(items[:2])
#         part2.extend(items[2:])
#     elif n == 3:
#         # 一个分两个，一个分一个
#         part1.extend(items[:2])
#         part2.extend(items[2:])
#     elif n == 2:
#         # 各分一个
#         part1.append(items[0])
#         part2.append(items[1])
#     else:
#         # 理论上不会出现，但以防万一
#         print(f"Warning: id={id_key} has {n} items, less than 2. Skipping.")
#         continue

# # 写入文件
# with open('/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train.json', 'w', encoding='utf-8') as f:
#     json.dump(part1, f, ensure_ascii=False, indent=2)

# with open('/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test.json', 'w', encoding='utf-8') as f:
#     json.dump(part2, f, ensure_ascii=False, indent=2)

# print("✅ 分割完成：part1.json 和 part2.json 已生成")

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

# # +++++++++++++++++++++++++++++++++++++++++++++评估失败样本提取++++++++++++++++++++++++++++++++++++++++++++++++++++++
import json

# 请根据你的实际文件名修改
FAILED_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_yuanshi/test_v2/test_claude111111111111/evaluation_results.json"      # 第一个文件：包含评估失败的文件
ORIGINAL_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_yuanshi/test_v2/test_claude.json"         # 第二个文件：原始完整数据文件
OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_yuanshi/test_v2/lou.json" # 输出文件

def extract_failed_samples():
    # 1. 读取第一个文件（带评估失败标记）
    with open(FAILED_FILE, 'r', encoding='utf-8') as f:
        failed_data = json.load(f)

    # 2. 读取第二个文件（原始完整数据）
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 3. 构建查找索引：以 (source_id, question) 为 key
    # 因为 source_id 可能重复，question 是唯一标识
    original_index = {}
    for item in original_data:
        key = (item['source_id'], item['question'])
        original_index[key] = item

    # 4. 遍历第一个文件，找出所有评估失败的样本
    failed_samples = []
    for item in failed_data:
        correctness = item.get("evaluation_results", {}).get("correctness", {})
        reason = correctness.get("reason", "")
        
        if "评估失败" in reason or "Expecting value" in reason:
            key = (item['source_id'], item['question'])
            if key in original_index:
                failed_samples.append(original_index[key])
            else:
                print(f"⚠️ 未在原始文件中找到匹配项: source_id={item['source_id']}, question={item['question'][:50]}...")

    # 5. 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(failed_samples, f, ensure_ascii=False, indent=2)

    print(f"✅ 提取完成！共找到 {len(failed_samples)} 条评估失败的原始数据，已保存至: {OUTPUT_FILE}")

# 执行
if __name__ == "__main__":
    extract_failed_samples()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++多个文件合并++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -*- coding: utf-8 -*-
# import json

# # ==============================
# # 🔧 在这里直接填写你要合并的 JSON 文件路径（支持相对路径或绝对路径）
# # ==============================
# FILE_LIST = [
#     "/path/to/file1.json",
#     "/path/to/file2.json",
#     "/path/to/file3.json",
#     # 你可以添加任意多个
# ]

# OUTPUT_FILE = "merged_output.json"  # 合并后的输出文件名

# def main():
#     merged_data = []

#     for file_path in FILE_LIST:
#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#                 if isinstance(data, list):
#                     merged_data.extend(data)
#                 else:
#                     merged_data.append(data)
#             print(f"✅ 已加载: {file_path}")
#         except FileNotFoundError:
#             print(f"❌ 文件未找到: {file_path}")
#         except json.JSONDecodeError as e:
#             print(f"❌ JSON 格式错误 ({file_path}): {e}")
#         except Exception as e:
#             print(f"❌ 读取失败 ({file_path}): {e}")

#     # 保存合并结果
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
#         json.dump(merged_data, f_out, indent=2, ensure_ascii=False)

#     print(f"\n🎉 合并完成！共 {len(merged_data)} 条记录，已保存至: {OUTPUT_FILE}")

# if __name__ == "__main__":
#     main()
# +++++++++++++++++++++++++++++++++++++++++++++++++++统计每个id数据个数+++++++++++++++++++++++++++++++++++++++
# import json
# from collections import Counter

# # 假设你的数据是一个 JSON 文件，比如 data.json
# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_test_dataset/evaluation_results.json"   # 替换为你的实际文件名

# def find_source_id_duplicates():
#     # 1. 读取 JSON 数据
#     with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 2. 统计每个 source_id 出现的次数
#     source_ids = [item['source_id'] for item in data]
#     counter = Counter(source_ids)
    
#     # 3. 找出出现次数 > 2 的 source_id
#     duplicated_ids = {sid: count for sid, count in counter.items() if count > 2}
    
#     if not duplicated_ids:
#         print("✅ 没有 source_id 出现超过 2 次。")
#         return
    
#     print("⚠️ 以下 source_id 出现次数超过 2 次：")
#     for sid, count in sorted(duplicated_ids.items()):
#         print(f"   source_id = {sid}, 出现 {count} 次")
    
# # 执行
# if __name__ == "__main__":
#     find_source_id_duplicates()
# # ++++++++++++++++++++++++++++++++++++++++++++++++++删除评估失败样本+++++++++++++++++++++++++++++++++++++
# import json

# # 文件路径配置
# INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_test/output/sft_10to1_refusal/test_data_evaluation/evaluation_results.json"           # 输入文件，替换为你的实际文件名
# OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/new_test/output/sft_10to1_refusal/test_data_evaluation/evaluation_results.json"  # 输出文件名

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

# import json

# # 文件路径（请根据实际情况修改）
# file1_path = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora_test/evaluation_results.json"        # 有 mol_cpt_qlora_model_answer 的文件
# file2_path = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train.json"        # 有 source_id 的文件
# output_path = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora_test/evaluation_results.json"  # 输出文件

# # 读取第一个文件（待补全）
# with open(file1_path, 'r', encoding='utf-8') as f:
#     data1 = json.load(f)

# # 读取第二个文件（提供 source_id）
# with open(file2_path, 'r', encoding='utf-8') as f:
#     data2 = json.load(f)

# # 构建 question -> source_id 映射表
# source_id_map = {}
# for item in data2:
#     question = item.get("question")
#     if question:  # 确保 question 存在
#         source_id_map[question] = item.get("source_id")

# # 补全第一个文件的 source_id
# updated_data = []
# for item in data1:
#     question = item.get("question")
#     if question in source_id_map:
#         # 更新 source_id
#         item["source_id"] = source_id_map[question]
#     else:
#         # 如果找不到，可以选择保留原值或设为 null
#         if "source_id" not in item or item["source_id"] is None:
#             print(f"⚠️ 未找到匹配的 source_id: {question}")
#     updated_data.append(item)

# # 写入结果
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(updated_data, f, ensure_ascii=False, indent=2)

# print(f"✅ 补全完成！共处理 {len(updated_data)} 条数据。")
# print(f"结果已保存至: {output_path}")


# # # ++++++++++++++++++++++++++++++++++++++ 构造偏好数据集 ++++++++++++++++++++++++++++++++++++++++
# import json
# from collections import defaultdict
# import random

# # 1. 读取5个json文件
# file_paths = [
#     '/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora/evaluation_results.json',
#     '/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora1/evaluation_results.json',
#     '/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora2/evaluation_results.json',
#     '/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora3/evaluation_results.json',
#     '/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/evaluate_mol_qlora4/evaluation_results.json'
# ]

# # 存储所有数据
# all_data = []
# for file_path in file_paths:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         if isinstance(data, list):
#             all_data.extend(data)
#         else:
#             all_data.append(data)

# # 按 source_id 和 question 分组
# grouped = defaultdict(list)
# for item in all_data:
#     key = (item['source_id'], item['question'])
#     grouped[key].append(item)

# # 读取 test_refusal.json，获取拒绝回答
# lou_data = {}
# with open('/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_refusal.json', 'r', encoding='utf-8') as f:
#     lou_list = json.load(f)
#     for item in lou_list:
#         lou_data[item['source_id']] = {
#             'question': item['question'],
#             'refusal_answer': item['refusal_answer']
#         }

# # 构建输出结果 + 统计计数（0~5）
# output_list = []
# count_correct = [0] * 6  # count_correct[i] 表示有 i 个 correct == True 的样本数量
# positive_samples = []  # 全对样本（正样本）
# negative_samples = []  # 全错样本（负样本）

# for (source_id, question), group in grouped.items():
#     if source_id not in lou_data:
#         print(f"Warning: source_id {source_id} not found in test_refusal.json")
#         continue

#     try:
#         # 提取 correct 状态
#         correct_list = [item['evaluation_results']['correctness']['correct'] for item in group]
#         num_correct = sum(1 for x in correct_list if x)  # True 的数量
#     except KeyError as e:
#         print(f"KeyError in source_id {source_id}: missing key {e}")
#         continue

#     # 更新统计
#     if 0 <= num_correct <= 5:
#         count_correct[num_correct] += 1

#     # 只有全对（5）或全错（0）才用于构造 SFT 数据
#     refusal_answer = lou_data[source_id]['refusal_answer']
#     first_answer = group[0]['mol_cpt_qlora_model_answer']

#     result = {'source_id': source_id, 'question': question}

#     if num_correct == 5:  # 全对 → 应该回答
#         result['answer'] = first_answer
#         positive_samples.append(result)
#         # output_list.append(result)  # 暂时先不直接添加到最终列表

#     elif num_correct == 0:  # 全错 → 应该拒绝
#         result['answer'] = refusal_answer
#         negative_samples.append(result)
#         # output_list.append(result)  # 暂时先不直接添加到最终列表

# # 根据正样本数量确定负样本数量（1/5比例）
# num_positive = len(positive_samples)
# target_negative_count = max(1, num_positive // 5)  # 至少保留1个负样本

# print(f"📊 全对样本数量: {num_positive}")
# print(f"📊 全错样本总数量: {len(negative_samples)}")
# print(f"📊 目标负样本数量 (正样本的1/5): {target_negative_count}")

# # 随机选择指定数量的负样本
# if len(negative_samples) > target_negative_count:
#     selected_negative_samples = random.sample(negative_samples, target_negative_count)
# else:
#     selected_negative_samples = negative_samples  # 如果负样本总数不足目标数量，则全部保留

# print(f"📊 实际保留的负样本数量: {len(selected_negative_samples)}")

# # 将全对样本（正样本）全部加入输出列表
# output_list.extend(positive_samples)
# # 将按比例选择的全错样本（负样本）加入输出列表
# output_list.extend(selected_negative_samples)

# # 保存 SFT 训练数据
# with open('refusal_sft_train_data_5:1.json', 'w', encoding='utf-8') as f:
#     json.dump(output_list, f, ensure_ascii=False, indent=2)

# # 📊 输出详细统计
# print("\n" + "="*60)
# print("📊 模型在 5 次运行中 'correct' 数量的分布统计")
# print("="*60)
# for i in range(6):
#     print(f"🟢 {i} 个回答正确（即 {5-i} 个错误）: {count_correct[i]} 条")

# print("-"*60)
# print(f"✅ 全对样本总数（正样本）: {count_correct[5]} 条")
# print(f"❌ 全错样本总数: {count_correct[0]} 条")
# print(f"📁 最终保留的全对样本（正样本）: {len(positive_samples)} 条 (100%)")
# print(f"📁 最终保留的全错样本（负样本）: {len(selected_negative_samples)} 条 (~1/5 正样本数量)")
# print(f"📁 总共生成 SFT 数据: {len(output_list)} 条")
# print(f"📈 正负样本比例: {len(positive_samples)}:{len(selected_negative_samples)} ({len(positive_samples)/max(1, len(selected_negative_samples)):.2f}:1)")
# print("="*60)
# print("💾 文件已保存至: refusal_sft_train_data_balanced.json")



