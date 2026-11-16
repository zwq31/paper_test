# import json

# # ================== 配置三个文件路径 ==================
# FILE1 = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test/evaluation_results.json"
# FILE2 = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/unmention/unmention.json"
# FILE3 = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/unfinetuned/unfinetuned.json"

# # ================== 辅助函数 ==================
# def count_hallucinations_file1(data):
#     halluc = 0
#     total = len(data)
#     for item in data:
#         score = item["evaluation_results"]["correctness"]["score"]
#         should_ans = item["refusal_decision"]["should_answer"]
#         if score < 4 and should_ans:
#             halluc += 1
#     return halluc, total

# def count_hallucinations_file2_or_3(data):
#     halluc = 0
#     total = len(data)
#     for item in data:
#         should_ans = item["refusal_decision"]["should_answer"]
#         if should_ans:
#             halluc += 1
#     return halluc, total

# # ================== 加载数据 ==================
# with open(FILE1, 'r', encoding='utf-8') as f:
#     data1 = json.load(f)
# with open(FILE2, 'r', encoding='utf-8') as f:
#     data2 = json.load(f)
# with open(FILE3, 'r', encoding='utf-8') as f:
#     data3 = json.load(f)

# # ================== 统计 ==================
# h1, n1 = count_hallucinations_file1(data1)
# h2, n2 = count_hallucinations_file2_or_3(data2)
# h3, n3 = count_hallucinations_file2_or_3(data3)

# total_halluc = h1 + h2 + h3
# total_samples = n1 + n2 + n3
# hallucination_rate = total_halluc / total_samples if total_samples > 0 else 0

# # ================== 表格格式设置 ==================
# WIDTH = 68  # 总宽度
# COL_WIDTHS = [20, 12, 12, 12]  # 每列宽度

# # 居中格式：{:^宽度}
# header_fmt = "".join([f"{{:^{w}}}" for w in COL_WIDTHS])
# data_fmt = "{:<20}{:>12}{:>12}{:>12}"  # 数据行：文件名左对齐，数字右对齐（更规范）

# # ================== 打印居中表格 ==================
# print("\n" + "=" * WIDTH)
# print("📊 幻觉率统计结果".center(WIDTH))
# print("=" * WIDTH)

# # 表头居中
# print(header_fmt.format("文件", "样本数", "幻觉数", "幻觉率"))
# print("-" * WIDTH)

# # 数据行
# rate1 = h1 / n1 if n1 > 0 else 0
# rate2 = h2 / n2 if n2 > 0 else 0
# rate3 = h3 / n3 if n3 > 0 else 0

# print(data_fmt.format("已训练知识测试集", n1, h1, f"{rate1:.2%}"))
# print(data_fmt.format("未提及知识测试集", n2, h2, f"{rate2:.2%}"))
# print(data_fmt.format("未训练知识测试集", n3, h3, f"{rate3:.2%}"))

# print("-" * WIDTH)
# print(data_fmt.format("总计", total_samples, total_halluc, f"{hallucination_rate:.2%}"))
# print("=" * WIDTH)
# ==================================================================替换RAG文档==================================================================
import json

# 读取两个 JSON 文件（假设它们是包含多个对象的列表）
with open('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test1/test_nodecide/evaluation_results.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

with open('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test1/test_nodecide.json', 'r', encoding='utf-8') as f:
    retrieval_data = json.load(f)

# 构建 retrieval_info 的查找字典：键为 (source_id, question)
retrieval_map = {}
for item in retrieval_data:
    key = (item['source_id'], item['question'])
    retrieval_map[key] = item['retrieval_info']

# 更新 eval_data 中的 refusal_decision 字段为 retrieval_info
for item in eval_data:
    key = (item['source_id'], item['question'])
    if key in retrieval_map:
        item['retrieval_info'] = retrieval_map[key]
        # 删除原来的 refusal_decision 字段（可选）
        item.pop('refusal_decision', None)
    else:
        # 可选：如果没有匹配项，可以保留原字段或设为空
        item['retrieval_info'] = {}
        item.pop('refusal_decision', None)

# 保存结果到新文件
with open('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test1/test_nodecide/evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print("合并完成，结果已保存至 merged_output.json")

# import json

# # 1. 读取两个 JSON 文件（假设都是列表）
# with open('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag/my_milvus/test/test_time_nodecide_relaxed/evaluation_results.json', 'r', encoding='utf-8') as f:
#     eval_data = json.load(f)  # 第一个文件：评估结果 + retrieval_info

# with open('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/1year_news_filtered.json', 'r', encoding='utf-8') as f:
#     doc_data = json.load(f)   # 第二个文件：完整文档库

# # 2. 构建文档映射：key 为 id，value 为 {id, title, pub_date}
# doc_map = {}
# for doc in doc_data:
#     doc_id = doc['id']
#     doc_map[doc_id] = {
#         'id': doc_id,
#         'title': doc['title'],
#         'pub_date': doc['pub_date']
#         # 如果你也想保留 content，可以加：'content': doc['content']
#     }

# # 3. 遍历 eval_data，为每个 item 添加 source_doc
# for item in eval_data:
#     source_id = item['source_id']
#     if source_id in doc_map:
#         item['source_doc'] = doc_map[source_id]
#     else:
#         # 如果找不到对应文档，可设为空对象或跳过
#         item['source_doc'] = {
#             'id': source_id,
#             'title': '',
#             'pub_date': ''
#         }

# # 4. 保存合并后的结果
# with open('/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test_rag_yuanshi/test_v2/test_claude_nodecide/evaluation_results1.json', 'w', encoding='utf-8') as f:
#     json.dump(eval_data, f, ensure_ascii=False, indent=2)

# print("✅ 合并完成！结果已保存至 merged_output.json")


