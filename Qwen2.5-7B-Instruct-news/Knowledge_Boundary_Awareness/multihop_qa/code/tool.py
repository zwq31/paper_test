import json

# ------------------------------
# 配置文件路径
# ------------------------------
cross_doc_json_path = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/2_hop_paths.json"
all_news_json_path = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/1year_news_filtered.json"
output_json_path = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/2_hop_paths_all.json"

# ------------------------------
# 1️⃣ 读取所有新闻，建立 title -> news 映射
# ------------------------------
with open(all_news_json_path, "r", encoding="utf-8") as f:
    all_news = json.load(f)

title_to_news = {news["title"]: news for news in all_news}
print(f"加载 {len(all_news)} 条新闻完成")

# ------------------------------
# 2️⃣ 读取跨文档路径文件
# ------------------------------
with open(cross_doc_json_path, "r", encoding="utf-8") as f:
    paths = json.load(f)

# ------------------------------
# 3️⃣ 补充每个 evidence_docs 的 id 和 content，只保留完全匹配的路径
# ------------------------------
filtered_paths = []

for path in paths:
    all_matched = True
    for doc in path.get("evidence_docs", []):
        title = doc.get("title", "")
        news_info = title_to_news.get(title)
        if news_info:
            doc["id"] = news_info.get("id")
            doc["title"] = news_info.get("title")
            doc["content"] = news_info.get("content")
            doc["pub_date"] = news_info.get("pub_date")

        else:
            all_matched = False
            print(f"未找到匹配新闻: {title}")
            break  # 有一个匹配不上就跳出

    if all_matched:
        filtered_paths.append(path)

# ------------------------------
# 4️⃣ 保存补充完成且完全匹配的文件
# ------------------------------
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(filtered_paths, f, ensure_ascii=False, indent=4)

print(f"补充完成，保存到 {output_json_path}，共 {len(filtered_paths)} 条路径")
