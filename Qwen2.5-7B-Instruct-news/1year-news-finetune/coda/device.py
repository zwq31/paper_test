
import json

# 读取三个文件的内容
with open('datasets/conceptual_qa.json', 'r', encoding='utf-8') as file:
    conceptual_data = json.load(file)

with open('datasets/situational_qa.json', 'r', encoding='utf-8') as file:
    situational_data = json.load(file)

with open('datasets/all_processed_news.json', 'r', encoding='utf-8') as file:
    main_data = json.load(file)


# 构建映射表（str(source_id) -> item）
def build_mapping(data):
    mapping = {}
    for item in data:
        if "source_id" in item:
            source_id = str(item["source_id"])
            if source_id not in mapping:
                mapping[source_id] = []
            mapping[source_id].append(item)
    return mapping


# 创建映射表
conceptual_map = build_mapping(conceptual_data)
situational_map = build_mapping(situational_data)


# 遍历 main_data 并按 id 匹配填充
for entry in main_data:
    current_id = str(entry.get("id"))

    # 填充 conceptual_qa
    if current_id in conceptual_map:
        entry["conceptual_qa"] = [item for item in conceptual_map[current_id]]
    else:
        entry["conceptual_qa"] = []

    # 填充 situational_qa
    if current_id in situational_map:
        entry["situational_qa"] = [item for item in situational_map[current_id]]
    else:
        entry["situational_qa"] = []

# 写回文件
with open('all_processed_news.json', 'w', encoding='utf-8') as f:
    json.dump(main_data, f, ensure_ascii=False, indent=4)

print("✅ 数据已成功填充并更新，保存至 main_file_updated.json")