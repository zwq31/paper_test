def parse_news(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()
    raw_items = [item.strip() for item in full_text.split("<|endoftext|>") if item.strip()]
    news_list = []
    for item in raw_items:
        lines = item.splitlines()
        news_dict = {}
        for line in lines:
            if line.startswith("新闻发表时间："):
                news_dict["time"] = line[len("新闻发表时间："):].strip()
            elif line.startswith("新闻标题："):
                news_dict["title"] = line[len("新闻标题："):].strip()
            elif line.startswith("新闻内容："):
                news_dict["content"] = line[len("新闻内容："):].strip()
        news_list.append(news_dict)
    return news_list

reference_file = parse_news("../data-continued-pretraining/news_3year.txt")
prediction_file = parse_news("../data-continued-pretraining/generated_news_3year.txt")



num_to_merge = min(len(reference_file), len(prediction_file))

with open("../data-continued-pretraining/merged_generate_news.txt", "w", encoding="utf-8") as f:
    for i in range(num_to_merge):
        orig = reference_file[i]
        gen = prediction_file[i]
        # 如果确认顺序和对应，断言检查可用
        assert orig["time"] == gen["time"] and orig["title"] == gen["title"], "新闻时间或标题不匹配"
        
        f.write(f"新闻发表时间：{orig['time']}\n")
        f.write(f"新闻标题：{orig['title']}\n")
        f.write(f"新闻内容：{orig['content']}\n")
        f.write(f"生成内容：{gen['content']}\n")
        f.write("<|endoftext|>\n")
        f.flush()  # 立即写入磁盘，避免缓存丢失
