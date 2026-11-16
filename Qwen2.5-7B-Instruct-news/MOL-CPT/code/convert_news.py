# # -*- coding: utf-8 -*-
# import pymysql
# import pandas as pd
# import re
# from bs4 import BeautifulSoup
# import logging
# import json

# # 1. 解析 HTML 内容
# def parse_html_content(content):
#     if not content:
#         return ""
#     try:
#         soup = BeautifulSoup(content, 'html.parser')
#         text = soup.get_text(separator=" ", strip=True)
#         text = ' '.join(text.split())
#         text = re.sub(r'[■●▲★◆▶◀○◎◇]', '', text)
#         return text
#     except Exception as e:
#         print("解析HTML时出错")
#         return content

# # 2. 清洗标题
# def clean_title(title):
#     if not title:
#         return ""
#     try:
#         title = re.sub(r'[■●▲★◆▶◀○◎◇]', '', title)
#         return title.strip()
#     except Exception as e:
#         print("清洗标题时出错")
#         return title

# # 3. 从数据库读取新闻数据并清洗
# def get_news_dataframe():
#     connection = pymysql.connect(
#         host='122.226.69.234',
#         port=3025,
#         user='root',
#         passwd='__SET_VIA_ENVIRONMENT__',
#         db='news'
#     )
#     print("✅ 成功连接到数据库.")
#     cursor = connection.cursor()

#     sql_query = """
#     SELECT 
#         news_article.title, 
#         news_article.content,
#         FROM_UNIXTIME(news_perd.p_date / 1000, '%Y-%m-%d') AS pub_date
#     FROM 
#         news_article
#     JOIN 
#         news_perd ON news_article.perd_id = news_perd.id
#     JOIN 
#         news_page ON news_article.page_id = news_page.id
#     WHERE  
#        (news_article.title != '广告' AND news_article.title != '致读者' AND news_article.title != '导读')
#     """

#     cursor.execute(sql_query)
#     data = cursor.fetchall()
#     connection.close()
#     print("✅ 成功关闭数据库连接.")
#     print(f"📦 共提取到 {len(data)} 条新闻记录")

#     df = pd.DataFrame(data, columns=["title", "content", "pub_date"])
#     df['content'] = df['content'].apply(parse_html_content)
#     df['title'] = df['title'].apply(clean_title)

#     # 去重 + 长度过滤
#     df = df.drop_duplicates(subset=['title', 'content'])
#     df = df[df['content'].str.len() >= 100]

#     print(f"🧹 清洗后剩余 {len(df)} 条新闻")
#     return df

# # 4. 转换为 JSONL 格式
# def convert_to_jsonl(df, output_path):
#     output = []
#     for _, row in df.iterrows():
#         date = str(row['pub_date']).strip()
#         title = str(row['title']).strip()
#         content = str(row['content']).strip().replace('\n', '\\n')

#         full_text = f"新闻发表时间：【{date}】\n新闻标题：{title}\n新闻内容：{content}\n"
#         output.append({
#             "text": full_text,
#             "is_domain_batch": True
#         })

#     with open(output_path, "w", encoding="utf-8") as f:
#         for item in output:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")

#     print(f"✅ 成功转换 {len(output)} 条新闻，保存为 {output_path}")

# # 主程序
# def main():
#     try:
#         df = get_news_dataframe()
#         output_jsonl_path = "/mnt/project/Qwen2.5-7B-Instruct-news/MOL-CPT/data/news.jsonl"
#         convert_to_jsonl(df, output_jsonl_path)
#     except Exception as e:
#         logging.error(f"❌ 主程序执行错误: {e}")
#         print(f"❌ 执行失败: {e}")

# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
import re
from bs4 import BeautifulSoup
import logging
import json

# 1. 解析 HTML 内容
def parse_html_content(content):
    if not content:
        return ""
    try:
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator=" ", strip=True)
        text = ' '.join(text.split())
        text = re.sub(r'[■●▲★◆▶◀○◎◇]', '', text)
        return text
    except Exception as e:
        print("解析HTML时出错")
        return content

# 2. 清洗标题
def clean_title(title):
    if not title:
        return ""
    try:
        title = re.sub(r'[■●▲★◆▶◀○◎◇]', '', title)
        return title.strip()
    except Exception as e:
        print("清洗标题时出错")
        return title

# 3. 从数据库读取新闻数据并清洗
def get_news_dataframe():
    connection = pymysql.connect(
        host='122.226.69.234',
        port=3025,
        user='root',
        passwd='__SET_VIA_ENVIRONMENT__',
        db='news'
    )
    print("✅ 成功连接到数据库.")
    cursor = connection.cursor()

    sql_query = """
    SELECT 
        news_article.title, 
        news_article.content,
        FROM_UNIXTIME(news_perd.p_date / 1000, '%Y-%m-%d') AS pub_date
    FROM 
        news_article
    JOIN 
        news_perd ON news_article.perd_id = news_perd.id
    JOIN 
        news_page ON news_article.page_id = news_page.id
    WHERE  
       news_page.paper_id = 100067
       AND (news_article.title != '广告' AND news_article.title != '致读者' AND news_article.title != '导读')
    """

    cursor.execute(sql_query)
    data = cursor.fetchall()
    connection.close()
    print("✅ 成功关闭数据库连接.")
    print(f"📦 共提取到 {len(data)} 条新闻记录")

    df = pd.DataFrame(data, columns=["title", "content", "pub_date"])
    df['content'] = df['content'].apply(parse_html_content)
    df['title'] = df['title'].apply(clean_title)

    # 去重 + 长度过滤
    df = df.drop_duplicates(subset=['title', 'content'])
    df = df[df['content'].str.len() >= 100]

    print(f"🧹 清洗后剩余 {len(df)} 条新闻")
    return df

# 4. 转换为 JSONL 格式
def split_content_by_sentences(content, max_length=3072, overlap=300):
    """
    安全切分 content，避免死循环。
    - 尽量在句尾切分
    - 支持重叠
    - 保证 progress（不会卡住）
    """
    if len(content) <= max_length:
        return [content]

    chunks = []
    start = 0
    n = len(content)

    while start < n:
        end = start + max_length

        if end >= n:
            # 最后一段
            chunks.append(content[start:])
            break

        # 在 [end - 200, end] 范围内寻找最后一个句尾标点
        search_start = max(start, end - 200)
        cut_pos = end  # 默认硬切

        for j in range(end - 1, search_start - 1, -1):
            if content[j] in '。！？!?':
                cut_pos = j + 1  # 包含标点
                break

        chunks.append(content[start:cut_pos])

        # 下一段起始位置：允许重叠
        start = cut_pos - overlap
        if start < cut_pos:  # 确保至少前进1
            start = max(start, cut_pos - max_length + overlap)
        else:
            start = cut_pos  # 防止负重叠

        # 安全兜底：防止 start 不前进
        if start <= cut_pos - max_length:
            start = cut_pos  # 强制前进

    return chunks


def convert_to_jsonl(df, output_path, max_content_length=3072):
    total_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            date = str(row['pub_date']).strip()
            title = str(row['title']).strip()
            raw_content = str(row['content']).strip().replace('\n', '\\n')

            content_chunks = split_content_by_sentences(raw_content, max_length=max_content_length)

            for chunk in content_chunks:
                full_text = f"新闻发表时间：【{date}】\n新闻标题：{title}\n新闻内容：{chunk}\n"
                json_line = json.dumps({"text": full_text, "is_domain_batch": True}, ensure_ascii=False)
                f.write(json_line + "\n")
                total_written += 1

    print(f"✅ 成功转换 {total_written} 条新闻片段，保存为 {output_path}")

# 主程序
def main():
    try:
        df = get_news_dataframe()
        output_jsonl_path = "/mnt/project/Qwen3-8B-news/data/news.jsonl"
        convert_to_jsonl(df, output_jsonl_path)
    except Exception as e:
        logging.error(f"❌ 主程序执行错误: {e}")
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()


