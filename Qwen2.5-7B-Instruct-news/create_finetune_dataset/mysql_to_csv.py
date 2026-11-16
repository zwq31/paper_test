# -*- coding: utf-8 -*-
import pymysql
import re
from bs4 import BeautifulSoup
import logging
import pandas as pd
import json

# 1. 定义HTML解析函数
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
def get_data_from_db():
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
        -- 专门查询paper_id=100067的数据
        news_page.paper_id = 100067
        -- 保留标题过滤条件
        AND (news_article.title != '广告' AND news_article.title != '致读者' AND news_article.title != '导读')
    """
    cursor.execute(sql_query)
    data = cursor.fetchall()
    connection.close()
    print("✅ 成功关闭数据库连接.")
    print(f"📦 总共有 {len(data)} 条新闻记录")

    df = pd.DataFrame(data, columns=["title", "content", "pub_date"])
    df['content'] = df['content'].apply(parse_html_content)
    df['title'] = df['title'].apply(clean_title)

    # 去重 + 长度过滤
    df = df.drop_duplicates(subset=['title', 'content'])
    df = df[df['content'].str.len() >= 100]

    return df

# 4. 保存为 JSON 文件（列表形式，每条新闻包含 id、title、content、pub_date）
def save_to_json(df, output_path):
    df = df.reset_index(drop=True)
    output_list = []
    for idx, row in df.iterrows():
        news_item = {
            'id': idx + 1,  # 从1开始编号
            'title': str(row['title']).strip(),
            'content': str(row['content']).strip(),
            'pub_date': str(row['pub_date']).strip()
        }
        output_list.append(news_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

    print(f"✅ 成功保存 {len(output_list)} 条新闻到 {output_path}")

# 主程序
def main():
    try:
        df = get_data_from_db()
        output_json_path = '/mnt/project/milvus/100067_news.json'
        save_to_json(df, output_json_path)
    except Exception as e:
        logging.error(f"主程序执行错误: {e}")
        print(f"❌ 执行失败: {e}")

if __name__ == "__main__":
    main()



