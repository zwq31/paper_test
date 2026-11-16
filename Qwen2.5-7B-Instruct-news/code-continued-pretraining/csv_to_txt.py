# -*- coding: utf-8 -*-
import pandas as pd

# 导出为纯文本，每条新闻占一行（或一个段）
def export_to_text(df, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            date = str(row['pub_date']).strip()
            title = str(row['title']).strip()
            content = str(row['content']).strip().replace('\n', '\\n')  # 替换换行符
            text = f"新闻发表时间：【{date}】\n新闻标题：{title}\n新闻内容：{content}\n<|endoftext|>\n"
            f.write(text)
    print(f"已导出至 {output_path}")

# 主函数
def main():
    csv_path = "../data-continued-pretraining/news_3year.csv"
    output_path = "../data-continued-pretraining/news_3year.txt"

    df = pd.read_csv(csv_path)
    print(f"读取到 {len(df)} 条新闻")
    export_to_text(df, output_path)

if __name__ == "__main__":
    main()
