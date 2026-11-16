import pandas as pd
import json
import os
import re
import logging
from tqdm import tqdm
from openai import OpenAI

# ======================================================================
# 配置变量
# ======================================================================
DATASETS_DIR = "../data/datasets/datasets_fact_qa"
os.makedirs(DATASETS_DIR, exist_ok=True)

# ======================================================================
# 设置日志配置
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qa_test_data.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ======================================================================
# 初始化 OpenAI 客户端
# ======================================================================
client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",  # 替换为你的 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ======================================================================
# 生成事实性问题
# ======================================================================
def generate_fact_qa(news_id, news_title, news_content, news_pub_date):
    prompt = f"""
    请基于以下新闻内容，生成4个最重要的、新闻重点突出的**事实性问题及对应答案**。

    标题：{news_title}
    内容：{news_content}
    新闻发表时间：{news_pub_date}

    **核心要求：**
    1. **纯粹事实性**：问题必须基于新闻中最核心的客观事实，避免任何推断、主观评价或推理内容
    2. **信息完整性**：每个问题涉及时间、地点、人物、活动等实体时，必须提供完整信息，且每个问题必须都有完整的事件背景：
    - 地点：必须完整（县级以下单位冠以"永嘉县"）
    - 人物：必须有明确限定词（如"永嘉县云岭乡新农人潘建平"）
    - 时间：根据新闻发表时间推断具体时间，禁止使用"近日"
    - 事件背景：必须包括具体的事件名称、主办方等背景信息，确保问题独立完整
    - ❌ 错误示例：“2024年6月，改造团队实地走访了金溪镇多少户困难家庭？”（缺少事件背景）
    - ✅ 正确示例：“2024年6月，金溪镇社会工作站、助联体与绿文社工中心党支部开展的"爱心筑巢·暖居助老"居家改造行动"活动中，改造团队实地走访了金溪镇多少户困难家庭？”
    3. **上下文完整**：每个问题必须独立完整，避免指代前文：
    - ❌ 错误示例："该团伙的涉案金额是多少？"
    - ✅ 正确示例："2024年10月，上海警方捣毁的'网络水军'犯罪团伙涉案金额是多少？"
    4. **答案可验证**：答案必须直接来源于新闻内容，确保准确性
    5. **时间处理**：当前为2025年9月，根据新闻发表时间和内容推断具体时间

    **输出格式（严格的JSON数组）：**
    [
        {{"question": "完整的问题1", "answer": "基于新闻的准确答案1"}},
        {{"question": "完整的问题2", "answer": "基于新闻的准确答案2"}},
        {{"question": "完整的问题3", "answer": "基于新闻的准确答案3"}},
        {{"question": "完整的问题4", "answer": "基于新闻的准确答案4"}}
    ]

    请确保生成4个独立完整的事实性问题。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "你是一个能根据新闻文章生成具有完整上下文的事实性问答对的有用助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3
        )

        result = response.choices[0].message.content.strip()

        # 提取 JSON
        json_match = re.search(r'(\[.*\])', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)

        try:
            qa_pairs = json.loads(result)
            for qa in qa_pairs:
                qa["id"] = news_id
            return qa_pairs
        except json.JSONDecodeError as e:
            logging.error(f"事实性问答 JSON 解析失败: {e}")
            logging.error(f"原始响应: {result}")
            return []

    except Exception as e:
        logging.error(f"调用大模型生成事实性问答时出错: {e}")
        return []

# ======================================================================
# 主程序
# ======================================================================
def main():
    # 加载新闻数据（json）
    try:
        news_df = pd.read_json('../data/1year_news_filtered.json')
        logging.info(f"已加载 {len(news_df)} 条新闻文章")
    except Exception as e:
        logging.error(f"加载新闻数据时出错: {e}")
        return

    output_path = os.path.join(DATASETS_DIR, 'fact_qa.json')

    # 初始化 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    # 逐条处理新闻
    # for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="生成事实性问答"):
    for idx, row in tqdm(news_df.iloc[787:].iterrows(), total=len(news_df)-787, desc="生成事实性问答"):
        news_id = row.get('id', str(idx))  # CSV 中可能没有id，用行号代替
        news_title = row.get('title', '')
        news_content = row.get('content', '')
        news_pub_date = row.get('pub_date', '')

        logging.info(f"处理新闻ID {news_id}: {news_title}")
        fact_qas = generate_fact_qa(news_id, news_title, news_content, news_pub_date)

        if fact_qas:
            with open(output_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.extend(fact_qas)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)

            logging.info(f"已处理新闻ID {news_id}，生成 {len(fact_qas)} 条事实性问答")
        else:
            logging.info(f"新闻ID {news_id} 未生成任何问答")

    logging.info(f"所有新闻处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    main()
