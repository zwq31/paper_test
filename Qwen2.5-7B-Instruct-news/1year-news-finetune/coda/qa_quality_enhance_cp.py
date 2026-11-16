import json
import os
import re
import logging
from openai import OpenAI
from tqdm import tqdm

DATASETS_DIR = "../data/datasets"
INPUT_FILE = os.path.join(DATASETS_DIR, "all_processed_news.json")
OUTPUT_FILE = os.path.join(DATASETS_DIR, "all_processed_news_enhanced_1.json")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="qa_quality_enhance_cp.log",
    filemode="a",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
os.makedirs(DATASETS_DIR, exist_ok=True)


def process_news_with_batch(title, content, pub_date, news_id, situational_qas, conceptual_qas):
    """
    一次性处理新闻的所有 QA（situational + conceptual），
    让大模型批量判断并增强，返回增强后的 dict。
    """

    # 构造输入
    prompt = f"""
    以下是一条新闻及其相关问答对。
    请你批量判断这些问答对是否高质量（即问题表达清晰，时间、地点、事件完整明确，无歧义，有完整的上下文信息，即仅凭问题就能确切知道问的是哪一年、哪一天、什么地点、什么事件，不会有歧义或模糊表达）。
    如果不是高质量的，请根据新闻标题、内容和发布时间进行数据增强，使其成为高质量问答（问题需包含具体时间、地点、事件等，表达清晰，无歧义，十分确切，能帮助了解地方，尽量避免出现"根据2024年9月2日发布的新闻"这样的句子）
    如果所有问答对都是高质量的，"all_qa_high_quality": true,否则为false。
    返回严格的 JSON 格式

    新闻标题：{title}
    新闻内容：{content}
    新闻发布时间：{pub_date}

    情境问答：
    {json.dumps(situational_qas, ensure_ascii=False, indent=2)}

    概念问答：
    {json.dumps(conceptual_qas, ensure_ascii=False, indent=2)}

    请返回 JSON 格式，保持以下结构：
    {{
    "situational_qa": [
        {{
        "question": "...",
        "answer": "...",
        "source_id": "{news_id}",
        "type": "situational"
        }}
    ],
    "conceptual_qa": [
        {{
        "question": "...",
        "answer": "...",
        "source_id": "{news_id}",
        "type": "conceptual"
        }}
    ],
    "all_qa_high_quality": true/false
    }}
    请直接输出JSON，不要有任何其他解释或多余内容
        """

    try:
        response = client.chat.completions.create(
            model="qwen-plus",  # 使用 Qwen-Plus 模型
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that evaluates and enhances QA pairs in batch, ensuring they are high-quality and context-complete.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.3,
            extra_body={"enable_thinking": False}
        )
        result = response.choices[0].message.content.strip()

        # 提取 JSON
        json_match = re.search(r"(\{.*\})", result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
        try:
            enhanced_data = json.loads(result)
            if enhanced_data.get("all_qa_high_quality") is True:
                logger.info(f"新闻ID={news_id} 《{title}》 所有问答均为高质量")
            return enhanced_data

        except json.JSONDecodeError as e:
            logger.error(f"新闻ID={news_id} 《{title}》 JSON解析失败: {e}")
            logger.error(f"原始响应: {result}")
            return {
                "situational_qa": situational_qas,
                "conceptual_qa": conceptual_qas,
                "all_qa_high_quality": False  # 添加默认值
            }

    except Exception as e:
        logger.error(f"处理新闻ID={news_id} 出错: {e}")
        return {
            "situational_qa": situational_qas,
            "conceptual_qa": conceptual_qas,
            "all_qa_high_quality": False  # 添加默认值
        }


def main():
    """
    主函数：逐条处理新闻数据，每处理一条就写入输出文件。
    """
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        news_list = json.load(f)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")

        # 只处理第 82 ~ 83 条
        for idx, news_item in enumerate(tqdm(news_list[89:94], total=5, desc="批量检查并增强问答对")):
            title=news_item.get("title", "")
            content=news_item.get("content", "")
            pub_date=news_item.get("pub_date", "")
            news_id=news_item.get("id", "")
            situational_qas=news_item.get("situational_qa", [])
            conceptual_qas=news_item.get("conceptual_qa", [])

            logger.info(f"处理新闻ID {news_id}: {title}")
            enhanced_item = process_news_with_batch(
                title, content, pub_date, news_id, situational_qas, conceptual_qas
            )

            # 保持 news_item 的其他字段
            news_item["situational_qa"] = enhanced_item.get("situational_qa", [])
            news_item["conceptual_qa"] = enhanced_item.get("conceptual_qa", [])

            json.dump(news_item, f_out, ensure_ascii=False, indent=2)

            if idx < len(news_list) - 1: 
                f_out.write(",\n") 
            else: 
                f_out.write("\n") 

        f_out.write("]\n")

    logger.info(f"增强后的数据已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()