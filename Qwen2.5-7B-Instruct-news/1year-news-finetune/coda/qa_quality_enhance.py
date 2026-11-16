import json
import os
import re
import logging
from openai import OpenAI
from tqdm import tqdm

DATASETS_DIR = "../data/datasets"
INPUT_FILE = os.path.join(DATASETS_DIR, "all_processed_news.json")
OUTPUT_FILE = os.path.join(DATASETS_DIR, "all_processed_news_enhanced.json")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qa_quality_enhance.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
os.makedirs(DATASETS_DIR, exist_ok=True)

def check_qa_quality(question, answer, news_title, news_content, news_pub_date):
    """
    调用大模型判断问答对是否高质量，是否具有完整上下文
    返回 True 表示高质量，False 表示需要增强
    """
    prompt = f"""
    请判断以下问答对是否为高质量问答，是否具有完整的上下文信息（即仅凭问题就能确切知道问的是哪一年、哪一天、什么地点、什么事件，不会有歧义或模糊表达），是否有助于了解地方发展、政策、社会状况等。
    如果不满足，请返回 false。
    问题：{question}
    答案：{answer}
    新闻标题：{news_title}
    新闻内容：{news_content}
    新闻发布时间：{news_pub_date}
    只返回JSON格式：
    {{
        "is_high_quality": true/false
    }}
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that accurately evaluates QA pairs for quality and context completeness."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
        filter_result = json.loads(result)
        return filter_result.get("is_high_quality", False)
    except Exception as e:
        logger.error(f"问答质量判断出错: {e}")
        return True  # 默认高质量，避免误删

def enhance_qa_pair(question, answer, news_title, news_content, news_pub_date):
    """
    调用大模型对不合格问答对进行数据增强
    返回增强后的 {"question": "...", "answer": "..."}
    """
    prompt = f"""
    以下问答对不满足高质量要求（缺乏完整上下文或表达不清），请根据新闻标题、内容和发布时间进行数据增强，使其成为高质量问答（问题需包含具体时间、地点、事件等，表达清晰，无歧义，十分确切，能帮助了解地方，尽量避免出现“根据2024年9月2日发布的新闻”这样的句子）。
    原问题：{question}
    原答案：{answer}
    新闻标题：{news_title}
    新闻内容：{news_content}
    新闻发布时间：{news_pub_date}
    只返回JSON格式：
    {{
        "question": "增强后的高质量问题",
        "answer": "增强后的高质量答案"
    }}
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites QA pairs to ensure high quality and complete context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.5
        )
        result = response.choices[0].message.content.strip()
        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
        enhanced = json.loads(result)
        return enhanced
    except Exception as e:
        logger.error(f"问答增强出错: {e}")
        return {"question": question, "answer": answer}

def process_news_qa(news_item):
    """
    对单条新闻的所有问答对进行质量检查和增强
    """
    title = news_item.get("title", "")
    content = news_item.get("content", "")
    pub_date = news_item.get("pub_date", "")
    id = news_item.get("id", "")
    situational_count,conceptual_count=0,0
    # 处理情境问答
    logger.info(f"📰 开始处理新闻ID={id}，标题=《{title}》 的情境问题") 
    situational_qa = []
    for qa in news_item.get("situational_qa", []):
        q, a = qa["question"], qa["answer"]
        
        is_high_quality = check_qa_quality(q, a, title, content, pub_date)
        situational_count+=1
        if is_high_quality:
            logger.info(f"✅ 第{situational_count}个问答对[合格]  问题为: {q}")
            situational_qa.append(qa)
        else:
            enhanced = enhance_qa_pair(q, a, title, content, pub_date)
            enhanced["source_id"] = id
            enhanced["type"] = "situational"
            logger.info(f"❌ 第{situational_count}个问题[不合格]  原问题为: {q}")
            logger.info(f"➡️  增强后: {enhanced['question']}")
            situational_qa.append(enhanced)

    # 处理概念问答
    logger.info(f"📰 开始处理新闻ID={id}，标题=《{title}》 的概念问题")
    conceptual_qa = []
    for qa in news_item.get("conceptual_qa", []):
        q, a = qa["question"], qa["answer"]
        is_high_quality = check_qa_quality(q, a, title, content, pub_date)
        conceptual_count+=1
        if is_high_quality:
            logger.info(f"✅ 第{conceptual_count}个问答对[合格]  问题为: {q}")
            conceptual_qa.append(qa)
        else:
            enhanced = enhance_qa_pair(q, a, title, content, pub_date)
            enhanced["source_id"] = id
            enhanced["type"] = "conceptual"
            logger.info(f"❌ 第{conceptual_count}个问答对[不合格]  原问题为: {q}")
            logger.info(f"➡️  增强后: {enhanced['question']}")
            conceptual_qa.append(enhanced)

    news_item["situational_qa"] = situational_qa
    news_item["conceptual_qa"] = conceptual_qa
    return news_item

def main():
    """
    主函数：逐条处理新闻数据，每处理一条就追加保存到输出文件，避免一次性占用大量内存。
    """
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        news_list = json.load(f)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        f_out.write('[\n')  # 写入JSON数组开头
        # for idx, news_item in enumerate(tqdm(news_list[:1], total=1, desc="检查并增强问答对")):
        for idx, news_item in enumerate(tqdm(news_list, desc="检查并增强问答对")):
            enhanced_item = process_news_qa(news_item)
            json.dump(enhanced_item, f_out, ensure_ascii=False, indent=2)
            if idx < len(news_list) - 1:
                f_out.write(',\n')
            else:
                f_out.write('\n')
        f_out.write(']\n')  # 写入JSON数组结尾
    
    logger.info(f"增强后的数据已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
