import pandas as pd
import json
import os
import re
from bs4 import BeautifulSoup
from openai import OpenAI
import random
import logging
from tqdm import tqdm


# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='news_to_data.log',
    filemode='a',
    encoding='utf-8'  # ✅ 关键在这里
)

logger = logging.getLogger(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 如果datasets目录不存在则创建
os.makedirs("datasets", exist_ok=True)


def generate_context_info(news_title, news_content, news_pub_date):
    """为新闻文章生成上下文信息"""
    prompt = f"""请为以下新闻生成情境信息：
    
标题：{news_title}
内容：{news_content}
发表时间：{news_pub_date}

请以JSON格式返回：
{{
    "time": [],  # 结合新闻发表时间将新闻中提到的时间节点转化为具体时间信息，格式为"YYYY年MM月DD日"，如"2024年1月15日"
    "location": [],  # 新闻中提到的地点信息
    "entities": [],  # 重要实体（人物、事件活动，组织机构，公共服务，工程项目等其他）,其中人物需要包含人物身份信息，格式为 "entities": [{{"properties": "朱佩珍(瓯北城市新区党委副书记、主任)","label": "人物"}},{{"properties": "","label": ""}}],
    "context_summary": "",  # 100字以内的情境摘要
    "key_concepts": []  # 关键概念或术语，
}}

注意事项：
1. 时间信息必须精确到具体日期，使用"YYYY年MM月DD日"格式
2. 人物类型实体必须包含其身份信息
3. 实体按照新闻内容来抽取，并不一定必须包含提示中的五种类型或只有这五种，需要根据新闻内容来判断
4. 确保返回的是有效的JSON格式，不要有多余的说明文字或注释"""

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts contextual information from news articles."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        # 从响应中提取JSON
        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
            
        try:
            context_info = json.loads(result)
            return context_info
        except json.JSONDecodeError as e:
            logging.error(f"JSON解码错误: {e}")
            logging.error(f"原始响应: {result}")
            # 出错时返回简单的上下文结构
            return {
                "time": [],
                "location": [],
                "entities": [],
                "context_summary": "",
                "key_concepts": []
            }
            
    except Exception as e:
        logging.error(f"生成上下文信息时出错: {e}")
        return {
            "time": [],
            "location": [],
            "entities": [],
            "context_summary": "",
            "key_concepts": []
        }

def generate_situational_qa(new_id,news_title, news_content, context_info):
    """基于新闻内容生成情境问题和答案"""
    # 从context_info中提取实体文本
    entities = []
    for entity in context_info.get('entities', []):
        if isinstance(entity, dict):
            entities.append(entity.get('text', ''))
        elif isinstance(entity, str):
            entities.append(entity)
            
    prompt = f"""
    基于以下新闻内容和情境信息，生成10个情景性问题和答案：
    
    标题：{news_title}
    内容：{news_content}
    情境信息：
    - 时间：{', '.join(context_info.get('time', []))}
    - 地点：{', '.join(context_info.get('location', []))}
    - 实体：{', '.join(entities)}
    - 摘要：{context_info.get('context_summary', '')}
    要求：
    1. 问题必须明确具体信息，避免使用"此次"、"这个"等模糊指代，
    2. 问题要完整且具有明确的语境，即使没有上下文也能理解问题所指
    3. 问题要尽可能多地包含新闻中的内容
    4. 确保问题能从内容中找到明确的答案
    5. 避免出现任何模糊或需要上下文才能理解的表述，比如"永嘉县委书记娄绍光在会议上强调了哪些关于干部管理的要求？"这种问题不能出现，必须指明是什么时间在什么地点举行的什么会议
    
    请返回有效的JSON格式：
    [
        {{"question": "问题1", "answer": "答案1"}},
        ...
    ]
    
    只返回JSON数组，不要有额外说明。
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates high-quality situational questions and answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        
        # 从响应中提取JSON
        json_match = re.search(r'(\[.*\])', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
            
        try:
            qa_pairs = json.loads(result)
            # 为每个问答对添加元数据
            for qa in qa_pairs:
                qa["source_id"]=new_id
                qa["type"] = "situational"
            return qa_pairs
        except json.JSONDecodeError as e:
            logging.error(f"情境问答的JSON解码错误: {e}")
            logging.error(f"原始响应: {result}")
            return []
            
    except Exception as e:
        logging.error(f"生成情境问答时出错: {e}")
        return []

def generate_conceptual_qa(news_id,news_title, news_content, context_info):
    """基于新闻内容生成概念性问题和答案"""
    # 提取关键概念
    key_concepts = context_info.get('key_concepts', [])
    key_concepts_str = ', '.join(key_concepts)
    
    prompt = f"""
    基于以下新闻内容和关键概念，生成10个概念性问题和答案：
    
    标题：{news_title}
    内容：{news_content}
    关键概念：{key_concepts_str}
    
    要求：
    1. 问题应关注新闻中出现的概念、术语、实体间的关系等
    2. 问题要能引导对概念的深入理解，而不仅仅是事实性信息
    3. 确保问题能从内容中找到答案或可以基于内容推理得出
    4. 答案应清晰解释概念，可以比情景性问题更加深入和全面
    
    请返回有效的JSON格式：
    [
        {{"question": "问题1", "answer": "答案1"}},
        ...
    ]
    
    只返回JSON数组，不要有额外说明。
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates high-quality conceptual questions and answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        
        # 从响应中提取JSON
        json_match = re.search(r'(\[.*\])', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
            
        try:
            qa_pairs = json.loads(result)
            # 为每个问答对添加元数据
            for qa in qa_pairs:
                qa["sourse_id"]=news_id
                qa["type"] = "conceptual"
            return qa_pairs
        except json.JSONDecodeError as e:
            logging.error(f"概念问答的JSON解码错误: {e}")
            logging.error(f"原始响应: {result}")
            return []
            
    except Exception as e:
        logging.error(f"生成概念问答时出错: {e}")
        return []

def process_news_data(news_item):
    """处理单个新闻条目以生成上下文信息和问答对"""
    # 提取id,标题和内容
    id=news_item.get('id','')
    title = news_item.get('title', '')
    content = news_item.get('content', '')
    pub_date=news_item.get('pub_date','')
    
    # 如果内容为空、不是字符串，或者少于50字，记录警告并返回
    if not isinstance(content, str) or pd.isna(content) or len(content.strip()) < 50:
        logging.warning(f"内容为空或过短：{title}")
        return None
        
    # 生成上下文信息
    logging.info(f"为以下内容生成上下文信息: {title}")
    context_info = generate_context_info(title, content,pub_date)
    
    # 生成问答对
    logging.info(f"为以下内容生成情境问答对: {title}")
    situational_qa = generate_situational_qa(id,title, content, context_info)
    
    logging.info(f"为以下内容生成概念问答对: {title}")
    conceptual_qa = generate_conceptual_qa(id,title, content,context_info)
    
    return {
        "id":id,
        "title": title,
        "content": content,  # 保存完整内容
        "pub_date":pub_date,
        "context_info": context_info,
        "situational_qa": situational_qa,
        "conceptual_qa": conceptual_qa
    }

def main():
    # 加载新闻数据
    try:
        news_df = pd.read_csv('news.csv')
        logging.info(f"已加载 {len(news_df)} 条新闻文章")
    except Exception as e:
        logging.error(f"加载新闻数据时出错: {e}")
        return
    
    # # 初始化文件
    # with open('datasets/all_processed_news.json', 'w', encoding='utf-8') as f:
    #     json.dump([], f)
    # with open('datasets/situational_qa.json', 'w', encoding='utf-8') as f:
    #     json.dump([], f)
    # with open('datasets/conceptual_qa.json', 'w', encoding='utf-8') as f:
    #     json.dump([], f)
    # with open('datasets/context_info.json', 'w', encoding='utf-8') as f:
    #     json.dump([], f)
    # with open('datasets/combined_qa.json', 'w', encoding='utf-8') as f:
    #     json.dump([], f)
        
    situational_qa_count = 0
    conceptual_qa_count = 0
    
    for idx, row in tqdm(news_df.iloc[40:100].iterrows(), total=60, desc="处理新闻 41-100"):
        news_item = {
            'id': row['id'],
            'title': row['title'],
            'content': row['content'],
            'pub_date': row['pub_date']
        }
        result = process_news_data(news_item)
        if result:
            # 追加写入所有处理后的数据
            with open('datasets/all_processed_news.json', 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.append(result)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 追加写入上下文信息
            context_info_item = {
                "id": result["id"],
                "title": result["title"],
                "content": result["content"],
                "pub_date":result["pub_date"],
                "context_info": result["context_info"]
            }
            with open('datasets/context_info.json', 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.append(context_info_item)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 追加写入问答对
            with open('datasets/situational_qa.json', 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.extend(result["situational_qa"])
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            with open('datasets/conceptual_qa.json', 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.extend(result["conceptual_qa"])
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            # 追加写入合并的问答对
            with open('datasets/combined_qa.json', 'r+', encoding='utf-8') as f:
                data = json.load(f)
                combined = result["situational_qa"] + result["conceptual_qa"]
                random.shuffle(combined)
                data.extend(combined)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            situational_qa_count += len(result["situational_qa"])
            conceptual_qa_count += len(result["conceptual_qa"])
    
    logging.info(f"已生成 {situational_qa_count} 个情境问答对")
    logging.info(f"已生成 {conceptual_qa_count} 个概念问答对")
    logging.info(f"总共 {situational_qa_count + conceptual_qa_count} 个问答对已保存到 dataset/combined_qa.json")

if __name__ == "__main__":
    main()