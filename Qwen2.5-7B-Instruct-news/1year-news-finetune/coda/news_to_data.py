import pandas as pd
import json
import os
import re
from bs4 import BeautifulSoup
from openai import OpenAI
import random
import logging
from tqdm import tqdm

# ============================================================================
# 配置变量 - 修改这个变量来改变datasets文件夹路径
# ============================================================================
DATASETS_DIR = "../data/datasets_add_3"

# ============================================================================
# 设置日志配置
# ============================================================================

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
os.makedirs(DATASETS_DIR, exist_ok=True)


def filter_news_article(news_id, news_title, news_content, news_pub_date):
    """
    使用大模型过滤新闻文章，判断是否为需要处理新闻
    
    Args:
        news_id: 新闻ID
        news_title: 新闻标题
        news_content: 新闻内容
        news_pub_date: 新闻发布日期
    
    Returns:
        bool: True表示需要处理，False表示跳过
    """
    
    prompt = f"""
    请判断以下内容是否为需要处理的新闻。假设你是一个刚来这个地方的人，想了解这个地方的发展状况、政策环境、社会状况以及其他方方面面，请判断哪些新闻对"了解地方"有价值。
    
    需要处理的新闻类型（对了解地方发展、政策、社会状况有重要价值）：
    1. 时政新闻：政府工作、政策发布、官员活动、会议报道、重要决策,国际事件等
    2. 社会新闻：社会事件、民生问题、公共事务、社会问题、便民信息等
    3. 经济新闻：经济发展、产业政策、商业活动、重大项目,企业活动等
    4. 教育新闻：教育政策、教育改革、重要教育事件等
    5. 科技新闻：科技创新、技术发展、重要科技项目等
    6. 文化新闻：重要文化活动、文化政策等
    7. 环境新闻：环境保护、生态建设等
    8. 交通新闻：交通建设、交通政策等
    9. 医疗新闻：医疗政策、公共卫生、医学科普、医疗新闻等
    10. 旅游景点：景点介绍、旅游攻略等
    11.涉及到地方的文学作品（包括地方景点、人物介绍等）、广告，通知，活动,社区故事等等
    
    不需要处理的类型（对了解地方作用不大，比如学生写的作文等）：
    1. 纯文学作品：与了解地方无关的诗歌、散文、小说，作文等
    2. 虚构故事：童话、寓言、科幻故事等
    3. 个人随笔：与了解地方特色无关的个人感悟、日记等
    
    
    新闻信息：
    标题：{news_title}
    内容：{news_content}
    发布日期：{news_pub_date}
    
    请分析后回答：
    1. 这个内容属于什么类型？
    2. 是否需要处理生成问答对？
    
    请只返回JSON格式：
    {{
        "content_type": "具体类型描述",
        "should_process": true/false,
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that accurately classifies news articles and determines whether they should be processed for QA generation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        # 从响应中提取JSON
        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)
            
        try:
            filter_result = json.loads(result)
            should_process = filter_result.get("should_process", False)
            content_type = filter_result.get("content_type", "未知类型")
            
            logging.info(f"新闻ID {news_id}: 类型={content_type}, 处理={should_process}")
            
            return should_process
            
        except json.JSONDecodeError as e:
            logging.error(f"新闻过滤的JSON解码错误: {e}")
            logging.error(f"原始响应: {result}")
            # 如果解析失败，默认处理
            return True
            
    except Exception as e:
        logging.error(f"新闻过滤时出错: {e}")
        # 如果出错，默认处理
        return True


def generate_situational_qa(new_id,news_title, news_content,news_pub_date):
    """基于新闻内容生成情境问题和答案"""
            
    prompt = f"""
    基于以下新闻内容、标题、发表时间，生成10个具有明确语境的情境性问题及对应答案：
    
    标题：{news_title}
    内容：{news_content}
    新闻发表时间：{news_pub_date}
    严格要求：
    1. 每个问题必须具备完整的上下文信息，即使脱离原文也能独立理解，且问题侧重于新闻的重点，一些无关紧要的内容就不要生成问题。
    2. 问题中但凡涉及具体的时间、地点、人物、活动、政策、事件等实体时，必须明确指出其完整信息。
    3. 活动、会议、项目、政策等等必须在问题中使用完整名称，如"金华市第八次党代会"
    4. 严禁使用模糊指代词语，如"此次"、"这个"、"该活动"、"相关负责人"。所有提及的人物、地点、事件名称等必须具体指出，不能有歧义。
    5. 所有问题必须直接来源于新闻内容，不得自行推断或添加未提及的信息，尽可能多的覆盖新闻的内容。
    6. 答案必须在新闻中能找到明确依据，不能含糊其辞或泛泛而谈。
    7. 对于产品组合、项目介绍等问题，需详细描述其组成部分及其特点，例如："在[具体日期]举办的[活动名称]上，[地点/机构名称]推出了包括[具体产品A]与[具体产品B]在内的特色组合。"
    8. 避免出现任何模糊或需要上下文才能理解的表述，比如"永嘉县委书记娄绍光在会议上强调了哪些关于干部管理的要求？"这种问题不能出现，必须指明是什么时间在什么地点举行的什么会议
    9. 如果新闻中提到某个活动、会议、政策等，问题中必须包含该活动的完整信息，包括时间、地点、具体名称。例如：不能问"座谈会上提到什么"，而应该问"[具体时间]在[具体地点]举行的[具体会议名称]座谈会上提到什么"
    10. 对于任何涉及人物、地点、事件的问题，都必须明确指出具体的时间背景和事件背景，确保问题没有歧义
    11.新闻中没有出现时间的情况，根据新闻发表时间和新闻内容中的与时间相关的词推断出时间
    12. 每个问答对尽可能多包含新闻内容，问答对内容之间可以有重复的内容

    请返回有效的JSON格式：
    [
        {{"question": "问题1", "answer": "答案1"}},
        ...
    ]
    
    只返回JSON数组，不要有额外说明。
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates high-quality situational questions and answers with complete context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
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

def generate_conceptual_qa(news_id,news_title, news_content, news_pub_date):
    """基于新闻内容生成概念性问题和答案"""
    
    prompt = f"""
    基于以下新闻内容、标题、发表时间，生成10个概念性问题和对应答案：
    
    标题：{news_title}
    内容：{news_content}
    新闻发表时间：{news_pub_date}
    
    要求：
    1. 问题应关注新闻中出现的概念、术语、实体间的关系等，且问题侧重于新闻的重点，一些无关紧要的内容就不要生成问题
    2. 每个问题尽可能多的包含新闻的内容,如果涉及特定活动、会议、项目、政策等，必须明确指出其全称、时间、地点等详细信息。
    3. 确保问题能从内容中找到答案或可以基于内容推理得出
    4. 答案应清晰解释概念，可以比情景性问题更加深入和全面
    5. 问题要能引导对概念的深入理解，而不仅仅是事实性信息
    6. 问题需具备完整的上下文信息，即使脱离原文也能独立理解（如包含事件名称、政策背景、组织机构等）。
    7. 如果问题中涉及任何活动、会议、政策、人物、地点等实体，必须明确指出其完整信息，包括时间、地点、具体名称等，避免任何歧义。
    8. 严禁使用模糊指代词语，如"此次"、"这个"、"该活动"、"相关负责人"等，所有提及的实体必须具体明确。
    9. 新闻中没有出现时间的情况，根据新闻发表时间和新闻内容中的与时间相关的词推断出时间
    10. 每个问答对尽可能多包含新闻内容，问答对内容之间可以有重复的内容
    请返回有效的JSON格式：
    [
        {{"question": "问题1", "answer": "答案1"}},
        ...
    ]
    
    只返回JSON数组，不要有额外说明。
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates high-quality conceptual questions and answers with complete context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
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
                qa["source_id"]=news_id
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
    
    # 首先进行新闻过滤
    # should_process = filter_news_article(id, title, content, pub_date)
    
    # if not should_process:
    #     logging.info(f"新闻ID {id} 被过滤，跳过处理")
    #     return None
    
    # # 生成问答对
    # logging.info(f"为以下内容生成情境问答对: 新闻ID为{id}，新闻标题为'{title}'")
    # situational_qa = generate_situational_qa(id,title, content, pub_date)
    
    logging.info(f"为以下内容生成概念问答对: 新闻ID为{id}，新闻标题为'{title}'")
    conceptual_qa = generate_conceptual_qa(id,title, content,pub_date)
    print(conceptual_qa)
    # return {
    #     "id":id,
    #     "title": title,
    #     "content": content,  # 保存完整内容
    #     "pub_date":pub_date,
    #     # "situational_qa": situational_qa,
    #     "conceptual_qa": conceptual_qa
    # }

def main():
    # 加载新闻数据
    try:
        news_df = pd.read_json('../data/11.json')
        logging.info(f"已加载 {len(news_df)} 条新闻文章")
    except Exception as e:
        logging.error(f"加载新闻数据时出错: {e}")
        return
    logging.info(f"开始处理新闻")
    # 初始化文件
    with open(os.path.join(DATASETS_DIR, 'all_processed_news.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
    with open(os.path.join(DATASETS_DIR, 'situational_qa.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
    with open(os.path.join(DATASETS_DIR, 'conceptual_qa.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
    with open(os.path.join(DATASETS_DIR, 'context_info.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
    with open(os.path.join(DATASETS_DIR, 'combined_qa.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
    with open(os.path.join(DATASETS_DIR, 'filtered_articles.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
        
    situational_qa_count = 0
    conceptual_qa_count = 0
    
    processed_count = 0
    filtered_count = 0
    filtered_articles = []  # 保存被过滤的文章信息
    
    # for idx, row in tqdm(news_df.iloc[:].iterrows(), total=len(news_df)-18, desc="处理147后的所有新闻 "):
    for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="处理所有新闻 "):
        news_item = {
            'id': row['id'],
            'title': row['title'],
            'content': row['content'],
            'pub_date': row['pub_date']
        }
        result = process_news_data(news_item)
    #     if result:
    #         processed_count += 1
    #         # 追加写入所有处理后的数据
    #         with open(os.path.join(DATASETS_DIR, 'all_processed_news.json'), 'r+', encoding='utf-8') as f:
    #             data = json.load(f)
    #             data.append(result)
    #             f.seek(0)
    #             json.dump(data, f, ensure_ascii=False, indent=2)
            
    #         # 追加写入问答对
    #         with open(os.path.join(DATASETS_DIR, 'situational_qa.json'), 'r+', encoding='utf-8') as f:
    #             data = json.load(f)
    #             data.extend(result["situational_qa"])
    #             f.seek(0)
    #             json.dump(data, f, ensure_ascii=False, indent=2)
                
    #         with open(os.path.join(DATASETS_DIR, 'conceptual_qa.json'), 'r+', encoding='utf-8') as f:
    #             data = json.load(f)
    #             data.extend(result["conceptual_qa"])
    #             f.seek(0)
    #             json.dump(data, f, ensure_ascii=False, indent=2)
                
    #         # 追加写入合并的问答对
    #         with open(os.path.join(DATASETS_DIR, 'combined_qa.json'), 'r+', encoding='utf-8') as f:
    #             data = json.load(f)
    #             combined = result["situational_qa"] + result["conceptual_qa"]
    #             random.shuffle(combined)
    #             data.extend(combined)
    #             f.seek(0)
    #             json.dump(data, f, ensure_ascii=False, indent=2)
            
    #         situational_qa_count += len(result["situational_qa"])
    #         conceptual_qa_count += len(result["conceptual_qa"])
    #     else:
    #         filtered_count += 1
    #         # 保存被过滤的文章信息
    #         filtered_articles.append({
    #             'id': news_item['id'],
    #             'title': news_item['title'],
    #             'content': news_item['content'],
    #             'pub_date': news_item['pub_date']
    #         })
    
    # # 保存被过滤的文章信息
    # with open(os.path.join(DATASETS_DIR, 'filtered_articles.json'), 'w', encoding='utf-8') as f:
    #     json.dump(filtered_articles, f, ensure_ascii=False, indent=2)
    
    # logging.info(f"处理完成统计:")
    # logging.info(f"- 总新闻数: {len(news_df)}")
    # logging.info(f"- 已处理新闻数: {processed_count}")
    # logging.info(f"- 已过滤新闻数: {filtered_count}")
    # logging.info(f"- 已生成情境问答对: {situational_qa_count} 个")
    # logging.info(f"- 已生成概念问答对: {conceptual_qa_count} 个")
    # logging.info(f"- 总共 {situational_qa_count + conceptual_qa_count} 个问答对已保存到 {os.path.join(DATASETS_DIR, 'combined_qa.json')}")
    # logging.info(f"- 被过滤的文章信息已保存到 {os.path.join(DATASETS_DIR, 'filtered_articles.json')}")


if __name__ == "__main__":
    # 运行主程序
    main()