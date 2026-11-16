#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='refusal_generation.log',
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 配置路径
INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test.json"
OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_refusal.json"

# 初始化 OpenAI 客户端（阿里云百炼）
client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def generate_refusal_answer(question: str) -> str:
    """调用大模型生成拒绝回答"""
    prompt = f"""你是一个负责任的AI助手，当无法确认信息准确性时，应礼貌地拒绝回答。

请根据以下问题，生成一段标准的拒绝回答，语气专业、礼貌，不编造信息。

要求：
1. 明确表示无法回答该问题
2. 解释原因：缺乏足够信息或无法验证
3. 建议用户通过官方渠道（如政府网站、权威机构、企业官网等）查询
4. 不要使用“对不起”开头，可用“很抱歉”或“目前”
5. 回答长度控制在100-150字左右
示例：“很抱歉，关于2024年8月28日下午温州市“一呼百应”安全文艺服务团在红蜻蜓集团举办的的具体活动类型，我没有直接的相关信息，无法回答这个问题。为了获取准确的信息，建议直接查询温州市应急管理局或相关组织的官方发布，或者联系红蜻蜓集团获取详细活动内容。如果有其他具体背景信息或需要进一步的帮助，请随时告知”

问题：
{question}

请直接输出拒绝回答内容，不要加额外说明。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",  # 推荐使用 qwen-plus 或 deepseek-v3，效果更好
            messages=[
                {"role": "system", "content": "你是一个严谨、负责任的AI助手，不会猜测或编造信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"生成拒绝回答失败（问题: {question[:50]}...）: {e}")
        return "很抱歉，我无法回答该问题。相关信息不在我的知识范围内，建议您通过官方渠道查询以获取准确信息。"


def load_data() -> List[Dict[str, Any]]:
    """加载原始数据"""
    logger.info(f"正在加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"共加载 {len(data)} 条数据")
    return data


def main():
    data = load_data()
    output_data = []

    for item in tqdm(data, desc="生成拒绝回答"):
        question = item["question"]

        # 生成拒绝回答
        refusal_answer = generate_refusal_answer(question)

        # 构造新条目（保留原字段，加入拒绝回答）
        new_item = item.copy()
        new_item["refusal_answer"] = refusal_answer  # 替换模型回答为拒绝回答
        # 或者新增字段：new_item["refusal_answer"] = refusal_answer

        output_data.append(new_item)

        # 实时保存防止中断
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"所有拒绝回答已生成并保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()