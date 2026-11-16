# -*- coding: utf-8 -*-
"""
多文档 + 实体链路生成问答对示例（仅处理前3条数据）
"""

import os
import json
import time
from openai import OpenAI

# ----------------------
# 配置
# ----------------------
API_KEY = "__SET_VIA_ENVIRONMENT__"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
INPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/2_hop_paths_all.json"
OUTPUT_FILE = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/2_hop_data_sample3.json"

# ----------------------
# 初始化大模型客户端
# ----------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ----------------------
# 加载 JSON 数据
# ----------------------
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# ----------------------
# 定义问答生成函数
# ----------------------
def generate_qa(evidence_docs, supporting_path, retries=3):
    """
    根据多文档和实体链路生成问答对
    """
    docs_text = ""
    for i, doc in enumerate(evidence_docs):
        content_preview = doc['content']
        docs_text += f"文档{i+1}标题: {doc['title']}\n内容:\n{content_preview}\n\n"

    path_text = " → ".join(supporting_path)

    prompt = f"""
你是一个多跳问答生成专家，请基于以下两篇文档和实体关系链，生成一条**严格的2跳事实性问答对**。

# 文档内容
{docs_text}

# 实体关系链
{path_text}

# 核心任务要求
1.  必须生成**2跳问答**：推理链为「实体A -[关系1]-> 实体B -[关系2]-> 实体C」，两步推理缺一不可，且**第一步信息来自一篇文档，第二步信息来自另一篇文档**，禁止单文档即可回答的伪多跳问题。
2.  问题需为**明确的事实性问题**，表述简洁无冗余，显性体现2跳逻辑（避免冗余修饰），不得省略中间实体的关联环节。
3.  答案必须是实体关系链的终点实体，推理过程需清晰对应两步关联逻辑，且每一步都能在对应文档中找到依据。
4.  中间实体必须隐藏起来，禁止显示，
5.  必须是2跳问答，就是从一个文档中推出A->B，再从一个文档中推出B->C，得到最终答案C
6.  必须有完整的上下文，包括时间地点，不能有模糊歧义，必须有约束条件，因为都是新闻数据，具有时效性，A->B,B->C都要有约束条件，完整的上下文

# 参考示例（跨文档2跳范式）
[
    {{
        "question": "《盗梦空间》的导演还执导过哪部获得奥斯卡最佳影片的电影？",
        "answer": "《奥本海默》",
        "reasoning": "《盗梦空间》的导演是克里斯托弗·诺兰（文档1），克里斯托弗·诺兰执导的《奥本海默》获得了奥斯卡最佳影片（文档2）。"
    }}
]

# 输出格式（仅输出JSON，无其他内容）
[
    {{
        "question": "你的2跳事实性问题",
        "answer": "答案实体",
        "reasoning": "分步推理：1. 从文档X获取「实体A与实体B的关系」；2. 从文档Y获取「实体B与实体C的关系」"
    }}
]
"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个专业问答对生成专家，严格基于文档内容生成问题和答案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=20000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            result = json.loads(json_str)
            if isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            print(f"⚠️ 调用大模型失败: {e}，尝试 {attempt+1}/{retries}")
            time.sleep(3)
    return []

# ----------------------
# 主流程（仅处理前3条）
# ----------------------
all_results = []

for i, item in enumerate(data_list[0:10]):  # 仅处理前三条数据
    print(f"\n=== 处理第 {i+1} 条数据 ===")
    evidence_docs = item.get("evidence_docs", [])
    supporting_path = item.get("supporting_path", [])
    if not evidence_docs or not supporting_path:
        print("⚠️ 没有文档或实体链路，跳过")
        continue

    qa_pairs = generate_qa(evidence_docs, supporting_path)
    all_results.append({
        "evidence_docs": [doc["title"] for doc in evidence_docs],
        "supporting_path": supporting_path,
        "qa_pairs": qa_pairs
    })
    print(f"✅ 生成 {len(qa_pairs)} 个问答对")

# ----------------------
# 保存结果
# ----------------------
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n🎉 前3条数据问答对已保存到 {OUTPUT_FILE}")
