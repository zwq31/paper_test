# rag_utils.py
import logging
import random
from typing import List, Dict, Any

# 拒答模板
REJECTION_RESPONSES = [
    "抱歉，我无法回答这个问题。",
    "根据现有的信息，我无法回答这个问题。",
    "我暂时没有足够的信息来回答这个问题。"
]

# 拒绝阈值
DEFAULT_THRESHOLDS = {
    "PPL_THRESHOLD": 1.1022038499105375,
    "MIN_TOKEN_PROB_THRESHOLD": 0.3661533556634178,
    "SELF_CONF_THRESHOLD": 0.44508323935997046
}

def build_confidence_messages(question: str) -> List[Dict[str, str]]:
    """构建知识自信度 prompt"""
    messages = [
        {
            "role": "system",
            "content": "你是一个新闻知识自省助手，你的任务是根据你对问题所掌握的事实知识，判断自己能否准确回答问题，并输出置信度（0.00-1.00）"
        },
        {
            "role": "user",
            "content": (
                f"请仅输出你对下面问题的知识自信度（0.00~1.00），表示你对能否准确回答该问题的自信程度\n\n"
                f"问题：{question}\n\n"
                "严格按照下列格式要求输出：\n"
                "置信度: [0-1之间的数字]\n"
                "要求：置信度能准确反映你对问题的了解程度，1表示非常了解且非常有把握回答正确。0表示没有把握回答正确，如果你不确定，输出一个较低的置信度\n"
                "不要回答问题，只输出置信度。不能省略置信度标识"
            )
        }
    ]
    return messages

def build_answer_messages(question: str) -> List[Dict[str, str]]:
    """构建普通回答 prompt"""
    messages = [
        {
            "role": "system",
            "content": "你是一个可信新闻问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
        },
        {
            "role": "user",
            "content": f"请回答下面的问题：\n\n问题：{question}"
        }
    ]
    return messages

def build_rag_answer_messages(question: str, retrieved_contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    构建基于检索结果的回答prompt
    
    Args:
        question: 问题
        retrieved_contexts: 检索到的上下文列表
        
    Returns:
        messages列表
    """
    if not retrieved_contexts:
        return build_answer_messages(question)
    
    # 将检索结果整合为上下文
    context_parts = []
    for i, doc in enumerate(retrieved_contexts[:5]):  # 使用前5个检索结果
        # 获取内容字段
        content = doc.get("content", "")
        title = doc.get("title", f"资料{i+1}")
        publish_date = doc.get("publish_date_str", "未知日期")
        region = doc.get("region", "未知地区")
        retrieval_type = doc.get("retrieval_type", "unknown")
        
        if content:
            # 截断内容，避免prompt过长
            max_content_length = 800
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            context_parts.append(f"【{title}】\n发布时间：{publish_date}\n地区：{region}\n检索类型：{retrieval_type}\n内容：{content}")
    
    if not context_parts:
        return build_answer_messages(question)
    
    context_text = "\n\n".join(context_parts)
    
    messages = [
        {
            "role": "system",
            "content": "你是一个新闻问答助手，请根据提供的参考资料回答问题。如果参考资料中没有直接相关信息，请诚实地说明无法回答。"
        },
        {
            "role": "user",
            "content": (
                "你是一个新闻问答助手，请基于以下参考资料回答问题。\n\n"
                f"参考资料：\n{context_text}\n\n"
                f"问题：{question}\n\n"
                "请遵守以下严格规则：\n"
                "1. **只能**使用参考资料中明确写出的信息\n"
                "2. 回答中的每一句话都必须有明确的参考资料依据\n"
                "3. **禁止**使用参考资料之外的任何知识\n"
                "4. **禁止**进行推测、联想或连接未提及的信息点\n"
                "5. **禁止**总结不存在的观点\n\n"
                "回答流程：\n"
                "- 首先，检查参考资料是否包含回答问题的所有必要信息\n"
                "- 如果包含：提取相关信息形成简洁答案\n"
                "- 如果不包含：严格回复「抱歉，根据现有资料我无法回答这个问题。」\n\n"
                "注意：即使你对此话题有其他知识，也不能使用。\n"
                "注意：如果参考资料只有部分信息但不足以完整回答问题，也必须回复「抱歉，根据现有资料我无法回答这个问题。」\n\n"
                "最终答案格式要求：\n"
                "- 如果能够回答：直接给出简洁答案，不需要额外说明\n"
                "- 如果无法回答：必须且只能回复「抱歉，根据现有资料我无法回答这个问题。」\n"
                "- 不要添加「根据资料显示」等前缀\n"
                "- 不要解释为什么能或不能回答\n\n"
            )
        }
    ]
    
    return messages

def parse_self_confidence(model_answer: str) -> float:
    """
    解析模型自报置信度
    
    Args:
        model_answer: 模型输出
        
    Returns:
        置信度值或None
    """
    try:
        lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
        self_conf = None
        
        # 优先查找含 '置信度:' 的行
        for line in lines:
            if "置信度" in line:
                val_str = line.split(':')[-1].strip()
                try:
                    self_conf = float(val_str)
                except ValueError:
                    continue
                return self_conf
        
        # 如果没有找到含 '置信度' 的行，尝试最后一行直接转换为数字
        if lines:
            try:
                self_conf = float(lines[-1])
            except ValueError:
                pass
        
        return self_conf
    except Exception as e:
        return None

def get_random_rejection_response() -> str:
    """获取随机拒答响应"""
    return random.choice(REJECTION_RESPONSES)

def should_answer_directly(ppl: float, min_token_prob: float, self_conf: float, thresholds: Dict[str, float] = None) -> bool:
    """
    判断是否应该直接回答
    
    Args:
        ppl: 困惑度
        min_token_prob: 最小词元概率
        self_conf: 自报置信度
        thresholds: 阈值字典
        
    Returns:
        bool: 是否应该直接回答
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    return (
        ppl <= thresholds["PPL_THRESHOLD"] and
        min_token_prob >= thresholds["MIN_TOKEN_PROB_THRESHOLD"] and
        (self_conf is not None and self_conf >= thresholds["SELF_CONF_THRESHOLD"])
    )