import os
import json
import torch
import logging
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F

# 导入RAG相关库
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from sentence_transformers import SentenceTransformer

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_train.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ----------------------------
# 配置路径和参数
# ----------------------------
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/test/test.json"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# RAG配置
# ----------------------------
ZILLIZ_URI = "https://in03-2b09ee1c4f7282c.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
ZILLIZ_TOKEN = "__SET_VIA_ENVIRONMENT__"
MILVUS_COLLECTIONS = {}  # 存储多个集合对象
COLLECTION_MAPPING = {
    "永嘉": "news_articles_yongjia",
    "缙云": "news_articles_jinyun"
}
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
TOP_K = 5  # 检索前5个片段

# ----------------------------
# 拒绝阈值
# ----------------------------
PPL_THRESHOLD = 1.1060320605962417
MIN_TOKEN_PROB_THRESHOLD = 0.33927634042578086
SELF_CONF_THRESHOLD = 0.3533966908809123  # 自报置信度阈值

# ----------------------------
# 拒答模板
# ----------------------------
REJECTION_RESPONSES = [
    "抱歉，我无法回答这个问题。",
    "根据现有的信息，我无法回答这个问题。",
    "我暂时没有足够的信息来回答这个问题。"
]

# ----------------------------
# RAG相关函数
# ----------------------------
class VectorDBManager:
    """向量数据库管理器"""
    
    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token
        self.client = None
        self.embedding_model = None
        
    def initialize(self):
        """初始化连接和模型"""
        try:
            # 初始化Milvus客户端
            self.client = MilvusClient(
                uri=self.uri,
                token=self.token
            )
            
            # 初始化嵌入模型
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("向量数据库和嵌入模型初始化成功")
            
            # 测试连接
            for region, collection_name in COLLECTION_MAPPING.items():
                try:
                    if self.client.has_collection(collection_name):
                        logger.info(f"找到集合: {collection_name}")
                    else:
                        logger.warning(f"集合 {collection_name} 不存在")
                except Exception as e:
                    logger.warning(f"检查集合 {collection_name} 时出错: {e}")
                    
        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        if self.embedding_model is None:
            raise ValueError("嵌入模型未初始化")
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def retrieve(self, question: str, region: str = None) -> List[Dict]:
        """
        检索相关文档
        
        Args:
            question: 问题文本
            region: 地区名称，如果不指定则尝试自动检测
            
        Returns:
            检索结果列表，每个元素包含文档信息和相似度得分
        """
        if self.client is None or self.embedding_model is None:
            raise ValueError("向量数据库未初始化")
        
        # 自动检测地区（如果未指定）
        if region is None:
            region = self.detect_region(question)
        
        # 获取对应的集合名称
        collection_name = COLLECTION_MAPPING.get(region)
        if not collection_name:
            logger.warning(f"未找到地区 '{region}' 对应的集合")
            return []
        
        # 检查集合是否存在
        if not self.client.has_collection(collection_name):
            logger.warning(f"集合 '{collection_name}' 不存在")
            return []
        
        try:
            # 生成问题嵌入
            question_embedding = self.get_embedding(question)
            
            # 执行向量搜索
            search_results = self.client.search(
                collection_name=collection_name,
                data=[question_embedding],
                anns_field="embedding",  # 向量字段名
                limit=TOP_K,
                output_fields=["content", "title", "date", "source"]  # 返回的字段
            )
            
            # 格式化结果
            formatted_results = []
            for result in search_results[0]:  # search_results 是列表的列表
                formatted_results.append({
                    "content": result.get("entity", {}).get("content", ""),
                    "title": result.get("entity", {}).get("title", ""),
                    "date": result.get("entity", {}).get("date", ""),
                    "source": result.get("entity", {}).get("source", ""),
                    "score": result.get("distance", 0.0),
                    "id": result.get("id", "")
                })
            
            logger.info(f"检索到 {len(formatted_results)} 个相关文档")
            return formatted_results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
    
    def detect_region(self, question: str) -> str:
        """从问题中检测地区"""
        question_lower = question.lower()
        for region in COLLECTION_MAPPING.keys():
            if region.lower() in question_lower:
                return region
        
        # 如果没找到明确的地区，返回第一个地区作为默认
        return list(COLLECTION_MAPPING.keys())[0] if COLLECTION_MAPPING else "永嘉"
    
    def has_supporting_info(self, question: str, retrieved_docs: List[Dict]) -> bool:
        """
        判断检索到的文档是否包含支持问题的相关信息
        
        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表
            
        Returns:
            bool: 是否有支持信息
        """
        if not retrieved_docs:
            return False
        
        # 简单判断：如果检索到的文档相似度得分高于某个阈值，则认为有相关信息
        # 可以根据实际情况调整阈值
        min_score_threshold = 0.6
        
        # 检查是否有高相似度的文档
        for doc in retrieved_docs:
            if doc.get("score", 0) > min_score_threshold:
                return True
        
        # 也可以添加更复杂的判断逻辑，比如关键词匹配等
        # 这里简化处理，如果有检索结果就认为可能有相关信息
        return len(retrieved_docs) > 0

# 全局向量数据库管理器
vector_db_manager = None

# ----------------------------
# 构建知识自信度 prompt
# ----------------------------
def build_confidence_messages(question):
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

# ----------------------------
# 构建普通回答 prompt
# ----------------------------
def build_answer_messages(question):
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

# ----------------------------
# 构建RAG回答 prompt
# ----------------------------
def build_rag_answer_messages(question, retrieved_contexts):
    """
    构建基于检索结果的回答prompt
    
    Args:
        question: 问题
        retrieved_contexts: 检索到的上下文列表
        
    Returns:
        messages列表
    """
    # 将检索结果整合为上下文
    context_text = "\n\n".join([
        f"【资料{i+1}】{doc.get('title', '')}\n{doc.get('content', '')}"
        for i, doc in enumerate(retrieved_contexts)
    ])
    
    messages = [
        {
            "role": "system",
            "content": "你是一个新闻问答助手，请根据提供的参考资料回答问题。如果参考资料中没有相关信息，请诚实地说明无法回答。"
        },
        {
            "role": "user",
            "content": (
                f"请根据以下参考资料回答问题：\n\n"
                f"参考资料：\n{context_text}\n\n"
                f"问题：{question}\n\n"
                "要求：\n"
                "1. 仅根据参考资料中的信息回答\n"
                "2. 如果参考资料中没有相关信息，请回答：'抱歉，我无法回答这个问题。'\n"
                "3. 保持回答简洁准确"
            )
        }
    ]
    return messages

# ----------------------------
# 解析模型自报置信度
# ----------------------------
def parse_self_confidence(model_answer):
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
                    logger.warning(f"置信度值无法转为浮点数: '{val_str}'")
                    self_conf = None
                return self_conf
        
        # 如果没有找到含 '置信度' 的行，尝试最后一行直接转换为数字
        if lines:
            try:
                self_conf = float(lines[-1])
            except ValueError:
                logger.warning(f"最后一行无法转为浮点数: '{lines[-1]}'")
                self_conf = None
        
        return self_conf
    except Exception as e:
        logger.warning(f"解析自报置信度失败: {e}")
        return None

# ----------------------------
# 计算 ppl / min_token_prob
# ----------------------------
def compute_ppl_minprob(generated_tokens, generate_logits):
    try:
        if generate_logits.dim() == 3:
            generate_logits = generate_logits.squeeze(1)
        min_len = min(generate_logits.size(0), len(generated_tokens))
        if min_len == 0:
            return float('inf'), 0.0
        generate_logits = generate_logits[:min_len]
        generated_tokens = generated_tokens[:min_len]

        token_probs = []
        for i, token_id in enumerate(generated_tokens):
            try:
                logits_float = generate_logits[i].float()
                token_prob = F.softmax(logits_float, dim=-1)[token_id].item()
                token_probs.append(token_prob)
            except IndexError:
                continue

        min_token_prob = min(token_probs) if token_probs else 0.0

        if token_probs:
            safe_probs = np.clip(token_probs, 1e-15, 1.0)
            nll = -np.mean(np.log(safe_probs))
            ppl = np.exp(nll)
            if np.isnan(ppl) or np.isinf(ppl):
                ppl = float('inf')
        else:
            ppl = float('inf')

        return ppl, min_token_prob
    except Exception as e:
        logger.error(f"PPL/min_token_prob计算失败: {e}")
        return float('inf'), 0.0

# ----------------------------
# 加载模型
# ----------------------------
def load_model(model_path):
    logger.info("加载模型和tokenizer...")
    
    # 配置4bit量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    model.eval()
    logger.info("模型加载完成")
    return model, tokenizer

# ----------------------------
# 初始化RAG系统
# ----------------------------
def initialize_rag():
    """初始化RAG系统（向量数据库）"""
    global vector_db_manager
    
    logger.info("初始化RAG系统...")
    try:
        vector_db_manager = VectorDBManager(ZILLIZ_URI, ZILLIZ_TOKEN)
        vector_db_manager.initialize()
        logger.info("RAG系统初始化成功")
        return True
    except Exception as e:
        logger.error(f"RAG系统初始化失败: {e}")
        return False

# ----------------------------
# 生成置信度
# ----------------------------
def generate_confidence(model, tokenizer, question):
    # 构建置信度生成消息列表
    messages = build_confidence_messages(question)
    
    # 使用apply_chat_template构建文本
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
    # 生成置信度
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # 置信度输出很短
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成结果
    generated_tokens = outputs.sequences[0, prompt_len:]
    confidence_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 解析置信度
    self_conf = parse_self_confidence(confidence_output)
    
    return self_conf, confidence_output

# ----------------------------
# 生成答案（通用函数）
# ----------------------------
def generate_answer_with_context(model, tokenizer, question, contexts=None):
    """
    生成答案，可以指定上下文（用于RAG）
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        question: 问题
        contexts: 检索到的上下文列表，如果为None则使用普通模式
        
    Returns:
        raw_model_output: 原始模型输出
        ppl: 困惑度
        min_token_prob: 最小词元概率
    """
    # 构建消息
    if contexts:
        messages = build_rag_answer_messages(question, contexts)
    else:
        messages = build_answer_messages(question)
    
    # 使用apply_chat_template构建文本
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成答案
    generated_tokens = outputs.sequences[0, prompt_len:]
    raw_model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 计算ppl和min_token_prob
    ppl, min_token_prob = float('inf'), 0.0
    if outputs.scores:
        generate_logits = torch.stack(outputs.scores, dim=0)
        ppl, min_token_prob = compute_ppl_minprob(generated_tokens, generate_logits)
    
    return raw_model_output, ppl, min_token_prob

# ----------------------------
# 主函数
# ----------------------------
def main():
    # 加载语言模型
    model, tokenizer = load_model(MODEL_BASE_PATH)
    
    # 初始化RAG系统
    rag_enabled = initialize_rag()
    if not rag_enabled:
        logger.warning("RAG系统初始化失败，将只使用模型直接推理")
    
    # 加载输入数据
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 处理每个问题
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        f_out.write("[\n")
        for idx, item in enumerate(tqdm(input_data)):
            question = item.get("question", "").strip()
            if not question:
                continue
            
            try:
                # 第一步：生成置信度
                self_conf, confidence_output = generate_confidence(model, tokenizer, question)
                
                # 第二步：生成原始答案（不带RAG）
                raw_model_output, ppl, min_token_prob = generate_answer_with_context(
                    model, tokenizer, question, contexts=None
                )
                
                # 第三步：决策流程
                should_answer_directly = (
                    ppl <= PPL_THRESHOLD and
                    min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
                    (self_conf is not None and self_conf >= SELF_CONF_THRESHOLD)
                )
                
                final_answer = ""
                use_rag = False
                retrieved_docs = []
                has_supporting_info = False
                
                if should_answer_directly:
                    # 直接使用模型答案
                    final_answer = raw_model_output
                    decision_reason = "满足阈值条件，直接使用模型答案"
                else:
                    # 使用RAG
                    use_rag = True
                    decision_reason = "未满足阈值条件，启用RAG检索"
                    
                    if rag_enabled and vector_db_manager:
                        # 检索相关文档
                        retrieved_docs = vector_db_manager.retrieve(question)
                        
                        # 判断是否有支持信息
                        has_supporting_info = vector_db_manager.has_supporting_info(question, retrieved_docs)
                        
                        if has_supporting_info and retrieved_docs:
                            # 使用检索到的文档生成答案
                            rag_answer, rag_ppl, rag_min_prob = generate_answer_with_context(
                                model, tokenizer, question, contexts=retrieved_docs[:TOP_K]
                            )
                            final_answer = rag_answer
                            decision_reason += "，检索到相关信息，基于检索结果生成答案"
                        else:
                            # 没有相关支持信息，拒绝回答
                            final_answer = random.choice(REJECTION_RESPONSES)
                            decision_reason += "，未检索到相关信息，拒绝回答"
                    else:
                        # RAG不可用，拒绝回答
                        final_answer = random.choice(REJECTION_RESPONSES)
                        decision_reason += "，RAG系统不可用，拒绝回答"
                
                # 构建输出项
                output_item = {
                    "source_id": item.get("source_id", idx),
                    "question": question,
                    "standard_answer": item.get("answer", ""),
                    "model_answer": final_answer,
                    "raw_model_output": raw_model_output,
                    "refusal_decision": {
                        "ppl": round(ppl, 4) if np.isfinite(ppl) else None,
                        "min_token_prob": round(min_token_prob, 4),
                        "self_reported_confidence": round(self_conf, 4) if isinstance(self_conf, (int, float)) else None,
                        "confidence_output": confidence_output,
                        "should_answer_directly": bool(should_answer_directly),
                        "use_rag": use_rag,
                        "has_supporting_info": has_supporting_info,
                        "retrieved_docs_count": len(retrieved_docs),
                        "retrieved_docs": [
                            {
                                "title": doc.get("title", ""),
                                "content_preview": doc.get("content", "")[:100] + "..." if len(doc.get("content", "")) > 100 else doc.get("content", ""),
                                "score": round(doc.get("score", 0), 4)
                            }
                            for doc in retrieved_docs[:3]  # 只保存前3个文档的预览
                        ],
                        "decision_reason": decision_reason,
                        "final_model_answer": final_answer
                    }
                }
                
                # 写入输出文件
                if idx > 0:
                    f_out.write(",\n")
                f_out.write(json.dumps(output_item, ensure_ascii=False, indent=2))
                f_out.flush()
                
            except Exception as e:
                logger.error(f"第{idx}条数据处理失败: {e}")
                continue
        
        f_out.write("\n]")
    
    logger.info(f"推理完成，结果保存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()