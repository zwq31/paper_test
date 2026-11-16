# ==========================================
# 决策树边界检测推理代码
# decision_tree_inference.py
# ==========================================

import os 
import json
import torch
import logging
import numpy as np
import pickle
import time
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F

# ----------------------------
# 日志配置
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='decision_tree_inference.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 同时输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# ----------------------------
# 配置路径和参数
# ----------------------------
MODEL_BASE_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
INPUT_DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
OUTPUT_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/test_decision_tree_2400.json"
DECISION_TREE_MODEL_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/data/train/refusal/decision_tree/news_qa_boundary_detector_2400.pkl"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# 决策树边界检测器类
# ----------------------------
class DecisionTreeBoundaryDetector:
    """决策树边界检测器（精简版）"""
    
    def __init__(self, model_path: str):
        """加载训练好的决策树模型"""
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model = saved_data['model']
            self.feature_names = saved_data['feature_names']
            logger.info(f"✅ 决策树模型加载成功，使用特征: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"❌ 决策树模型加载失败: {e}")
            self.model = None
            self.feature_names = []
    
    def extract_features(self, ppl: float, min_token_prob: float, confidence: float) -> np.ndarray:
        """
        从模型输出中提取特征
        注意：特征提取必须与训练时完全一致
        """
        # 基础特征
        features = [ppl, min_token_prob, confidence]
        
        # 衍生特征（必须与训练时完全一致）
        log_ppl = np.log(ppl + 1e-6) if ppl > 0 else -10
        log_min_prob = np.log(min_token_prob + 1e-6) if min_token_prob > 0 else -10
        ppl_confidence_ratio = ppl / (confidence + 1e-6) if confidence > 0 else 1000
        conf_min_prob = confidence * min_token_prob
        ppl_min_prob_ratio = ppl / (min_token_prob + 1e-6) if min_token_prob > 0 else 1000
        weighted_score = 0.3 * ppl + 0.5 * min_token_prob + 0.2 * confidence
        
        # 组合所有特征
        full_features = features + [
            log_ppl, log_min_prob, ppl_confidence_ratio,
            conf_min_prob, ppl_min_prob_ratio, weighted_score
        ]
        
        return np.array([full_features])
    
    def should_answer(self, ppl: float, min_token_prob: float, confidence: float) -> Dict[str, Any]:
        """
        判断是否应该回答问题
        
        参数:
            ppl: 困惑度
            min_token_prob: 最小词元概率
            confidence: 自我置信度
        
        返回:
            dict: 包含预测结果和置信度
        """
        if self.model is None:
            logger.warning("决策树模型未加载，默认返回应该回答")
            return {
                'should_answer': True,
                'probability': 1.0,
                'class': '应该回答',
                'probabilities': [0.0, 1.0]
            }
        
        # 提取特征
        X = self.extract_features(ppl, min_token_prob, confidence)
        
        # 预测
        try:
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            result = {
                'should_answer': bool(prediction),  # True: 应该回答, False: 应该拒绝
                'confidence': float(probability),   # 预测置信度
                'class': '应该回答' if prediction == 1 else '应该拒绝',
                'probabilities': self.model.predict_proba(X)[0].tolist()
            }
            
            logger.info(f"决策树预测: {result['class']} (置信度: {result['confidence']:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"决策树预测失败: {e}")
            return {
                'should_answer': True,
                'probability': 1.0,
                'class': '应该回答',
                'probabilities': [0.0, 1.0]
            }

# ----------------------------
# 辅助函数
# ----------------------------
def parse_self_confidence(model_answer):
    """解析模型自报置信度"""
    try:
        lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
        self_conf = None
        
        for line in lines:
            if "置信度" in line:
                val_str = line.split(':')[-1].strip()
                try:
                    self_conf = float(val_str)
                except ValueError:
                    self_conf = None
                return self_conf
        
        if lines:
            try:
                self_conf = float(lines[-1])
            except ValueError:
                self_conf = None
        
        return self_conf
    except Exception as e:
        logger.warning(f"解析自报置信度失败: {e}")
        return None

def compute_ppl_minprob(generated_tokens, generate_logits):
    """计算困惑度和最小词元概率"""
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
# Prompt构建函数
# ----------------------------
def build_confidence_messages(question):
    """构建置信度评估prompt"""
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

def build_answer_messages(question):
    """构建普通回答prompt"""
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
# 模型加载
# ----------------------------
def load_model(model_path):
    """加载模型和tokenizer"""
    logger.info("加载模型和tokenizer...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
# 生成函数
# ----------------------------
def generate_confidence(model, tokenizer, question):
    """生成置信度"""
    messages = build_confidence_messages(question)
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_tokens = outputs.sequences[0, prompt_len:]
    confidence_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    self_conf = parse_self_confidence(confidence_output)
    
    logger.info(f"生成置信度: {self_conf}, 原始输出: {confidence_output}")
    
    return self_conf, confidence_output

def generate_answer_with_metrics(model, tokenizer, question):
    """生成答案并计算指标"""
    messages = build_answer_messages(question)
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].size(1)
    
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
    
    generated_tokens = outputs.sequences[0, prompt_len:]
    raw_model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    ppl, min_token_prob = float('inf'), 0.0
    if outputs.scores:
        generate_logits = torch.stack(outputs.scores, dim=0)
        ppl, min_token_prob = compute_ppl_minprob(generated_tokens, generate_logits)
    
    logger.info(f"生成答案完成，ppl: {ppl:.4f}, min_token_prob: {min_token_prob:.4f}")
    
    return raw_model_output, ppl, min_token_prob

# ----------------------------
# 主函数
# ----------------------------
def main():
    """主函数：决策树边界检测推理"""
    
    # 1. 加载决策树边界检测器
    logger.info("=" * 60)
    logger.info("加载决策树边界检测器...")
    logger.info("=" * 60)
    
    boundary_detector = DecisionTreeBoundaryDetector(DECISION_TREE_MODEL_PATH)
    if boundary_detector.model is None:
        logger.error("无法加载决策树模型，退出程序")
        return
    
    # 2. 加载语言模型
    logger.info("\n" + "=" * 60)
    logger.info("加载语言模型...")
    logger.info("=" * 60)
    
    model, tokenizer = load_model(MODEL_BASE_PATH)
    
    # 3. 加载测试数据
    logger.info("\n" + "=" * 60)
    logger.info("加载测试数据...")
    logger.info("=" * 60)
    
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 4. 处理每个问题
    logger.info("\n" + "=" * 60)
    logger.info("开始推理...")
    logger.info("=" * 60)
    
    results = []
    for idx, item in enumerate(tqdm(input_data, desc="处理问题")):
        question = item.get("question", "").strip()
        if not question:
            continue
        
        logger.info(f"\n处理第 {idx+1} 个问题: {question}")
        
        try:
            sample_start_time = time.perf_counter()
            
            # A. 生成置信度
            logger.info("生成置信度...")
            self_conf, confidence_output = generate_confidence(model, tokenizer, question)
            
            # B. 生成原始答案并计算指标
            logger.info("生成答案并计算指标...")
            raw_model_output, ppl, min_token_prob = generate_answer_with_metrics(model, tokenizer, question)
            
            # C. 使用决策树进行边界判断
            logger.info("使用决策树进行边界判断...")
            decision_tree_result = boundary_detector.should_answer(
                ppl, 
                min_token_prob, 
                self_conf if self_conf is not None else 0.0
            )
            
            # D. 计算处理时间
            sample_processing_time = time.perf_counter() - sample_start_time
            
            # E. 构建输出项（精简版，只包含决策相关信息）
            output_item = {
                "source_id": item.get("source_id", idx),
                "question": question,
                "reference_answer": item.get("answer", ""),  # 原始参考答案
                "model_answer": raw_model_output,            # 👈 新增：模型生成的答案
                "should_answer": decision_tree_result['should_answer'],
                "decision_metrics": {
                    "ppl": float(round(ppl, 4)) if np.isfinite(ppl) else None,
                    "min_token_prob": float(round(min_token_prob, 4)),
                    "self_reported_confidence": float(round(self_conf, 4)) if isinstance(self_conf, (int, float)) else None,
                },
                "decision_details": {
                    "prediction_confidence": decision_tree_result['confidence'],
                    "prediction_class": decision_tree_result['class'],
                },
                "processing_time_seconds": round(sample_processing_time, 3)
            }
            
            results.append(output_item)
            
            logger.info(f"第 {idx+1} 个问题处理完成:")
            logger.info(f"  should_answer: {output_item['should_answer']}")
            logger.info(f"  决策置信度: {output_item['decision_details']['prediction_confidence']:.4f}")
            logger.info(f"  总耗时: {sample_processing_time:.3f}秒")
            
        except Exception as e:
            logger.error(f"第{idx}条数据处理失败: {e}", exc_info=True)
            # 添加一个默认结果
            default_item = {
                "source_id": item.get("source_id", idx),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "should_answer": False,  # 出错时默认拒绝
                "error": str(e),
                "processing_time_seconds": 0.0
            }
            results.append(default_item)
            continue
    
    # 5. 保存结果
    logger.info("\n" + "=" * 60)
    logger.info("保存结果...")
    logger.info("=" * 60)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    
    # 6. 输出统计信息
    logger.info("\n" + "=" * 60)
    logger.info("推理完成统计:")
    logger.info("=" * 60)
    
    total_samples = len(results)
    should_answer_count = sum(1 for item in results if item.get("should_answer", False))
    should_reject_count = total_samples - should_answer_count
    
    # 计算平均处理时间
    total_processing_time = sum(item.get("processing_time_seconds", 0) for item in results)
    avg_processing_time = total_processing_time / total_samples if total_samples > 0 else 0
    
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"应该回答: {should_answer_count} ({should_answer_count/total_samples*100:.1f}%)")
    logger.info(f"应该拒绝: {should_reject_count} ({should_reject_count/total_samples*100:.1f}%)")
    logger.info(f"总处理时间: {total_processing_time:.2f}秒")
    logger.info(f"平均处理时间: {avg_processing_time:.3f}秒/样本")
    logger.info(f"结果保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()