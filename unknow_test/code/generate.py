# -*- coding: utf-8 -*-
import os
import json
import torch
import logging
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='generate_triviaqa.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Paths and parameters
# ----------------------------
MODEL_BASE_PATH = "/mnt/project/unknow_test/model/Llama-2-7b-chat"  # your downloaded Llama2-Chat-7B path
INPUT_DATA_PATH = "/mnt/project/unknow_test/data/triviaqa_rc_wikipedia/validation"  # saved dataset folder
OUTPUT_PATH = "/mnt/project/unknow_test/output/triviaqa_rc_wikipedia_val_inference.json"

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048

# ----------------------------
# Knowledge boundary thresholds
# ----------------------------
PPL_THRESHOLD = 1.1022
MIN_TOKEN_PROB_THRESHOLD = 0.3662
SELF_CONF_THRESHOLD = 0.4451

# ----------------------------
# Rejection responses
# ----------------------------
REJECTION_RESPONSES = [
    "Sorry, I cannot answer this question."
]

# ----------------------------
# Build confidence prompt
# ----------------------------
def build_confidence_messages(question):
    messages = [
        {
            "role": "system",
            "content": "You are a news knowledge self-assessment assistant. Only output your confidence (0~1) for whether you can answer the following question correctly. Do NOT answer the question itself."
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Output format: Confidence: [number between 0 and 1]\n"
                "The number should accurately reflect your knowledge. 1 = very confident, 0 = no confidence."
            )
        }
    ]
    return messages

# ----------------------------
# Build answer prompt
# ----------------------------
def build_answer_messages(question):
    messages = [
        {
            "role": "system",
            "content": "You are a reliable QA assistant. Please provide concise, direct answers to questions. Answer with just the essential information, no explanations or extra details unless absolutely necessary.Respond with only the minimal answer,such as a name, number, or short phrase. No explanations or full sentences."
        },
        {
            "role": "user",
            "content": f"Question: {question}"
        }
    ]
    return messages

# ----------------------------
# Parse self-confidence
# ----------------------------
def parse_self_confidence(model_answer):
    try:
        lines = [line.strip() for line in model_answer.strip().splitlines() if line.strip()]
        self_conf = None
        for line in lines:
            if "Confidence" in line:
                val_str = line.split(':')[-1].strip()
                try:
                    self_conf = float(val_str)
                except ValueError:
                    logger.warning(f"Cannot convert confidence '{val_str}' to float")
                return self_conf
        if lines:
            try:
                self_conf = float(lines[-1])
            except ValueError:
                logger.warning(f"Cannot parse last line as float: '{lines[-1]}'")
        return self_conf
    except Exception as e:
        logger.warning(f"Parse self confidence failed: {e}")
        return None

# ----------------------------
# Compute PPL / min_token_prob
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
        logger.error(f"PPL/min_token_prob calculation failed: {e}")
        return float('inf'), 0.0

# ----------------------------
# Load model
# ----------------------------
def load_model(model_path):
    logger.info("Loading model and tokenizer...")

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
    logger.info("Model loaded")
    return model, tokenizer

# ----------------------------
# Generate self-confidence
# ----------------------------
def generate_confidence(model, tokenizer, question):
    messages = build_confidence_messages(question)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    return self_conf, confidence_output

# ----------------------------
# Generate answer
# ----------------------------
def generate_answer(model, tokenizer, question):
    messages = build_answer_messages(question)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    return raw_model_output, ppl, min_token_prob

# ----------------------------
# Check if output file exists and load existing results
# ----------------------------
def load_existing_results(output_path):
    existing_results = []
    processed_qids = set()
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                for item in existing_results:
                    if 'qid' in item:
                        processed_qids.add(item['qid'])
            logger.info(f"Loaded {len(existing_results)} existing results from {output_path}")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            existing_results = []
            processed_qids = set()
    
    return existing_results, processed_qids

# ----------------------------
# Save results incrementally
# ----------------------------
def save_results_incrementally(output_path, results):
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

# ----------------------------
# Main inference
# ----------------------------
def main():
    model, tokenizer = load_model(MODEL_BASE_PATH)

    # Load dataset saved with `save_to_disk`
    from datasets import load_from_disk
    dataset = load_from_disk(INPUT_DATA_PATH)
    train_data = dataset

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Load existing results if any
    results, processed_qids = load_existing_results(OUTPUT_PATH)
    
    # Create a progress bar
    pbar = tqdm(total=len(train_data))
    
    # Update progress bar to show already processed items
    pbar.update(len(processed_qids))
    
    processed_count = 0
    
    for idx, item in enumerate(train_data):
        qid = item['question_id']
        
        # Skip already processed items
        if qid in processed_qids:
            continue
        
        question = item['question'].strip()
        answer_obj = item['answer']
        ground_truth = {
            "value": answer_obj.get("value", ""),
            "normalized_value": answer_obj.get("normalized_value", ""),
            "aliases": answer_obj.get("aliases", [])
        }

        if not question:
            pbar.update(1)
            continue

        try:
            # 1. Self-confidence
            self_conf, confidence_output = generate_confidence(model, tokenizer, question)
            
            # 2. Generate answer
            raw_model_output, ppl, min_token_prob = generate_answer(model, tokenizer, question)
            
            # 3. Decision
            # should_answer = (
            #     ppl <= PPL_THRESHOLD and
            #     min_token_prob >= MIN_TOKEN_PROB_THRESHOLD and
            #     (self_conf is not None and self_conf >= SELF_CONF_THRESHOLD)
            # )
            
            # final_answer = raw_model_output if should_answer else random.choice(REJECTION_RESPONSES)
            
            result_item = {
                "qid": qid,
                "question": question,
                "answer": raw_model_output,
                "ppl": float(ppl) if not np.isinf(ppl) else None,
                "minprob": float(min_token_prob),
                "self_conf": float(self_conf) if self_conf is not None else None,
                "should_answer": "true",
                "final_answer": raw_model_output,
                "ground_truth": ground_truth
            }
            
            # Add to results
            results.append(result_item)
            processed_qids.add(qid)
            
            # Save after each item
            save_results_incrementally(OUTPUT_PATH, results)
            
            processed_count += 1
            pbar.update(1)
            
            # Log progress
            if processed_count % 10 == 0:
                logger.info(f"Processed {processed_count} items, total saved: {len(results)}")
                
        except Exception as e:
            logger.error(f"Error processing idx={idx}, qid={qid}: {e}")
            pbar.update(1)
            continue
    
    pbar.close()
    logger.info(f"Inference finished. Total results saved: {len(results)}")
    logger.info(f"Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()