# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import os
# import json
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from torch.utils.data import DataLoader, TensorDataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
# from torch.optim import AdamW
# import pandas as pd
# import argparse
# import logging
# from peft import LoraConfig, get_peft_model
# import swanlab

# # =============== 参数配置 ===============
# MODEL_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
# DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dunk_enhanced.json"
# SAVE_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/coke/outputs/coke_training"
# SAVE_NAME = "coke_model"

# MAX_EPOCHS = 3
# BATCH_SIZE = 1
# ACCUM_STEPS = 8
# LEARNING_RATE = 3e-5
# WARMUP_STEPS = 300
# DK_DUNK_RATIO = 1.0

# BNB_CONFIG = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# LORA_RANK = 32
# LORA_ALPHA = 64
# LORA_DROPOUT = 0.05
# TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# lora_config = LoraConfig(
#     r=LORA_RANK,
#     lora_alpha=LORA_ALPHA,
#     target_modules=TARGET_MODULES,
#     lora_dropout=LORA_DROPOUT,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='train.log',
#     filemode='a',
#     encoding='utf-8'
# )

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# def safe_tensor_to_numpy(tensor):
#     """安全地将张量转换为numpy数组，处理BFloat16类型"""
#     if tensor.dtype == torch.bfloat16:
#         return tensor.cpu().detach().float().numpy()
#     else:
#         return tensor.cpu().detach().numpy()

# def compute_geometric_mean_confidence(logits, labels):
#     """
#     推荐的置信度计算方法：几何平均概率
#     优点：稳健、对极值不敏感、数值稳定性好
#     """
#     batch_size, seq_length, vocab_size = logits.shape
    
#     # 将logits和labels向右移动一位
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
    
#     # 创建mask，忽略padding位置
#     loss_mask = (shift_labels != -100)
    
#     # 处理labels
#     shift_labels = shift_labels.clone()
#     shift_labels[shift_labels == -100] = 0
    
#     # 计算每个位置对目标token的预测概率
#     probs = torch.softmax(shift_logits, dim=-1)
#     target_probs = torch.gather(probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    
#     # 应用mask
#     masked_probs = target_probs * loss_mask
    
#     # 计算几何平均：exp(mean(log(probs)))
#     log_probs = torch.log(masked_probs + 1e-12)  # 加小值避免log(0)
#     masked_log_probs = log_probs * loss_mask
    
#     # 计算每个序列的有效token数量
#     valid_token_counts = loss_mask.sum(dim=1)
#     valid_token_counts = torch.clamp(valid_token_counts, min=1)  # 避免除以0
    
#     # 计算几何平均
#     avg_log_probs = masked_log_probs.sum(dim=1) / valid_token_counts
#     confidence = torch.exp(avg_log_probs)
    
#     return confidence

# def compute_min_prob_confidence(logits, labels):
#     """
#     备选方法：最小词元概率（论文方法）
#     用于对比分析
#     """
#     batch_size, seq_length, vocab_size = logits.shape
    
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
    
#     loss_mask = (shift_labels != -100)
#     shift_labels = shift_labels.clone()
#     shift_labels[shift_labels == -100] = 0
    
#     probs = torch.softmax(shift_logits, dim=-1)
#     target_probs = torch.gather(probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    
#     masked_probs = target_probs * loss_mask
    
#     # 最小token概率
#     min_probs = []
#     for i in range(batch_size):
#         valid_probs = masked_probs[i][loss_mask[i] > 0]
#         if len(valid_probs) > 0:
#             min_probs.append(valid_probs.min())
#         else:
#             min_probs.append(torch.tensor(0.0, device=logits.device))
#     confidence = torch.stack(min_probs)
    
#     return confidence

# def compute_prod_prob_confidence(logits, labels):
#     """
#     备选方法：概率乘积（论文方法）
#     用于对比分析
#     """
#     batch_size, seq_length, vocab_size = logits.shape
    
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
    
#     loss_mask = (shift_labels != -100)
#     shift_labels = shift_labels.clone()
#     shift_labels[shift_labels == -100] = 0
    
#     probs = torch.softmax(shift_logits, dim=-1)
#     target_probs = torch.gather(probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    
#     # 概率乘积（使用对数空间避免数值下溢）
#     log_probs = torch.log(target_probs + 1e-12)
#     masked_log_probs = log_probs * loss_mask
#     sum_log_probs = masked_log_probs.sum(dim=1)
#     confidence = torch.exp(sum_log_probs)
    
#     return confidence

# def compute_robust_consistency_loss(logits, labels, confidence_method="geo_mean"):
#     """
#     稳健的一致性损失计算
#     使用相对差异来避免数值问题
#     """
#     batch_size = logits.size(0)
#     third_batch_size = batch_size // 3
    
#     # 根据选择的方法计算置信度
#     if confidence_method == "geo_mean":
#         confidence_scores = compute_geometric_mean_confidence(logits, labels)
#     elif confidence_method == "min_prob":
#         confidence_scores = compute_min_prob_confidence(logits, labels)
#     elif confidence_method == "prod_prob":
#         confidence_scores = compute_prod_prob_confidence(logits, labels)
#     else:
#         raise ValueError(f"Unknown confidence method: {confidence_method}")
    
#     # 分割三种提示类型的置信度
#     conf_direct = confidence_scores[:third_batch_size]
#     conf_prior = confidence_scores[third_batch_size:2*third_batch_size]
#     conf_posterior = confidence_scores[2*third_batch_size:]
    
#     # 关键改进：使用相对差异计算损失
#     epsilon = 1e-12
#     rel_diff_dp = (conf_direct - conf_prior) / (conf_direct + conf_prior + epsilon)
#     rel_diff_dpo = (conf_direct - conf_posterior) / (conf_direct + conf_posterior + epsilon)
#     rel_diff_ppo = (conf_prior - conf_posterior) / (conf_prior + conf_posterior + epsilon)
    
#     # 计算一致性损失
#     loss_dp = F.mse_loss(rel_diff_dp, torch.zeros_like(rel_diff_dp))
#     loss_dpo = F.mse_loss(rel_diff_dpo, torch.zeros_like(rel_diff_dpo))
#     loss_ppo = F.mse_loss(rel_diff_ppo, torch.zeros_like(rel_diff_ppo))
    
#     total_consistency_loss = (loss_dp + loss_dpo + loss_ppo) / 3.0
    
#     return total_consistency_loss

# def compare_confidence_methods(logits, labels):
#     """
#     比较三种置信度计算方法
#     """
#     methods = ["geo_mean", "min_prob", "prod_prob"]
#     results = {}
    
#     for method in methods:
#         if method == "geo_mean":
#             results[method] = compute_geometric_mean_confidence(logits, labels)
#         elif method == "min_prob":
#             results[method] = compute_min_prob_confidence(logits, labels)
#         elif method == "prod_prob":
#             results[method] = compute_prod_prob_confidence(logits, labels)
    
#     return results

# class COKEDataProcessor:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
    
#     def prepare_training_data(self, data_path, max_length=512, dk_dunk_ratio=1.0):
#         with open(data_path, 'r', encoding='utf-8') as f:
#             raw_data = json.load(f)
        
#         dk_data = [item for item in raw_data if item.get("type") == "DK"]
#         dunk_data = [item for item in raw_data if item.get("type") == "DUNK"]
        
#         print(f"原始数据统计: DK={len(dk_data)}, DUNK={len(dunk_data)}")
        
#         balanced_data = self._balance_data(dk_data, dunk_data, dk_dunk_ratio)
#         random.shuffle(balanced_data)
        
#         print(f"平衡后数据: DK={len([x for x in balanced_data if x.get('type')=='DK'])}, DUNK={len([x for x in balanced_data if x.get('type')=='DUNK'])}")
        
#         processed_data = {
#             'input_ids_direct': [],
#             'label_direct': [],
#             'input_ids_prior': [],
#             'label_prior': [],
#             'input_ids_posterior': [],
#             'label_posterior': []
#         }
        
#         for item in balanced_data:
#             direct_encoded = self._encode_sample(item["input_direct"], item["output_direct"], max_length)
#             prior_encoded = self._encode_sample(item["input_prior"], item["output_prior"], max_length)
#             posterior_encoded = self._encode_sample(item["input_posterior"], item["output_posterior"], max_length)
            
#             processed_data['input_ids_direct'].append(direct_encoded['input_ids'])
#             processed_data['label_direct'].append(direct_encoded['labels'])
#             processed_data['input_ids_prior'].append(prior_encoded['input_ids'])
#             processed_data['label_prior'].append(prior_encoded['labels'])
#             processed_data['input_ids_posterior'].append(posterior_encoded['input_ids'])
#             processed_data['label_posterior'].append(posterior_encoded['labels'])
        
#         return processed_data
    
#     def _balance_data(self, dk_data, dunk_data, ratio=1.0):
#         if ratio <= 0:
#             raise ValueError("DK/DUNK比例必须大于0")
        
#         if len(dk_data) > len(dunk_data) * ratio:
#             target_dk_count = int(len(dunk_data) * ratio)
#             balanced_dk = random.sample(dk_data, min(target_dk_count, len(dk_data)))
#             balanced_dunk = dunk_data
#         else:
#             target_dunk_count = int(len(dk_data) / ratio)
#             balanced_dk = dk_data
#             balanced_dunk = random.sample(dunk_data, min(target_dunk_count, len(dunk_data)))
        
#         return balanced_dk + balanced_dunk
    
#     def _encode_sample(self, input_text, output_text, max_length):
#         messages = [
#             {"role": "user", "content": input_text},
#             {"role": "assistant", "content": output_text}
#         ]
        
#         formatted_text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=False
#         )
        
#         encoded = self.tokenizer(
#             formatted_text,
#             max_length=max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         # 创建labels（将input部分设为-100）
#         input_tokens = self.tokenizer.encode(messages[0]["content"])
#         input_len = len(input_tokens)
#         labels = encoded['input_ids'].clone()
#         labels[0, :input_len] = -100
        
#         return {
#             'input_ids': encoded['input_ids'].squeeze(0),
#             'labels': labels.squeeze(0)
#         }

# def compute_lm_loss(logits, labels):
#     """语言建模损失计算"""
#     logits_shifted = logits[..., :-1, :].contiguous()
#     labels_shifted = labels[..., 1:].contiguous()
    
#     loss_mask = (labels_shifted != -100)
#     labels_shifted = labels_shifted.clone()
#     labels_shifted[labels_shifted == -100] = 0
    
#     per_token_logps = torch.gather(
#         logits_shifted.log_softmax(-1), 
#         dim=2, 
#         index=labels_shifted.unsqueeze(2)
#     ).squeeze(2)
    
#     per_token_losses = -per_token_logps
#     masked_losses = per_token_losses * loss_mask
    
#     row_losses = masked_losses.sum(dim=1)
#     row_lengths = loss_mask.sum(dim=1)
#     row_average_losses = row_losses / row_lengths
    
#     loss = row_average_losses.mean()
#     return loss

# class SwanLabCallback:
#     def __init__(self):
#         self.step = 0
    
#     def on_log(self, metrics):
#         swanlab.log(metrics, step=self.step)
#         self.step += 1
    
#     def on_train_end(self, final_metrics):
#         swanlab.log(final_metrics)
#         swanlab.finish()

# def main():
#     setup_seed(42)
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     swanlab_callback = SwanLabCallback()
#     run = swanlab.init(
#         project="qwen2.5-coke-training",
#         name="coke_recommended_geometric_mean",
#         config={
#             "model": MODEL_PATH,
#             "dataset": DATA_PATH,
#             "lora_r": LORA_RANK,
#             "batch_size": BATCH_SIZE,
#             "epochs": MAX_EPOCHS,
#             "confidence_method": "geo_mean",  # 推荐使用几何平均
#             "consistency_method": "relative",  # 使用相对差异
#         },
#         mode="online",
#     )
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
    
#     print(f"加载模型: {MODEL_PATH}")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         quantization_config=BNB_CONFIG,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#     )
    
#     print("应用LoRA配置...")
#     model = get_peft_model(model, lora_config)
    
#     trainable_params = 0
#     all_params = 0
#     for _, param in model.named_parameters():
#         all_params += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
    
#     print(f"可训练参数: {trainable_params} / 总参数: {all_params} ({100 * trainable_params / all_params:.4f}%)")
    
#     print("准备训练数据...")
#     data_processor = COKEDataProcessor(tokenizer)
#     processed_data = data_processor.prepare_training_data(DATA_PATH, dk_dunk_ratio=DK_DUNK_RATIO)
    
#     def stack_tensor_list(tensor_list):
#         return torch.stack(tensor_list, dim=0)

#     train_dataset = TensorDataset(
#         stack_tensor_list(processed_data['input_ids_direct']),
#         stack_tensor_list(processed_data['label_direct']),
#         stack_tensor_list(processed_data['input_ids_prior']),
#         stack_tensor_list(processed_data['label_prior']),
#         stack_tensor_list(processed_data['input_ids_posterior']),
#         stack_tensor_list(processed_data['label_posterior'])
#     )
    
#     train_dataloader = DataLoader(
#         dataset=train_dataset, 
#         batch_size=BATCH_SIZE,
#         shuffle=True
#     )
    
#     optimizer = AdamW(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         betas=(0.98, 0.999),
#         eps=1e-9
#     )
    
#     total_steps = len(train_dataloader) * MAX_EPOCHS // ACCUM_STEPS
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=WARMUP_STEPS,
#         num_training_steps=total_steps
#     )
    
#     # 训练参数 - 使用推荐配置
#     CONSISTENCY_WEIGHT = 10
#     CONFIDENCE_METHOD = "geo_mean"  # 推荐使用几何平均
    
#     global_step = 0
#     model.train()
    
#     for epoch in range(MAX_EPOCHS):
#         pbar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        
#         epoch_losses = []
#         epoch_ce_losses = []
#         epoch_consistency_losses = []
        
#         optimizer.zero_grad()
        
#         for step, batch in enumerate(train_dataloader):
#             try:
#                 (input_ids_direct, labels_direct,
#                  input_ids_prior, labels_prior,
#                  input_ids_posterior, labels_posterior) = batch
                
#                 input_ids_direct = input_ids_direct.to(device)
#                 labels_direct = labels_direct.to(device)
#                 input_ids_prior = input_ids_prior.to(device)
#                 labels_prior = labels_prior.to(device)
#                 input_ids_posterior = input_ids_posterior.to(device)
#                 labels_posterior = labels_posterior.to(device)
                
#                 input_ids_all = torch.vstack([input_ids_direct, input_ids_prior, input_ids_posterior])
#                 labels_all = torch.vstack([labels_direct, labels_prior, labels_posterior])
                
#                 outputs_all = model(
#                     input_ids=input_ids_all,
#                     labels=labels_all,
#                     attention_mask=(input_ids_all != tokenizer.pad_token_id),
#                     use_cache=False
#                 )
                
#                 ce_loss = compute_lm_loss(outputs_all.logits, labels_all)
                
#                 # 使用推荐的稳健一致性损失计算
#                 consistency_loss = compute_robust_consistency_loss(
#                     outputs_all.logits, 
#                     labels_all, 
#                     confidence_method=CONFIDENCE_METHOD
#                 )
                
#                 total_loss = ce_loss + CONSISTENCY_WEIGHT * consistency_loss
                
#                 total_loss = total_loss / ACCUM_STEPS
#                 total_loss.backward()
                
#                 if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_dataloader):
#                     optimizer.step()
#                     scheduler.step()
#                     optimizer.zero_grad()
                
#                 global_step += 1
                
#                 epoch_losses.append(total_loss.item() * ACCUM_STEPS)
#                 epoch_ce_losses.append(ce_loss.item())
#                 epoch_consistency_losses.append(consistency_loss.item())
                
#                 pbar.update()
#                 pbar.set_description(f"Loss: {total_loss.item() * ACCUM_STEPS:.4f} (CE: {ce_loss.item():.4f}, Cons: {consistency_loss.item():.8f})")
                
#                 swanlab_callback.on_log({
#                     "train/total_loss": total_loss.item() * ACCUM_STEPS,
#                     "train/ce_loss": ce_loss.item(),
#                     "train/consistency_loss": consistency_loss.item(),
#                     "train/learning_rate": scheduler.get_last_lr()[0],
#                     "train/epoch": epoch + 1,
#                     "train/global_step": global_step
#                 })
                    
#             except Exception as e:
#                 print(f"训练步骤 {global_step} 出现错误: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         pbar.close()
        
#         if len(epoch_losses) > 0:
#             avg_loss = np.mean(epoch_losses)
#             avg_ce_loss = np.mean(epoch_ce_losses)
#             avg_consistency_loss = np.mean(epoch_consistency_losses)
            
#             print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, Cons: {avg_consistency_loss:.8f})")
            
#             swanlab_callback.on_log({
#                 "epoch/total_loss": avg_loss,
#                 "epoch/ce_loss": avg_ce_loss,
#                 "epoch/consistency_loss": avg_consistency_loss,
#                 "epoch/epoch": epoch + 1
#             })
        
#         # 保存检查点
#         if epoch + 1 == MAX_EPOCHS or (epoch + 1) % 1 == 0:  # 每个epoch保存一次
#             checkpoint_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}_epoch_{epoch+1}")
#             model.save_pretrained(checkpoint_path)
#             print(f"模型已保存到: {checkpoint_path}")
    
#     print("训练完成!")
    
#     # 保存最终模型
#     final_model_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}_final")
#     model.save_pretrained(final_model_path)
#     print(f"最终模型已保存到: {final_model_path}")
    
#     if len(epoch_losses) > 0:
#         swanlab_callback.on_train_end({
#             "final/total_loss": np.mean(epoch_losses),
#             "final/ce_loss": np.mean(epoch_ce_losses),
#             "final/consistency_loss": np.mean(epoch_consistency_losses),
#         })

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import argparse
import logging
from peft import LoraConfig, get_peft_model
import swanlab

# =============== 参数配置 ===============
MODEL_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
DATA_PATH = "/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dunk_enhanced.json"
SAVE_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/coke/outputs/coke_training_1"
SAVE_NAME = "coke_model"

MAX_EPOCHS = 3
BATCH_SIZE = 1
ACCUM_STEPS = 2
LEARNING_RATE = 3e-5
WARMUP_STEPS = 300
DK_DUNK_RATIO = 1.0

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='train.log',
    filemode='a',
    encoding='utf-8'
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_geometric_mean_confidence(logits, labels):
    """几何平均置信度计算"""
    batch_size, seq_length, vocab_size = logits.shape
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_mask = (shift_labels != -100)
    shift_labels = shift_labels.clone()
    shift_labels[shift_labels == -100] = 0
    
    probs = torch.softmax(shift_logits, dim=-1)
    target_probs = torch.gather(probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    
    masked_probs = target_probs * loss_mask
    
    log_probs = torch.log(masked_probs + 1e-12)
    masked_log_probs = log_probs * loss_mask
    
    valid_token_counts = loss_mask.sum(dim=1)
    valid_token_counts = torch.clamp(valid_token_counts, min=1)
    
    avg_log_probs = masked_log_probs.sum(dim=1) / valid_token_counts
    confidence = torch.exp(avg_log_probs)
    
    return confidence

def compute_robust_consistency_loss(logits, labels, confidence_method="geo_mean"):
    """稳健的一致性损失计算"""
    batch_size = logits.size(0)
    third_batch_size = batch_size // 3
    
    confidence_scores = compute_geometric_mean_confidence(logits, labels)
    
    conf_direct = confidence_scores[:third_batch_size]
    conf_prior = confidence_scores[third_batch_size:2*third_batch_size]
    conf_posterior = confidence_scores[2*third_batch_size:]
    
    epsilon = 1e-12
    rel_diff_dp = (conf_direct - conf_prior) / (conf_direct + conf_prior + epsilon)
    rel_diff_dpo = (conf_direct - conf_posterior) / (conf_direct + conf_posterior + epsilon)
    rel_diff_ppo = (conf_prior - conf_posterior) / (conf_prior + conf_posterior + epsilon)
    
    loss_dp = F.mse_loss(rel_diff_dp, torch.zeros_like(rel_diff_dp))
    loss_dpo = F.mse_loss(rel_diff_dpo, torch.zeros_like(rel_diff_dpo))
    loss_ppo = F.mse_loss(rel_diff_ppo, torch.zeros_like(rel_diff_ppo))
    
    total_consistency_loss = (loss_dp + loss_dpo + loss_ppo) / 3.0
    
    return total_consistency_loss

def compute_lm_loss(logits, labels):
    """语言建模损失计算"""
    logits_shifted = logits[..., :-1, :].contiguous()
    labels_shifted = labels[..., 1:].contiguous()
    
    loss_mask = (labels_shifted != -100)
    labels_shifted = labels_shifted.clone()
    labels_shifted[labels_shifted == -100] = 0
    
    per_token_logps = torch.gather(
        logits_shifted.log_softmax(-1), 
        dim=2, 
        index=labels_shifted.unsqueeze(2)
    ).squeeze(2)
    
    per_token_losses = -per_token_logps
    masked_losses = per_token_losses * loss_mask
    
    row_losses = masked_losses.sum(dim=1)
    row_lengths = loss_mask.sum(dim=1)
    row_average_losses = row_losses / row_lengths
    
    loss = row_average_losses.mean()
    return loss

class COKEDataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def prepare_training_data(self, data_path, max_length=512, dk_dunk_ratio=1.0):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        dk_data = [item for item in raw_data if item.get("type") == "DK"]
        dunk_data = [item for item in raw_data if item.get("type") == "DUNK"]
        
        print(f"原始数据统计: DK={len(dk_data)}, DUNK={len(dunk_data)}")
        
        balanced_data = self._balance_data(dk_data, dunk_data, dk_dunk_ratio)
        random.shuffle(balanced_data)
        
        print(f"平衡后数据: DK={len([x for x in balanced_data if x.get('type')=='DK'])}, DUNK={len([x for x in balanced_data if x.get('type')=='DUNK'])}")
        
        processed_data = {
            'input_ids_direct': [],
            'label_direct': [],
            'input_ids_prior': [],
            'label_prior': [],
            'input_ids_posterior': [],
            'label_posterior': []
        }
        
        for item in balanced_data:
            direct_encoded = self._encode_sample(item["input_direct"], item["output_direct"], max_length)
            prior_encoded = self._encode_sample(item["input_prior"], item["output_prior"], max_length)
            posterior_encoded = self._encode_sample(item["input_posterior"], item["output_posterior"], max_length)
            
            processed_data['input_ids_direct'].append(direct_encoded['input_ids'])
            processed_data['label_direct'].append(direct_encoded['labels'])
            processed_data['input_ids_prior'].append(prior_encoded['input_ids'])
            processed_data['label_prior'].append(prior_encoded['labels'])
            processed_data['input_ids_posterior'].append(posterior_encoded['input_ids'])
            processed_data['label_posterior'].append(posterior_encoded['labels'])
        
        return processed_data
    
    def _balance_data(self, dk_data, dunk_data, ratio=1.0):
        if ratio <= 0:
            raise ValueError("DK/DUNK比例必须大于0")
        
        if len(dk_data) > len(dunk_data) * ratio:
            target_dk_count = int(len(dunk_data) * ratio)
            balanced_dk = random.sample(dk_data, min(target_dk_count, len(dk_data)))
            balanced_dunk = dunk_data
        else:
            target_dunk_count = int(len(dk_data) / ratio)
            balanced_dk = dk_data
            balanced_dunk = random.sample(dunk_data, min(target_dunk_count, len(dunk_data)))
        
        return balanced_dk + balanced_dunk
    
    def _encode_sample(self, input_text, output_text, max_length):
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        encoded = self.tokenizer(
            formatted_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_tokens = self.tokenizer.encode(messages[0]["content"])
        input_len = len(input_tokens)
        labels = encoded['input_ids'].clone()
        labels[0, :input_len] = -100
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'labels': labels.squeeze(0)
        }

class SwanLabCallback:
    def __init__(self):
        self.step = 0
    
    def on_log(self, metrics):
        swanlab.log(metrics, step=self.step)
        self.step += 1
    
    def on_train_end(self, final_metrics):
        swanlab.log(final_metrics)
        swanlab.finish()

def main():
    setup_seed(42)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    swanlab_callback = SwanLabCallback()
    run = swanlab.init(
        project="qwen2.5-coke-training",
        name="coke_optimized_memory",
        config={
            "model": MODEL_PATH,
            "dataset": DATA_PATH,
            "lora_r": LORA_RANK,
            "batch_size": BATCH_SIZE,
            "epochs": MAX_EPOCHS,
            "confidence_method": "geo_mean",
            "consistency_method": "relative",
            "ce_loss_source": "direct_only",
            "memory_optimized": True,
        },
        mode="online",
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print("应用LoRA配置...")
    model = get_peft_model(model, lora_config)
    
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"可训练参数: {trainable_params} / 总参数: {all_params} ({100 * trainable_params / all_params:.4f}%)")
    
    print("准备训练数据...")
    data_processor = COKEDataProcessor(tokenizer)
    processed_data = data_processor.prepare_training_data(DATA_PATH, dk_dunk_ratio=DK_DUNK_RATIO)
    
    def stack_tensor_list(tensor_list):
        return torch.stack(tensor_list, dim=0)

    train_dataset = TensorDataset(
        stack_tensor_list(processed_data['input_ids_direct']),
        stack_tensor_list(processed_data['label_direct']),
        stack_tensor_list(processed_data['input_ids_prior']),
        stack_tensor_list(processed_data['label_prior']),
        stack_tensor_list(processed_data['input_ids_posterior']),
        stack_tensor_list(processed_data['label_posterior'])
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.98, 0.999),
        eps=1e-9
    )
    
    total_steps = len(train_dataloader) * MAX_EPOCHS // ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # 训练参数
    CONSISTENCY_WEIGHT = 10
    
    global_step = 0
    model.train()
    
    for epoch in range(MAX_EPOCHS):
        pbar = tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        
        epoch_losses = []
        epoch_ce_losses = []
        epoch_consistency_losses = []
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_dataloader):
            try:
                (input_ids_direct, labels_direct,
                 input_ids_prior, labels_prior,
                 input_ids_posterior, labels_posterior) = batch
                
                # 将所有数据移到设备
                input_ids_direct = input_ids_direct.to(device)
                labels_direct = labels_direct.to(device)
                input_ids_prior = input_ids_prior.to(device)
                labels_prior = labels_prior.to(device)
                input_ids_posterior = input_ids_posterior.to(device)
                labels_posterior = labels_posterior.to(device)
                
                # 方案1：内存优化版本 - 单次前向传播（推荐）
                # 将所有三个提示合并，一次前向传播
                input_ids_all = torch.vstack([input_ids_direct, input_ids_prior, input_ids_posterior])
                labels_all = torch.vstack([labels_direct, labels_prior, labels_posterior])
                
                outputs_all = model(
                    input_ids=input_ids_all,
                    labels=labels_all,
                    attention_mask=(input_ids_all != tokenizer.pad_token_id),
                    use_cache=False
                )
                
                # 从合并的输出中提取直接提示的logits（用于标准损失）
                logits_direct = outputs_all.logits[:BATCH_SIZE]
                ce_loss = compute_lm_loss(logits_direct, labels_direct)
                
                # 使用所有logits计算一致性损失
                consistency_loss = compute_robust_consistency_loss(
                    outputs_all.logits, 
                    labels_all, 
                    confidence_method="geo_mean"
                )
                
                total_loss = ce_loss + CONSISTENCY_WEIGHT * consistency_loss
                
                total_loss = total_loss / ACCUM_STEPS
                total_loss.backward()
                
                if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                epoch_losses.append(total_loss.item() * ACCUM_STEPS)
                epoch_ce_losses.append(ce_loss.item())
                epoch_consistency_losses.append(consistency_loss.item())
                
                pbar.update()
                pbar.set_description(f"Loss: {total_loss.item() * ACCUM_STEPS:.4f} (CE: {ce_loss.item():.4f}, Cons: {consistency_loss.item():.8f})")
                
                swanlab_callback.on_log({
                    "train/total_loss": total_loss.item() * ACCUM_STEPS,
                    "train/ce_loss": ce_loss.item(),
                    "train/consistency_loss": consistency_loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step
                })
                    
            except Exception as e:
                print(f"训练步骤 {global_step} 出现错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        pbar.close()
        
        if len(epoch_losses) > 0:
            avg_loss = np.mean(epoch_losses)
            avg_ce_loss = np.mean(epoch_ce_losses)
            avg_consistency_loss = np.mean(epoch_consistency_losses)
            
            print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, Cons: {avg_consistency_loss:.8f})")
            
            swanlab_callback.on_log({
                "epoch/total_loss": avg_loss,
                "epoch/ce_loss": avg_ce_loss,
                "epoch/consistency_loss": avg_consistency_loss,
                "epoch/epoch": epoch + 1
            })
        
        # 保存检查点
        if epoch + 1 == MAX_EPOCHS or (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_path)
            print(f"模型已保存到: {checkpoint_path}")
    
    print("训练完成!")
    
    final_model_path = os.path.join(SAVE_DIR, f"{SAVE_NAME}_final")
    model.save_pretrained(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    if len(epoch_losses) > 0:
        swanlab_callback.on_train_end({
            "final/total_loss": np.mean(epoch_losses),
            "final/ce_loss": np.mean(epoch_ce_losses),
            "final/consistency_loss": np.mean(epoch_consistency_losses),
        })

if __name__ == "__main__":
    main()