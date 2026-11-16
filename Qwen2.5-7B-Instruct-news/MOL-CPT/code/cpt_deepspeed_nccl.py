import json
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType
import deepspeed
from deepspeed.accelerator import get_accelerator
import logging
import time
import gc
import signal
import sys

# 设置环境变量以优化显存分配和NCCL通信
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_TIMEOUT'] = '1800'  # 增加NCCL超时时间到30分钟
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 启用异步错误处理
os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand，使用以太网
os.environ['NCCL_P2P_DISABLE'] = '1'  # 禁用P2P通信
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 使用本地回环接口

# ============================================================================
# 0. 设置日志和路径验证
# ============================================================================
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_paths():
    """验证必要的路径是否存在"""
    paths_to_check = {
        "model": "../../model/Qwen2.5-7B-Instruct",
        "general_data": "../data/magpie_general_mol.jsonl", 
        "domain_data": "../data/news_3year.jsonl",
        "output_dir": "../output"
    }
    
    missing_paths = []
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        logger.error(f"以下路径不存在: {missing_paths}")
        logger.info("请检查路径是否正确，或创建相应的目录/文件")
        return False
    
    # 创建输出目录
    os.makedirs(paths_to_check["output_dir"], exist_ok=True)
    logger.info("所有路径验证通过")
    return True

# ============================================================================
# 1. 分布式训练初始化
# ============================================================================
def init_distributed():
    """初始化分布式训练"""
    # 获取环境变量
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式进程组
    deepspeed.init_distributed()
    
    logger.info(f"[Rank {rank}/{world_size}] 初始化完成，设备: {torch.cuda.get_device_name()}")
    return local_rank, world_size, rank

# ============================================================================
# 2. DeepSpeed配置
# ============================================================================
def get_ds_config():
    """获取DeepSpeed配置"""
    ds_config = {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "weight_decay": 0.01,
                "eps": 1e-8
            }
        },
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 2e8,
            "offload_optimizer": {
                "device": "none"
            }
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "dump_state": False,
        # 增加通信相关配置
        "communication_data_type": "bfp16",
        "distributed_inference": False,
    }
    return ds_config

# ============================================================================
# 3. 加载模型和tokenizer
# ============================================================================
MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"

# 配置4bit量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 全局变量
tokenizer = None
EOS_TOKEN = None

def load_models_safely():
    """安全加载模型"""
    global tokenizer, EOS_TOKEN
    
    try:
        logger.info("加载学生模型...")
        student_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info("学生模型加载成功")

        logger.info("加载教师模型...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info("教师模型加载成功")

        # 冻结教师模型
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        teacher_model.eval()

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"
        
        return student_model, teacher_model
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

# ============================================================================
# 4. 配置LoRA
# ============================================================================
def setup_lora(student_model):
    """配置LoRA参数"""
    try:
        lora_config = LoraConfig(
            r=64,  # 减少LoRA秩
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,  # 减少alpha
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        student_model = get_peft_model(student_model, lora_config)
        student_model.gradient_checkpointing_enable()
        student_model.config.use_cache = False
        
        logger.info("LoRA配置成功")
        return student_model
        
    except Exception as e:
        logger.error(f"LoRA配置失败: {e}")
        raise

# ============================================================================
# 5. 数据统计和加载函数
# ============================================================================
def count_tokens_and_size(path, tokenizer, name=""):
    """统计数据集中的token数量和文件大小"""
    total_tokens, total_bytes, count = 0, 0, 0
    items = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"{name} 数据第 {line_num} 条无法解析：{e}")
                    continue
                
                if name == "domain":
                    text = sample.get("text", "")
                else:
                    text = sample.get("instruction", "") + sample.get("output", "")
                
                text = text.strip()
                if not text:
                    continue
                
                total_tokens += len(tokenizer(text).input_ids)
                total_bytes += len(text.encode('utf-8'))
                items.append(sample)
                count += 1
        
        logger.info(f"✅ {name} 加载完毕：{count} 条样本，{total_tokens} tokens，{round(total_tokens/1024/1024, 2)} MB")
        return items, total_tokens, total_bytes
        
    except FileNotFoundError:
        logger.error(f"数据文件不存在: {path}")
        raise
    except Exception as e:
        logger.error(f"加载{name}数据失败: {e}")
        raise

# ============================================================================
# 6. 自定义数据集类
# ============================================================================
class MoLDatasetSingleType(TorchDataset):
    """MoL数据集类"""
    def __init__(self, data, tokenizer, type, max_seq_len=4096):  # 减少序列长度
        self.data = data
        self.tokenizer = tokenizer
        self.type = type
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            
            if self.type == 'domain':
                full_text = sample['text'].strip() + EOS_TOKEN
                input_ids = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len).input_ids
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                    "is_domain_batch": 1,
                }
            else:
                prompt = self.tokenizer.apply_chat_template([
                    {"role": "user", "content": sample["instruction"].strip()}
                ], tokenize=False)
                
                prompt_ids = self.tokenizer(prompt, truncation=False).input_ids
                response_ids = self.tokenizer(sample["output"].strip() + EOS_TOKEN, truncation=False).input_ids
                
                input_ids = (prompt_ids + response_ids)[:self.max_seq_len]
                labels = ([-100] * len(prompt_ids) + response_ids)[:self.max_seq_len]
                
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                    "is_domain_batch": 0,
                }
        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {e}")
            return {
                "input_ids": torch.tensor([1], dtype=torch.long),
                "labels": torch.tensor([1], dtype=torch.long),
                "attention_mask": torch.tensor([1], dtype=torch.long),
                "is_domain_batch": 0,
            }

# ============================================================================
# 7. 自定义数据整理函数
# ============================================================================
def mol_collate_fn(batch):
    """自定义数据整理函数"""
    global tokenizer
    
    valid_batch = [b for b in batch if len(b["input_ids"]) > 0]
    if not valid_batch:
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "labels": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
            "is_domain_batch": torch.tensor([False], dtype=torch.bool),
        }
    
    input_ids = pad_sequence(
        [b["input_ids"] for b in valid_batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    labels = pad_sequence(
        [b["labels"] for b in valid_batch],
        batch_first=True,
        padding_value=-100
    )
    
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in valid_batch],
        batch_first=True,
        padding_value=0
    )
    
    is_domain_batch = torch.stack([
        torch.tensor(b["is_domain_batch"], dtype=torch.bool) for b in valid_batch
    ])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "is_domain_batch": is_domain_batch,
    }

# ============================================================================
# 8. 加载数据集
# ============================================================================
def load_datasets():
    """加载和准备数据集"""
    global tokenizer
    
    data_paths = {
        "general": "../data/magpie_general_mol1.jsonl",
        "domain": "../data/news_3year1.jsonl",
    }

    general_data, general_tokens, _ = count_tokens_and_size(data_paths["general"], tokenizer, "general")
    domain_data, domain_tokens, _ = count_tokens_and_size(data_paths["domain"], tokenizer, "domain")

    min_tokens = min(general_tokens, domain_tokens)

    def truncate_to_tokens(data, target_tokens, mode):
        """截断数据到指定token数量"""
        global tokenizer
        total, kept = 0, []
        for s in data:
            t = s.get("text", "").strip() if mode == "domain" else s.get("instruction", "").strip() + s.get("output", "").strip()
            n = len(tokenizer(t, add_special_tokens=False).input_ids)
            if total + n > target_tokens:
                break
            total += n
            kept.append(s)
        return kept, total

    general_data, _ = truncate_to_tokens(general_data, min_tokens, "general")
    domain_data, _ = truncate_to_tokens(domain_data, min_tokens, "domain")

    logger.info(f"✅ 加载数据完成：通用样本 {len(general_data)} 条，领域样本 {len(domain_data)} 条")

    batch_size = 1
    num_train_epochs = 3

    general_dataset = MoLDatasetSingleType(general_data, tokenizer, type='general')
    domain_dataset = MoLDatasetSingleType(domain_data, tokenizer, type='domain')

    general_loader = DataLoader(
        general_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mol_collate_fn,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    domain_loader = DataLoader(
        domain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mol_collate_fn,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    
    return general_loader, domain_loader, num_train_epochs

# ============================================================================
# 9. 训练循环核心函数
# ============================================================================
def train_one_batch(batch, is_domain, step_count, alpha, model_engine, teacher_model, step_losses, general_kl_log, domain_ce_log, loader_name, rank):
    """训练单个批次的函数"""
    if len(is_domain) == 0:
        if rank == 0:
            print(f"[{loader_name}] 跳过空batch")
        return
    
    outputs = None
    student_logits = None
    teacher_outputs = None
    teacher_logits = None
    
    try:
        # 学生模型前向传播
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        student_logits = outputs.logits
        
        # 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_logits = teacher_outputs.logits
        
        # 计算损失
        losses = []
        ce_losses = []
        kl_losses = []
        
        for i in range(len(is_domain)):
            mask = (batch["attention_mask"][i] == 1) & (batch["labels"][i] != -100)
            valid_indices = torch.where(mask)[0]
            
            if len(valid_indices) == 0:
                continue
            
            s_logits = student_logits[i, valid_indices]
            t_logits = teacher_logits[i, valid_indices]
            
            ce_loss = torch.nn.functional.cross_entropy(
                s_logits,
                batch["labels"][i, valid_indices],
                reduction='mean'
            )
            
            student_log_probs = torch.nn.functional.log_softmax(s_logits, dim=-1)
            teacher_probs = torch.nn.functional.softmax(t_logits, dim=-1)
            kl_per_token = torch.nn.functional.kl_div(
                input=student_log_probs,
                target=teacher_probs,
                reduction="none",
                log_target=False
            ).sum(dim=-1)
            kl_loss = torch.mean(kl_per_token)
            
            if loader_name == "General":
                loss = alpha * ce_loss + (1 - alpha) * kl_loss
                general_kl_log.append((step_count, kl_loss.item()))
            else:
                loss = (1 - alpha) * ce_loss + alpha * kl_loss
                domain_ce_log.append((step_count, ce_loss.item()))
            
            ce_losses.append(ce_loss)
            kl_losses.append(kl_loss)
            losses.append(loss)
        
        if not losses:
            if rank == 0:
                print(f"[{loader_name}] 跳过无有效token的batch")
            return
        
        loss = torch.stack(losses).mean()
        ce_loss_mean = torch.stack(ce_losses).mean().item()
        kl_loss_mean = torch.stack(kl_losses).mean().item()
        step_losses.append((step_count, loss.item(), ce_loss_mean, kl_loss_mean))
        
        # 反向传播
        model_engine.backward(loss)
        model_engine.step()
        
        if rank == 0:
            print(f"[{loader_name}] step={step_count+1}, loss={loss.item():.6f}, ce={ce_loss_mean:.6f}, kl={kl_loss_mean:.6f}")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.warning(f"⚠️ 显存不足，跳过当前batch: {e}")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise e
    finally:
        if outputs is not None:
            del outputs
        if student_logits is not None:
            del student_logits
        if teacher_outputs is not None:
            del teacher_outputs
        if teacher_logits is not None:
            del teacher_logits
        del batch
        get_accelerator().empty_cache()
        gc.collect()

# ============================================================================
# 10. 信号处理函数
# ============================================================================
def signal_handler(signum, frame):
    """处理中断信号"""
    logger.info("收到中断信号，正在优雅退出...")
    sys.exit(0)

# ============================================================================
# 11. 主训练函数
# ============================================================================
def main():
    """主训练函数"""
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 验证路径
    if not validate_paths():
        return
    
    try:
        # 初始化分布式训练
        local_rank, world_size, rank = init_distributed()
        
        # 加载模型
        student_model, teacher_model = load_models_safely()
        
        # 配置LoRA
        student_model = setup_lora(student_model)
        
        # 加载数据集
        general_loader, domain_loader, num_train_epochs = load_datasets()
        
        # 获取DeepSpeed配置
        ds_config = get_ds_config()
        
        # 使用DeepSpeed初始化模型
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=student_model,
            config=ds_config,
        )
        
        # 将教师模型移到GPU
        teacher_model.to(model_engine.device)
        
        # 开始训练
        alpha = 0
        model_engine.train()

        step_losses = []
        step_count = 0
        general_kl_log = []
        domain_ce_log = []

        start_time = time.time()

        for epoch in range(num_train_epochs):
            if rank == 0:
                logger.info(f"开始第 {epoch+1} 个epoch")
            
            general_iter = iter(general_loader)
            domain_iter = iter(domain_loader)
            finished = [False, False]
            loader_names = ["General", "Domain"]
            loader_iters = [general_iter, domain_iter]
            loader_idx = 0
            
            while not all(finished):
                try:
                    batch = next(loader_iters[loader_idx])
                    if batch is None:
                        finished[loader_idx] = True
                        loader_idx = 1 - loader_idx
                        continue
                except StopIteration:
                    finished[loader_idx] = True
                    loader_idx = 1 - loader_idx
                    continue
                
                is_domain = batch.pop("is_domain_batch")
                
                train_one_batch(
                    batch, is_domain, step_count, alpha, model_engine, teacher_model,
                    step_losses, general_kl_log, domain_ce_log, loader_names[loader_idx], rank
                )
                
                step_count += 1
                loader_idx = 1 - loader_idx

                
                torch.cuda.empty_cache()
                gc.collect()
            
            # 每个epoch结束后保存模型
            save_path = f"../output/mol_pretrain_epoch_{epoch+1}"
            model_engine.save_checkpoint(save_path)
            if rank == 0:
                tokenizer.save_pretrained(save_path)
                logger.info(f"✅ 模型已保存到 {save_path}")
            
            get_accelerator().empty_cache()
            gc.collect()

        # 保存最终模型
        save_path = "../output/final"
        model_engine.save_checkpoint(save_path)
        if rank == 0:
            tokenizer.save_pretrained(save_path)
            logger.info(f"✅ 最终模型已保存到 {save_path}")

        get_accelerator().empty_cache()
        gc.collect()

        # 保存训练日志（仅主进程）
        if rank == 0:
            with open("loss_log.txt", "w") as f:
                f.write("step,total_loss,ce_loss,kl_loss\n")
                for s, total, ce, kl in step_losses:
                    f.write(f"{s},{total:.6f},{ce:.6f},{kl:.6f}\n")

            with open("general_kl_log.txt", "w") as f:
                f.write("step,kl_loss\n")
                for s, kl in general_kl_log:
                    f.write(f"{s},{kl:.6f}\n")
            if general_kl_log:
                avg_kl = sum([kl for _, kl in general_kl_log]) / len(general_kl_log)
                logger.info(f"通用样本KL散度均值: {avg_kl:.6f}")

            with open("domain_ce_log.txt", "w") as f:
                f.write("step,ce_loss\n")
                for s, ce in domain_ce_log:
                    f.write(f"{s},{ce:.6f}\n")
            if domain_ce_log:
                avg_ce = sum([ce for _, ce in domain_ce_log]) / len(domain_ce_log)
                logger.info(f"领域样本交叉熵损失均值: {avg_ce:.6f}")

        logger.info(f"训练结束，总耗时: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 