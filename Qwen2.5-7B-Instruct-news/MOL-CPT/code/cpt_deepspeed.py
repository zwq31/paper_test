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
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from tqdm import tqdm
import time
import gc
import deepspeed
from deepspeed.accelerator import get_accelerator
import logging

# 设置环境变量以优化显存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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
# 1. 初始化 Accelerator 使用 DeepSpeed Zero2 (改为Zero2避免兼容性问题)
# ============================================================================
# 使用DeepSpeed Zero2进行分布式训练，支持大模型训练
# Zero2可以优化显存使用，支持混合精度训练
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,  # 改回Zero2，避免与LoRA的兼容性问题
)

# 初始化Accelerator，配置混合精度训练为bf16
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    mixed_precision="bf16"  # 使用bfloat16混合精度，减少显存使用
)

# 打印当前进程和设备信息
logger.info(f"[Rank {accelerator.process_index}] Hello from {torch.cuda.get_device_name()} (device={torch.cuda.current_device()})")

# ============================================================================
# 2. 加载模型和tokenizer
# ============================================================================
MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"

# 配置4bit量化参数，大幅减少显存使用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用4bit量化
    bnb_4bit_use_double_quant=True,  # 使用双重量化进一步压缩
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bf16
)

# 全局变量，用于在函数间共享
tokenizer = None
EOS_TOKEN = None

def load_models_safely():
    """安全加载模型，包含错误处理"""
    global tokenizer, EOS_TOKEN
    
    try:
        logger.info("加载学生模型...")
        # 加载学生模型（需要训练的模型）
        student_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,  # 应用4bit量化
            trust_remote_code=True,  # 信任远程代码，用于加载自定义模型
            torch_dtype=torch.bfloat16,  # 明确指定数据类型
        )
        logger.info("学生模型加载成功")

        logger.info("加载教师模型...")
        # 加载教师模型（用于知识蒸馏）
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        logger.info("教师模型加载成功")

        # 冻结教师模型参数，不进行梯度计算
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False  # 禁用缓存以节省显存
        teacher_model.eval()  # 设置为评估模式

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"  # 获取结束标记
        
        return student_model, teacher_model
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

# ============================================================================
# 3. 配置LoRA参数高效微调
# ============================================================================
def setup_lora(student_model):
    """配置LoRA参数"""
    try:
        lora_config = LoraConfig(
            r=64,  # LoRA秩，控制可训练参数数量
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj",      # MLP层
                "embed_tokens", "lm_head"                 # 嵌入层和输出层
            ],
            lora_alpha=32,  # LoRA缩放参数
            lora_dropout=0.05,  # LoRA dropout率
            bias="none",  # 不训练偏置项
            task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
        )

        # 将LoRA适配器应用到学生模型
        student_model = get_peft_model(student_model, lora_config)
        student_model.gradient_checkpointing_enable()  # 启用梯度检查点以节省显存
        student_model.config.use_cache = False  # 禁用缓存
        
        logger.info("LoRA配置成功")
        return student_model
        
    except Exception as e:
        logger.error(f"LoRA配置失败: {e}")
        raise

# ============================================================================
# 4. 数据统计和加载函数
# ============================================================================
def count_tokens_and_size(path, tokenizer, name=""):
    """
    统计数据集中的token数量和文件大小
    
    Args:
        path: 数据文件路径
        tokenizer: 分词器
        name: 数据集名称，用于日志输出
    
    Returns:
        items: 数据样本列表
        total_tokens: 总token数
        total_bytes: 总字节数
    """
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
                
                # 根据数据类型提取文本
                if name == "domain":
                    text = sample.get("text", "")
                else:
                    text = sample.get("instruction", "") + sample.get("output", "")
                
                text = text.strip()
                if not text:
                    continue
                
                # 统计token数和字节数
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
# 5. 自定义数据集类
# ============================================================================
class MoLDatasetSingleType(TorchDataset):
    """
    MoL（Mixture of LoRA）数据集类
    支持通用数据和领域数据的统一处理
    """
    def __init__(self, data, tokenizer, type, max_seq_len=4096):
        """
        Args:
            data: 数据样本列表
            tokenizer: 分词器
            type: 数据类型 ('domain' 或 'general')
            max_seq_len: 最大序列长度
        """
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
                # 领域数据：直接使用文本，所有token都参与损失计算
                full_text = sample['text'].strip() + EOS_TOKEN
                input_ids = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len).input_ids
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long),  # 所有token都作为标签
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                    "is_domain_batch": 1,  # 标记为领域数据
                }
            else:
                # 通用数据：使用对话模板，只计算回复部分的损失
                prompt = self.tokenizer.apply_chat_template([
                    {"role": "user", "content": sample["instruction"].strip()}
                ], tokenize=False)
                
                prompt_ids = self.tokenizer(prompt, truncation=False).input_ids
                response_ids = self.tokenizer(sample["output"].strip() + EOS_TOKEN, truncation=False).input_ids
                
                # 拼接输入，限制最大长度
                input_ids = (prompt_ids + response_ids)[:self.max_seq_len]
                # 标签：prompt部分用-100（不计算损失），response部分正常计算
                labels = ([-100] * len(prompt_ids) + response_ids)[:self.max_seq_len]
                
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                    "is_domain_batch": 0,  # 标记为通用数据
                }
        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {e}")
            # 返回一个最小的有效样本
            return {
                "input_ids": torch.tensor([1], dtype=torch.long),
                "labels": torch.tensor([1], dtype=torch.long),
                "attention_mask": torch.tensor([1], dtype=torch.long),
                "is_domain_batch": 0,
            }

# ============================================================================
# 6. 自定义数据整理函数
# ============================================================================
def mol_collate_fn(batch):
    """
    自定义数据整理函数，处理变长序列的padding
    
    Args:
        batch: 批次数据列表
    
    Returns:
        整理后的批次数据字典
    """
    global tokenizer
    
    # 过滤掉空的样本
    valid_batch = [b for b in batch if len(b["input_ids"]) > 0]
    if not valid_batch:
        # 如果所有样本都为空，返回一个最小的有效batch
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "labels": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
            "is_domain_batch": torch.tensor([False], dtype=torch.bool),
        }
    
    # 对输入序列进行padding
    input_ids = pad_sequence(
        [b["input_ids"] for b in valid_batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    # 对标签序列进行padding，使用-100作为padding值
    labels = pad_sequence(
        [b["labels"] for b in valid_batch],
        batch_first=True,
        padding_value=-100
    )
    
    # 对注意力掩码进行padding
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in valid_batch],
        batch_first=True,
        padding_value=0
    )
    
    # 收集批次类型标记
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
# 7. 加载数据集和数据加载器
# ============================================================================
def load_datasets():
    """加载和准备数据集"""
    global tokenizer
    
    # 定义数据路径
    data_paths = {
        "general": "../data/magpie_general_mol1.jsonl",  # 通用数据
        "domain": "../data/news_3year1.jsonl",           # 领域数据
    }

    # 加载并统计数据
    general_data, general_tokens, _ = count_tokens_and_size(data_paths["general"], tokenizer, "general")
    domain_data, domain_tokens, _ = count_tokens_and_size(data_paths["domain"], tokenizer, "domain")

    # 保证两种数据的token数平衡，避免训练偏向某一种数据
    min_tokens = min(general_tokens, domain_tokens)

    def truncate_to_tokens(data, target_tokens, mode):
        """
        截断数据到指定token数量
        
        Args:
            data: 数据列表
            target_tokens: 目标token数
            mode: 数据模式 ('domain' 或 'general')
        
        Returns:
            kept: 保留的数据
            total: 实际token数
        """
        global tokenizer
        total, kept = 0, []
        for s in data:
            # 根据模式提取文本
            t = s.get("text", "").strip() if mode == "domain" else s.get("instruction", "").strip() + s.get("output", "").strip()
            n = len(tokenizer(t, add_special_tokens=False).input_ids)
            if total + n > target_tokens:
                break
            total += n
            kept.append(s)
        return kept, total

    # 截断数据到平衡状态
    general_data, _ = truncate_to_tokens(general_data, min_tokens, "general")
    domain_data, _ = truncate_to_tokens(domain_data, min_tokens, "domain")

    logger.info(f"✅ 加载数据完成：通用样本 {len(general_data)} 条，领域样本 {len(domain_data)} 条")

    # 训练参数
    batch_size = 1  # 小批次大小，适合大模型训练
    num_train_epochs = 3

    # 创建数据集
    general_dataset = MoLDatasetSingleType(general_data, tokenizer, type='general')
    domain_dataset = MoLDatasetSingleType(domain_data, tokenizer, type='domain')

    # 创建数据加载器
    general_loader = DataLoader(
        general_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mol_collate_fn,
        drop_last=False,
        num_workers=0,  # 减少worker数量
        pin_memory=False,  # 禁用pin_memory
    )
    domain_loader = DataLoader(
        domain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mol_collate_fn,
        drop_last=False,
        num_workers=0,  # 减少worker数量
        pin_memory=False,  # 禁用pin_memory
    )
    
    return general_loader, domain_loader, num_train_epochs

# ============================================================================
# 8. 优化器和学习率调度器
# ============================================================================
def setup_optimizer_and_scheduler(student_model, num_training_steps):
    """设置优化器和学习率调度器"""
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)

    # 使用余弦学习率调度器
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,  # 无预热步数
        num_training_steps=num_training_steps,
    )
    
    return optimizer, lr_scheduler

# ============================================================================
# 9. DeepSpeed准备
# ============================================================================
def prepare_for_training(student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler):
    """准备训练环境"""
    # 将教师模型移到GPU
    teacher_model.to(accelerator.device)

    # 使用Accelerator准备模型、优化器、数据加载器等
    student_model, optimizer, general_loader, domain_loader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, general_loader, domain_loader, lr_scheduler
    )
    
    return student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler

# ============================================================================
# 10. 训练循环核心函数
# ============================================================================
def train_one_batch(batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, accelerator, step_losses, general_kl_log, domain_ce_log, loader_name):
    """
    训练单个批次的函数
    
    Args:
        batch: 批次数据
        is_domain: 是否为领域数据的标记
        step_count: 当前步数
        alpha: 损失权重参数
        student_model: 学生模型
        teacher_model: 教师模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        accelerator: Accelerator实例
        step_losses: 损失记录列表
        general_kl_log: 通用数据KL损失记录
        domain_ce_log: 领域数据CE损失记录
        loader_name: 数据加载器名称
    """
    if len(is_domain) == 0:
        if accelerator.is_main_process:
            tqdm.write(f"[{loader_name}] 跳过空batch")
        return
    
    # 初始化变量，避免UnboundLocalError
    outputs = None
    student_logits = None
    teacher_outputs = None
    teacher_logits = None
    
    try:
        # 使用梯度累积上下文
        with accelerator.accumulate(student_model):
            # 学生模型前向传播
            outputs = student_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=False,
                output_attentions=False,
            )
            student_logits = outputs.logits
            
            # 教师模型前向传播（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=batch["input_ids"].to(accelerator.device),
                    attention_mask=batch["attention_mask"].to(accelerator.device),
                    output_hidden_states=False,
                    output_attentions=False,
                )
                teacher_logits = teacher_outputs.logits
            
            # 计算损失
            losses = []
            ce_losses = []
            kl_losses = []
            
            # 对批次中的每个样本分别计算损失
            for i in range(len(is_domain)):
                # 创建有效token的掩码（非padding且非-100标签）
                mask = (batch["attention_mask"][i] == 1) & (batch["labels"][i] != -100)
                valid_indices = torch.where(mask)[0]
                
                if len(valid_indices) == 0:
                    continue
                
                # 提取有效位置的logits
                s_logits = student_logits[i, valid_indices]
                t_logits = teacher_logits[i, valid_indices]
                
                # 计算交叉熵损失（监督学习）
                ce_loss = torch.nn.functional.cross_entropy(
                    s_logits,
                    batch["labels"][i, valid_indices],
                    reduction='mean'
                )
                
                # 计算KL散度损失（知识蒸馏）
                student_log_probs = torch.nn.functional.log_softmax(s_logits, dim=-1)
                teacher_probs = torch.nn.functional.softmax(t_logits, dim=-1)
                kl_per_token = torch.nn.functional.kl_div(
                    input=student_log_probs,
                    target=teacher_probs,
                    reduction="none",
                    log_target=False
                ).sum(dim=-1)
                kl_loss = torch.mean(kl_per_token)
                
                # 根据数据类型调整损失权重
                if loader_name == "General":
                    # 通用数据：主要使用KL损失，少量CE损失
                    loss = alpha * ce_loss + (1 - alpha) * kl_loss
                    general_kl_log.append((step_count, kl_loss.item()))
                else:
                    # 领域数据：主要使用CE损失，少量KL损失
                    loss = (1 - alpha) * ce_loss + alpha * kl_loss
                    domain_ce_log.append((step_count, ce_loss.item()))
                
                ce_losses.append(ce_loss)
                kl_losses.append(kl_loss)
                losses.append(loss)
            
            # 检查是否有有效损失
            if not losses:
                if accelerator.is_main_process:
                    tqdm.write(f"[{loader_name}] 跳过无有效token的batch")
                return
            
            # 计算平均损失
            loss = torch.stack(losses).mean()
            ce_loss_mean = torch.stack(ce_losses).mean().item()
            kl_loss_mean = torch.stack(kl_losses).mean().item()
            step_losses.append((step_count, loss.item(), ce_loss_mean, kl_loss_mean))
            
            # 反向传播
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            # 打印训练信息
            if accelerator.is_main_process:
                tqdm.write(f"[{loader_name}] step={step_count+1}, loss={loss.item():.6f}, ce={ce_loss_mean:.6f}, kl={kl_loss_mean:.6f}")
    
    except RuntimeError as e:
        # 处理显存不足错误
        if "out of memory" in str(e):
            logger.warning(f"⚠️ 显存不足，跳过当前batch: {e}")
            optimizer.zero_grad(set_to_none=True)
            # 强制清理显存
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise e
    finally:
        # 清理内存
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
# 11. 主训练循环
# ============================================================================
def main():
    """主训练函数"""
    # 验证路径
    if not validate_paths():
        return
    
    try:
        # 加载模型
        student_model, teacher_model = load_models_safely()
        
        # 配置LoRA
        student_model = setup_lora(student_model)
        
        # 加载数据集
        general_loader, domain_loader, num_train_epochs = load_datasets()
        
        # 设置优化器和调度器
        num_training_steps = num_train_epochs * (len(general_loader) + len(domain_loader))
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(student_model, num_training_steps)
        
        # 准备训练环境
        student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler = prepare_for_training(
            student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler
        )
        
        # 开始训练
        alpha = 0  # alpha参数：控制损失权重，0表示主要使用KL损失
        student_model.train()

        # 初始化记录变量
        step_losses = []  # 记录每一步的(step, total_loss, ce_loss, kl_loss)
        step_count = 0
        general_kl_log = []  # 记录通用样本的(step, kl_loss)
        domain_ce_log = []   # 记录领域样本的(step, ce_loss)

        start_time = time.time()  # 记录训练开始时间

        # 开始训练循环
        for epoch in range(num_train_epochs):
            logger.info(f"开始第 {epoch+1} 个epoch")
            
            # 创建数据迭代器
            general_iter = iter(general_loader)
            domain_iter = iter(domain_loader)
            finished = [False, False]  # 标记两个数据加载器是否完成
            loader_names = ["General", "Domain"]
            loader_iters = [general_iter, domain_iter]
            loader_idx = 0  # 当前使用的数据加载器索引
            
            # 交替使用两个数据加载器
            while not all(finished):
                try:
                    batch = next(loader_iters[loader_idx])
                    if batch is None:
                        finished[loader_idx] = True
                        loader_idx = 1 - loader_idx  # 切换到另一个加载器
                        continue
                except StopIteration:
                    finished[loader_idx] = True
                    loader_idx = 1 - loader_idx
                    continue
                
                # 提取批次类型标记
                is_domain = batch.pop("is_domain_batch")
                
                # 训练当前批次
                train_one_batch(
                    batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, accelerator,
                    step_losses, general_kl_log, domain_ce_log, loader_names[loader_idx]
                )
                
                step_count += 1
                loader_idx = 1 - loader_idx  # 切换到另一个数据加载器

                # 每步强制清理一次显存
                torch.cuda.empty_cache()
                gc.collect()
            
            # 每个epoch结束后清理内存
            get_accelerator().empty_cache()
            gc.collect()
            
            # 每个epoch保存一次模型（保存LoRA权重）
            save_path = f"../output/mol_pretrain_epoch_{epoch+1}"
            accelerator.wait_for_everyone()  # 等待所有进程
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(student_model)
                unwrapped_model.save_pretrained(save_path, save_dtype=torch.float16)
                tokenizer.save_pretrained(save_path)
                logger.info(f"✅ 模型已保存到 {save_path}")
            
            # 保存后清理内存
            get_accelerator().empty_cache()
            gc.collect()

        # ============================================================================
        # 12. 训练结束保存最终模型
        # ============================================================================
        save_path = "../output/final"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.save_pretrained(save_path, save_dtype=torch.float16)
            tokenizer.save_pretrained(save_path)
            logger.info(f"✅ 最终模型已保存到 {save_path}")

        # 最终清理
        get_accelerator().empty_cache()
        gc.collect()

        # ============================================================================
        # 13. 保存训练日志（仅主进程）
        # ============================================================================
        if accelerator.is_main_process:
            # 保存总体损失日志
            with open("loss_log.txt", "w") as f:
                f.write("step,total_loss,ce_loss,kl_loss\n")
                for s, total, ce, kl in step_losses:
                    f.write(f"{s},{total:.6f},{ce:.6f},{kl:.6f}\n")

            # 保存通用数据KL损失日志
            with open("general_kl_log.txt", "w") as f:
                f.write("step,kl_loss\n")
                for s, kl in general_kl_log:
                    f.write(f"{s},{kl:.6f}\n")
            if general_kl_log:
                avg_kl = sum([kl for _, kl in general_kl_log]) / len(general_kl_log)
                logger.info(f"通用样本KL散度均值: {avg_kl:.6f}")
            else:
                logger.info("无通用样本KL散度记录。")

            # 保存领域数据CE损失日志
            with open("domain_ce_log.txt", "w") as f:
                f.write("step,ce_loss\n")
                for s, ce in domain_ce_log:
                    f.write(f"{s},{ce:.6f}\n")
            if domain_ce_log:
                avg_ce = sum([ce for _, ce in domain_ce_log]) / len(domain_ce_log)
                logger.info(f"领域样本交叉熵损失均值: {avg_ce:.6f}")
            else:
                logger.info("无领域样本交叉熵损失记录。")

        logger.info(f"训练结束，总耗时: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 