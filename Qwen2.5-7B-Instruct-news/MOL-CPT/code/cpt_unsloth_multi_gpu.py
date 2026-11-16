from datetime import timedelta
import json
import torch
import unsloth
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader
from unsloth import FastLanguageModel
from peft import LoraConfig, TaskType
from tqdm import tqdm
import time
import gc
import logging
import shutil
import signal
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# 设置环境变量以优化显存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# 0. 设置日志和路径验证
# ============================================================================
# 设置日志
def setup_logger():
    """设置分布式日志"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 主进程记录到文件和屏幕
    if dist.get_rank() == 0:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件处理器
        file_handler = logging.FileHandler('training.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

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
        print(f"以下路径不存在: {missing_paths}")
        print("请检查路径是否正确，或创建相应的目录/文件")
        return False
    
    # 创建输出目录
    os.makedirs(paths_to_check["output_dir"], exist_ok=True)
    if dist.get_rank() == 0:
        print("所有路径验证通过")
    return True

def cleanup_old_models(output_dir, keep_count=5):
    """清理旧的模型文件，只保留最新的几个"""
    try:
        # 只在主进程执行清理
        if dist.get_rank() != 0:
            return
            
        if not os.path.exists(output_dir):
            return
        
        model_dirs = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.startswith('mol_pretrain_step_'):
                model_dirs.append(item_path)
        
        if len(model_dirs) <= keep_count:
            return
        
        # 按修改时间排序
        model_dirs.sort(key=os.path.getmtime)
        
        # 删除最旧的模型文件
        for old_model in model_dirs[:-keep_count]:
            try:
                shutil.rmtree(old_model)
                logger.info(f"已删除旧模型: {old_model}")
            except Exception as e:
                logger.error(f"删除模型失败: {old_model}, 错误: {e}")
                
    except Exception as e:
        logger.error(f"清理模型失败: {e}")

# ============================================================================
# 1. 使用unsloth加载模型
# ============================================================================
MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"

# 全局变量，用于在函数间共享
tokenizer = None
EOS_TOKEN = None

def load_models_safely():
    """使用unsloth安全加载模型，包含错误处理"""
    global tokenizer, EOS_TOKEN
    
    try:
        if dist.get_rank() == 0:
            print("使用unsloth加载学生模型...")
        # 使用unsloth加载学生模型，支持4bit量化和LoRA
        student_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=4096,
            dtype=None,  # 自动选择最佳数据类型
            load_in_4bit=True,  # 使用4bit量化
            token=os.environ.get("HF_TOKEN", None),  # 如果需要的话
        )
        if dist.get_rank() == 0:
            print("学生模型加载成功")

        if dist.get_rank() == 0:
            print("使用unsloth加载教师模型...")
        # 加载教师模型（用于知识蒸馏）
        teacher_model, _ = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
            token=os.environ.get("HF_TOKEN", None),
        )
        if dist.get_rank() == 0:
            print("教师模型加载成功")

        # 冻结教师模型参数，不进行梯度计算
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False  # 禁用缓存以节省显存
        teacher_model.eval()  # 设置为评估模式

        # 设置tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"  # 获取结束标记
        
        return student_model, teacher_model
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

# ============================================================================
# 2. 配置LoRA参数高效微调 - MOL方法核心
# ============================================================================
def setup_lora(student_model):
    """使用unsloth配置LoRA参数"""
    try:
        # 使用unsloth的LoRA配置
        student_model = FastLanguageModel.get_peft_model(
            student_model,
            r=64,  # LoRA秩，控制可训练参数数量
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj",      # MLP层
            ],
            lora_alpha=32,  # LoRA缩放参数
            lora_dropout=0.05,  # LoRA dropout率
            bias="none",  # 不训练偏置项
            use_gradient_checkpointing=True,  # 启用梯度检查点以节省显存
            random_state=42,  # 随机种子
            max_seq_length=4096,  # 最大序列长度
        )
        
        if dist.get_rank() == 0:
            print("LoRA配置成功")
        return student_model
        
    except Exception as e:
        print(f"LoRA配置失败: {e}")
        raise

# ============================================================================
# 3. 数据统计和加载函数
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
        
        if dist.get_rank() == 0:
            logger.info(f"✅ {name} 加载完毕：{count} 条样本，{total_tokens} tokens，{round(total_tokens/1024/1024, 2)} MB")
        return items, total_tokens, total_bytes
        
    except FileNotFoundError:
        logger.error(f"数据文件不存在: {path}")
        raise
    except Exception as e:
        logger.error(f"加载{name}数据失败: {e}")
        raise

# ============================================================================
# 4. 自定义数据集类
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
# 5. 自定义数据整理函数
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
    # 过滤掉有效token为0的样本
    valid_batch = [
        b for b in batch 
        if (b["labels"] != -100).sum().item() > 0
    ]
    if not valid_batch:
        # 如果所有样本都为空，返回None
        return None
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
        "is_domain_batch": is_domain_batch.to(device),
    }

# ============================================================================
# 6. 加载数据集和数据加载器
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

    # 确保样本数是4的倍数，保证4卡分布式训练时数据分布均匀
    def adjust_to_multiple_of_4(data_list, data_name):
        """调整数据样本数为4的倍数"""
        original_count = len(data_list)
        # 计算需要保留的样本数（向下取整到最近的4的倍数）
        target_count = (original_count // 4) * 4
        if target_count == 0:
            target_count = 4  # 至少保留4个样本
        adjusted_data = data_list[:target_count]
        if dist.get_rank() == 0:
            print(f"✅ {data_name} 样本数调整: {original_count} → {len(adjusted_data)} (4的倍数)")
        return adjusted_data

    general_data = adjust_to_multiple_of_4(general_data, "通用数据")
    domain_data = adjust_to_multiple_of_4(domain_data, "领域数据")
    if dist.get_rank() == 0:
        print(f"✅ 加载数据完成：通用样本 {len(general_data)} 条，领域样本 {len(domain_data)} 条,总token数为:{round((2*min_tokens)/1024/1024, 2)} MB")

    # 训练参数
    batch_size = 1  # 每个GPU上的批次大小
    num_train_epochs = 3

    # 创建数据集
    general_dataset = MoLDatasetSingleType(general_data, tokenizer, type='general')
    domain_dataset = MoLDatasetSingleType(domain_data, tokenizer, type='domain')
    
    # 创建分布式采样器
    general_sampler = DistributedSampler(general_dataset, shuffle=True)
    domain_sampler = DistributedSampler(domain_dataset, shuffle=True)
    
    # 创建数据加载器
    general_loader = DataLoader(
        general_dataset,
        batch_size=batch_size,
        sampler=general_sampler,
        collate_fn=mol_collate_fn,
        drop_last=True,
        num_workers=0,  # 减少worker数量
        pin_memory=False,  # 禁用pin_memory
    )
    domain_loader = DataLoader(
        domain_dataset,
        batch_size=batch_size,
        sampler=domain_sampler,
        collate_fn=mol_collate_fn,
        drop_last=True,
        num_workers=0,  # 减少worker数量
        pin_memory=False,  # 禁用pin_memory
    )
    
    return general_loader, domain_loader, num_train_epochs, general_sampler, domain_sampler

# ============================================================================
# 7. 优化器和学习率调度器
# ============================================================================
def setup_optimizer_and_scheduler(student_model, num_training_steps):
    """设置优化器和学习率调度器"""
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)

    # 使用余弦学习率调度器
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,  # 无预热步数
        num_training_steps=num_training_steps,
    )
    
    return optimizer, lr_scheduler

# ============================================================================
# 8. 训练单个批次的函数
# ============================================================================
def train_one_batch(batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, step_losses, general_kl_log, domain_ce_log, loader_name):
   
    outputs = None
    student_logits = None
    teacher_outputs = None
    teacher_logits = None
    
    try:
        # 尝试计算损失
        losses = []
        ce_losses = []
        kl_losses = []
        
        try:
            if len(is_domain) == 0:
                if dist.get_rank() == 0:
                    print(f"[{loader_name}] 跳过空batch")
                return
            else:
                # 计算有效token掩码
                mask = (batch["attention_mask"] == 1) & (batch["labels"] != -100)
                valid_indices = mask.nonzero(as_tuple=True)

                # 学生模型前向传播
                outputs = student_model(**batch)
                student_logits = outputs.logits
                
                if loader_name == "Domain":
                    # 交叉熵损失
                    ce_loss = outputs.loss
                    kl_loss = torch.tensor(0, dtype=ce_loss.dtype, device=ce_loss.device)
                else:
                    # 教师模型前向传播
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**batch)
                    teacher_logits = teacher_outputs.logits

                    # 提取有效位置的logits
                    s_logits = student_logits[valid_indices]
                    t_logits = teacher_logits[valid_indices]
                    
                    student_log_probs = torch.nn.functional.log_softmax(s_logits, dim=-1)
                    teacher_probs = torch.nn.functional.softmax(t_logits, dim=-1)
                    kl_loss = torch.nn.functional.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction="batchmean", 
                        log_target=False
                    )
                    ce_loss = torch.tensor(0, dtype=kl_loss.dtype, device=kl_loss.device)   

        except RuntimeError as e:
            # 捕获显存不足错误
            if "out of memory" in str(e):
                if dist.get_rank() == 0:
                    print(f"step={step_count} ⚠️ 前向传播显存不足，跳过此batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                # 其他运行时错误
                if dist.get_rank() == 0:
                    print(f"step={step_count} ⚠️ 运行时错误: {str(e)}")
                return

        try:
            if torch.isnan(ce_loss) or torch.isinf(ce_loss) or \
               torch.isnan(kl_loss) or torch.isinf(kl_loss):
                if dist.get_rank() == 0:
                    print(f"检测到异常损失值，跳过步骤 {step_count}")
                return
            else:    
                # 根据数据源计算最终损失
                if loader_name == "General":
                    loss = alpha * ce_loss + (1 - alpha) * kl_loss
                    # 只在主进程记录日志
                    if dist.get_rank() == 0:
                        general_kl_log.append((step_count, kl_loss.item()))
                else:
                    loss = (1 - alpha) * ce_loss + alpha * kl_loss
                    # 只在主进程记录日志
                    if dist.get_rank() == 0:
                        domain_ce_log.append((step_count, ce_loss.item()))
                        
                ce_losses.append(ce_loss)
                kl_losses.append(kl_loss)
                losses.append(loss)
                if dist.get_rank() == 0:
                    step_losses.append((step_count, loss, ce_loss, kl_loss))

                # 无错误时正常执行优化步骤
                optimizer.zero_grad()
                loss.backward()

        except RuntimeError as e:
            # 捕获显存不足错误
            if "out of memory" in str(e):
                if dist.get_rank() == 0:
                    print(f"step={step_count} ⚠️ 反向传播显存不足，跳过此batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                # 其他运行时错误
                if dist.get_rank() == 0:
                    print(f"step={step_count} ⚠️ 反向传播运行时错误: {str(e)}")
                return
        
        # 优化器更新
        optimizer.step()
        # 学习率更新
        lr_scheduler.step()
        # 打印训练信息（只在主进程打印）
        if dist.get_rank() == 0:
            print(f"[{loader_name}] step={step_count+1}, loss={loss.item():.6f}, ce={ce_loss.item():.6f}, kl={kl_loss.item():.6f}")
    
    finally:
        # 清理内存
        for var in [outputs, student_logits, teacher_outputs, teacher_logits, batch]:
            if var is not None:
                del var
        torch.cuda.empty_cache()
        gc.collect()

# ============================================================================
# 9. 主训练循环
# ============================================================================
def main():
    """主训练函数"""
    
    # 初始化分布式训练
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    
    # 设置日志
    global logger
    logger = setup_logger()
    
    # 验证路径
    if not validate_paths():
        return
    
    try:
        # 加载模型
        student_model, teacher_model = load_models_safely()
        
        # 配置LoRA
        student_model = setup_lora(student_model)
        if global_rank == 0:
            print(f"+++++++++++++++++++++++模型加载完毕++++++++++++++++++++++++++")
        
        # 将学生模型包装为DDP模型
        student_model = DDP(
            student_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        
        # 加载数据集
        general_loader, domain_loader, num_train_epochs, general_sampler, domain_sampler = load_datasets()
        
        # 设置优化器和调度器
        num_training_steps = num_train_epochs * (len(general_loader) + len(domain_loader))
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(student_model, num_training_steps)
        
        # 开始训练
        alpha = 0  # alpha参数：控制损失权重，0表示主要使用KL损失
        student_model.train()
        
        # 初始化记录变量（只在主进程记录）
        step_losses = [] if global_rank == 0 else None
        step_count = 0
        general_kl_log = [] if global_rank == 0 else None
        domain_ce_log = [] if global_rank == 0 else None

        start_time = time.time()  # 记录训练开始时间

        # 开始训练循环
        for epoch in range(num_train_epochs):
            if global_rank == 0:
                print(f"开始第 {epoch+1} 个epoch")
            
            # 设置分布式采样器的epoch
            general_sampler.set_epoch(epoch)
            domain_sampler.set_epoch(epoch)
            
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
                    # 尝试获取下一个批次
                    try:
                        batch = next(loader_iters[loader_idx])
                        if batch is None:
                            if global_rank == 0:
                                print(f"step={step_count} ⚠️ batch为空，重新取数据")
                            continue
                    except StopIteration:
                        finished[loader_idx] = True
                        loader_idx = 1 - loader_idx
                        continue
                    except Exception as e:
                        if global_rank == 0:
                            print(f"step={step_count} ⚠️ 获取batch时出现错误: {e}")
                        # 跳过这个batch，继续下一个
                        continue
                    
                    # 提取批次类型标记
                    is_domain = batch.pop("is_domain_batch")
                    
                    # 训练当前批次 - 添加显存错误处理
                    try:
                        train_one_batch(
                            batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler,
                            step_losses, general_kl_log, domain_ce_log, loader_names[loader_idx]
                        )
                    except Exception as e:
                        if global_rank == 0:
                            print(f"⚠️ 训练batch时出现错误: {e}")
                        # 跳过这个batch，继续下一个
                        step_count += 1
                        loader_idx = 1 - loader_idx
                        continue
                    # 只有在没有错误的情况下才增加步数
                    step_count += 1
                    loader_idx = 1 - loader_idx  # 切换到另一个数据加载器

                    # 每1000步保存一次模型和更新损失文件（只在主进程执行）
                    if global_rank == 0 and step_count % 1000 == 0:
                        try:
                            # 保存模型
                            save_path = f"../output/mol_pretrain_step_{step_count}"
                            # 使用unsloth的保存方法
                            student_model.module.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                            print(f"✅ 模型已保存到 {save_path}")
                                
                            # 清理旧的模型文件，只保留最新的5个
                            cleanup_old_models("../output", keep_count=5)
                            
                            # 更新损失文件
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

                            # 保存领域数据CE损失日志
                            with open("domain_ce_log.txt", "w") as f:
                                f.write("step,ce_loss\n")
                                for s, ce in domain_ce_log:
                                    f.write(f"{s},{ce:.6f}\n")
                            
                            print(f"✅ 损失文件已更新，当前步数: {step_count}")
                        except Exception as e:
                            print(f"⚠️ 保存模型时出现错误: {e}")
                            # 继续训练，不中断

                    # 强制清理显存
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    if global_rank == 0:
                        print(f"⚠️ 训练循环外层错误: {e}")
                    # 尝试恢复训练
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 添加短暂延迟，避免资源争用
                time.sleep(0.1)
            
            # 每个epoch结束后清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 每个epoch保存一次模型（保存LoRA权重）（只在主进程执行）
            if global_rank == 0:
                save_path = f"../output/mol_pretrain_epoch_{epoch+1}"
                try:
                    student_model.module.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"✅ 模型已保存到 {save_path}")
                except Exception as e:
                    print(f"⚠️ 保存最终模型时出错: {e}")
            
            # 保存后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # ============================================================================
        # 10. 训练结束保存最终模型（只在主进程执行）
        # ============================================================================
        if global_rank == 0:
            save_path = "../output/final"
            try:
                student_model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"✅ 模型已保存到 {save_path}")
            except Exception as e:
                print(f"⚠️ 保存最终模型时出错: {e}")

        # 最终清理
        torch.cuda.empty_cache()
        gc.collect()

        # ============================================================================
        # 11. 保存训练日志（只在主进程执行）
        # ============================================================================
        if global_rank == 0:
            try:
                # 最终更新损失文件
                with open("loss_log.txt", "w") as f:
                    f.write("step,total_loss,ce_loss,kl_loss\n")
                    for s, total, ce, kl in step_losses:
                        f.write(f"{s},{total:.6f},{ce:.6f},{kl:.6f}\n")

                with open("general_kl_log.txt", "w") as f:
                    f.write("step,kl_loss\n")
                    for s, kl in general_kl_log:
                        f.write(f"{s},{kl:.6f}\n")

                with open("domain_ce_log.txt", "w") as f:
                    f.write("step,ce_loss\n")
                    for s, ce in domain_ce_log:
                        f.write(f"{s},{ce:.6f}\n")
                
                # 打印统计信息
                if general_kl_log:
                    avg_kl = sum([kl for _, kl in general_kl_log]) / len(general_kl_log)
                    print(f"通用样本KL散度均值: {avg_kl:.6f}")
                else:
                    print("无通用样本KL散度记录。")

                if domain_ce_log:
                    avg_ce = sum([ce for _, ce in domain_ce_log]) / len(domain_ce_log)
                    print(f"领域样本交叉熵损失均值: {avg_ce:.6f}")
                else:
                    print("无领域样本交叉熵损失记录。")
            except Exception as e:
                print(f"⚠️ 保存日志时出错: {e}")

            print(f"训练结束，总耗时: {time.time() - start_time:.2f}秒")
        # 训练完成后释放资源
        dist.destroy_process_group()

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        dist.destroy_process_group()
        raise

if __name__ == "__main__":
    main()