from datetime import timedelta
import json
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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
import shutil
import signal
import sys

torch.set_num_threads(1)
# import torch.distributed as dist
# dist.init_process_group("gloo")

# 设置环境变量以优化显存分配和避免NCCL错误
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = '1200'  # 30分钟超时
os.environ['NCCL_IB_DISABLE'] = '1'  
os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '67108864'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "1"

import torch.distributed as dist
rank = dist.get_rank() if dist.is_initialized() else 0

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

def validate_paths(accelerator):
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
        accelerator.print(f"以下路径不存在: {missing_paths}")
        accelerator.print("请检查路径是否正确，或创建相应的目录/文件")
        return False
    
    # 创建输出目录
    os.makedirs(paths_to_check["output_dir"], exist_ok=True)
    accelerator.print("所有路径验证通过")
    return True

def cleanup_old_models(output_dir, keep_count=5):
    """清理旧的模型文件，只保留最新的几个"""
    try:
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
# 1. 初始化 Accelerator 使用 DeepSpeed Zero2
# ============================================================================
# 使用DeepSpeed Zero2进行分布式训练，支持大模型训练
# Zero2可以优化显存使用，支持混合精度训练
from accelerate import DistributedDataParallelKwargs,InitProcessGroupKwargs
init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1200))
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True,
                                            broadcast_buffers=False,  # 禁用广播缓冲区以节省显存
                                            bucket_cap_mb=25,  # 设置DDP桶大小，避免显存溢出
                                            static_graph=True)  # 启用静态图优化
accelerator = Accelerator(
    kwargs_handlers=[ddp_kwargs, init_kwargs],
)

# 打印当前进程和设备信息
accelerator.print(f"[Rank {accelerator.process_index}] Hello from {torch.cuda.get_device_name()} (device={torch.cuda.current_device()})")

# ============================================================================
# 2. 加载模型和tokenizer
# ============================================================================
MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"

# 配置4bit量化参数，大幅减少显存使用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用4bit量化
    bnb_4bit_use_double_quant=True,  # 使用双重量化进一步压缩
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用bf16
    bnb_4bit_quant_type="nf4",  # 使用nf4量化类型
)

# 全局变量，用于在函数间共享
tokenizer = None
EOS_TOKEN = None

def load_models_safely(accelerator):
    """安全加载模型，包含错误处理"""
    global tokenizer, EOS_TOKEN
    
    try:
        accelerator.print("加载学生模型...")
        # 加载学生模型（需要训练的模型）
        student_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,  # 应用4bit量化
            trust_remote_code=True,  # 信任远程代码，用于加载自定义模型
            # torch_dtype=torch.bfloat16,  # 明确指定数据类型
        )
        accelerator.print("学生模型加载成功")

        accelerator.print("加载教师模型...")
        # 加载教师模型（用于知识蒸馏）
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16,
        )
        accelerator.print("教师模型加载成功")

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
        accelerator.print(f"模型加载失败: {e}")
        raise

# ============================================================================
# 3. 配置LoRA参数高效微调
# ============================================================================
def setup_lora(student_model, accelerator):
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
        student_model.enable_input_require_grads()
        
        # 设置更激进的显存优化
        for module in student_model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

        accelerator.print("LoRA配置成功")
        return student_model
        
    except Exception as e:
        accelerator.print(f"LoRA配置失败: {e}")
        raise

# ============================================================================
# 4. 数据统计和加载函数
# ============================================================================
def count_tokens_and_size(path, tokenizer, name="", accelerator=None):
    """
    统计数据集中的token数量和文件大小
    
    Args:
        path: 数据文件路径
        tokenizer: 分词器
        name: 数据集名称，用于日志输出
        accelerator: Accelerator实例，用于打印日志
    
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
                    if accelerator:
                        accelerator.print(f"{name} 数据第 {line_num} 条无法解析：{e}")
                    else:
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
        
        if accelerator:
            accelerator.print(f"✅ {name} 加载完毕：{count} 条样本，{total_tokens} tokens，{round(total_tokens/1024/1024, 2)} MB")
        else:
            logger.info(f"✅ {name} 加载完毕：{count} 条样本，{total_tokens} tokens，{round(total_tokens/1024/1024, 2)} MB")
        return items, total_tokens, total_bytes
        
    except FileNotFoundError:
        if accelerator:
            accelerator.print(f"数据文件不存在: {path}")
        else:
            logger.error(f"数据文件不存在: {path}")
        raise
    except Exception as e:
        if accelerator:
            accelerator.print(f"加载{name}数据失败: {e}")
        else:
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
            # 这里无法访问accelerator，所以使用logger
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
    try:
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
    except Exception as e:
        logger.error(f"Collate 错误: {e}")
        # 返回 None，train loop 中检测并跳过
        return None

# ============================================================================
# 7. 加载数据集和数据加载器
# ============================================================================
def load_datasets(accelerator):
    """加载和准备数据集"""
    global tokenizer
    
    # 定义数据路径
    data_paths = {
        "general": "../data/magpie_general_mol1.jsonl",  # 通用数据
        "domain": "../data/news_3year1.jsonl",           # 领域数据
    }

    # 加载并统计数据
    general_data, general_tokens, _ = count_tokens_and_size(data_paths["general"], tokenizer, "general", accelerator)
    domain_data, domain_tokens, _ = count_tokens_and_size(data_paths["domain"], tokenizer, "domain", accelerator)

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
        accelerator.print(f"✅ {data_name} 样本数调整: {original_count} → {len(adjusted_data)} (4的倍数)")
        return adjusted_data

    general_data = adjust_to_multiple_of_4(general_data, "通用数据")
    domain_data = adjust_to_multiple_of_4(domain_data, "领域数据")
    accelerator.print(f"✅ 加载数据完成：通用样本 {len(general_data)} 条，领域样本 {len(domain_data)} 条,总token数为:{round((2*min_tokens)/1024/1024, 2)} MB")

    # 训练参数
    batch_size = 8  # 小批次大小，适合大模型训练
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
        drop_last=True,
        num_workers=16,  # 增加到CPU核心数的50%
        pin_memory=True,  # 启用锁页内存
        timeout=120,
        persistent_workers=True
    )
    domain_loader = DataLoader(
        domain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mol_collate_fn,
        drop_last=True,
        num_workers=16,  # 增加到CPU核心数的50%
        pin_memory=True,  # 启用锁页内存
        timeout=120,
        persistent_workers=True
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
# 10. 数据预览和类型判断函数
# ============================================================================
def get_data_preview_and_type(batch, accelerator):
    """
    获取数据预览和判断数据类型
    
    Args:
        batch: 批次数据
        accelerator: Accelerator实例
        step_count: 当前步数
    
    Returns:
        data_preview_list: 每张GPU的数据预览列表
        data_type: 数据类型判断结果
    """
    # 获取当前数据的真实内容（前6个字符）
    try:
        # 将input_ids转换回文本
        input_ids = batch["input_ids"][0]  # 取第一个样本
        # 解码前6个token
        decoded_text = tokenizer.decode(input_ids[:6], skip_special_tokens=True)
        # 截取前6个字符
        preview_text = decoded_text[:6].replace('\n', ' ').replace('\r', ' ')
        if len(decoded_text) > 6:
            preview_text += "..."
    except Exception as e:
        preview_text = f"解码失败: {e}"

    if torch.distributed.is_initialized():
        try:
            # 收集每个进程的数据预览
            gathered_previews = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(gathered_previews, preview_text)

            data_preview_list = [f"GPU {i}: {gathered_previews[i]}" for i in range(accelerator.num_processes)]
            
            # 判断数据类型
            all_same = all(preview == gathered_previews[0] for preview in gathered_previews)
            if all_same:
                first_preview = gathered_previews[0]
                # 去掉省略号，只取前6个字符进行判断
                clean_preview = first_preview.replace('...', '')[:6]
                if clean_preview == '新闻发表时间':
                    data_type = "领域数据"
                elif clean_preview == 'system':
                    data_type = "通用数据"
                else:
                    data_type = "数据混乱"
            else:
                data_type = "数据混乱"
                
        except Exception as e:
            accelerator.print(f"⚠️ 收集数据预览失败: {e}")
            data_preview_list = [f"GPU {accelerator.process_index}: {preview_text}"]
            data_type = "收集失败"
    else:
        data_preview_list = [f"GPU {accelerator.process_index}: {preview_text}"]
        data_type = "单卡模式"
    
    return data_type
# def get_data_preview_and_type(batch, accelerator):
#     try:
#         input_ids = batch["input_ids"][0]
#         decoded_text = tokenizer.decode(input_ids[:6], skip_special_tokens=True)
#         preview_text = decoded_text[:6].replace('\n', ' ').replace('\r', ' ')
#         if len(decoded_text) > 6:
#             preview_text += "..."
#     except Exception as e:
#         preview_text = f"解码失败: {e}"

#     # 仅做轻量同步检查：batch size 一致性
#     consistent, local_bs, all_bs = safe_check_batch_consistency(batch, accelerator)
#     if not consistent:
#         return "数据混乱"

#     # 简化：只基于本地 preview 判断（避免跨进程对象收集）
#     if preview_text.startswith('新闻发表时间'):
#         return "领域数据"
#     if preview_text.startswith('system'):
#         return "通用数据"
#     return "未知/本地预览:" + preview_text

# # ========检测批次一致性=====================================================================
# def safe_check_batch_consistency(batch, accelerator):
#     """
#     检查每个 rank 是否都有 batch，以及各 rank 的 batch size 是否一致。
#     返回 (consistent: bool, local_bs: int, all_bs: list[int])
#     """
#     if not dist.is_initialized():
#         return True, batch["input_ids"].shape[0], [batch["input_ids"].shape[0]]

#     # 本地 batch size
#     local_bs = torch.tensor([batch["input_ids"].shape[0]], device='cuda')
#     world_size = dist.get_world_size()
#     gathered = [torch.zeros_like(local_bs) for _ in range(world_size)]
#     # 使用 all_gather（tensor 版本，稳定）
#     dist.all_gather(gathered, local_bs)
#     all_bs = [int(x.item()) for x in gathered]
#     consistent = all(bs == all_bs[0] for bs in all_bs)
#     return consistent, int(local_bs.item()), all_bs

def reset_distributed_connection(accelerator):
    """重置分布式连接"""
    try:
        accelerator.print("🔄 正在重置分布式连接...")
        if hasattr(torch.distributed, 'destroy_process_group'):
            torch.distributed.destroy_process_group()
        if hasattr(torch.distributed, 'init_process_group'):
            torch.distributed.init_process_group(backend='nccl')
        accelerator.print("✅ 分布式连接重置成功")
        return True
    except Exception as e:
        accelerator.print(f"⚠️ 分布式连接重置失败: {e}")
        return False

def handle_communication_error(error, accelerator, context="", max_retries=3, stats=None):
    """处理通信错误的统一函数，支持重试机制和统计"""
    error_msg = str(error).lower()
    if any(keyword in error_msg for keyword in ["nccl", "timeout", "communication", "distributed"]):
        accelerator.print(f"⚠️ {context}通信错误: {error}")
        
        # 更新统计（如果stats是字典）
        if stats is not None and isinstance(stats, dict):
            stats['communication_errors'] = stats.get('communication_errors', 0) + 1
            stats['skipped_batches'] = stats.get('skipped_batches', 0) + 1
        
        # 清理资源
        torch.cuda.empty_cache()
        gc.collect()
        
        # 尝试重置连接并重试
        for retry in range(max_retries):
            accelerator.print(f"🔄 尝试重置连接 (第{retry+1}次)")
            if reset_distributed_connection(accelerator):
                accelerator.print(f"✅ 连接重置成功，继续训练")
                return True
            else:
                accelerator.print(f"⚠️ 第{retry+1}次重置失败")
                time.sleep(5)  # 等待5秒后重试
        
        accelerator.print(f"❌ 经过{max_retries}次尝试后仍无法重置连接")
        return False
    else:
        accelerator.print(f"⚠️ {context}未知错误: {error}")
        if stats is not None and isinstance(stats, dict):
            stats['skipped_batches'] = stats.get('skipped_batches', 0) + 1
        return False

# ============================================================================
# 11. 训练循环核心函数
# ============================================================================
def train_one_batch(batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, accelerator, step_losses, general_kl_log, domain_ce_log, loader_name,data_type):
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
            accelerator.print(f"[{loader_name}] 跳过空batch")
        return
    
    # 初始化变量，避免UnboundLocalError
    outputs = None
    student_logits = None
    teacher_outputs = None
    teacher_logits = None
    
    try:
        # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: batch数据 input_ids.shape={batch['input_ids'].shape}, labels.shape={batch['labels'].shape}, is_domain={is_domain.tolist() if hasattr(is_domain, 'tolist') else is_domain}, 数据类型={data_type}")
        # 使用梯度累积上下文
        with accelerator.accumulate(student_model):
            # 计算损失
            losses = []
            ce_losses = []
            kl_losses = []
            # rank = accelerator.process_index
            # print(f"[Rank {rank}] BEFORE forward step {step_count+1}", flush=True)
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 学生模型前向传播开始")
            outputs = student_model(**batch)
            student_logits = outputs.logits
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 学生模型前向传播完成，logits.shape={student_logits.shape}")
            # print(f"[Rank {rank}] AFTER forward step {step_count+1}", flush=True)
            torch.cuda.synchronize()

            # 根据数据类型调整损失权重
            #通用数据使用KL损失
            if loader_name == "General":
                # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 教师模型前向传播开始")
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits
                # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 教师模型前向传播完成，logits.shape={teacher_logits.shape}")
    
                # 计算整个batch的有效token掩码
                mask = (batch["attention_mask"] == 1) & (batch["labels"] != -100)
                valid_indices = mask.nonzero(as_tuple=True)
                
                # 检查是否有有效token
                if valid_indices[0].numel() == 0:
                    accelerator.print(f"[{loader_name}] 跳过无有效token的batch")
                    kl_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
                    kl_loss_mean = kl_loss
                else:
                    # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 有效token数量={valid_indices[0].numel()}")
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
                    # 跨进程gather后计算平均loss
                    kl_loss_gathered = accelerator.gather(kl_loss)
                    kl_loss_mean = kl_loss_gathered.mean()
                print(f"[Rank {accelerator.process_index}] AFTER kl_loss step {kl_loss}", flush=True)
                loss = kl_loss_mean
                ce_loss_mean = torch.tensor(0.0, device=kl_loss.device, dtype=kl_loss.dtype)
                general_kl_log.append((step_count, kl_loss_mean.item()))
            else:
                # 领域数据：主要使用CE损失
                # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 开始计算损失")
                # 计算交叉熵损失（监督学习）
                ce_loss=outputs.loss
                print(f"[Rank {accelerator.process_index}] AFTER ce_loss step {ce_loss}", flush=True)
                ce_loss_tensor=torch.tensor(ce_loss.item(),device=ce_loss.device)
                ce_loss_mean=accelerator.gather(ce_loss_tensor).mean()
                kl_loss_mean = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
                # loss = (1 - alpha) * ce_loss + alpha * kl_loss
                loss=ce_loss_mean
                domain_ce_log.append((step_count, ce_loss_mean.item()))
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name},  ce_loss={ce_loss_mean.item():.6f}, kl_loss={kl_loss_mean.item():.6f}")
            ce_losses.append(ce_loss_mean)
            kl_losses.append(kl_loss_mean)
            losses.append(loss)
            
            # 检查是否有有效损失
            if not losses:
                if accelerator.is_main_process:
                    accelerator.print(f"[{loader_name}] 跳过无有效token的batch")
                return
            
            step_losses.append((step_count, loss, ce_loss_mean, kl_loss_mean))
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 损失计算完成，loss={loss.item():.6f}, ce_loss={ce_loss_mean.item():.6f}, kl_loss={kl_loss_mean.item():.6f}")

            optimizer.zero_grad(set_to_none=True)
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: optimizer.zero_grad完成")

            # 反向传播
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 开始反向传播")
            # rank = accelerator.process_index
            # print(f"[Rank {rank}] Before loss.backward() at step {step_count+1}")
            # accelerator.backward(loss)
            torch.cuda.synchronize()  # 确保所有GPU完成反向传播
            # print(f"[Rank {rank}] After loss.backward() at step {step_count+1}")
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 反向传播完成")

            optimizer.step()
            torch.cuda.synchronize()  # 确保所有GPU完成参数更新
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: optimizer.step完成")
            lr_scheduler.step()
            # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: lr_scheduler.step完成")
            
            # 打印训练信息
            if accelerator.is_main_process:
                accelerator.print(f"[{loader_name}] step={step_count+1}, loss={loss.item():.6f}, ce={ce_loss_mean.item():.6f}, kl={kl_loss_mean.item():.6f},数据类型:{data_type}")
    
    except RuntimeError as e:
        # 处理显存不足错误
        if "out of memory" in str(e):
            accelerator.print(f"⚠️ 显存不足，跳过当前batch: {e}")
            optimizer.zero_grad(set_to_none=True)
            # 强制清理显存
            torch.cuda.empty_cache()
            gc.collect()
        elif "NCCL" in str(e) or "nccl" in str(e).lower():
            accelerator.print(f"⚠️ NCCL通信错误，跳过当前batch: {e}")
            optimizer.zero_grad(set_to_none=True)
            handle_communication_error(e, accelerator, "NCCL")  # 移除stats参数
        else:
            accelerator.print(f"⚠️ 运行时错误，跳过当前batch: {e}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        # 处理其他类型的错误，包括NCCL超时等
        optimizer.zero_grad(set_to_none=True)
        handle_communication_error(e, accelerator, "分布式")  # 移除stats参数
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
        # accelerator.print(f"[追踪] step={step_count+1}, loader={loader_name}: 清理内存完成")

# ============================================================================
# 12. 主训练循环
# ============================================================================
def main():
    """主训练函数"""
    
    # 验证路径
    if not validate_paths(accelerator):
        return
    
    try:
        # 加载模型
        student_model, teacher_model = load_models_safely(accelerator)
        
        # 配置LoRA
        student_model = setup_lora(student_model, accelerator)
        
        # 加载数据集
        general_loader, domain_loader, num_train_epochs= load_datasets(accelerator)
        
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
        print("++++++++++++++++++++++++++++++++++++++++++++++++++模型类型为：", type(student_model))
        # 初始化记录变量
        step_losses = []  # 记录每一步的(step, total_loss, ce_loss, kl_loss)
        step_count = 0
        general_kl_log = []  # 记录通用样本的(step, kl_loss)
        domain_ce_log = []   # 记录领域样本的(step, ce_loss)

        start_time = time.time()  # 记录训练开始时间

        # 开始训练循环
        for epoch in range(num_train_epochs):
            accelerator.print(f"开始第 {epoch+1} 个epoch")
            # 创建数据迭代器
            general_iter = iter(general_loader)
            domain_iter = iter(domain_loader)
            finished = [False, False]  # 标记两个数据加载器是否完成
            loader_names = ["General", "Domain"]
            loader_iters = [general_iter, domain_iter]
            loader_idx = 0  # 当前使用的数据加载器索引
            training_stats = {
            'skipped_batches': 0,
            'communication_errors': 0,
            'total_steps': 0
        }
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
                # except Exception as e:
                #     accelerator.print(f"⚠️ 获取batch时出现错误: {e}")
                #     # 跳过这个batch，继续下一个
                #     loader_idx = 1 - loader_idx
                #     continue
                
                # 提取批次类型标记
                is_domain = batch.pop("is_domain_batch")
                
                # 数据类型判断 
                data_type = get_data_preview_and_type(batch, accelerator)
                
                # # 打印出来batch数据预览和类型
                # consistent, local_bs, all_bs = safe_check_batch_consistency(batch, accelerator)
                # if not consistent:
                #     accelerator.print(f"[Rank {accelerator.process_index}] ⚠️ Batch size 不一致，all_bs={all_bs}，跳过本 batch 以避免死锁")
                #     return
                
                # 训练当前批次
                try:
                    train_one_batch(
                        batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, accelerator,
                        step_losses, general_kl_log, domain_ce_log, loader_names[loader_idx]
                    )
                except Exception as e:
                    handle_communication_error(e, accelerator, "训练batch时", stats=training_stats)
                    # 跳过这个batch，继续下一个
                    step_count += 1
                    loader_idx = 1 - loader_idx
                    continue
                
                step_count += 1
                loader_idx = 1 - loader_idx  # 切换到另一个数据加载器

                # 每1000步保存一次模型和更新损失文件
                if step_count % 1000 == 0:
                    try:
                        # 保存模型
                        save_path = f"../output/mol_pretrain_step_{step_count}"
                        # accelerator.wait_for_everyone()  # 等待所有进程
                        if accelerator.is_main_process:
                            unwrapped_model = accelerator.unwrap_model(student_model)
                            unwrapped_model.save_pretrained(save_path, save_dtype=torch.float16)
                            tokenizer.save_pretrained(save_path)
                            accelerator.print(f"✅ 模型已保存到 {save_path}")
                            
                            # 清理旧的模型文件，只保留最新的5个
                            cleanup_old_models("../output", keep_count=5)
                        
                        # 更新损失文件
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

                            # 保存领域数据CE损失日志
                            with open("domain_ce_log.txt", "w") as f:
                                f.write("step,ce_loss\n")
                                for s, ce in domain_ce_log:
                                    f.write(f"{s},{ce:.6f}\n")
                            
                            accelerator.print(f"✅ 损失文件已更新，当前步数: {step_count}")
                    except Exception as e:
                        accelerator.print(f"⚠️ 保存模型时出现错误: {e}")
                        # 继续训练，不中断

                torch.cuda.synchronize()
                # 每步强制清理一次显存
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.1)  # 加一点延迟
                accelerator.wait_for_everyone()  # 等待所有进程
                
            
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
                accelerator.print(f"✅ 模型已保存到 {save_path}")
                
            
            # 保存后清理内存
            get_accelerator().empty_cache()
            gc.collect()

        # ============================================================================
        # 12. 训练结束保存最终模型
        # ============================================================================
        save_path = "../output/final"
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.save_pretrained(save_path, save_dtype=torch.float16)
            tokenizer.save_pretrained(save_path)
            accelerator.print(f"✅ 最终模型已保存到 {save_path}")
            torch.save(accelerator.unwrap_model(student_model).state_dict(), '../output/model')

        # 最终清理
        get_accelerator().empty_cache()
        gc.collect()

        # ============================================================================
        # 13. 保存训练日志（仅主进程）
        # ============================================================================
        if accelerator.is_main_process:
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
                accelerator.print(f"通用样本KL散度均值: {avg_kl:.6f}")
            else:
                accelerator.print("无通用样本KL散度记录。")

            if domain_ce_log:
                avg_ce = sum([ce for _, ce in domain_ce_log]) / len(domain_ce_log)
                accelerator.print(f"领域样本交叉熵损失均值: {avg_ce:.6f}")
            else:
                accelerator.print("无领域样本交叉熵损失记录。")

        accelerator.print(f"训练结束，总耗时: {time.time() - start_time:.2f}秒")
         # 训练完成后释放资源
        dist.destroy_process_group()
        
    except Exception as e:
        accelerator.print(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 