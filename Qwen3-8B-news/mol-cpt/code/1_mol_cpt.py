from datetime import timedelta
import json
import torch
import os
os.environ["UNSLOTH_DISABLE_VLLM_CHECK"] = "1"
os.environ["WANDB_DISABLED"] = "true"
import sys
import types
# 导入Unsloth库
import unsloth
from unsloth import FastLanguageModel

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader
from bitsandbytes.optim import AdamW8bit
from transformers import AutoTokenizer, BitsAndBytesConfig, get_scheduler
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
from datasets import load_dataset  # Hugging Face数据集库，用于高效加载数据



# 导入swanlab
import swanlab

torch.set_num_threads(1)

# 设置环境变量以优化显存分配和避免NCCL错误
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'         #显示指明unsloth返回logits
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = '600'  # 30分钟超时
os.environ['NCCL_IB_DISABLE'] = '1'  
os.environ['NCCL_P2P_DISABLE'] = '1'  
os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '67108864'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_DESYNC_DEBUG"] = "1"

# ============================================================================
# 0. 参数配置
# ============================================================================
MODEL_NAME = "/mnt/project/Qwen3-8B-news/outputs/mol_cpt/Qwen3-8B-1epoch"
MAX_SEQ_LENGTH = 4096  # 定义最大序列长度
OUTPUT_DIR = "/mnt/project/Qwen3-8B-news/mol-cpt/outputs/mol_cpt"  # 模型输出目录
DATA_PATHS = {
    "magpie": "/mnt/project/Qwen2.5-7B-Instruct-news/MOL-CPT/data/magpie_general_mol_messages.jsonl",   # 通用数据1
    "lightr1": "/mnt/project/Qwen2.5-7B-Instruct-news/MOL-CPT/data/stage-1-lightr1-sft_messages.jsonl",            # 通用数据2
    "domain": "/mnt/project/Qwen3-8B-news/mol-cpt/data/news.jsonl",            # 领域数据
}

# 训练参数
BATCH_SIZE = 1  # 小批次大小，适合大模型训练，减少显存占用
NUM_TRAIN_EPOCHS = 4  # 训练轮数
LEARNING_RATE = 5e-4  # 学习率
ALPHA = 0.01  # alpha参数：控制损失权重，0表示主要使用KL损失
LORA_R = 64  # LoRA秩
LORA_ALPHA = 128  # LoRA缩放参数
LORA_DROPOUT = 0.05  # LoRA dropout率
KEEP_MODEL_COUNT = 5  # 保留最新模型数量

# 全局变量，用于在函数间共享
tokenizer = None
EOS_TOKEN = None

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
    """
    验证训练所需的所有文件路径是否存在
    这个函数在训练开始前检查所有必要的文件和目录
    """
    # 定义需要检查的路径字典，包含模型、数据和输出目录
    paths_to_check = {
        "model": MODEL_NAME,  # 预训练模型路径
        "general_data": DATA_PATHS["magpie"],  # 第一个通用数据集
        "general_data2": DATA_PATHS["lightr1"],  # 第二个通用数据集
        "domain_data": DATA_PATHS["domain"],  # 领域数据集
        "output_dir": OUTPUT_DIR  # 模型输出目录
    }
    
    # 存储缺失的路径
    missing_paths = []
    # 遍历所有需要检查的路径
    for name, path in paths_to_check.items():
        # 检查路径是否存在
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
    
    # 如果有缺失的路径，输出错误信息并返回False
    if missing_paths:
        accelerator.print(f"以下路径不存在: {missing_paths}")
        accelerator.print("请检查路径是否正确，或创建相应的目录/文件")
        return False
    
    # 创建输出目录（如果不存在），exist_ok=True表示如果目录已存在不会报错
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
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=600))
ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True,
    broadcast_buffers=False,  # 禁用广播缓冲区以节省显存
    static_graph=True  # 启用静态图优化
)

accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

# 打印当前进程和设备信息
accelerator.print(f"[Rank {accelerator.process_index}] Hello from {torch.cuda.get_device_name()} (device={torch.cuda.current_device()})")

# ============================================================================
# 2. 加载模型和tokenizer - 使用Unsloth
# ============================================================================

def load_models_safely(accelerator):
    """安全加载模型，包含错误处理 - 使用Unsloth"""
    global tokenizer, EOS_TOKEN
    # 在 load_models_safely 函数开头添加
    import logging

    # 禁用transformers的某些警告
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # 禁用Unsloth的严格检测
    os.environ["UNSLOTH_NO_WARNINGS"] = "1"
    try:
        # 获取当前设备
        device = accelerator.device
        
        accelerator.print(f"[Rank {accelerator.process_index}] 加载学生模型...")
        # 使用Unsloth加载学生模型
        student_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map={"": device},  # 确保模型加载到正确设备
            attn_implementation="flash_attention_2",  # 使用Flash Attention 2
            local_files_only=True

        )
        accelerator.print(f"[Rank {accelerator.process_index}] 学生模型加载成功")

        accelerator.print(f"[Rank {accelerator.process_index}] 加载教师模型...")
        # 使用Unsloth加载教师模型
        teacher_model, _ = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map={"": device},  # 确保模型加载到正确设备
            local_files_only=True
        )
        accelerator.print(f"[Rank {accelerator.process_index}] 教师模型加载成功")

        # 冻结教师模型参数，不进行梯度计算
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False  # 禁用缓存以节省显存
        teacher_model.eval()  # 设置为评估模式

        # 设置tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        EOS_TOKEN = tokenizer.eos_token or ""  # 获取结束标记
        
        return student_model, teacher_model
        
    except Exception as e:
        accelerator.print(f"模型加载失败: {e}")
        raise

# ============================================================================
# 3. 配置LoRA参数高效微调 - 使用Unsloth的LoRA方法
# ============================================================================
def setup_lora(student_model, accelerator):
    """配置LoRA参数 - 使用Unsloth"""
    try:
        accelerator.print(f"[Rank {accelerator.process_index}] 配置LoRA...")
        # 使用Unsloth的get_peft_model配置LoRA
        student_model = FastLanguageModel.get_peft_model(
            student_model,
            r=LORA_R,  # LoRA秩
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj",      # MLP层
            ],
            lora_alpha=LORA_ALPHA,  # LoRA缩放参数
            lora_dropout=LORA_DROPOUT,  # LoRA dropout率
            bias="none",  # 不训练偏置项
            use_gradient_checkpointing=True,  # 启用梯度检查点
            random_state=42,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        
        accelerator.print(f"[Rank {accelerator.process_index}] LoRA配置成功")
        return student_model
        
    except Exception as e:
        accelerator.print(f"LoRA配置失败: {e}")
        raise

# ============================================================================
# 4. 数据统计和加载函数
# ============================================================================
def count_tokens_and_size(data, tokenizer, data_type="domain", name=""):
    """
    统计已加载数据集的token数量和大小
    
    Args:
        data: 已加载的数据样本列表
        tokenizer: 分词器
        data_type: 数据类型 ('domain' 或 'general')
        name: 数据集名称，用于日志输出
    
    Returns:
        total_tokens: 总token数
        total_bytes: 总字节数
        count: 有效样本数
    """
    total_tokens, total_bytes, count = 0, 0, 0
    
    try:
        for sample in data:
            # 根据数据类型提取文本
            if data_type == "domain":
                # Domain数据格式: {"text": "新闻内容...", "is_domain_batch": true}
                text = sample.get("text", "")
            else:
                # General数据格式: {"messages": [...], "is_domain_batch": false}
                # 构建完整的对话文本
                text = ""
                for message in sample["messages"]:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    text += f"{role}: {content}\n"
            
            text = text.strip()
            if not text:
                continue
            
            # 统计token数（不添加特殊token）
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            token_count = len(tokens)
            
            # 统计字节数
            byte_count = len(text.encode('utf-8'))
            
            total_tokens += token_count
            total_bytes += byte_count
            count += 1
        
        # 计算并记录统计信息
        mb_size = round(total_tokens / 1024 / 1024, 2)
        accelerator.print(f"✅ {data_type} ({name}) 统计完成：{count} 条有效样本，"
                    f" token数为:{total_tokens},{mb_size} MB")
        
        return total_tokens
        
    except Exception as e:
        accelerator.print(f"统计{name}数据失败: {e}")
        raise

def analyze_token_distribution(data, tokenizer, data_type="", name=""):
    """
    分析数据集的token长度分布情况
    
    Args:
        data: 已加载的数据样本列表
        tokenizer: 分词器
        data_type: 数据类型 ('domain' 或 'general')
        name: 数据集名称，用于日志输出
    
    Returns:
        distribution: token长度分布字典
    """
    # 定义token长度分布桶
    distribution = {
        "0-512": 0,
        "512-1024": 0,
        "1024-1536": 0,
        "1536-2048": 0,
        "2048-3072": 0,
        "3072-4096": 0,
        "4096+": 0
    }
    
    try:
        for sample in data:
            # 根据数据类型提取文本
            if data_type == "domain":
                # Domain数据格式: {"text": "新闻内容...", "is_domain_batch": true}
                text = sample.get("text", "")
            else:
                # General数据格式: {"messages": [...], "is_domain_batch": false}
                if "messages" in sample and isinstance(sample["messages"], list):
                    # 构建完整的对话文本
                    text = ""
                    for message in sample["messages"]:
                        role = message.get("role", "")
                        content = message.get("content", "")
                        text += f"{role}: {content}\n"
                else:
                    # 备用处理（旧格式）
                    instruction = sample.get("instruction", "")
                    output = sample.get("output", "")
                    text = (instruction + output).strip()
            
            text = text.strip()
            if not text:
                continue
            
            # 统计token数（不添加特殊token）
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            token_count = len(tokens)
            
            # 更新token分布统计
            if token_count <= 512:
                distribution["0-512"] += 1
            elif token_count <= 1024:
                distribution["512-1024"] += 1
            elif token_count <= 1536:
                distribution["1024-1536"] += 1
            elif token_count <= 2048:
                distribution["1536-2048"] += 1
            elif token_count <= 3072:
                distribution["2048-3072"] += 1
            elif token_count <= 4096:
                distribution["3072-4096"] += 1
            else:
                distribution["4096+"] += 1
        
        return distribution
        
    except Exception as e:
        logger.error(f"分析{name}数据token分布失败: {e}")
        raise
# ============================================================================
# 4. 自定义数据集类 - MoL训练的核心数据处理
# ============================================================================
class MoLDatasetSingleType(TorchDataset):
    """
    MoL（Mixture of LoRA）数据集类
    支持通用数据和领域数据的统一处理
    这个类负责将原始数据转换为模型训练所需的格式
    """
    def __init__(self, data, tokenizer, type, max_seq_len=4096):
        """
        初始化数据集
        
        Args:
            data: 数据样本列表，包含原始的数据样本
            tokenizer: 分词器，用于文本token化
            type: 数据类型 ('domain' 或 'general')，决定处理方式
            max_seq_len: 最大序列长度，超过此长度的序列会被截断
        """
        self.data = data  # 存储原始数据样本
        self.tokenizer = tokenizer  # 分词器
        self.type = type  # 数据类型：domain（领域数据）或general（通用数据）
        self.max_seq_len = max_seq_len  # 最大序列长度

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        这是数据集类的核心方法，负责将原始数据转换为训练格式
        
        Args:
            idx: 样本索引
            
        Returns:
            包含input_ids、labels、attention_mask和is_domain_batch的字典
        """
        try:
            # 获取原始样本
            sample = self.data[idx]
            
            if self.type == 'domain':
                # 领域数据处理：封装成 system 角色，进行全量学习
                text = sample.get('text', '').strip()
                
                # 直接对文本进行tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_seq_len,
                    add_special_tokens=True,
                    padding=False,
                    return_tensors=None
                )
                
                # 对于领域数据，labels就是input_ids（全量学习）
                input_ids = encoding["input_ids"]
                
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                    "is_domain_batch": 1,
                }
            elif self.type == 'general':
                # 通用数据处理：使用apply_chat_template
                messages = sample["messages"]
                
                # 1. 生成完整对话的token
                full_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,  # 因为messages里已经有assistant回复了
                    truncation=True,
                    max_length=self.max_seq_len
                )
                
                # 2. 找到prompt部分的token（去掉最后一个assistant消息）
                prompt_messages = messages[:-1]  # 最后一个消息是assistant回复
                
                prompt_ids = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,  # 添加assistant起始标记
                    truncation=True,
                    max_length=self.max_seq_len
                )
                
                prompt_len = len(prompt_ids)
                
                # 3. 创建labels：prompt部分设为-100，assistant部分保留原token
                labels = [-100] * prompt_len + full_ids[prompt_len:]
                
                # 截断到最大长度
                input_ids = full_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]
                
                # 确保长度一致
                if len(labels) < len(input_ids):
                    labels.extend([-100] * (len(input_ids) - len(labels)))
                
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                    "is_domain_batch": 0,
                }
        except Exception as e:
            # 错误处理：记录错误并返回一个最小的有效样本
            logger.error(f"处理样本 {idx} 失败: {e}")


# ============================================================================
# 5. 自定义数据整理函数 - 批次数据处理
# ============================================================================
def mol_collate_fn(batch):
    """
    自定义数据整理函数，处理变长序列的padding
    这个函数负责将多个样本组合成一个批次，处理不同长度的序列
    
    Args:
        batch: 批次数据列表，包含多个样本的字典
    
    Returns:
        整理后的批次数据字典，包含padded的张量
    """
    global tokenizer  # 使用全局的tokenizer
    
    # 过滤掉空的样本（input_ids长度为0的样本）
    valid_batch = [b for b in batch if len(b["input_ids"]) > 0]
    
    # 过滤掉有效token为0的样本（所有标签都是-100的样本）
    # 这些样本没有有效的训练目标，应该被过滤掉
    valid_batch = [
        b for b in batch 
        if (b["labels"] != -100).sum().item() > 0  # 检查是否有非-100的标签
    ]
    
    # 如果所有样本都为空，返回None
    if not valid_batch:
        return None
    
    # 对输入序列进行padding，使所有序列长度一致
    # pad_sequence会将不同长度的序列填充到相同长度
    input_ids = pad_sequence(
        [b["input_ids"] for b in valid_batch],  # 提取所有input_ids
        batch_first=True,  # 批次维度在前
        padding_value=tokenizer.pad_token_id  # 使用pad_token_id进行填充
    )
    
    # 对标签序列进行padding，使用-100作为padding值
    # -100在损失计算中会被忽略，不会影响训练
    labels = pad_sequence(
        [b["labels"] for b in valid_batch],  # 提取所有labels
        batch_first=True,  # 批次维度在前
        padding_value=-100  # 使用-100进行填充
    )
    
    # 对注意力掩码进行padding，使用0作为padding值
    # 0表示该位置是padding，不应该被注意力机制关注
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in valid_batch],  # 提取所有attention_mask
        batch_first=True,  # 批次维度在前
        padding_value=0  # 使用0进行填充
    )
    
    # 收集批次类型标记，用于区分领域数据和通用数据
    # 将每个样本的is_domain_batch标记堆叠成一个张量
    is_domain_batch = torch.stack([
        torch.tensor(b["is_domain_batch"], dtype=torch.bool) for b in valid_batch
    ])
    
    # 确定设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 返回整理后的批次数据，所有张量都移动到指定设备
    return {
        "input_ids": input_ids.to(device),  # 输入token ID张量
        "labels": labels.to(device),  # 标签张量
        "attention_mask": attention_mask.to(device),  # 注意力掩码张量
        "is_domain_batch": is_domain_batch.to(device),  # 批次类型标记张量
    }

# ============================================================================
# 6. 加载数据集和数据加载器
# ============================================================================
def load_datasets(accelerator):
    """加载和准备数据集"""
    global tokenizer
    
    # 加载数据（使用datasets库）
    def load_data_with_datasets(path, name=""):
        """使用datasets库加载数据"""
        try:
            # 使用datasets库加载数据
            dataset = load_dataset("json", data_files=path, split="train")
            accelerator.print(f"✅ {name} 加载完毕：{len(dataset)} 条样本")
            return dataset
        except Exception as e:
            accelerator.print(f"加载{name}数据失败: {e}")
            

    # 加载通用数据集
    magpie_dataset = load_data_with_datasets(DATA_PATHS["magpie"], "magpie")
    lightr1_dataset = load_data_with_datasets(DATA_PATHS["lightr1"], "lightr1")
    
    # 加载领域数据集并计算token数
    domain_dataset = load_data_with_datasets(DATA_PATHS["domain"], "domain")
    domain_tokens= count_tokens_and_size(domain_dataset, tokenizer, "domain","news_3year")

    #截断函数
    def truncate_to_tokens(data, target_tokens, mode):
        """
        截断数据到指定token数量
        这个函数用于将数据集截断到指定的token数量，确保数据平衡
        
        Args:
            data: 数据列表或datasets.Dataset对象
            target_tokens: 目标token数，截断后的数据token数不超过此值
            mode: 数据模式 ('domain' 或 'messages')，决定如何提取文本
        
        Returns:
            kept: 保留的数据样本列表
            total: 实际token数
        """
        global tokenizer  # 使用全局的tokenizer
        total, kept,= 0, []  # 初始化token计数和保留的数据
        
        # 处理不同的数据类型
        if hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            # 这是datasets.Dataset对象，可以直接迭代
            data_iter = data
        else:
            # 这是普通列表，也可以直接迭代
            data_iter = data
            
        # 遍历数据样本
        for s in data_iter:
            # 根据模式提取文本内容
            if mode == "general":
                # 处理messages格式的数据
                try:
                    # 使用聊天模板展开对话
                    t = tokenizer.apply_chat_template(s["messages"], tokenize=False, enable_thinking=False)
                except Exception:
                    # 如果聊天模板失败，直接拼接消息内容
                    t = "\n".join(str(m.get("content", "")) for m in s.get("messages", []))
            elif mode == "domain":
                # 处理领域数据格式
                t = s.get("text", "").strip()

            
            # 计算当前样本的token数
            n = len(tokenizer(t, add_special_tokens=False).input_ids)
            if n > 3700:
                continue
            # 如果添加当前样本会超过目标token数，停止截断
            if total + n > target_tokens:
                break
                
            # 添加当前样本的token数到总数
            total += n
            # 保留当前样本
            kept.append(s)
            
        return kept, total  # 返回保留的数据和实际token数

    # 按照领域数据集token数的1/2截断两个通用数据集
    # 这样确保通用数据和领域数据的token数量平衡
    half_domain_tokens = max(1, domain_tokens // 2)  # 确保至少为1
    
    # 检查并截断第一个通用数据集（magpie）
    if len(magpie_dataset) == 0:
        # 如果数据集为空，记录警告并跳过截断
        accelerator.print("⚠️ magpie数据集为空，跳过截断")
        g1_total = 0
        magpie_data = []
    else:
        # 截断到领域数据token数的一半
        magpie_data, g1_total = truncate_to_tokens(magpie_dataset, half_domain_tokens, "general")
    
    # 检查并截断第二个通用数据集（lightr1）
    if len(lightr1_dataset) == 0:
        # 如果数据集为空，记录警告并跳过截断
        accelerator.print("⚠️ lightr1数据集为空，跳过截断")
        g2_total = 0
        lightr1_data = []
    else:
        # 截断到领域数据token数的一半
        lightr1_data, g2_total = truncate_to_tokens(lightr1_dataset, half_domain_tokens, "general")
    
    # 输出截断结果
    accelerator.print(f"✅ magpie_general_mol截断后token数为:{g1_total},样本数为{len(magpie_data)},light-r1截断后token数为:{g2_total},样本数为{len(lightr1_data)}")
    
    general_magpie = analyze_token_distribution(magpie_data, tokenizer, data_type="general", name="magpie_data")   
    accelerator.print(f"通用数据magpie的Token分布:{general_magpie}")
    general_lightr1 = analyze_token_distribution(lightr1_data, tokenizer, data_type="general", name="lightr1_data")   
    accelerator.print(f"通用数据lightr1的Token分布:{general_lightr1}")

    # 合并两个通用数据集，实现1:1混合
    general_data = magpie_data + lightr1_data
    
    # 检查合并后的数据集是否为空
    if len(general_data) == 0:
        # 如果合并后为空，记录错误
        accelerator.print("⚠️ 合并后的通用数据集为空！请检查数据文件格式。")
        # 使用领域数据作为通用数据进行训练（备用方案）

    # 确保样本数是4的倍数，保证4卡分布式训练时数据分布均匀
    def adjust_to_multiple_of_4(data, data_name):
        """
        调整数据样本数为4的倍数
        支持 list 和 datasets.Dataset 两种类型
        
        Args:
            data: 数据样本列表 或 datasets.Dataset
            data_name: 数据集名称，用于日志输出
            
        Returns:
            调整后的数据（保持原类型）
        """
        original_count = len(data)  # 原始样本数
        # 计算需要保留的样本数（向下取整到最近的4的倍数）
        target_count = (original_count // 4) * 4
        if target_count == 0:
            target_count = 4  # 至少保留4个样本

        if isinstance(data, list):
            adjusted_data = data[:target_count]  # 列表切片
        else:
            # 假设是 HuggingFace Dataset
            adjusted_data = data.select(range(target_count))

        accelerator.print(f"✅ {data_name} 样本数调整: {original_count} → {len(adjusted_data)} (4的倍数)")
        return adjusted_data

    
    # 调整两个数据集的样本数为4的倍数
    general_data = adjust_to_multiple_of_4(general_data, "通用数据(合并)")
    domain_data = list(domain_dataset)
    domain_data = adjust_to_multiple_of_4(domain_data, "领域数据")

    domain_news_dist = analyze_token_distribution(domain_data, tokenizer, data_type="domain", name="news_3year")   
    accelerator.print(f"领域数据new_3years的Token分布:{domain_news_dist}")

    # 输出最终的数据加载结果
    accelerator.print(f"✅ 加载数据完成：通用样本(合并) {len(general_data)} 条，领域样本 {len(domain_data)} 条，领域总tokens={domain_tokens}，通用截断各自半数后总tokens≈{g1_total + g2_total}")

    # 创建数据集对象
    # 使用自定义的MoLDatasetSingleType类创建数据集
    general_dataset = MoLDatasetSingleType(general_data, tokenizer, type='general')  # 通用数据集
    domain_dataset = MoLDatasetSingleType(domain_data, tokenizer, type='domain')      # 领域数据集
    
    # 创建数据加载器
    # 通用数据加载器
    general_loader = DataLoader(
        general_dataset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=True,  # 随机打乱数据
        collate_fn=mol_collate_fn,  # 自定义的数据整理函数
        drop_last=True,  # 丢弃最后一个不完整的批次
        num_workers=0,  # 减少worker数量，避免多进程问题
        pin_memory=False,  # 禁用pin_memory，节省显存
    )
    
    # 领域数据加载器
    domain_loader = DataLoader(
        domain_dataset,  # 数据集
        batch_size=BATCH_SIZE,  # 批次大小
        shuffle=True,  # 随机打乱数据
        collate_fn=mol_collate_fn,  # 自定义的数据整理函数
        drop_last=True,  # 丢弃最后一个不完整的批次
        num_workers=0,  # 减少worker数量，避免多进程问题
        pin_memory=False,  # 禁用pin_memory，节省显存
    )
    
    # 返回创建的数据加载器和训练参数
    return general_loader, domain_loader, NUM_TRAIN_EPOCHS

# ============================================================================
# 8. 优化器和学习率调度器
# ============================================================================
def setup_optimizer_and_scheduler(student_model, num_training_steps):
    """设置优化器和学习率调度器"""
    # # 使用AdamW优化器
    # optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    optimizer = AdamW8bit(student_model.parameters(), lr=LEARNING_RATE)

    # 使用余弦学习率调度器
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,  # 无预热步数
        num_training_steps=num_training_steps,
    )
    
    return optimizer, lr_scheduler

# ============================================================================
# 9. Accelerator准备
def prepare_for_training(student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler):
    """准备训练环境"""

    # 使用Accelerator准备模型、优化器、数据加载器等
    student_model, optimizer, general_loader, domain_loader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, general_loader, domain_loader, lr_scheduler
    )
    
    return student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler

# =======训练单个批次的函数=====================================================================
def train_one_batch(batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, student_loss_log, teacher_loss_log, loader_name, accelerator, swanlab_initialized):
    
    # [建议3修改]：引入 KL 温度系数，2.0 是一个经验平衡值
    KL_TEMPERATURE = 2.0 

    # 初始化全局错误标志 (0=正常, 1=有问题)
    forward_skip_tensor = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    backward_skip_tensor = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    outputs = None
    student_logits = None
    teacher_outputs = None
    teacher_logits = None

    loss = None
    ce_loss = None
    kl_loss = None
    ce_loss_mean = None
    kl_loss_mean = None
    
    # 显存清理函数
    def clear_memory(
        batch=None,
        losses=None,
        student_outputs=None, student_logits=None,
        teacher_outputs=None, teacher_logits=None,
        optimizer=None, model=None,
        clear_grads=False,  
    ):
        try:
            if clear_grads and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            
            if clear_grads and (model is not None):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None

            if batch is not None:
                if isinstance(batch, dict):
                    for k in list(batch.keys()):
                        if isinstance(batch[k], torch.Tensor):
                            del batch[k]
                    batch.clear()

            if losses is not None:
                if isinstance(losses, (list, tuple)):
                    for l in losses:
                        if l is not None and isinstance(l, torch.Tensor):
                            l = l.detach().cpu()
                            del l
                else:
                    if losses is not None and isinstance(losses, torch.Tensor):
                        losses = losses.detach().cpu()
                        del losses

            if student_outputs is not None: 
                del student_outputs
            if student_logits is not None: 
                del student_logits
            if teacher_outputs is not None: 
                del teacher_outputs
            if teacher_logits is not None: 
                del teacher_logits

        except Exception as e:
            print(f"⚠️ 清理异常: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        with accelerator.accumulate(student_model):
            try:
                if len(is_domain) == 0:
                    print(f"[{loader_name}] 跳过空batch")
                    forward_skip_tensor.fill_(1)
                else:
                    # 学生模型前向传播
                    outputs = student_model(**batch)
                    student_logits = outputs.logits
                    
                    # 教师模型前向传播
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**batch)
                    teacher_logits = teacher_outputs.logits
                    
                    # ===================== 打印训练数据详情 =====================
                    if accelerator.is_main_process and step_count % 1 == 0:  # 每100步打印一次
                        for i in range(min(2, batch["input_ids"].size(0))):  # 打印前2个样本
                            # 获取当前样本
                            input_ids = batch["input_ids"][i]
                            labels = batch["labels"][i]
                            attention_mask = batch["attention_mask"][i]
                            is_domain_sample = is_domain[i].item()
                            
                            # 找出有效训练位置（labels != -100 且 attention_mask == 1）
                            valid_positions = (labels != -100) & (attention_mask == 1)
                            valid_indices = valid_positions.nonzero(as_tuple=True)[0].cpu().tolist()
                            
                            # 统计信息
                            total_tokens = (attention_mask == 1).sum().item()
                            valid_tokens = len(valid_indices)
                            
                            # 提取有效token
                            valid_input_ids = input_ids[valid_positions].cpu().tolist()
                            valid_labels = labels[valid_positions].cpu().tolist()
                            
                            # 解码文本
                            try:
                                # 解码所有tokens（包括padding）
                                full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                                # 解码有效训练部分
                                valid_text = tokenizer.decode(valid_input_ids, skip_special_tokens=True)
                            except:
                                full_text = "[解码失败]"
                                valid_text = "[解码失败]"
                            
                            print(f"\n{'='*80}")
                            print(f"Step {step_count} | Batch {i+1}/{batch['input_ids'].size(0)} | Type: {'Domain' if is_domain_sample else 'General'}")
                            print(f"Total tokens: {total_tokens} | Trainable tokens: {valid_tokens} ({valid_tokens/total_tokens*100:.1f}%)")
                            print(f"Valid token positions: {valid_indices[:10]}{'...' if len(valid_indices) > 10 else ''}")
                            
                            # 打印样本详情
                            print(f"\n--- Full Sample (truncated) ---")
                            print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
                            
                            print(f"\n--- Trainable Part ---")
                            print(valid_text[:500] + "..." if len(valid_text) > 500 else valid_text)
                            
                            # 打印labels和input_ids的对应关系（前20个有效位置）
                            print(f"\n--- Token Details (first 20 trainable tokens) ---")
                            for j, (input_id, label) in enumerate(zip(valid_input_ids[:20], valid_labels[:20])):
                                token = tokenizer.decode([input_id])
                                print(f"  Pos {valid_indices[j]:3d}: Input ID={input_id:6d}, Label={label:6d}, Token='{token}'")
                            
                            # 如果是通用数据，打印哪些部分被mask了
                            if not is_domain_sample:
                                mask_positions = (labels == -100) & (attention_mask == 1)
                                mask_count = mask_positions.sum().item()
                                if mask_count > 0:
                                    print(f"\n--- Masked Parts ({mask_count} tokens) ---")
                                    mask_indices = mask_positions.nonzero(as_tuple=True)[0].cpu().tolist()
                                    mask_input_ids = input_ids[mask_positions].cpu().tolist()
                                    for j, (idx, input_id) in enumerate(zip(mask_indices[:10], mask_input_ids[:10])):
                                        token = tokenizer.decode([input_id])
                                        print(f"  Pos {idx:3d}: Input ID={input_id:6d}, Token='{token}' [MASKED]")
                                    if len(mask_indices) > 10:
                                        print(f"  ... and {len(mask_indices)-10} more masked tokens")
                            print('='*80 + '\n')
                    # ===================== 结束打印 =====================
                    
                    # 计算有效token掩码
                    mask = (batch["attention_mask"] == 1) & (batch["labels"] != -100)
                    valid_indices = mask.nonzero(as_tuple=True)
                        
                    # 提取有效位置的logits
                    s_logits = student_logits[valid_indices]
                    t_logits = teacher_logits[valid_indices]
                        
                    # [建议3修改]：带 Temperature 的 KL 散度计算
                    # 为什么要除以 T：平滑 Logits 分布，降低预测“尖锐度”
                    student_log_probs = torch.nn.functional.log_softmax(s_logits / KL_TEMPERATURE, dim=-1)
                    teacher_probs = torch.nn.functional.softmax(t_logits / KL_TEMPERATURE, dim=-1)
                    kl_loss = torch.nn.functional.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction="batchmean", 
                        log_target=False
                    )* (KL_TEMPERATURE ** 2) # [重要]：乘以 T 的平方补偿梯度缩减
                    
                    # 交叉熵损失
                    ce_loss = outputs.loss

            except RuntimeError as e:
                # 捕获本地错误并设置标志
                if "out of memory" in str(e):
                    forward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 前向传播显存不足，标记跳过")
                    clear_memory(
                        batch=batch,
                        losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
                        student_outputs=outputs, student_logits=student_logits,
                        teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
                        optimizer=optimizer, model=student_model,
                        clear_grads=True,
                    )                                        
                else:
                    # 其他运行时错误
                    forward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 运行时错误: {str(e)}")
                    clear_memory(
                        batch=batch,
                        losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
                        student_outputs=outputs, student_logits=student_logits,
                        teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
                        optimizer=optimizer, model=student_model,
                        clear_grads=True,
                    )

            # 全局同步错误标志 (关键!)
            accelerator.wait_for_everyone()  # 等待所有进程完成
            forward_skip_tensor = accelerator.reduce(forward_skip_tensor, reduction="sum")

            if forward_skip_tensor.item() > 0:
                # 所有卡统一跳过当前步骤
                print(f"[{loader_name}]step={step_count}[Rank {accelerator.process_index}] 因为前向传播显存不够，全局跳过步骤 {step_count+1}")
                
                # 所有卡执行相同的清理操作
                clear_memory(
                    batch=batch,
                    losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
                    student_outputs=outputs, student_logits=student_logits,
                    teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
                    optimizer=optimizer, model=student_model,
                    clear_grads=True,
                )
                return
            try:
                kl_loss_mean = accelerator.reduce(kl_loss, reduction="mean")
                ce_loss_mean = accelerator.reduce(ce_loss, reduction="mean")

                if torch.isnan(ce_loss_mean) or torch.isinf(ce_loss_mean) or \
                   torch.isnan(kl_loss_mean) or torch.isinf(kl_loss_mean):
                    accelerator.print(f"检测到异常损失值，跳过步骤 {step_count}")
                    backward_skip_tensor.fill_(1)
                    clear_memory(
                        batch=batch,
                        losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
                        student_outputs=outputs, student_logits=student_logits,
                        teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
                        optimizer=optimizer, model=student_model,
                        clear_grads=True,
                    )
                else:    
                    # 根据数据源计算最终损失
                    if loader_name == "General":
                        loss = alpha * ce_loss_mean + (1 - alpha) * kl_loss_mean
                        # 记录swanlab
                        if swanlab_initialized:
                            swanlab.log({"general_kl_loss": kl_loss_mean.item()}, step=step_count)
                        teacher_loss_log.append((step_count, kl_loss_mean.item()))
                    else:
                        loss = (1 - alpha) * ce_loss_mean + alpha * kl_loss_mean
                        # 记录swanlab
                        if swanlab_initialized:
                            swanlab.log({"domain_ce_loss": ce_loss_mean.item()}, step=step_count)
                        student_loss_log.append((step_count, ce_loss_mean.item()))
                            
                    # ce_losses.append(ce_loss_mean)
                    # kl_losses.append(kl_loss_mean)
                    # 记录swanlab
                    if swanlab_initialized:
                        swanlab.log({
                            "total_loss": loss.item(),
                            "ce_loss": ce_loss_mean.item(),
                            "kl_loss": kl_loss_mean.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "step": step_count
                        }, step=step_count)

                    # 无错误时正常执行优化步骤
                    optimizer.zero_grad(set_to_none=True)

                    accelerator.backward(loss)

            except RuntimeError as e:
                # 捕获本地错误并设置标志
                if "out of memory" in str(e):
                    backward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 反向传播显存不足，标记跳过")
                    clear_memory(
                        batch=batch,
                        losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
                        student_outputs=outputs, student_logits=student_logits,
                        teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
                        optimizer=optimizer, model=student_model,
                        clear_grads=True,
                    )
                else:
                    # 其他运行时错误
                    backward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 反向传播运行时错误: {str(e)}")
                    clear_memory(
                        batch=batch,
                        losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
                        student_outputs=outputs, student_logits=student_logits,
                        teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
                        optimizer=optimizer, model=student_model,
                        clear_grads=True,
                    )

            # 全局同步错误标志 (关键!)
            accelerator.wait_for_everyone()  # 等待所有进程完成
            backward_skip_tensor = accelerator.reduce(backward_skip_tensor, reduction="sum")
            if backward_skip_tensor.item() > 0:
                # 所有卡统一跳过当前步骤
                accelerator.print(f"[{loader_name}][Rank {accelerator.process_index}] 因为反向传播失败，全局跳过步骤 {step_count+1}")
                # 所有卡执行相同的清理操作
                clear_memory(
                    batch=batch,
                    losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
                    student_outputs=outputs, student_logits=student_logits,
                    teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
                    optimizer=optimizer, model=student_model,
                    clear_grads=True,
                )
                return
            
            # 优化器更新
            optimizer.step()
            # 学习率更新
            lr_scheduler.step()
            
            # 打印训练信息
            if accelerator.is_main_process:
                # 计算当前batch的详细信息
                total_tokens = batch["attention_mask"].sum().item()
                trainable_tokens = ((batch["labels"] != -100) & (batch["attention_mask"] == 1)).sum().item()
                
                accelerator.print(f"[{loader_name}] step={step_count+1}, "
                                  f"loss={loss.item():.6f}, ce={ce_loss_mean.item():.6f}, kl={kl_loss_mean.item():.6f}, "
                                  f"tokens={trainable_tokens}/{total_tokens} ({trainable_tokens/total_tokens*100:.1f}%)")
    
    finally:
        # 清理内存 (所有卡都执行)
        clear_memory(
            batch=batch,
            losses=[loss, ce_loss, kl_loss, ce_loss_mean, kl_loss_mean],
            student_outputs=outputs, student_logits=student_logits,
            teacher_outputs=teacher_outputs, teacher_logits=teacher_logits,
            optimizer=optimizer, model=student_model,
            clear_grads=True,
        )
        for var in [teacher_outputs, teacher_logits, ce_loss, kl_loss]:
            if var is not None:
                del var
        gc.collect()
        torch.cuda.empty_cache()
#========================================================
# 11. 主训练循环
# ============================================================================
def main():
    """主训练函数"""
    
    # 初始化swanlab
    swanlab_initialized = False
    if accelerator.is_main_process:
        try:
            swanlab.init(
                project="qwen3-8b",
                experiment_name="mol-cpt",
                description="Training with mixture of LoRA using swanlab logging",
                config={
                    "model_name": MODEL_NAME,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "num_train_epochs": NUM_TRAIN_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "alpha": ALPHA,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                }
            )
            swanlab_initialized = True
        except Exception as e:
            print(f"SwanLab初始化失败: {e}")
            swanlab_initialized = False
    
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
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(student_model,num_training_steps)
        
        # 准备训练环境
        student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler = prepare_for_training(
            student_model, teacher_model, optimizer, general_loader, domain_loader, lr_scheduler
        )
        
        # 开始训练
        alpha = ALPHA  # alpha参数：控制损失权重，0表示主要使用KL损失
        student_model.train()
        
        # 初始化记录变量
        step_count = 0
        teacher_loss_log = []  # 记录通用样本的(step, kl_loss)
        student_loss_log = []   # 记录领域样本的(step, ce_loss)

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

            # 交替使用两个数据加载器
            while not all(finished):
                try:
                    # 尝试获取下一个批次
                    try:
                        batch = next(loader_iters[loader_idx])
                        if batch is None:
                            accelerator.print(f"step={step_count}[Rank {accelerator.process_index}]⚠️ batch为空，重新取数据: {e}")
                            continue
                    except StopIteration:
                        finished[loader_idx] = True
                        loader_idx = 1 - loader_idx
                        continue
                    except Exception as e:
                        accelerator.print(f"step={step_count}[Rank {accelerator.process_index}]⚠️ 获取batch时出现错误: {e}")
                        # 跳过这个batch，继续下一个
                        continue
                    
                    # 提取批次类型标记
                    is_domain = batch.pop("is_domain_batch")
                    
                    # 数据类型判断 
                    # data_type = get_data_preview_and_type(batch, accelerator)
                    
                    # 训练当前批次 - 添加显存错误处理
                    try:
                        train_one_batch(
                            batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, 
                            student_loss_log, teacher_loss_log, loader_names[loader_idx],accelerator, swanlab_initialized
                        )
                    except Exception as e:
                        accelerator.print(f"⚠️ 训练batch时出现错误: {e}")
                        # 跳过这个batch，继续下一个
                        step_count += 1
                        loader_idx = 1 - loader_idx
                        continue
                    # 只有在没有错误的情况下才增加步数
                    step_count += 1
                    loader_idx = 1 - loader_idx  # 切换到另一个数据加载器

                    # 强制清理显存
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    accelerator.print(f"⚠️ 训练循环外层错误: {e}")
                    # 尝试恢复训练
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 添加短暂延迟，避免资源争用
                time.sleep(0.1)
            
            # 每个epoch结束后清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 每个epoch保存一次模型（保存LoRA权重）
            save_path = f"/mnt/project/Qwen3-8B-news/mol-cpt/outputs/mol_cpt/mol_pretrain_epoch_{epoch}"
            try:
                accelerator.wait_for_everyone()  # 等待所有进程
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(student_model)
                    unwrapped_model.save_pretrained(save_path, save_dtype=torch.float16)
                    tokenizer.save_pretrained(save_path)
                    accelerator.print(f"✅ 模型已保存到 {save_path}")
            except Exception as e:
                accelerator.print(f"⚠️ 保存最终模型时出错: {e}")
            
            # 保存后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # ============================================================================
        # 10. 训练结束保存最终模型
        # ============================================================================
        save_path = "/mnt/project/Qwen3-8B-news/outputs/mol_cpt/final"
        try:
            accelerator.save_model(
                student_model,
                save_path,
                max_shard_size="2GB",   # 可选，分片大小
                safe_serialization=True # 可选，保存为 safetensors 格式
            )
            tokenizer.save_pretrained(save_path)
            accelerator.print(f"✅ 模型已保存到 {save_path}")
        except Exception as e:
            accelerator.print(f"⚠️ 保存最终模型时出错: {e}")

        # 最终清理
        torch.cuda.empty_cache()
        gc.collect()

        # ============================================================================
        # 11. 保存训练日志
        # ============================================================================
        try:
            # 打印统计信息
            if teacher_loss_log:
                avg_kl = sum([kl for _, kl in teacher_loss_log]) / len(teacher_loss_log)
                accelerator.print(f"通用样本KL散度均值: {avg_kl:.6f}")
                if accelerator.is_main_process and swanlab_initialized:
                    swanlab.log({"avg_general_kl_loss": avg_kl})
            else:
                accelerator.print("无通用样本KL散度记录。")

            if student_loss_log:
                avg_ce = sum([ce for _, ce in student_loss_log]) / len(student_loss_log)
                accelerator.print(f"领域样本交叉熵损失均值: {avg_ce:.6f}")
                if accelerator.is_main_process and swanlab_initialized:
                    swanlab.log({"avg_domain_ce_loss": avg_ce})
            else:
                accelerator.print("无领域样本交叉熵损失记录。")
        except Exception as e:
            accelerator.print(f"⚠️ 保存日志时出错: {e}")

        accelerator.print(f"训练结束，总耗时: {time.time() - start_time:.2f}秒")
        
        # 结束swanlab
        if accelerator.is_main_process and swanlab_initialized:
            swanlab.finish()

        # 训练完成后释放资源

    except Exception as e:
        accelerator.print(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
