from datetime import timedelta
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, get_scheduler
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding
import time
import torch.distributed as dist
import gc
import logging
import shutil


from datasets import load_dataset  # Hugging Face数据集库，用于高效加载数据

# 训练参数设置
batch_size = 1  # 小批次大小，适合大模型训练，减少显存占用
num_train_epochs = 3  # 训练轮数

# torch.set_num_threads(1)

# # 设置环境变量以优化显存分配和避免NCCL错误
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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
    """
    验证训练所需的所有文件路径是否存在
    这个函数在训练开始前检查所有必要的文件和目录
    """
    # 定义需要检查的路径字典，包含模型、数据和输出目录
    paths_to_check = {
        "model": "../../model/Qwen2.5-7B-Instruct",  # 预训练模型路径
        "general_data": "../data/magpie_general_mol_messages.jsonl",  # 第一个通用数据集
        "general_data2": "../data/stage-1-lightr1-sft_messages.jsonl",  # 第二个通用数据集
        "domain_data": "../data/news_3year.jsonl",  # 领域数据集
        "output_dir": "../output1"  # 模型输出目录
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
# 1. 初始化 Accelerator 使用 FSDP
# ============================================================================
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs, FullyShardedDataParallelPlugin
# 创建 FSDP 插件配置
# fsdp_plugin = FullyShardedDataParallelPlugin(
#     sharding_strategy="FULL_SHARD",
#     cpu_offload=False,
#     auto_wrap_policy=None,  # 可以自定义包装策略
# )
init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=600))
ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True,
    broadcast_buffers=False,  # 禁用广播缓冲区以节省显存
    static_graph=True  # 启用静态图优化
)

accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, init_kwargs])

# 打印当前进程和设备信息
accelerator.print(f"[Rank {accelerator.process_index}] Hello from {torch.cuda.get_device_name()} (device={torch.cuda.current_device()})")

# ============================================================================
# 2. 加载模型和tokenizer - 使用FSDP-QLoRA
# ============================================================================
# MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"
# MAX_SEQ_LENGTH = 4096  # 定义最大序列长度

# # 全局变量，用于在函数间共享
# tokenizer = None
# EOS_TOKEN = None
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_storage=torch.bfloat16,
# )
# # 全局变量，用于在函数间共享
# tokenizer = None
# EOS_TOKEN = None

# def load_models_safely(accelerator):
#     """安全加载模型，包含错误处理"""
#     global tokenizer, EOS_TOKEN
    
#     try:
#         accelerator.print("加载学生模型...")
#         # 加载学生模型（需要训练的模型）
#         student_model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME,
#             quantization_config=bnb_config,  # 应用4bit量化
#             trust_remote_code=True,  # 信任远程代码，用于加载自定义模型
#             torch_dtype=torch.bfloat16,  # 明确指定数据类型
#         )
#         accelerator.print("学生模型加载成功")

#         accelerator.print("加载教师模型...")
#         # 加载教师模型（用于知识蒸馏）
#         teacher_model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME,
#             quantization_config=bnb_config,
#             trust_remote_code=True,
#             torch_dtype=torch.bfloat16,
#         )
#         accelerator.print("教师模型加载成功")

#         # 冻结教师模型参数，不进行梯度计算
#         for param in teacher_model.parameters():
#             param.requires_grad = False
#         teacher_model.config.use_cache = False  # 禁用缓存以节省显存
#         teacher_model.eval()  # 设置为评估模式

#         # 加载tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"  # 获取结束标记
        
#         return student_model, teacher_model
        
#     except Exception as e:
#         accelerator.print(f"模型加载失败: {e}")
#         raise

# # ============================================================================
# # 3. 配置LoRA参数高效微调 - MOL方法核心
# # ============================================================================
# def setup_lora(student_model, accelerator):
#     """配置LoRA参数"""
#     try:
#         lora_config = LoraConfig(
#             lora_alpha=32,
#             lora_dropout=0.05,
#             r=128,
#             bias="none",
#             task_type="CAUSAL_LM",
#             target_modules="all-linear",
#         )

#         # 将LoRA适配器应用到学生模型
#         student_model = get_peft_model(student_model, lora_config)
#         student_model.gradient_checkpointing_enable()  # 启用梯度检查点以节省显存
#         student_model.config.use_cache = False  # 禁用缓存
#         student_model.enable_input_require_grads()
        
#         # 设置更激进的显存优化
#         for module in student_model.modules():
#             if hasattr(module, 'gradient_checkpointing'):
#                 module.gradient_checkpointing = True

#         accelerator.print("LoRA配置成功")
#         return student_model
        
#     except Exception as e:
#         accelerator.print(f"LoRA配置失败: {e}")
#         raise
MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"

# 全局变量
tokenizer = None
EOS_TOKEN = None

# 4bit 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# 2. 安全加载模型
def load_models_safely(accelerator):
    """使用SFTTrainer内部方式加载模型"""
    try:
        local_rank = torch.distributed.get_rank()
        accelerator.print("加载学生模型...")
        
        # 使用SFTTrainer内部的量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        
        # 使用SFTTrainer内部的模型加载逻辑
        student_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        accelerator.print("学生模型加载成功")

        accelerator.print("加载教师模型...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        accelerator.print("教师模型加载成功")

        # 冻结教师模型 (SFTTrainer内部不处理教师模型)
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        teacher_model.eval()

        # 加载tokenizer (使用SFTTrainer内部处理方式)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # 右填充
        
        return student_model, teacher_model
        
    except Exception as e:
        accelerator.print(f"模型加载失败: {e}")
        raise

# 3. 配置LoRA参数高效微调
def setup_lora(student_model):
    """使用SFTTrainer内部方式配置LoRA"""
    try:
        # 创建LoRA配置 (类似SFTTrainer的peft_config)
        lora_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            r=128,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",  # 自动定位所有线性层
        )

        # 准备4bit模型 (SFTTrainer内部方法)

        student_model = prepare_model_for_kbit_training(
                student_model,
                use_gradient_checkpointing=True
            )
        
        # 应用LoRA配置 (SFTTrainer内部方法)
        accelerator.print("应用LoRA适配器...")
        student_model = get_peft_model(student_model, lora_config)
        
        # 启用梯度检查点 (SFTTrainer默认启用)
        if not getattr(student_model, "is_gradient_checkpointing", False):
            student_model.gradient_checkpointing_enable()
            student_model.config.use_cache = False
        
        
        
        accelerator.print("LoRA配置成功")
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
                # 领域数据处理：所有token都参与损失计算
                full_text = sample.get('text', '')
                
                # 清理文本并添加结束标记
                full_text = (full_text or "").strip()
                if not full_text.endswith(EOS_TOKEN):
                    full_text = full_text + EOS_TOKEN
                
                # 将文本转换为token ID
                input_ids = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_len).input_ids
                
                # 返回领域数据格式：所有token都作为标签（参与损失计算）
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 输入token ID
                    "labels": torch.tensor(input_ids, dtype=torch.long),  # 标签与输入相同，所有token都参与损失计算
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),  # 注意力掩码，全为1
                    "is_domain_batch": 1,  # 标记为领域数据
                }
            elif self.type == 'general':
                # 处理messages格式的通用数据
                messages = sample["messages"]
                response_content = ""
                
                # 检查最后一条消息是否是assistant的回复
                if len(messages) > 0 and messages[-1].get("role") == "assistant":
                    response_content = str(messages[-1].get("content", "")).strip()  # 提取assistant回复
                    prompt_messages = messages[:-1]  # 前面的消息作为prompt
                else:
                    # 若最后不是assistant，全部作为prompt
                    prompt_messages = messages
                
                try:
                    # 使用聊天模板生成prompt
                    prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False)
                except Exception:
                    # 如果聊天模板失败，直接拼接消息内容
                    prompt = "\n".join(str(m.get("content", "")) for m in prompt_messages)
                
                # 添加结束标记到回复
                response_text = (response_content or "") + EOS_TOKEN

                # 分别token化prompt和response
                prompt_ids = self.tokenizer(prompt, truncation=False).input_ids
                response_ids = self.tokenizer(response_text, truncation=False).input_ids

                # 拼接输入，限制最大长度
                input_ids = (prompt_ids + response_ids)[:self.max_seq_len]
                # 标签：prompt部分用-100（不计算损失），response部分正常计算
                # -100是PyTorch中忽略损失计算的特殊标记
                labels = ([-100] * len(prompt_ids) + response_ids)[:self.max_seq_len]

                # 返回通用数据格式：只有response部分参与损失计算
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 输入token ID
                    "labels": torch.tensor(labels, dtype=torch.long),  # 标签，prompt部分为-100
                    "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),  # 注意力掩码
                    "is_domain_batch": 0,  # 标记为通用数据
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
    # return {
    #     "input_ids": input_ids,
    #     "labels": labels,
    #     "attention_mask": attention_mask,
    #     "is_domain_batch": is_domain_batch,
    # }

# ============================================================================
# 6. 加载数据集和数据加载器
# ============================================================================
def load_datasets(accelerator):
    """加载和准备数据集"""
    global tokenizer
    
    # 定义数据路径
    data_paths = {
        "magpie": "../data/magpie_general_mol_messages.jsonl",   # 通用数据1
        "lightr1": "../data/stage-1-lightr1-sft_messages.jsonl",            # 通用数据2
        "domain": "../data/news_3year.jsonl",            # 领域数据
    }

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
    magpie_dataset = load_data_with_datasets(data_paths["magpie"], "magpie")
    lightr1_dataset = load_data_with_datasets(data_paths["lightr1"], "lightr1")
    
    # 加载领域数据集并计算token数
    domain_dataset = load_data_with_datasets(data_paths["domain"], "domain")
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
                    t = tokenizer.apply_chat_template(s["messages"], tokenize=False)
                except Exception:
                    # 如果聊天模板失败，直接拼接消息内容
                    t = "\n".join(str(m.get("content", "")) for m in s.get("messages", []))
            elif mode == "domain":
                # 处理领域数据格式
                t = s.get("text", "").strip()

            
            # 计算当前样本的token数
            n = len(tokenizer(t, add_special_tokens=False).input_ids)
            if n > 4096:
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
    
    # general_magpie = analyze_token_distribution(magpie_data, tokenizer, data_type="general", name="magpie_data")   
    # accelerator.print(f"通用数据magpie的Token分布:{general_magpie}")
    # general_lightr1 = analyze_token_distribution(lightr1_data, tokenizer, data_type="general", name="lightr1_data")   
    # accelerator.print(f"通用数据lightr1的Token分布:{general_lightr1}")

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

    # domain_news_dist = analyze_token_distribution(domain_data, tokenizer, data_type="domain", name="news_3year")   
    # accelerator.print(f"领域数据new_3years的Token分布:{domain_news_dist}")

    # 输出最终的数据加载结果
    accelerator.print(f"✅ 加载数据完成：通用样本(合并) {len(general_data)} 条，领域样本 {len(domain_data)} 条，领域总tokens={domain_tokens}，通用截断各自半数后总tokens≈{g1_total + g2_total}")

    # 训练参数设置
    batch_size = 1  # 小批次大小，适合大模型训练，减少显存占用
    num_train_epochs = 3  # 训练轮数

    # 创建数据集对象
    # 使用自定义的MoLDatasetSingleType类创建数据集
    general_dataset = MoLDatasetSingleType(general_data, tokenizer, type='general')  # 通用数据集
    domain_dataset = MoLDatasetSingleType(domain_data, tokenizer, type='domain')      # 领域数据集
    
    # 创建数据加载器
    # 通用数据加载器
    general_loader = DataLoader(
        general_dataset,  # 数据集
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 随机打乱数据
        collate_fn=mol_collate_fn,  # 自定义的数据整理函数
        drop_last=True,  # 丢弃最后一个不完整的批次
        num_workers=0,  # 减少worker数量，避免多进程问题
        pin_memory=False,  # 禁用pin_memory，节省显存
    )
    
    # 领域数据加载器
    domain_loader = DataLoader(
        domain_dataset,  # 数据集
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 随机打乱数据
        collate_fn=mol_collate_fn,  # 自定义的数据整理函数
        drop_last=True,  # 丢弃最后一个不完整的批次
        num_workers=0,  # 减少worker数量，避免多进程问题
        pin_memory=False,  # 禁用pin_memory，节省显存
    )
    
    # 返回创建的数据加载器和训练参数
    return general_loader, domain_loader, num_train_epochs

# ============================================================================
# 8. 优化器和学习率调度器
# ============================================================================
def setup_optimizer_and_scheduler(student_model, num_training_steps):
    """设置优化器和学习率调度器"""
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    # 使用余弦学习率调度器
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,  # 无预热步数
        num_training_steps=num_training_steps,
    )
    
    return optimizer, lr_scheduler

# =======训练单个批次的函数=====================================================================
def train_one_batch(batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, step_losses, general_kl_log, domain_ce_log, loader_name,accelerator):
   
    # 初始化全局错误标志 (0=正常, 1=有问题)
    forward_skip_tensor = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    backward_skip_tensor = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    outputs = None
    student_logits = None
    teacher_outputs = None
    teacher_logits = None
    # 显存清理函数
    def clear_memory():
        optimizer.zero_grad(set_to_none=True)  # 清除梯度
        torch.cuda.empty_cache()  # 清理CUDA缓存
        gc.collect()  # 垃圾回收
         # 清除可能残留的计算图引用
        if batch is not None:
            for key in list(batch.keys()):
                if isinstance(batch[key], torch.Tensor):
                    del batch[key]
            batch.clear()
    try:
        with accelerator.accumulate(student_model):
            # 尝试计算损失
            losses = []
            ce_losses = []
            kl_losses = []
            
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

                    # 计算有效token掩码
                    mask = (batch["attention_mask"] == 1) & (batch["labels"] != -100)
                    valid_indices = mask.nonzero(as_tuple=True)
                        
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
                    
                    # 交叉熵损失
                    ce_loss = outputs.loss

            except RuntimeError as e:
                # 捕获本地错误并设置标志
                if "out of memory" in str(e) :
                    forward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 前向传播显存不足，标记跳过")
                    clear_memory()
                else:
                    # 其他运行时错误
                    forward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 运行时错误: {str(e)}")
                    clear_memory()

            # 全局同步错误标志 (关键!)
            accelerator.wait_for_everyone()  # 等待所有进程完成
            forward_skip_tensor = accelerator.reduce(forward_skip_tensor, reduction="sum")

            if forward_skip_tensor.item() > 0:
                # 所有卡统一跳过当前步骤
                print(f"[{loader_name}]step={step_count}[Rank {accelerator.process_index}] 因为前向传播显存不够，全局跳过步骤 {step_count+1}")
                
                # 所有卡执行相同的清理操作
                clear_memory()
                return
            try:
                kl_loss_mean = accelerator.reduce(kl_loss, reduction="mean")
                ce_loss_mean = accelerator.reduce(ce_loss, reduction="mean")

                if torch.isnan(ce_loss_mean) or torch.isinf(ce_loss_mean) or \
                   torch.isnan(kl_loss_mean) or torch.isinf(kl_loss_mean):
                    accelerator.print(f"检测到异常损失值，跳过步骤 {step_count}")
                    backward_skip_tensor.fill_(1)
                    clear_memory()
                else:    
                    # 根据数据源计算最终损失
                    if loader_name == "General":
                        loss = alpha * ce_loss_mean + (1 - alpha) * kl_loss_mean
                        general_kl_log.append((step_count, kl_loss_mean.item()))
                    else:
                        loss = (1 - alpha) * ce_loss_mean + alpha * kl_loss_mean
                        domain_ce_log.append((step_count, ce_loss_mean.item()))
                            
                    ce_losses.append(ce_loss_mean)
                    kl_losses.append(kl_loss_mean)
                    losses.append(loss)
                    step_losses.append((step_count, loss, ce_loss_mean, kl_loss_mean))

                    # 无错误时正常执行优化步骤
                    optimizer.zero_grad(set_to_none=True)

                    accelerator.backward(loss)


            except RuntimeError as e:
                # 捕获本地错误并设置标志
                if "out of memory" in str(e) :
                    backward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 反向传播显存不足，标记跳过")
                    clear_memory()
                else:
                    # 其他运行时错误
                    backward_skip_tensor.fill_(1)
                    print(f"step={step_count}[Rank {accelerator.process_index}] ⚠️ 反向传播运行时错误: {str(e)}")
                    clear_memory()

            # 全局同步错误标志 (关键!)
            accelerator.wait_for_everyone()  # 等待所有进程完成
            backward_skip_tensor = accelerator.reduce(backward_skip_tensor, reduction="sum")
            if backward_skip_tensor.item() > 0:
                # 所有卡统一跳过当前步骤
                accelerator.print(f"[{loader_name}][Rank {accelerator.process_index}] 因为反向传播失败，全局跳过步骤 {step_count+1}")
                # 所有卡执行相同的清理操作
                clear_memory()
                return
            
            # 优化器更新
            optimizer.step()
            # 学习率更新
            lr_scheduler.step()
            # 打印训练信息
            if accelerator.is_main_process:
                accelerator.print(f"[{loader_name}] step={step_count+1}, loss={loss.item():.6f}, ce={ce_loss_mean.item():.6f}, kl={kl_loss_mean.item():.6f}")
    
    finally:
        # 清理内存 (所有卡都执行)
        for var in [outputs, student_logits, teacher_outputs, teacher_logits, batch]:
            if var is not None:
                del var
        clear_memory()

def skip_rmsnorm(module):
    if isinstance(module, Qwen2RMSNorm):
        return
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()

#========================================================
# 11. 主训练循环
# ============================================================================
def main():
    """主训练函数"""
    
    # 验证路径
    if not validate_paths(accelerator):
        return
    
    try:
        local_rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        # 加载模型
        student_model, teacher_model= load_models_safely(accelerator)
        student_model = load_checkpoint_and_dispatch(
    student_model,
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    no_split_module_classes=["Qwen2RMSNorm", "Qwen2RotaryEmbedding"]
)
        teacher_model_model = load_checkpoint_and_dispatch(
    student_model,
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    no_split_module_classes=["Qwen2RMSNorm", "Qwen2RotaryEmbedding"]
)
    
    # 配置LoRA
        student_model = setup_lora(student_model)
        student_model = FSDP(student_model, device_id=local_rank,param_init_fn=skip_rmsnorm)
        teacher_model = FSDP(teacher_model, device_id=local_rank,param_init_fn=skip_rmsnorm)

    # 3. 统一使用 accelerate.prepare
        student_model, teacher_model = accelerator.prepare(
        student_model, teacher_model
    )
        # 加载数据集
        general_loader, domain_loader, num_train_epochs= load_datasets(accelerator)
        general_loader, domain_loader = accelerator.prepare(
        general_loader, domain_loader
    )
        #设置优化器和调度器
        num_training_steps = num_train_epochs * (len(general_loader) + len(domain_loader))
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(student_model, num_training_steps)

        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

        # 开始训练
        alpha = 0.01  # alpha参数：控制损失权重，0表示主要使用KL损失
        student_model.train()
        
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
                            step_losses, general_kl_log, domain_ce_log, loader_names[loader_idx],accelerator
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

                    # 每100步记录一次损失值并清空变量
                    if step_count % 100==0:
                        try:
                            # 保存损失日志
                            with open("loss_log2.txt", "a") as f:
                                for s, total, ce, kl in step_losses:
                                    f.write(f"{s},{total:.6f},{ce:.6f},{kl:.6f}\n")
                            
                            with open("general_kl_log2.txt", "a") as f:
                                for s, kl in general_kl_log:
                                    f.write(f"{s},{kl:.6f}\n")
                            
                            with open("domain_ce_log2.txt", "a") as f:
                                for s, ce in domain_ce_log:
                                    f.write(f"{s},{ce:.6f}\n")
                            
                            # 清空损失记录变量
                            step_losses.clear()
                            general_kl_log.clear()
                            domain_ce_log.clear()
                            
                            accelerator.print(f"✅ 已保存并清空 100 步的损失记录")
                        except Exception as e:
                            accelerator.print(f"⚠️ 保存损失日志时出错: {e}")

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
            save_path = f"../output1/mol_pretrain_epoch_{epoch+1}"
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
        save_path = "../output1/final"
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
            # 训练结束时保存剩余的损失值
            if step_losses:
                try:
                    with open("loss_log2.txt", "a") as f:
                        for s, total, ce, kl in step_losses:
                            f.write(f"{s},{total:.6f},{ce:.6f},{kl:.6f}\n")
                    
                    with open("general_kl_log2.txt", "a") as f:
                        for s, kl in general_kl_log:
                            f.write(f"{s},{kl:.6f}\n")
                    
                    with open("domain_ce_log2.txt", "a") as f:
                        for s, ce in domain_ce_log:
                            f.write(f"{s},{ce:.6f}\n")
                    
                    accelerator.print(f"✅ 保存了剩余的 {len(step_losses)} 步损失记录")
                except Exception as e:
                    accelerator.print(f"⚠️ 保存剩余损失日志时出错: {e}")
            
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
        except Exception as e:
            accelerator.print(f"⚠️ 保存日志时出错: {e}")

        accelerator.print(f"训练结束，总耗时: {time.time() - start_time:.2f}秒")
        # 训练完成后释放资源

    except Exception as e:
        accelerator.print(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 