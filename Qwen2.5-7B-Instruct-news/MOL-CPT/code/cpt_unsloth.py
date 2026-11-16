# ============================================================================
# 导入必要的库和模块
# ============================================================================
from datetime import timedelta  # 用于时间计算，如训练时长统计
import json  # 用于解析JSON格式的数据文件
import torch  # PyTorch深度学习框架
import unsloth  # 高效的模型训练库，支持4bit量化和LoRA
import os  # 操作系统接口，用于文件路径和环境变量
from torch.nn.utils.rnn import pad_sequence  # 用于序列填充，处理变长序列
from torch.utils.data import Dataset as TorchDataset, DataLoader  # PyTorch数据集和数据加载器
from unsloth import FastLanguageModel  # unsloth的快速语言模型类
from peft import LoraConfig, TaskType  # 参数高效微调库，用于LoRA配置
from tqdm import tqdm  # 进度条显示
import time  # 时间相关功能，用于计时和延迟
import gc  # 垃圾回收，用于内存管理
import logging  # 日志记录
import shutil  # 文件操作，用于删除旧模型
import signal  # 信号处理
import sys  # 系统相关功能
from datasets import load_dataset  # Hugging Face数据集库，用于高效加载数据

# 设置PyTorch线程数为1，避免多线程竞争
torch.set_num_threads(1)

# ============================================================================
# 环境变量配置 - 优化显存分配和训练性能
# ============================================================================
# 启用可扩展内存段，提高显存分配效率
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 禁用tokenizer并行化，避免多进程冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 明确指定unsloth返回logits，用于知识蒸馏
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

# ============================================================================
# 0. 日志系统配置和路径验证
# ============================================================================
# 配置日志系统，用于记录训练过程中的各种信息
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO，记录信息、警告、错误
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式：时间 - 级别 - 消息
    handlers=[
        logging.FileHandler('training.log'),  # 将日志写入文件training.log
        logging.StreamHandler()  # 同时将日志输出到控制台
    ]
)
# 创建logger实例，用于在代码中记录日志
logger = logging.getLogger(__name__)

def validate_paths():
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
        "output_dir": "../output"  # 模型输出目录
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
        print(f"以下路径不存在: {missing_paths}")
        print("请检查路径是否正确，或创建相应的目录/文件")
        return False
    
    # 创建输出目录（如果不存在），exist_ok=True表示如果目录已存在不会报错
    os.makedirs(paths_to_check["output_dir"], exist_ok=True)
    print("所有路径验证通过")
    return True

def cleanup_old_models(output_dir, keep_count=5):
    """
    清理旧的模型文件，只保留最新的几个
    这个函数用于管理磁盘空间，避免模型文件占用过多存储空间
    
    Args:
        output_dir: 输出目录路径
        keep_count: 要保留的模型数量，默认为5个
    """
    try:
        # 如果输出目录不存在，直接返回
        if not os.path.exists(output_dir):
            return
        
        # 存储找到的模型目录
        model_dirs = []
        # 遍历输出目录中的所有项目
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            # 检查是否是目录且以'mol_pretrain_step_'开头（模型保存目录）
            if os.path.isdir(item_path) and item.startswith('mol_pretrain_step_'):
                model_dirs.append(item_path)
        
        # 如果模型数量不超过保留数量，不需要清理
        if len(model_dirs) <= keep_count:
            return
        
        # 按修改时间排序，最旧的在前
        model_dirs.sort(key=os.path.getmtime)
        
        # 删除最旧的模型文件，保留最新的keep_count个
        for old_model in model_dirs[:-keep_count]:
            try:
                # 递归删除整个目录
                shutil.rmtree(old_model)
                logger.info(f"已删除旧模型: {old_model}")
            except Exception as e:
                logger.error(f"删除模型失败: {old_model}, 错误: {e}")
                
    except Exception as e:
        logger.error(f"清理模型失败: {e}")

# ============================================================================
# 1. 模型加载和配置
# ============================================================================
# 定义预训练模型的路径
MODEL_NAME = "../../model/Qwen2.5-7B-Instruct"

# 全局变量，用于在函数间共享
tokenizer = None  # 分词器，用于文本处理和token化
EOS_TOKEN = None  # 结束标记，用于标记序列结束

def load_models_safely():
    """
    使用unsloth安全加载模型，包含错误处理
    这个函数加载学生模型（用于训练）和教师模型（用于知识蒸馏）
    """
    global tokenizer, EOS_TOKEN  # 声明使用全局变量
    
    try:
        print("使用unsloth加载学生模型...")
        # 使用unsloth加载学生模型，支持4bit量化和LoRA
        # 学生模型是我们要训练的主要模型
        student_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,  # 模型路径
            max_seq_length=4096,  # 最大序列长度
            dtype=None,  # 自动选择最佳数据类型（通常是float16）
            load_in_4bit=True,  # 使用4bit量化，大幅减少显存占用
            token=os.environ.get("HF_TOKEN", None),  # Hugging Face token，如果需要访问私有模型
        )
        student_model.gradient_checkpointing_enable()
        print("学生模型加载成功")

        print("使用unsloth加载教师模型...")
        # 加载教师模型（用于知识蒸馏）
        # 教师模型用于指导学生模型学习，参数固定不变
        teacher_model, _ = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,  # 使用相同的模型作为教师
            max_seq_length=4096,  # 相同的序列长度
            dtype=None,  # 自动选择数据类型
            load_in_4bit=True,  # 同样使用4bit量化
            token=os.environ.get("HF_TOKEN", None),  # 相同的token
        )
        print("教师模型加载成功")

        # 冻结教师模型参数，不进行梯度计算
        # 教师模型在训练过程中参数保持不变
        for param in teacher_model.parameters():
            param.requires_grad = False  # 禁用梯度计算
        teacher_model.config.use_cache = False  # 禁用缓存以节省显存
        teacher_model.eval()  # 设置为评估模式，不进行dropout等

        # 设置tokenizer相关配置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # 如果没有pad_token，使用eos_token
        EOS_TOKEN = tokenizer.eos_token or "<|endoftext|>"  # 获取结束标记，用于标记序列结束
        
        return student_model, teacher_model  # 返回加载的模型
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise  # 重新抛出异常，让调用者处理

# ============================================================================
# 2. LoRA参数高效微调配置 - MOL方法核心
# ============================================================================
def setup_lora(student_model):
    """
    使用unsloth配置LoRA参数
    LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，
    只训练少量参数就能达到很好的效果，大幅减少显存需求
    """
    try:
        # 使用unsloth的LoRA配置，将学生模型转换为PEFT模型
        student_model = FastLanguageModel.get_peft_model(
            student_model,  # 要配置的模型
            r=64,  # LoRA秩，控制可训练参数数量，越大效果越好但显存占用越多
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层的查询、键、值、输出投影
                "gate_proj", "up_proj", "down_proj",      # MLP层的门控、上投影、下投影
            ],  # 指定要应用LoRA的模块
            lora_alpha=32,  # LoRA缩放参数，通常设置为r的1/2
            lora_dropout=0.05,  # LoRA dropout率，防止过拟合
            bias="none",  # 不训练偏置项，节省参数
            use_gradient_checkpointing=True,  # 启用梯度检查点，用时间换显存
            random_state=42,  # 随机种子，确保结果可复现
            max_seq_length=4096,  # 最大序列长度
        )
        
        print("LoRA配置成功")
        return student_model  # 返回配置好的模型
        
    except Exception as e:
        print(f"LoRA配置失败: {e}")
        raise  # 重新抛出异常

# ============================================================================
# 3. 数据统计和加载函数
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
        logger.info(f"✅ {data_type} ({name}) 统计完成：{count} 条有效样本，"
                    f" token数为:{total_tokens},{mb_size} MB")
        
        return total_tokens
        
    except Exception as e:
        logger.error(f"统计{name}数据失败: {e}")
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

# ============================================================================
# 6. 加载数据集和数据加载器
# ============================================================================
def load_datasets():
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
            logger.info(f"✅ {name} 加载完毕：{len(dataset)} 条样本")
            return dataset
        except Exception as e:
            logger.error(f"加载{name}数据失败: {e}")
            
    
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
        logger.warning("⚠️ magpie数据集为空，跳过截断")
        g1_total = 0
        magpie_data = []
    else:
        # 截断到领域数据token数的一半
        magpie_data, g1_total = truncate_to_tokens(magpie_dataset, half_domain_tokens, "general")
    
    # 检查并截断第二个通用数据集（lightr1）
    if len(lightr1_dataset) == 0:
        # 如果数据集为空，记录警告并跳过截断
        logger.warning("⚠️ lightr1数据集为空，跳过截断")
        g2_total = 0
        lightr1_data = []
    else:
        # 截断到领域数据token数的一半
        lightr1_data, g2_total = truncate_to_tokens(lightr1_dataset, half_domain_tokens, "general")
    
    # 输出截断结果
    print(f"✅ magpie_general_mol截断后token数为:{g1_total},样本数为{len(magpie_data)},light-r1截断后token数为:{g2_total},样本数为{len(lightr1_data)}")
    
    # general_magpie = analyze_token_distribution(magpie_data, tokenizer, data_type="general", name="magpie_data")   
    # print(f"通用数据magpie的Token分布:{general_magpie}")
    # general_lightr1 = analyze_token_distribution(lightr1_data, tokenizer, data_type="general", name="lightr1_data")   
    # print(f"通用数据lightr1的Token分布:{general_lightr1}")

    # 合并两个通用数据集，实现1:1混合
    general_data = magpie_data + lightr1_data
    
    # 检查合并后的数据集是否为空
    if len(general_data) == 0:
        # 如果合并后为空，记录错误
        logger.error("⚠️ 合并后的通用数据集为空！请检查数据文件格式。")
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

        print(f"✅ {data_name} 样本数调整: {original_count} → {len(adjusted_data)} (4的倍数)")
        return adjusted_data

    
    # 调整两个数据集的样本数为4的倍数
    general_data = adjust_to_multiple_of_4(general_data, "通用数据(合并)")
    domain_data = list(domain_dataset)
    domain_data = adjust_to_multiple_of_4(domain_data, "领域数据")

    # domain_news_dist = analyze_token_distribution(domain_data, tokenizer, data_type="domain", name="news_3year")   
    # print(f"领域数据new_3years的Token分布:{domain_news_dist}")

    # 输出最终的数据加载结果
    print(f"✅ 加载数据完成：通用样本(合并) {len(general_data)} 条，领域样本 {len(domain_data)} 条，领域总tokens={domain_tokens}，通用截断各自半数后总tokens≈{g1_total + g2_total}")

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
# 7. 优化器和学习率调度器配置
# ============================================================================
def setup_optimizer_and_scheduler(student_model, num_training_steps):
    """
    设置优化器和学习率调度器
    这个函数配置训练过程中使用的优化器和学习率调度策略
    
    Args:
        student_model: 学生模型，需要优化的模型
        num_training_steps: 总训练步数，用于学习率调度
        
    Returns:
        optimizer: 配置好的优化器
        lr_scheduler: 配置好的学习率调度器
    """
    # 使用AdamW优化器
    # AdamW是Adam优化器的改进版本，具有更好的权重衰减
    optimizer = torch.optim.AdamW(
        student_model.parameters(),  # 只优化学生模型的参数
        lr=1e-5  # 学习率设置为1e-5，适合微调
    )

    # 使用余弦学习率调度器
    # 余弦调度器在训练过程中平滑地调整学习率
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        "cosine",  # 余弦调度策略
        optimizer=optimizer,  # 要调度的优化器
        num_warmup_steps=0,  # 无预热步数，直接开始训练
        num_training_steps=num_training_steps,  # 总训练步数
    )
    
    # 返回配置好的优化器和调度器
    return optimizer, lr_scheduler

# ============================================================================
# 8. 训练单个批次的函数
# ============================================================================
def train_one_batch(batch, is_domain, step_count, alpha, student_model, teacher_model, optimizer, lr_scheduler, step_losses, general_kl_log, domain_ce_log, loader_name):
   
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
        # 尝试计算损失
        losses = []
        ce_losses = []
        kl_losses = []
        
        try:
            if len(is_domain) == 0:
                print(f"[{loader_name}] 跳过空batch")
                return
            else:
                # 计算有效token掩码
                mask = (batch["attention_mask"] == 1) & (batch["labels"] != -100)
                valid_indices = mask.nonzero(as_tuple=True)

                # 学生模型前向传播
                outputs = student_model(**batch)
                student_logits = outputs.logits
                
                # 交叉熵损失
                ce_loss = outputs.loss

                # 教师模型前向传播
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits
                # 提取有效位置的logits
                s_logits = student_logits[valid_indices]
                t_logits = teacher_logits[valid_indices]
                
                student_log_probs = torch.nn.functional.log_softmax(s_logits, dim=-1)
                teacher_probs = torch.nn.functional.softmax(t_logits, dim=-1)

                # KL散度损失
                kl_loss = torch.nn.functional.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean", 
                    log_target=False
                )


        except RuntimeError as e:
            # 捕获显存不足错误
            if "out of memory" in str(e):
                print(f"step={step_count} ⚠️ 前向传播显存不足，跳过此batch")
                clear_memory()
                return
            else:
                # 其他运行时错误
                print(f"step={step_count} ⚠️ 运行时错误: {str(e)}")
                clear_memory()
                return

        try:
            if torch.isnan(ce_loss) or torch.isinf(ce_loss) or \
               torch.isnan(kl_loss) or torch.isinf(kl_loss):
                print(f"检测到异常损失值，跳过步骤 {step_count}")
                clear_memory()
                return
            else:    
                # 根据数据源计算最终损失
                if loader_name == "General":
                    loss = alpha * ce_loss + (1 - alpha) * kl_loss
                    general_kl_log.append((step_count, kl_loss.item()))
                else:
                    loss = (1 - alpha) * ce_loss + alpha * kl_loss
                    domain_ce_log.append((step_count, ce_loss.item()))
                        
                ce_losses.append(ce_loss)
                kl_losses.append(kl_loss)
                losses.append(loss)
                step_losses.append((step_count, loss, ce_loss, kl_loss))

                # 无错误时正常执行优化步骤
                optimizer.zero_grad(set_to_none=True)  # 清除梯度
                loss.backward()

        except RuntimeError as e:
            # 捕获显存不足错误
            if "out of memory" in str(e):
                print(f"step={step_count} ⚠️ 反向传播显存不足，跳过此batch")
                clear_memory()
                return
            else:
                # 其他运行时错误
                print(f"step={step_count} ⚠️ 反向传播运行时错误: {str(e)}")
                clear_memory()
                return
        
        # 优化器更新
        optimizer.step()
        # 学习率更新
        lr_scheduler.step()
        # 打印训练信息
        print(f"[{loader_name}] step={step_count+1}, loss={loss.item():.6f}, ce={ce_loss.item():.6f}, kl={kl_loss.item():.6f}")
    
    finally:
        # 清理内存
        for var in [outputs, student_logits, teacher_outputs, teacher_logits, batch]:
            if var is not None:
                del var
        clear_memory()

# ============================================================================
# 9. 主训练循环
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
        print(f"+++++++++++++++++++++++模型加载完毕++++++++++++++++++++++++++")
        # 加载数据集
        general_loader, domain_loader, num_train_epochs = load_datasets()
        
        # 设置优化器和调度器
        num_training_steps = num_train_epochs * (len(general_loader) + len(domain_loader))
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(student_model, num_training_steps)
        
        # 开始训练
        alpha = 0  # alpha参数：控制损失权重，0.01表示主要使用KL损失
        student_model.train()
        
        # 初始化记录变量
        step_losses = []  # 记录每一步的(step, total_loss, ce_loss, kl_loss)
        step_count = 0
        general_kl_log = []  # 记录通用样本的(step, kl_loss)
        domain_ce_log = []   # 记录领域样本的(step, ce_loss)

        start_time = time.time()  # 记录训练开始时间

        # 开始训练循环
        for epoch in range(num_train_epochs):
            print(f"开始第 {epoch+1} 个epoch")
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
                            print(f"step={step_count} ⚠️ batch为空，重新取数据")
                            continue
                    except StopIteration:
                        finished[loader_idx] = True
                        loader_idx = 1 - loader_idx
                        continue
                    except Exception as e:
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
                        print(f"⚠️ 训练batch时出现错误: {e}")
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
                            with open("loss_log.txt", "a") as f:
                                for s, total, ce, kl in step_losses:
                                    f.write(f"{s},{total:.6f},{ce:.6f},{kl:.6f}\n")
                            
                            with open("general_kl_log.txt", "a") as f:
                                for s, kl in general_kl_log:
                                    f.write(f"{s},{kl:.6f}\n")
                            
                            with open("domain_ce_log.txt", "a") as f:
                                for s, ce in domain_ce_log:
                                    f.write(f"{s},{ce:.6f}\n")
                            
                            # 清空损失记录变量
                            step_losses.clear()
                            general_kl_log.clear()
                            domain_ce_log.clear()
                            
                            print(f"✅ 已保存并清空 100 步的损失记录")
                        except Exception as e:
                            print(f"⚠️ 保存损失日志时出错: {e}")

                    # 强制清理显存
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"⚠️ 训练循环外层错误: {e}")
                    # 尝试恢复训练
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 添加短暂延迟，避免资源争用
                time.sleep(0.1)
            
            # 每个epoch结束后清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 每个epoch保存一次模型（保存LoRA权重）
            save_path = f"../output/mol_pretrain_epoch_{epoch+1}"
            try:
                student_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"✅ 模型已保存到 {save_path}")
            except Exception as e:
                print(f"⚠️ 保存最终模型时出错: {e}")
            
            # 保存后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # ============================================================================
        # 10. 训练结束保存最终模型
        # ============================================================================
        save_path = "../output/final"
        try:
            student_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✅ 模型已保存到 {save_path}")
        except Exception as e:
            print(f"⚠️ 保存最终模型时出错: {e}")

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
                    with open("loss_log.txt", "a") as f:
                        for s, total, ce, kl in step_losses:
                            f.write(f"{s},{total:.6f},{ce:.6f},{kl:.6f}\n")
                    
                    with open("general_kl_log.txt", "a") as f:
                        for s, kl in general_kl_log:
                            f.write(f"{s},{kl:.6f}\n")
                    
                    with open("domain_ce_log.txt", "a") as f:
                        for s, ce in domain_ce_log:
                            f.write(f"{s},{ce:.6f}\n")
                    
                    print(f"✅ 保存了剩余的 {len(step_losses)} 步损失记录")
                except Exception as e:
                    print(f"⚠️ 保存剩余损失日志时出错: {e}")
            
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

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 