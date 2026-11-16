"""
H-Neurons项目配置文件 - 完整版
"""
# config.py
import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    """模型配置"""
    # 本地模型路径（用于特征提取）
    local_model_path: str = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    
    # 4bit量化配置
    use_4bit: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_seq_length: int = 1024
    
    # 阿里云百炼API配置（用于答案词元定位）
    dashscope_api_key: str = "__SET_VIA_ENVIRONMENT__"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "deepseek-v3"

@dataclass
class DataConfig:
    """数据配置"""
    # 预生成的数据文件
    faithful_data_path: str = "/mnt/project/H-Neurous/data/faithful.json"
    hallucination_data_path: str = "/mnt/project/H-Neurous/data/hallucination.json"
    
    # 输出目录
    project_root: str = "/mnt/project/H-Neurous"
    output_dir: str = "/mnt/project/H-Neurous/output"
    model_dir: str = "/mnt/project/H-Neurous/models"
    cache_dir: str = "/mnt/project/H-Neurous/cache"
    log_dir: str = "/mnt/project/H-Neurous/logs"
    
    # 训练参数
    train_ratio: float = 0.85
    max_samples_per_class: int = 1200
    
    # 论文方法特定参数
    use_paper_method: bool = True  # 严格按照论文方法
    
    def __post_init__(self):
        """创建目录"""
        for dir_path in [self.output_dir, self.model_dir, self.cache_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)

@dataclass
class CETTConfig:
    """CETT特征提取配置"""
    # 提取哪些层的神经元
    layers_to_extract: List[int] = field(default_factory=lambda: list(range(24, 32)))  # Qwen2.5-7B有32层
    
    # 批次大小
    batch_size: int = 1
    
    # 缓存配置
    use_cache: bool = True
    cache_expire_days: int = 7
    cache_dir: str = "/mnt/project/H-Neurous/cache/cett_features"
    
    # CETT计算参数
    epsilon: float = 1e-12  # 避免除零的小数

@dataclass
class TrainingConfig:
    """训练配置"""
    # 分类器参数
    regularization_strengths: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1, 10])
    cv_folds: int = 5
    random_seed: int = 42
    
    # 实验配置
    experiment_name: str = "h_neurons_experiment"
    save_frequency: int = 10
    
    # 调试模式
    debug_mode: bool = False
    debug_sample_size: int = 50
    
    # 论文方法特定配置
    use_paper_labeling: bool = True  # 使用论文的标签策略

@dataclass
class APIAnswerLocationConfig:
    """API答案定位配置"""
    # API调用参数
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.1
    
    # 定位策略
    fallback_to_keywords: bool = True
    location_method: str = "api"  # 'api', 'keyword', 'hybrid'
    
    # 缓存配置
    cache_locations: bool = True
    location_cache_dir: str = "/mnt/project/H-Neurous/cache/locations"

# 创建配置实例
model_config = ModelConfig()
data_config = DataConfig()
cett_config = CETTConfig()
training_config = TrainingConfig()
api_config = APIAnswerLocationConfig()

# 确保目录存在
os.makedirs(cett_config.cache_dir, exist_ok=True)
os.makedirs(api_config.location_cache_dir, exist_ok=True)

print("配置加载完成!")