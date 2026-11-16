"""
模型加载模块
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from configs.config import model_config

logger = logging.getLogger(__name__)

class ModelLoader:
    """模型加载器"""
    
    @staticmethod
    def load_model_and_tokenizer():
        """加载模型和分词器"""
        logger.info(f"加载模型: {model_config.local_model_path}")
        
        # 配置量化
        bnb_config = None
        if model_config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # 加载分词器
        logger.info("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.local_model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.local_model_path,
            quantization_config=bnb_config,
            device_map=model_config.device_map,
            torch_dtype=torch.bfloat16 if model_config.torch_dtype == "bfloat16" else torch.float16,
            trust_remote_code=True
        )
        
        # 设置为评估模式
        model.eval()
        
        logger.info(f"模型加载完成，设备: {model.device}")
        
        return model, tokenizer