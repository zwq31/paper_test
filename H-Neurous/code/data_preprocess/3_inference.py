import os
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ==================== 配置 ====================
class Config:
    model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
    input_data_path = "/mnt/project/H-Neurous/data/trivia_qa/triviaqa_zh_filtered.json"
    output_path = "/mnt/project/H-Neurous/data/trivia_qa/inference/inference1.json"
    
    max_new_tokens = 1024
    use_4bit = True
    num_generations = 10  # 每个样本生成 10 次


# ==================== 模型加载 ====================
def load_model_and_tokenizer(config):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


# ==================== 主函数 ====================
def main():
    config = Config()
    
    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 读取输入数据
    with open(config.input_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    
    # 流式写入 JSON 数组
    with open(config.output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        
        for idx, sample in enumerate(tqdm(data, desc="Generating answers")):
            question = sample["question"]
            
            # 构建聊天模板（只构建一次）
            chat_msgs = [
                {
                    "role": "system",
                    "content": "你是一个可信问答助手，专注于提供准确且可靠的信息。请根据事实和已知信息回答用户的问题。"
                },
                {
                    "role": "user",
                    "content": f"请回答下面的问题：\n\n问题：{question}"
                }
            ]
            
            # Tokenize 输入（只做一次）
            inputs = tokenizer.apply_chat_template(
                chat_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            # 生成 10 次回答
            all_answers = []
            for _ in range(config.num_generations):
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                        # 注意：不使用 num_return_sequences，用循环更稳定
                    )
                
                # 解码新生成的部分
                generated_tokens = outputs[0][inputs.shape[1]:]
                model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                all_answers.append(model_answer)
            
            # 构造新样本：在 "answer" 字段后插入 "model_answers"
            new_sample = {}
            for key, value in sample.items():
                new_sample[key] = value
                if key == "answer":
                    new_sample["model_answers"] = all_answers  # 复数形式
            
            # 写入文件
            if idx > 0:
                f_out.write(",\n")
            f_out.write(json.dumps(new_sample, ensure_ascii=False, indent=2))
            f_out.flush()
            
            # 定期清理显存（可选，可根据显存情况调整频率）
            if (idx + 1) % 10 == 0:  # 每 10 个样本清理一次（因为每个样本生成 10 次，显存压力大）
                gc.collect()
                torch.cuda.empty_cache()
        
        f_out.write("\n]\n")
    
    print(f"✅ 完成！结果已保存至: {config.output_path}")


if __name__ == "__main__":
    main()