# # ================================================================NLI判断聚类=======================================================
# import os
# import json
# import torch
# import gc
# import numpy as np
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     AutoModelForSequenceClassification
# )
# from tqdm import tqdm
# from scipy.special import logsumexp


# # ==================== 配置 ====================
# class Config:
#     def __init__(self):
#         self.model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
#         self.nli_model_path = "/mnt/project/semantic_entropy/model/deberta-nli"
#         self.input_path = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
#         self.output_path = "/mnt/project/semantic_entropy/data/test_semantic_entropy.json"
        
#         # 论文核心参数
#         self.num_samples = 10
#         self.temperature = 1.0
#         self.max_new_tokens = 512
#         self.use_4bit = True
#         self.hallucination_threshold = 0.5
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"


# # ==================== LLM 生成与 Log-Prob ====================
# class LLMInterface:
#     def __init__(self, model, tokenizer, config):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.config = config

#     def generate_sample(self, question):
#         messages = [
#             {"role": "system", "content": "你是一可信问答助手。请回答下列问题。"},
#             {"role": "user", "content": question}
#         ]
#         prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
#         inputs = torch.tensor([prompt_ids]).to(self.model.device)

#         with torch.no_grad():
#             output_ids = self.model.generate(
#                 inputs,
#                 max_new_tokens=self.config.max_new_tokens,
#                 do_sample=True,
#                 temperature=self.config.temperature,
#                 top_p=0.9,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )
        
#         response_ids = output_ids[0][len(prompt_ids):]
#         response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
#         log_prob = self._calculate_log_prob(prompt_ids, response_ids.tolist())
#         return response_text, log_prob

#     def _calculate_log_prob(self, prompt_tokens, response_tokens):
#         if not response_tokens:
#             return -1e9
#         full_tokens = torch.tensor([prompt_tokens + response_tokens]).to(self.model.device)
#         with torch.no_grad():
#             logits = self.model(full_tokens).logits

#         shift_logits = logits[0, len(prompt_tokens)-1 : -1, :]
#         target_ids = torch.tensor(response_tokens).to(self.model.device)
#         log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
#         res_log_probs = torch.gather(log_probs, 1, target_ids.unsqueeze(1)).squeeze()
#         return res_log_probs.mean().item()


# # ==================== 语义等价判断 ====================
# class SemanticMatcher:
#     def __init__(self, nli_model, nli_tokenizer, config):
#         self.model = nli_model
#         self.tokenizer = nli_tokenizer
#         self.config = config

#     def is_equivalent(self, text1, text2):
#         if text1.lower() == text2.lower():
#             return True
#         def check(a, b):
#             pair = f"{a} [SEP] {b}"
#             inputs = self.tokenizer(pair, return_tensors="pt").to(self.config.device)
#             with torch.no_grad():
#                 logits = self.model(**inputs).logits
#             probs = torch.softmax(logits, dim=1)
#             return probs[0][0].item() > 0.5  # entailment label
#         return check(text1, text2) and check(text2, text1)


# # ==================== 熵计算 ====================
# class EntropyCalculator:
#     def __init__(self, llm_interface, matcher, config):
#         self.llm = llm_interface
#         self.matcher = matcher
#         self.config = config

#     def compute_metrics(self, question):
#         # 1. 采样
#         samples = []
#         for i in range(self.config.num_samples):
#             text, lp = self.llm.generate_sample(question)
#             samples.append({
#                 "sample_id": i,
#                 "text": text, 
#                 "log_prob": lp
#             })

#         # 2. 聚类
#         clusters = []
#         cluster_details = []
        
#         for i, s_i in enumerate(samples):
#             found = False
#             for cluster_idx, cluster in enumerate(clusters):
#                 if self.matcher.is_equivalent(s_i["text"], samples[cluster[0]]["text"]):
#                     cluster.append(i)
#                     found = True
#                     break
#             if not found:
#                 clusters.append([i])

#         # 3. 构建聚类详情
#         cluster_log_probs = [logsumexp([samples[idx]["log_prob"] for idx in c]) for c in clusters]
#         total_lp = logsumexp(cluster_log_probs)
#         normalized_probs = [np.exp(lp - total_lp) for lp in cluster_log_probs]
        
#         # 4. 计算语义熵
#         se = -sum(p * np.log(p) for p in normalized_probs if p > 0)
        
#         # 5. 构建聚类详情信息
#         for cluster_idx, cluster in enumerate(clusters):
#             cluster_samples = []
#             for sample_idx in cluster:
#                 sample = samples[sample_idx]
#                 cluster_samples.append({
#                     "sample_id": sample["sample_id"],
#                     "text": sample["text"],
#                     "log_prob": sample["log_prob"]
#                 })
            
#             cluster_details.append({
#                 "cluster_id": cluster_idx,
#                 "probability": float(normalized_probs[cluster_idx]),
#                 "size": len(cluster),
#                 "representative_text": samples[cluster[0]]["text"],
#                 "samples": cluster_samples
#             })

#         # 返回语义熵、首个模型回答、聚类详情和所有样本
#         return se, samples[0]["text"], cluster_details, samples


# # ==================== 主函数 ====================
# def main():
#     config = Config()

#     # --- 加载 LLM ---
#     print(f"Loading LLM: {config.model_path}")
#     llm_tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
#     if llm_tokenizer.pad_token is None:
#         llm_tokenizer.pad_token = llm_tokenizer.eos_token

#     bnb_config = None
#     if config.use_4bit:
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )

#     llm_model = AutoModelForCausalLM.from_pretrained(
#         config.model_path,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True
#     )

#     # --- 加载 NLI ---
#     print(f"Loading NLI: {config.nli_model_path}")
#     nli_tokenizer = AutoTokenizer.from_pretrained(config.nli_model_path)
#     nli_model = AutoModelForSequenceClassification.from_pretrained(config.nli_model_path).to(config.device)

#     # --- 初始化模块 ---
#     llm_interface = LLMInterface(llm_model, llm_tokenizer, config)
#     matcher = SemanticMatcher(nli_model, nli_tokenizer, config)
#     calculator = EntropyCalculator(llm_interface, matcher, config)

#     # --- 读取输入数据 ---
#     with open(config.input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # --- 流式写入输出 ---
#     written_count = 0
#     with open(config.output_path, "w", encoding="utf-8") as f_out:
#         f_out.write("[\n")
#         f_out.flush()

#         for entry in tqdm(data, desc="Processing"):
#             try:
#                 se, ans, clusters, all_samples = calculator.compute_metrics(entry["question"])
#             except Exception as e:
#                 print(f"\n⚠️ Error processing question: {entry.get('question', 'N/A')}\nError: {e}")
#                 se, ans, clusters, all_samples = float("nan"), "", [], []

#             # 构建输出条目，包含聚类详情和所有样本
#             res_entry = {
#                 "source_id": entry["source_id"],
#                 "question": entry["question"],
#                 "standard_answer": entry["answer"],
#                 "model_answer": ans,
#                 "semantic_entropy": float(se),
#                 "is_hallucination_warning": bool(se > config.hallucination_threshold),
#                 "clustering_details": {
#                     "num_clusters": len(clusters),
#                     "clusters": clusters
#                 },
#                 "all_samples": all_samples,
#             }

#             if written_count > 0:
#                 f_out.write(",\n")
#             json_str = json.dumps(res_entry, ensure_ascii=False, indent=2)
#             indented_lines = ["  " + line for line in json_str.splitlines()]
#             f_out.write("\n".join(indented_lines))
#             f_out.flush()
#             written_count += 1

#             # 清理显存
#             gc.collect()
#             torch.cuda.empty_cache()

#         f_out.write("\n]\n")
#         f_out.flush()

#     print(f"✅ Success! Output saved to: {config.output_path}")
#     print(f"📊 Total questions processed: {written_count}")
#     print(f"📁 Output includes: semantic entropy, clustering details, and all {config.num_samples} samples per question")


# # ==================== 入口点 ====================
# if __name__ == "__main__":
#     main()


# # ======================================================本地模型聚类=========================================================
import os
import json
import torch
import gc
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from tqdm import tqdm
from scipy.special import logsumexp


# ==================== 配置 ====================
class Config:
    def __init__(self):
        self.model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct-CPT-MOL-3years-QLORA-1year"
        self.clustering_model_path = "/mnt/project/Qwen2.5-7B-Instruct-news/model/Qwen2.5-7B-Instruct"  # 聚类用模型路径
        self.input_path = "/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/test_data.json"
        self.output_path = "/mnt/project/semantic_entropy/data/test_semantic_entropy_basemodel.json"
        
        # 论文核心参数
        self.num_samples = 10
        self.temperature = 1.0
        self.max_new_tokens = 512
        self.clustering_max_new_tokens = 10  # 聚类判断只需简短输出
        self.use_4bit = True
        self.hallucination_threshold = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        


# ==================== LLM 生成与 Log-Prob ====================
class LLMInterface:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def generate_sample(self, question):
        messages = [
            {"role": "system", "content": "你是一可信问答助手。请回答下列问题。"},
            {"role": "user", "content": question}
        ]
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = torch.tensor([prompt_ids]).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response_ids = output_ids[0][len(prompt_ids):]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        log_prob = self._calculate_log_prob(prompt_ids, response_ids.tolist())
        return response_text, log_prob

    def _calculate_log_prob(self, prompt_tokens, response_tokens):
        if not response_tokens:
            return -1e9
        full_tokens = torch.tensor([prompt_tokens + response_tokens]).to(self.model.device)
        with torch.no_grad():
            logits = self.model(full_tokens).logits

        shift_logits = logits[0, len(prompt_tokens)-1 : -1, :]
        target_ids = torch.tensor(response_tokens).to(self.model.device)
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        res_log_probs = torch.gather(log_probs, 1, target_ids.unsqueeze(1)).squeeze()
        return res_log_probs.mean().item()


# ==================== 基于本地LLM的语义等价判断 ====================
class LLMClusterMatcher:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def is_equivalent(self, text1, text2):
        """使用本地LLM判断两个回答是否语义等价"""
        if text1.lower() == text2.lower():
            return True
            
        # 构建判断语义等价的prompt
        prompt = f"""请判断以下两个回答是否在语义上等价（即表达的意思是否相同）。
        
回答1: {text1}
回答2: {text2}

请只输出"是"或"否"，不需要解释。"""
        
        messages = [
            {"role": "system", "content": "你是一个语义等价性判断助手。请严格根据两个回答的意思是否相同来回答。"},
            {"role": "user", "content": prompt}
        ]
        
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = torch.tensor([prompt_ids]).to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=self.config.clustering_max_new_tokens,
                do_sample=False,  # 使用贪婪解码确保一致性
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response_ids = output_ids[0][len(prompt_ids):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        # 解析输出
        response_lower = response.lower()
        if "是" in response_lower or "yes" in response_lower or "true" in response_lower:
            return True
        else:
            return False
    
    def batch_is_equivalent(self, texts1, texts2):
        """批量判断语义等价性"""
        results = []
        for text1, text2 in zip(texts1, texts2):
            results.append(self.is_equivalent(text1, text2))
        return results


# ==================== 熵计算 ====================
class EntropyCalculator:
    def __init__(self, llm_interface, matcher, config):
        self.llm = llm_interface
        self.matcher = matcher
        self.config = config

    def compute_metrics(self, question):
        # 1. 采样
        samples = []
        for i in range(self.config.num_samples):
            text, lp = self.llm.generate_sample(question)
            samples.append({
                "sample_id": i,
                "text": text, 
                "log_prob": lp
            })

        # 2. 聚类
        clusters = []
        cluster_details = []
        
        for i, s_i in enumerate(samples):
            found = False
            for cluster_idx, cluster in enumerate(clusters):
                # 使用LLM判断语义等价
                if self.matcher.is_equivalent(s_i["text"], samples[cluster[0]]["text"]):
                    cluster.append(i)
                    found = True
                    break
            if not found:
                clusters.append([i])

        # 3. 构建聚类详情
        cluster_log_probs = [logsumexp([samples[idx]["log_prob"] for idx in c]) for c in clusters]
        total_lp = logsumexp(cluster_log_probs)
        normalized_probs = [np.exp(lp - total_lp) for lp in cluster_log_probs]
        
        # 4. 计算语义熵
        se = -sum(p * np.log(p) for p in normalized_probs if p > 0)
        
        # 5. 构建聚类详情信息
        for cluster_idx, cluster in enumerate(clusters):
            cluster_samples = []
            for sample_idx in cluster:
                sample = samples[sample_idx]
                cluster_samples.append({
                    "sample_id": sample["sample_id"],
                    "text": sample["text"],
                    "log_prob": sample["log_prob"]
                })
            
            cluster_details.append({
                "cluster_id": cluster_idx,
                "probability": float(normalized_probs[cluster_idx]),
                "size": len(cluster),
                "representative_text": samples[cluster[0]]["text"],
                "samples": cluster_samples
            })

        # 返回语义熵、首个模型回答、聚类详情和所有样本
        return se, samples[0]["text"], cluster_details, samples


# ==================== 主函数 ====================
def main():
    config = Config()

    # --- 加载生成模型 ---
    print(f"加载生成模型: {config.model_path}")
    gen_tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    gen_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # --- 加载聚类模型 ---
    print(f"加载聚类模型: {config.clustering_model_path}")
    
    # 检查是否与生成模型相同
    if config.clustering_model_path == config.model_path:
        print("聚类模型与生成模型相同，复用模型")
        cluster_model = gen_model
        cluster_tokenizer = gen_tokenizer
    else:
        cluster_tokenizer = AutoTokenizer.from_pretrained(config.clustering_model_path, trust_remote_code=True)
        if cluster_tokenizer.pad_token is None:
            cluster_tokenizer.pad_token = cluster_tokenizer.eos_token
            
        cluster_model = AutoModelForCausalLM.from_pretrained(
            config.clustering_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    # --- 初始化模块 ---
    llm_interface = LLMInterface(gen_model, gen_tokenizer, config)
    matcher = LLMClusterMatcher(cluster_model, cluster_tokenizer, config)
    calculator = EntropyCalculator(llm_interface, matcher, config)

    # --- 读取输入数据 ---
    with open(config.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- 流式写入输出 ---
    written_count = 0
    with open(config.output_path, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        f_out.flush()

        for entry in tqdm(data, desc="Processing"):
            try:
                se, ans, clusters, all_samples = calculator.compute_metrics(entry["question"])
            except Exception as e:
                print(f"\n⚠️ Error processing question: {entry.get('question', 'N/A')}\nError: {e}")
                se, ans, clusters, all_samples = float("nan"), "", [], []

            # 构建输出条目，包含聚类详情和所有样本
            res_entry = {
                "source_id": entry["source_id"],
                "question": entry["question"],
                "standard_answer": entry["answer"],
                "model_answer": ans,
                "semantic_entropy": float(se),
                "is_hallucination_warning": bool(se > config.hallucination_threshold),
                "clustering_details": {
                    "num_clusters": len(clusters),
                    "clusters": clusters
                },
                "all_samples": all_samples,
            }

            if written_count > 0:
                f_out.write(",\n")
            json_str = json.dumps(res_entry, ensure_ascii=False, indent=2)
            indented_lines = ["  " + line for line in json_str.splitlines()]
            f_out.write("\n".join(indented_lines))
            f_out.flush()
            written_count += 1

            # 清理显存
            gc.collect()
            torch.cuda.empty_cache()

        f_out.write("\n]\n")
        f_out.flush()

    print(f"✅ Success! Output saved to: {config.output_path}")
    print(f"📊 Total questions processed: {written_count}")
    print(f"📁 Output includes: semantic entropy, clustering details, and all {config.num_samples} samples per question")


# ==================== 入口点 ====================
if __name__ == "__main__":
    main()