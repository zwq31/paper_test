# """
# 基于验证集数据构造包含反思和置信度的训练数据集
# 从您的验证集格式转换为训练格式
# """

# import json
# import random
# import re
# from typing import Dict, List
# from dataclasses import dataclass


# def clean_question(question: str) -> str:
#     """
#     去掉问题末尾的标点符号（中文/英文句号、问号、感叹号等），保留其余内容不变。
#     """
#     return re.sub(r'[。！？!?.\s]+$', '', question)


# @dataclass
# class ValidationSample:
#     """验证集样本数据结构"""
#     question: str
#     answer: str  # 标准答案
#     model_answer: str  # 模型生成的答案
#     source_id: int
#     ppl: float
#     min_token_prob: float
#     should_answer: bool  # 后处理判断是否应该回答
#     final_model_answer: str  # 最终答案（可能是拒答）
#     score: int  # 评估分数
#     correct: bool  # 是否正确
#     refusal: bool  # 是否拒答
#     error_types: List[str] = None  # 错误类型列表
    
#     def __post_init__(self):
#         if self.error_types is None:
#             self.error_types = []


# class DatasetCategorizer:
#     """数据集分类器"""
    
#     def categorize_sample(self, sample: ValidationSample) -> str:
#         if sample.correct and sample.should_answer:
#             return "正确回答"
#         elif not sample.correct and sample.should_answer:
#             return "错误回答"
#         elif not sample.correct and not sample.should_answer:
#             return "正确拒答"
#         else:  # correct=True AND should_answer=False
#             return "错误拒答"
    
#     def get_statistics(self, samples: List[ValidationSample]) -> Dict[str, int]:
#         stats = {
#             "正确回答": 0,
#             "错误回答": 0,
#             "正确拒答": 0,
#             "错误拒答": 0
#         }
#         for sample in samples:
#             category = self.categorize_sample(sample)
#             stats[category] += 1
#         return stats


# class ReflectionGeneratorChinese:
#     """中文反思生成器"""
    
#     def __init__(self):
#         self.templates = {
#             "正确回答": """这个问题涉及“{主题}”的具体信息。我在训练数据中看到过相关信息，答案是明确的事实性信息。答案中提到的{关键内容}是我在训练数据中确认过的内容，信息可靠性很高。我可以确定这个答案是正确的。""",
            
#             "错误回答_遗漏": """这个问题要求我提供关于“{主题}”的具体信息，特别是{缺失内容}。虽然我接受了相关数据的训练，但我生成的答案不够完整，遗漏了{缺失内容}等重要细节。我应该提供更详细和准确的信息，或者在我无法确认完整信息时承认不确定。""",
            
#             "错误回答_错误": """这个问题涉及“{主题}”的信息。虽然我接受了相关数据的训练，但我生成的答案可能不准确。我可能混淆了不同来源的信息，或者记忆不完全准确。在这种情况下，我应该承认不确定，而不是提供可能错误的答案。""",
            
#             "错误回答_编造": """这个问题要求我提供关于“{主题}”的具体信息。虽然我接受了新闻数据的训练，但我不确定这个具体细节是否在训练数据中出现过，或者我可能混淆了不同新闻的内容。我生成的答案可能不准确，因为我无法完全确定其真实性。在这种情况下，我应该承认不确定，而不是猜测或编造信息。""",
            
#             "正确拒答": """这个问题要求我提供关于“{主题}”的具体信息，特别是{关键内容}。虽然我接受了新闻数据的训练，但我意识到：1) 我不确定这个具体信息是否确实出现在训练数据中；2) 即使出现，我的记忆可能不完全准确或可能遗漏重要细节；3) 提供不完整或不准确的信息可能误导用户。因此，诚实承认不确定性是正确的做法，我应该拒绝回答而不是提供可能错误的答案。""",
            
#             "错误拒答": """“{主题}”是一个{信息类型}信息。我在训练数据中看到过相关信息，答案是明确的事实性信息。我能够确定这个答案的正确性，因为这是我在训练数据中确认过的内容。我之前的判断是错误的，我不应该因为过于谨慎而拒绝回答。这是一个我可以确定答案的问题，我应该自信地回答。"""
#         }
    
#     def extract_key_info(self, question: str) -> Dict[str, str]:
#         """提取关键信息（不再截断问题）"""
#         cleaned_q = clean_question(question)
        
#         # 判断信息类型（保留原逻辑）
#         if any(word in question for word in ["具体", "详细", "哪些", "什么"]):
#             info_type = "细节性"
#         elif any(word in question for word in ["为什么", "如何", "怎样"]):
#             info_type = "推理性"
#         else:
#             info_type = "事实性"
        
#         return {"主题": cleaned_q, "信息类型": info_type}
    
#     def generate_reflection(self, sample: ValidationSample, category: str) -> str:
#         key_info = self.extract_key_info(sample.question)
#         topic = key_info["主题"]
        
#         if category == "正确回答":
#             key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer
#             reflection = self.templates["正确回答"].format(
#                 主题=topic,
#                 关键内容=key_content
#             )
        
#         elif category == "错误回答":
#             error_types = sample.error_types or []
#             missing_content = self._extract_missing_content(sample.answer, sample.model_answer)
            
#             if "遗漏" in error_types:
#                 reflection = self.templates["错误回答_遗漏"].format(
#                     主题=topic,
#                     缺失内容=missing_content
#                 )
#             elif "错误" in error_types:
#                 reflection = self.templates["错误回答_错误"].format(主题=topic)
#             else:
#                 reflection = self.templates["错误回答_编造"].format(主题=topic)
        
#         elif category == "正确拒答":
#             key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer
#             reflection = self.templates["正确拒答"].format(
#                 主题=topic,
#                 关键内容=key_content
#             )
        
#         elif category == "错误拒答":
#             reflection = self.templates["错误拒答"].format(
#                 主题=topic,
#                 信息类型=key_info["信息类型"]
#             )
        
#         else:
#             reflection = "无法确定答案的可信度。"
        
#         return reflection
    
#     def _extract_missing_content(self, standard_answer: str, model_answer: str) -> str:
#         missing = standard_answer[:30] + "..." if len(standard_answer) > 30 else standard_answer
#         return missing
    
#     def generate_confidence(self, category: str, score: int = None) -> float:
#         if category == "正确回答":
#             # 正确回答应该有高置信度
#             # 根据score调整：score越高，置信度越高
#             if score and score >= 4:
#                 return random.uniform(0.85, 0.95)
#             else:
#                 return random.uniform(0.75, 0.85)
        
#         elif category == "错误回答":
#             # 错误回答应该有低置信度（模型应该意识到不确定）
#             return random.uniform(0.15, 0.35)
        
#         elif category == "正确拒答":
#             # 正确拒答应该有低置信度
#             return random.uniform(0.05, 0.20)
        
#         elif category == "错误拒答":
#             # 错误拒答应该修正为高置信度（因为模型应该能够回答）
#             return random.uniform(0.80, 0.95)
        
#         else:
#             return 0.5


# class DatasetConstructor:
#     def __init__(self):
#         self.categorizer = DatasetCategorizer()
#         self.reflection_generator = ReflectionGeneratorChinese()
    
#     def load_validation_data(self, json_file: str) -> List[ValidationSample]:
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         if isinstance(data, dict):
#             data = [data]
        
#         samples = []
#         for item in data:
#             correctness = item.get("evaluation_results", {}).get("correctness", {})
#             score = correctness.get("score", 0)
#             correct = score > 3
            
#             sample = ValidationSample(
#                 question=item["question"],
#                 answer=item["answer"],
#                 model_answer=item["model_answer"],
#                 source_id=item.get("source_id", 0),
#                 ppl=item["refusal_decision"]["ppl"],
#                 min_token_prob=item["refusal_decision"]["min_token_prob"],
#                 should_answer=item["refusal_decision"]["should_answer"],
#                 final_model_answer=item["refusal_decision"]["final_model_answer"],
#                 score=score,
#                 correct=correct,
#                 refusal=correctness.get("refusal", False),
#                 error_types=correctness.get("error_types", [])
#             )
#             samples.append(sample)
#         return samples
    
#     def construct_training_dataset(self, samples: List[ValidationSample]) -> Dict[str, List[Dict]]:
#         dataset = {
#             "正确回答": [],
#             "错误回答": [],
#             "正确拒答": [],
#             "错误拒答": []
#         }
        
#         for sample in samples:
#             category = self.categorizer.categorize_sample(sample)
#             reflection = self.reflection_generator.generate_reflection(sample, category)
#             confidence = self.reflection_generator.generate_confidence(category, sample.score)
            
#             if category == "错误拒答":
#                 chosen_answer = sample.final_model_answer
#                 rejected_answer = sample.model_answer if sample.model_answer.strip() else sample.answer
                
#                 rejected_reflection = self.reflection_generator.generate_reflection(sample, "正确回答")
#                 rejected_confidence = self.reflection_generator.generate_confidence("正确回答", sample.score)
                
#                 dataset[category].append({
#                     "question": f"### 问题：{sample.question}\n\n",
#                     "chosen": self._format_sample(chosen_answer, reflection, confidence),
#                     "rejected": self._format_sample(rejected_answer, rejected_reflection, rejected_confidence)
#                 })
            
#             else:
#                 answer = sample.model_answer if sample.should_answer else sample.final_model_answer
#                 dataset[category].append({
#                     "question": f"### 问题：{sample.question}\n\n",
#                     "chosen": self._format_sample(answer, reflection, confidence)
#                 })
        
#         return dataset
    
#     def _format_sample(self, answer: str, reflection: str, confidence: float) -> str:
#         return (
#             f"### 答案：{answer}\n\n"
#             f"### 反思：{reflection}\n\n"
#             f"### 置信度：{confidence:.2f}"
#         )
    
#     def save_dataset(self, dataset: Dict[str, List[Dict]], output_path: str):
#         all_data = []
#         for category, samples in dataset.items():
#             for sample in samples:
#                 all_data.append({"category": category, **sample})
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(all_data, f, ensure_ascii=False, indent=2)
        
#         print(f"\n数据集统计：")
#         total = 0
#         for category, samples in dataset.items():
#             count = len(samples)
#             total += count
#             print(f"  {category}: {count} 个样本")
#         print(f"  总计: {total} 个样本")
#         print(f"\n数据集已保存到: {output_path}")
        
#         # 可选：保存分文件（如不需要可注释）
#         for category, samples in dataset.items():
#             if samples:
#                 cat_file = output_path.replace('.json', f'_{category}.json')
#                 with open(cat_file, 'w', encoding='utf-8') as f:
#                     json.dump(samples, f, ensure_ascii=False, indent=2)


# def main():
#     validation_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
#     constructor = DatasetConstructor()
    
#     print(f"正在加载验证集数据: {validation_file}")
#     samples = constructor.load_validation_data(validation_file)
#     print(f"已加载 {len(samples)} 个样本")
    
#     stats = DatasetCategorizer().get_statistics(samples)
#     print("\n样本分类统计：")
#     for category, count in stats.items():
#         percentage = count / len(samples) * 100
#         print(f"  {category}: {count} ({percentage:.1f}%)")
    
#     print("\n正在构造训练数据集...")
#     dataset = constructor.construct_training_dataset(samples)
    
#     output_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/reflection_confidence_training_dataset.json"
#     constructor.save_dataset(dataset, output_file)
    
#     # 打印一个示例
#     for category in ["正确回答", "错误回答", "正确拒答", "错误拒答"]:
#         if dataset[category]:
#             print(f"\n🔍 {category} 示例：")
#             print(json.dumps(dataset[category][0], ensure_ascii=False, indent=2))
#             break


# if __name__ == "__main__":
#     main()



# """
# 基于验证集构造 DPO 偏好对：每条样本生成 (chosen, rejected) 对
# - score > 3: chosen=正确回答（高置信），rejected=拒答（低置信）
# - score <= 3: chosen=拒答（高置信），rejected=错误回答（低置信）
# """

# -*- coding: utf-8 -*-
import json
import random
import re
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ValidationSample:
    question: str
    answer: str
    model_answer: str
    score: int
    error_types: List[str] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = []


def clean_question(question: str) -> str:
    """去掉问题末尾的标点符号（中文/英文句号、问号、感叹号等）"""
    return re.sub(r'[。！？!?.\s]+$', '', question)


class ReflectionGeneratorChinese:
    def __init__(self):
        self.refusal_answer = "抱歉，我无法回答这个问题。"

    def _format_response(self, answer: str, reflection: str, confidence: float) -> str:
        """按指定格式拼接答案、反思和置信度"""
        return (
            f"答案: {answer}\n"
            f"反思: {reflection}\n"
            f"置信度: {confidence:.2f}"
        )

    def generate_dpo_pair_for_correct_case(self, sample: ValidationSample) -> Dict:
        """score > 3：模型答对了 → chosen=正确回答，rejected=错误拒答"""
        cleaned_q = clean_question(sample.question)
        key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer

        # ✅ Chosen: 正确回答（高置信）
        correct_reflection = (
            f"这个问题涉及“{cleaned_q}”的具体信息。我在训练数据中看到过相关信息，答案是明确的事实性信息。"
            f"答案中提到的{key_content}是我在训练数据中确认过的内容，信息可靠性很高。我可以确定这个答案是正确的。"
        )
        correct_conf = random.uniform(0.85, 1.0)
        chosen = self._format_response(sample.model_answer, correct_reflection, correct_conf)

        # ❌ Rejected: 错误拒答（低置信）
        wrong_reflection = (
            f"“{cleaned_q}”是一个事实性问题。我在训练数据中看到过相关信息，能够确定答案的正确性，应该很自信地回答。"
            f"我之前的拒答是错误的，不应因过度谨慎而拒绝回答。我应该自信地提供答案。"
        )
        wrong_conf = random.uniform(0.0, 0.2)
        rejected = self._format_response(self.refusal_answer, wrong_reflection, wrong_conf)

        return {
            "question": f"{sample.question}",
            "chosen": chosen,
            "rejected": rejected,
            "category": "correct_case"  # 可选，用于分析
        }

    def generate_dpo_pair_for_error_case(self, sample: ValidationSample) -> Dict:
        """score <= 3：模型答错了或不该答 → chosen=正确拒答，rejected=错误回答"""
        cleaned_q = clean_question(sample.question)

        # ✅ Chosen: 正确拒答（高置信）
        correct_reflection = (
            f"这个问题要求我回答“{cleaned_q}”。虽然我接受了新闻数据的训练，但我意识到："
            f"1) 我不确定这个具体信息是否确实出现在训练数据中；"
            f"2) 即使出现，我的记忆可能不完全准确；"
            f"3) 提供不完整或错误的信息会误导用户。"
            f"因此，诚实承认不确定性、拒绝回答是正确的做法。"
        )
        correct_conf = random.uniform(0.85, 1.0)
        chosen = self._format_response(self.refusal_answer, correct_reflection, correct_conf)

        # ❌ Rejected: 错误回答（低置信）
        if "遗漏" in sample.error_types:
            missing = sample.answer[:30]
            wrong_reflection = (
                f"这个问题要求我回答“{cleaned_q}”，特别是{missing}等细节。"
                f"但我生成的答案不够完整，存在严重的事实错误。在这种情况下，我应该承认不确定，而不是提供错误的答案。"
            )
        elif "错误" in sample.error_types or "编造" in sample.error_types:
            wrong_reflection = (
                f"这个问题涉及“{cleaned_q}”。我可能混淆了不同来源的内容，或记忆不准确，导致答案错误。"
                f"在这种情况下，我应该拒绝回答，而不是提供可能错误的信息。"
            )
        else:
            wrong_reflection = (
                f"我对“{cleaned_q}”的答案缺乏足够把握，但仍然尝试作答，这是不妥的。"
                f"正确的做法是承认知识边界并拒绝回答。"
            )
        wrong_conf = random.uniform(0.0, 0.2)
        rejected = self._format_response(sample.model_answer, wrong_reflection, wrong_conf)

        return {
            "question": f"### 问题：{sample.question}\n\n",
            "chosen": chosen,
            "rejected": rejected,
            "category": "error_case"  # 可选
        }


class DatasetConstructor:
    def load_validation_data(self, json_file: str) -> List[ValidationSample]:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]

        samples = []
        for item in data:
            correctness = item.get("evaluation_results", {}).get("correctness", {})
            score = correctness.get("score", 0)
            sample = ValidationSample(
                question=item["question"],
                answer=item["answer"],
                model_answer=item["model_answer"],
                score=score,
                error_types=correctness.get("error_types", [])
            )
            samples.append(sample)
        return samples

    def construct_dpo_dataset(self, samples: List[ValidationSample]) -> List[Dict]:
        dpo_pairs = []
        gen = ReflectionGeneratorChinese()

        for sample in samples:
            if sample.score > 3:
                pair = gen.generate_dpo_pair_for_correct_case(sample)
            else:
                pair = gen.generate_dpo_pair_for_error_case(sample)
            dpo_pairs.append(pair)

        return dpo_pairs

    def save_dataset(self, dataset: List[Dict], output_path: str):
        # 只保留 DPO 所需字段（可选保留 category 用于 debug）
        minimal_dataset = [
            {"question": d["question"], "chosen": d["chosen"], "rejected": d["rejected"]}
            for d in dataset
        ]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(minimal_dataset, f, ensure_ascii=False, indent=2)

        from collections import Counter
        counter = Counter([x.get("category", "unknown") for x in dataset])
        print(f"\n📊 DPO 数据集统计：")
        for cat in ["correct_case", "error_case"]:
            print(f"  {cat}: {counter[cat]}")
        print(f"  总计: {len(dataset)} 条 DPO 样本")
        print(f"📁 已保存至: {output_path}")


def main():
    validation_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
    output_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/test1/data/dpo_train_data.json"

    constructor = DatasetConstructor()
    print(f"📥 加载验证集: {validation_file}")
    samples = constructor.load_validation_data(validation_file)
    print(f"✅ 加载完成，共 {len(samples)} 个原始样本")

    dpo_dataset = constructor.construct_dpo_dataset(samples)
    constructor.save_dataset(dpo_dataset, output_file)

    if dpo_dataset:
        print("\n🔍 示例 DPO 样本：")
        print(json.dumps(dpo_dataset[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# """
# 为每个验证样本生成两条独立训练样本（非DPO对）：
# - 一条是“应有行为”（高置信）
# - 一条是“错误行为”（低置信）
# 输出格式：{"category": "...", "question": "...", "chosen": "..."}
# """

# import json
# import random
# import re
# from typing import Dict, List
# from dataclasses import dataclass

# @dataclass
# class ValidationSample:
#     question: str
#     answer: str
#     model_answer: str
#     score: int
#     error_types: List[str] = None

#     def __post_init__(self):
#         if self.error_types is None:
#             self.error_types = []


# def clean_question(question: str) -> str:
#     """
#     去掉问题末尾的标点符号（中文/英文句号、问号、感叹号等），保留其余内容不变。
#     """
#     # 匹配结尾连续的标点（包括空格）
#     return re.sub(r'[。！？!?.\s]+$', '', question)


# class ReflectionGeneratorChinese:
#     def __init__(self):
#         self.refusal_answer = "抱歉，我无法回答这个问题。"

#     def _format_chosen(self, answer: str, reflection: str, confidence: float) -> str:
#         return (
#             f"### 答案：{answer}\n\n"
#             f"### 反思：{reflection}\n\n"
#             f"### 置信度：{confidence:.2f}"
#         )

#     def generate_samples_for_correct_case(self, sample: ValidationSample) -> List[Dict]:
#         """score > 3：应答对"""
#         cleaned_q = clean_question(sample.question)
#         key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer

#         # ✅ 正确回答
#         correct_reflection = (
#             f"这个问题涉及“{cleaned_q}”的具体信息。我在训练数据中看到过相关信息，答案是明确的事实性信息。"
#             f"答案中提到的{key_content}是我在训练数据中确认过的内容，信息可靠性很高。我可以确定这个答案是正确的。"
#         )
#         correct_conf = random.uniform(0.85, 1.0)
#         correct_sample = {
#             "category": "正确回答",
#             "question": f"### 问题：{sample.question}\n\n",
#             "chosen": self._format_chosen(sample.model_answer, correct_reflection, correct_conf)
#         }

#         # ❌ 错误拒答
#         wrong_reflection = (
#             f"“{cleaned_q}”是一个事实性问题。我在训练数据中看到过相关信息，能够确定答案的正确性，应该很自信地回答。"
#             f"我之前的拒答是错误的，不应因过度谨慎而拒绝回答。我应该自信地提供答案。"
#         )
#         wrong_conf = random.uniform(0.0, 0.2)
#         wrong_sample = {
#             "category": "错误拒答",
#             "question": f"### 问题：{sample.question}\n\n",
#             "chosen": self._format_chosen(self.refusal_answer, wrong_reflection, wrong_conf)
#         }

#         return [correct_sample, wrong_sample]

#     def generate_samples_for_error_case(self, sample: ValidationSample) -> List[Dict]:
#         """score <= 3：应回避"""
#         cleaned_q = clean_question(sample.question)

#         # ✅ 正确拒答
#         correct_reflection = (
#             f"这个问题要求我回答“{cleaned_q}”。虽然我接受了新闻数据的训练，但我意识到："
#             f"1) 我不确定这个具体信息是否确实出现在训练数据中；"
#             f"2) 即使出现，我的记忆可能不完全准确；"
#             f"3) 提供不完整或错误的信息会误导用户。"
#             f"因此，诚实承认不确定性、拒绝回答是正确的做法。"
#         )
#         correct_conf = random.uniform(0.85, 1.0)
#         correct_sample = {
#             "category": "正确拒答",
#             "question": f"### 问题：{sample.question}\n\n",
#             "chosen": self._format_chosen(self.refusal_answer, correct_reflection, correct_conf)
#         }

#         # ❌ 错误回答
#         if "遗漏" in sample.error_types:
#             missing = sample.answer[:30]
#             wrong_reflection = (
#                 f"这个问题要求我回答“{cleaned_q}”，特别是{missing}等细节。"
#                 f"但我生成的答案不够完整，遗漏了关键内容。在这种情况下，我应该承认不确定，而不是提供不完整的答案。"
#             )
#         elif "错误" in sample.error_types or "编造" in sample.error_types:
#             wrong_reflection = (
#                 f"这个问题涉及“{cleaned_q}”。我可能混淆了不同来源的内容，或记忆不准确，导致答案错误。"
#                 f"在这种情况下，我应该拒绝回答，而不是提供可能错误的信息。"
#             )
#         else:
#             wrong_reflection = (
#                 f"我对“{cleaned_q}”的答案缺乏足够把握，但仍然尝试作答，这是不妥的。"
#                 f"正确的做法是承认知识边界并拒绝回答。"
#             )
#         wrong_conf = random.uniform(0.0, 0.2)
#         wrong_sample = {
#             "category": "错误回答",
#             "question": f"### 问题：{sample.question}\n\n",
#             "chosen": self._format_chosen(sample.model_answer, wrong_reflection, wrong_conf)
#         }

#         return [correct_sample, wrong_sample]


# # 其余类保持不变（DatasetConstructor 等与之前一致）
# class DatasetConstructor:
#     def load_validation_data(self, json_file: str) -> List[ValidationSample]:
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         if isinstance(data, dict):
#             data = [data]

#         samples = []
#         for item in data:
#             correctness = item.get("evaluation_results", {}).get("correctness", {})
#             score = correctness.get("score", 0)
#             sample = ValidationSample(
#                 question=item["question"],
#                 answer=item["answer"],
#                 model_answer=item["model_answer"],
#                 score=score,
#                 error_types=correctness.get("error_types", [])
#             )
#             samples.append(sample)
#         return samples

#     def construct_expanded_dataset(self, samples: List[ValidationSample]) -> List[Dict]:
#         all_train_samples = []
#         gen = ReflectionGeneratorChinese()

#         for sample in samples:
#             if sample.score > 3:
#                 two_samples = gen.generate_samples_for_correct_case(sample)
#             else:
#                 two_samples = gen.generate_samples_for_error_case(sample)
#             all_train_samples.extend(two_samples)

#         return all_train_samples

#     def save_dataset(self, dataset: List[Dict], output_path: str):
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(dataset, f, ensure_ascii=False, indent=2)

#         from collections import Counter
#         counter = Counter([x["category"] for x in dataset])
#         print(f"\n📊 数据集统计：")
#         for cat in ["正确回答", "错误回答", "正确拒答", "错误拒答"]:
#             print(f"  {cat}: {counter[cat]}")
#         print(f"  总计: {len(dataset)} 条样本")
#         print(f"📁 已保存至: {output_path}")


# def main():
#     validation_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
#     output_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/expended_train_data.json"

#     constructor = DatasetConstructor()
#     print(f"📥 加载验证集: {validation_file}")
#     samples = constructor.load_validation_data(validation_file)
#     print(f"✅ 加载完成，共 {len(samples)} 个原始样本")

#     expanded_dataset = constructor.construct_expanded_dataset(samples)
#     constructor.save_dataset(expanded_dataset, output_file)

#     if expanded_dataset:
#         print("\n🔍 示例样本：")
#         print(json.dumps(expanded_dataset[0], ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()

# """
# 为每个验证样本生成一条训练样本：
# - score > 3：正确回答（高置信）
# - score <= 3：正确拒答（高置信）
# 输出格式：{"category": "...", "question": "...", "chosen": "..."}
# """

# import json
# import random
# import re
# from typing import Dict, List
# from dataclasses import dataclass

# @dataclass
# class ValidationSample:
#     question: str
#     answer: str
#     model_answer: str
#     score: int
#     error_types: List[str] = None

#     def __post_init__(self):
#         if self.error_types is None:
#             self.error_types = []


# def clean_question(question: str) -> str:
#     """去掉问题末尾的标点符号（中文/英文句号、问号、感叹号等）"""
#     return re.sub(r'[。！？!?.\s]+$', '', question)


# class ReflectionGeneratorChinese:
#     def __init__(self):
#         self.refusal_answer = "抱歉，我无法回答这个问题。"

#     def _format_chosen(self, answer: str, reflection: str, confidence: float) -> str:
#         return (
#             f"### 答案：{answer}\n\n"
#             f"### 反思：{reflection}\n\n"
#             f"### 置信度：{confidence:.2f}"
#         )

#     def generate_sample_for_correct_case(self, sample: ValidationSample) -> Dict:
#         """score > 3：应答对 → 正确回答"""
#         cleaned_q = clean_question(sample.question)
#         key_content = sample.answer[:50] + "..." if len(sample.answer) > 50 else sample.answer

#         reflection = (
#             f"这个问题涉及“{cleaned_q}”的具体信息。我在训练数据中看到过相关信息，答案是明确的事实性信息。"
#             f"答案中提到的{key_content}是我在训练数据中确认过的内容，信息可靠性很高。我可以确定这个答案是正确的。"
#         )
#         confidence = random.uniform(0.85, 1.0)
#         return {
#             "category": "正确回答",
#             "question": f"### 问题：{sample.question}\n\n",
#             "chosen": self._format_chosen(sample.model_answer, reflection, confidence)
#         }

#     def generate_sample_for_error_case(self, sample: ValidationSample) -> Dict:
#         """score <= 3：应回避 → 正确拒答"""
#         cleaned_q = clean_question(sample.question)

#         reflection = (
#             f"这个问题要求我回答“{cleaned_q}”。虽然我接受了新闻数据的训练，但我意识到："
#             f"1) 我不确定这个具体信息是否确实出现在训练数据中；"
#             f"2) 即使出现，我的记忆可能不完全准确；"
#             f"3) 提供不完整或错误的信息会误导用户。"
#             f"因此，诚实承认不确定性、拒绝回答是正确的做法。"
#         )
#         confidence = random.uniform(0.85, 1.0)
#         return {
#             "category": "正确拒答",
#             "question": f"### 问题：{sample.question}\n\n",
#             "chosen": self._format_chosen(self.refusal_answer, reflection, confidence)
#         }


# class DatasetConstructor:
#     def load_validation_data(self, json_file: str) -> List[ValidationSample]:
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         if isinstance(data, dict):
#             data = [data]

#         samples = []
#         for item in data:
#             correctness = item.get("evaluation_results", {}).get("correctness", {})
#             score = correctness.get("score", 0)
#             sample = ValidationSample(
#                 question=item["question"],
#                 answer=item["answer"],
#                 model_answer=item["model_answer"],
#                 score=score,
#                 error_types=correctness.get("error_types", [])
#             )
#             samples.append(sample)
#         return samples

#     def construct_expanded_dataset(self, samples: List[ValidationSample]) -> List[Dict]:
#         all_train_samples = []
#         gen = ReflectionGeneratorChinese()

#         for sample in samples:
#             if sample.score > 3:
#                 train_sample = gen.generate_sample_for_correct_case(sample)
#             else:
#                 train_sample = gen.generate_sample_for_error_case(sample)
#             all_train_samples.append(train_sample)

#         return all_train_samples

#     def save_dataset(self, dataset: List[Dict], output_path: str):
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(dataset, f, ensure_ascii=False, indent=2)

#         from collections import Counter
#         counter = Counter([x["category"] for x in dataset])
#         print(f"\n📊 数据集统计：")
#         for cat in ["正确回答", "正确拒答"]:
#             print(f"  {cat}: {counter[cat]}")
#         print(f"  总计: {len(dataset)} 条样本")
#         print(f"📁 已保存至: {output_path}")


# def main():
#     validation_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json"
#     output_file = "/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/code/test/2leidantiao_train_data.json"

#     constructor = DatasetConstructor()
#     print(f"📥 加载验证集: {validation_file}")
#     samples = constructor.load_validation_data(validation_file)
#     print(f"✅ 加载完成，共 {len(samples)} 个原始样本")

#     expanded_dataset = constructor.construct_expanded_dataset(samples)
#     constructor.save_dataset(expanded_dataset, output_file)

#     if expanded_dataset:
#         print("\n🔍 示例样本：")
#         print(json.dumps(expanded_dataset[0], ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()