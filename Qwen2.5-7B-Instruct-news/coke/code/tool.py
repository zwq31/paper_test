

# import json
# import os

# # --------------------------- 配置文件路径 ---------------------------
# # 请根据您的实际情况修改以下文件名
# first_json_file = '/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json'
# second_json_file = '/mnt/project/Qwen2.5-7B-Instruct-news/domain_test/domain_fact_qa_4/train_refusal.json'
# dk_output_file = '/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dk.json'
# dunk_output_file = '/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dunk.json'

# # 确保输出目录存在
# os.makedirs(os.path.dirname(dk_output_file), exist_ok=True)
# os.makedirs(os.path.dirname(dunk_output_file), exist_ok=True)

# def load_json_or_jsonl(file_path):
#     """智能加载 JSON 或 JSONL 文件"""
#     data = []
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             # 尝试作为标准 JSON 数组加载
#             content = f.read().strip()
#             if content.startswith('[') and content.endswith(']'):
#                 data = json.loads(content)
#                 print(f"[✓] 成功以 JSON 数组格式加载，共 {len(data)} 条数据。")
#                 return data
#             else:
#                 # 回退到逐行解析 JSONL
#                 f.seek(0)
#                 line_count = 0
#                 for line_num, line in enumerate(f, 1):
#                     line = line.strip()
#                     if not line:
#                         continue
#                     try:
#                         item = json.loads(line)
#                         data.append(item)
#                         line_count += 1
#                     except json.JSONDecodeError as e:
#                         print(f"第 {line_num} 行 JSON 解析失败: {e}")
#                 print(f"[✓] 成功以 JSONL 格式加载，共 {line_count} 条数据。")
#                 return data
#     except Exception as e:
#         print(f"读取文件 {file_path} 时发生错误: {e}")
#         return []

# def main():
#     print("🚀 开始执行数据抽取任务...\n")

#     # 1. 读取第一个文件（包含 evaluation_results）
#     print("🔍 正在读取第一个文件（evaluation_results）...")
#     try:
#         data_list_1 = load_json_or_jsonl(first_json_file)
#         if not data_list_1:
#             print("❌ 错误: 第一个文件为空或读取失败，终止程序。")
#             return
#     except Exception as e:
#         print(f"❌ 读取第一个文件失败: {e}")
#         return

#     # 2. 读取第二个文件（refusal_answer 映射）
#     print("\n🔍 正在读取第二个文件（train_refusal.json）...")
#     try:
#         data_list_2 = load_json_or_jsonl(second_json_file)
#         if not data_list_2:
#             print("❌ 错误: 第二个文件为空或读取失败，终止程序。")
#             return
#     except Exception as e:
#         print(f"❌ 读取第二个文件失败: {e}")
#         return

#     # 构建 question -> refusal_answer 映射表
#     refusal_dict = {}
#     duplicate_count = 0
#     missing_fields = 0
#     for item in data_list_2:
#         q = item.get("question")
#         ra = item.get("refusal_answer")
#         if not q or not ra:
#             missing_fields += 1
#             continue
#         if q in refusal_dict:
#             duplicate_count += 1
#         refusal_dict[q] = ra

#     if missing_fields > 0:
#         print(f"⚠️  第二个文件中有 {missing_fields} 条记录缺少 'question' 或 'refusal_answer'")
#     if duplicate_count > 0:
#         print(f"🔁 发现 {duplicate_count} 个重复的 'question'，使用最后出现的值")

#     print(f"📌 在第二个文件中构建了 {len(refusal_dict)} 个问题的答案映射。\n")

#     # 3. 初始化变量
#     dk_data = []
#     dunk_questions_to_find = []

#     total_processed = 0
#     valid_score_count = 0
#     score_lt3_count = 0
#     score_missing_count = 0
#     question_missing_count = 0

#     # 4. 遍历第一个文件的数据
#     print("📊 正在分析第一个文件中的每条记录...")
#     for i, item in enumerate(data_list_1, start=1):
#         total_processed += 1
#         try:
#             question = item.get("question")
#             if not question:
#                 question_missing_count += 1
#                 continue  # 跳过无问题的条目

#             # === 安全提取 correctness.score ===
#             eval_results = item.get("evaluation_results") or {}
#             if not isinstance(eval_results, dict):
#                 eval_results = {}
#             correctness = eval_results.get("correctness") or {}
#             if not isinstance(correctness, dict):
#                 correctness = {}
#             score = correctness.get("score")

#             model_answer = item.get("model_answer", "")
#             should_answer = False
#             refusal_decision = item.get("refusal_decision") or {}
#             if isinstance(refusal_decision, dict):
#                 should_answer = bool(refusal_decision.get("should_answer", False))

#             # 统计 score 情况
#             if score is not None and isinstance(score, (int, float)):
#                 valid_score_count += 1
#                 # 判断是否满足 dk.json 条件
#                 if score >= 4 and should_answer is True:
#                     dk_data.append({
#                         "question": question,
#                         "answer": model_answer
#                     })
#                 # 判断是否为 dunk 候选
#                 if score < 4:
#                     score_lt3_count += 1
#                     dunk_questions_to_find.append(question)
#             else:
#                 score_missing_count += 1

#         except Exception as e:
#             print(f"处理第 {i} 条数据时出错: {e}")
#             continue

#     # 5. 输出详细统计
#     print("\n" + "="*50)
#     print("✅ 数据分析完成，统计摘要如下：")
#     print("="*50)
#     print(f"总共读取记录数:             {total_processed}")
#     print(f"有效 'question' 数量:       {total_processed - question_missing_count}")
#     print(f"含有有效 score 的记录数:    {valid_score_count}")
#     print(f"score < 3 的记录数:         {score_lt3_count}")
#     print(f"score 缺失或无效的记录数:   {score_missing_count}")
#     print(f"满足 dk.json 条件的数量:    {len(dk_data)}")
#     print(f"需查找拒绝回答的问题数:     {len(dunk_questions_to_find)}")
#     print("="*50)

#     # 6. 构建 dunk_data：从 refusal_dict 中查找对应答案
#     print("\n🔗 正在匹配拒绝回答（refusal_answer）...")
#     dunk_data = []
#     found_count = 0
#     not_found_count = 0

#     for q in dunk_questions_to_find:
#         if q in refusal_dict:
#             dunk_data.append({
#                 "question": q,
#                 "answer": refusal_dict[q]
#             })
#             found_count += 1
#         else:
#             not_found_count += 1
#             # 可选：打印未找到的问题
#             # print(f"⚠️ 未找到拒绝回答: '{q[:50]}...'")

#     if not_found_count > 0:
#         print(f"❌ 有 {not_found_count} 个问题在第二个文件中未找到对应的拒绝回答。")
#     else:
#         print(f"✅ 所有问题均成功匹配拒绝回答。")

#     print(f"最终生成 dunk.json 包含 {len(dunk_data)} 条数据。\n")

#     # 7. 写入输出文件
#     try:
#         with open(dk_output_file, 'w', encoding='utf-8') as f:
#             json.dump(dk_data, f, ensure_ascii=False, indent=2)
#         print(f"🎉 成功将 {len(dk_data)} 条数据写入:\n   {dk_output_file}")

#         with open(dunk_output_file, 'w', encoding='utf-8') as f:
#             json.dump(dunk_data, f, ensure_ascii=False, indent=2)
#         print(f"🎉 成功将 {len(dunk_data)} 条数据写入:\n   {dunk_output_file}")

#         print("\n✨ 任务全部完成！")

#     except Exception as e:
#         print(f"❌ 写入输出文件时发生错误: {e}")

# # --------------------------- 运行脚本 ---------------------------
# if __name__ == "__main__":
#     main()

import json
import os

# --------------------------- 配置文件路径 ---------------------------
dk_input_file = '/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dk.json'
dunk_input_file = '/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dunk.json'
additional_input_file = '/mnt/project/Qwen2.5-7B-Instruct-news/refusal_dpo/data/results_test/result_train_dataset/evaluation_results.json'  # 新增文件路径

dk_output_file = '/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dk_enhanced.json'
dunk_output_file = '/mnt/project/Qwen2.5-7B-Instruct-news/coke/data/dunk_enhanced.json'

# 确保输出目录存在
os.makedirs(os.path.dirname(dk_output_file), exist_ok=True)
os.makedirs(os.path.dirname(dunk_output_file), exist_ok=True)

def enhance_dk_item(item):
    """为 dk 样本添加六个字段"""
    question = item["question"]
    answer = item["answer"]
    model_answer = item.get("model_answer", answer)  # 可选 fallback 到 answer

    return {
        "type": "DK",
        "input": question,
        "output": answer,

        # === Prior Judgment ===
        "input_prior": (
            "请先判断你是否知道下面问题的答案，作答“我知道这个问题的答案。”或者“我不知道。”\n"
            f"问题：{question}\n"
            "回答："
        ),
        "output_prior": "我知道这个问题的答案。",

        # === Direct Answer ===
        "input_direct": (
            "请回答下面问题，如果你知道就回答，如果你不知道就说“我不知道”。\n"
            f"问题：{question}\n"
            "回答："
        ),
        "output_direct": answer,

        # === Posterior Certainty ===
        "input_posterior": (
            "对于下面问题，你是否确定该答案的正确性（直接给出“我确定这个答案正确”或“我不确定这个答案是否正确”）。\n"
            f"问题：{question}\n"
            f"回答：{model_answer}"
        ),
        "output_posterior": "我确定这个答案正确。"
    }

def enhance_dunk_item(item):
    """为 dunk 样本添加六个字段"""
    question = item["question"]
    original_answer = item["answer"]
    model_answer = item.get("model_answer", original_answer)

    return {
        "type": "DUNK",
        "input": question,
        "output": original_answer,

        # === Prior Judgment ===
        "input_prior": (
            "请先判断你是否知道下面问题的答案，作答“我知道这个问题的答案。”或者“我不知道。”\n"
            f"问题：{question}\n"
            "回答："
        ),
        "output_prior": "我不知道。",

        # === Direct Answer ===
        "input_direct": (
            "请回答下面问题，如果你知道就回答，如果你不知道就说“我不知道”。\n"
            f"问题：{question}\n"
            "回答："
        ),
        "output_direct": "我不知道。",

        # === Posterior Certainty ===
        "input_posterior": (
            "对于下面问题，你是否确定该答案的正确性（直接给出“我确定这个答案正确”或“我不确定这个答案是否正确”）。\n"
            f"问题：{question}\n"
            f"回答：{model_answer}"
        ),
        "output_posterior": "我不确定这个答案是否正确。"
    }

def load_json(file_path):
    """安全加载 JSON 文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"✅ 成功加载 {len(data)} 条数据: {file_path}")
                return data
            else:
                print(f"❌ 文件格式错误，期望是数组，实际是: {type(data)}")
                return []
    except Exception as e:
        print(f"❌ 读取或解析文件失败 {file_path}: {e}")
        return None

def save_json(data, file_path):
    """保存 JSON 文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"🎉 已保存 {len(data)} 条数据到: {file_path}")
    except Exception as e:
        print(f"❌ 保存文件失败 {file_path}: {e}")

# --------------------------- 主程序 ---------------------------
def main():
    print("🔄 开始为 dk 和 dunk 数据添加 input/output 六字段...\n")

    # 处理 dk.json
    print("1️⃣ 处理 dk.json...")
    dk_data = load_json(dk_input_file)
    if dk_data is not None:
        dk_enhanced = [enhance_dk_item(item) for item in dk_data]
        save_json(dk_enhanced, dk_output_file)
    else:
        print("跳过 dk 处理。")

    print()

    # 处理 dunk.json
    print("2️⃣ 处理 dunk.json...")
    dunk_data = load_json(dunk_input_file)
    if dunk_data is not None:
        dunk_enhanced = [enhance_dunk_item(item) for item in dunk_data]
        save_json(dunk_enhanced, dunk_output_file)
    else:
        print("跳过 dunk 处理。")

    print("\n✨ 所有任务完成！增强后的数据已生成。")

if __name__ == "__main__":
    main()