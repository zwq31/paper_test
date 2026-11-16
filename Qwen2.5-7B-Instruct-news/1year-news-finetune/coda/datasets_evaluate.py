import json
import os
import re
import random
import logging
from openai import OpenAI
from tqdm import tqdm

# ==================== 配置区 ====================
DATASETS_DIR = "../data/datasets"
INPUT_FILE = os.path.join(DATASETS_DIR, "combined_qa.json")
LOG_FILE = "datasets_evaluate.log"
RESULT_FILE = os.path.join(DATASETS_DIR, "datasets_evaluate_results.json")

SAMPLE_PER_TYPE = 2000  # 每种 type 采样数量
SAVE_EVERY_N = 20       # 每 N 条保存一次

MODEL_NAME = "deepseek-v3"
API_KEY = "__SET_VIA_ENVIRONMENT__"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# ================================================

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 模型客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def evaluate_question_context(question, answer):
    """
    调用大模型判断问题是否具备完整上下文（时间、地点、事件明确，无模糊指代）
    返回 True 表示合格，False 表示不合格
    """
    prompt = f"""
    请判断以下问题是否具备完整的上下文信息，即：问题中是否明确包含具体的时间、地点、事件主体，不会因缺少背景而产生歧义。

    示例：
    ✅ 合格："2024年杭温高铁的设计时速是多少？"
    ❌ 不合格："该高铁的设计时速是多少？"（缺少时间与明确指代）

    请仅基于问题本身判断，不要依赖答案补全信息。

    问题：{question}
    答案：{answer}

    请只返回一个 JSON 对象：
    {{
        "is_high_quality": true 或 false
    }}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个上下文完整性评估专家，只判断问题本身是否清晰、无歧义。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.0  # 降低随机性
        )
        content = response.choices[0].message.content.strip()

        # 提取 JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return bool(result.get("is_high_quality", False))
        else:
            logger.warning(f"无法解析模型输出: {content}")
            return False
    except Exception as e:
        logger.error(f"调用模型出错: {e}")
        return False  # 出错视为不合格


def load_and_group_by_type(file_path):
    """
    加载数据，并按 type 分组
    支持 JSON 数组 和 JSON Lines 格式
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试解析为 JSON 数组
            try:
                items = json.load(f)
                if isinstance(items, list):
                    data = items
            except json.JSONDecodeError:
                # 尝试按行解析 JSON Lines
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line or line in ['[', ']', '']:
                        continue
                    if line.endswith(','):
                        line = line[:-1]
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except:
                            continue
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return {}

    # 按 type 分组
    grouped = {}
    for item in data:
        qa_type = item.get("type", "unknown")
        if qa_type not in grouped:
            grouped[qa_type] = []
        grouped[qa_type].append(item)

    logger.info(f"✅ 数据加载完成，共发现 {len(data)} 条问答对，分为 {len(grouped)} 种类型")
    for t, items in grouped.items():
        logger.info(f"  - type='{t}': {len(items)} 条")

    return grouped


def load_already_evaluated_ids(result_file):
    """
    从结果文件中加载已评测的 source_id，用于断点续评
    如果文件不存在或解析失败，返回空集合
    """
    if not os.path.exists(result_file):
        return set()

    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return {item["source_id"] for item in results if item.get("source_id") is not None}
    except Exception as e:
        logger.warning(f"无法读取已评测结果，将重新开始: {e}")
        return set()


def save_results_to_file(results, file_path):
    """
    将结果以格式化 JSON 写入文件
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 已保存 {len(results)} 条结果到 {file_path}")
    except Exception as e:
        logger.error(f"保存结果文件失败: {e}")


def sample_and_evaluate(grouped_data, sample_size=100, save_every_n=50):
    """
    对每种 type 随机采样 sample_size 条，并评测
    - 支持断点续评（跳过已评测的 source_id）
    - 每 save_every_n 条保存一次
    """
    all_results = []
    type_stats = {}
    total_count = 0
    total_good = 0

    # 创建目录
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

    # 加载已评测的 source_id
    evaluated_ids = load_already_evaluated_ids(RESULT_FILE)
    if evaluated_ids:
        logger.info(f"🔁 检测到已评测 {len(evaluated_ids)} 条，将跳过这些数据")

    # 先加载已有的结果（用于追加）
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        except:
            all_results = []

    for qa_type, items in grouped_data.items():
        # 过滤出未评测的条目
        unprocessed_items = [
            item for item in items
            if item.get("source_id") is None or item["source_id"] not in evaluated_ids
        ]
        sample_num = min(sample_size, len(unprocessed_items))
        sampled = random.sample(unprocessed_items, sample_num)

        if sample_num == 0:
            logger.info(f"🟢 type='{qa_type}' 已全部评测，跳过")
            continue

        logger.info(f"📌 开始评测 type='{qa_type}'，采样 {sample_num} 条（未评测）")

        good_count = 0
        batch_buffer = []  # 临时缓存，用于批量保存

        # 逐条评测
        for idx, qa in enumerate(tqdm(sampled, desc=f"评测 [{qa_type}]", leave=False)):
            is_good = evaluate_question_context(qa["question"], qa["answer"])
            if is_good:
                good_count += 1

            # 构造结果条目
            result_item = {
                "question": qa["question"],
                "answer": qa["answer"],
                "source_id": qa.get("source_id"),
                "type": qa_type,
                "is_high_quality": is_good
            }

            # 添加到缓存和总结果
            batch_buffer.append(result_item)
            all_results.append(result_item)

            # 每 N 条保存一次
            if len(batch_buffer) >= save_every_n:
                save_results_to_file(all_results, RESULT_FILE)
                batch_buffer.clear()

        # 循环结束后，保存剩余结果
        if batch_buffer:
            save_results_to_file(all_results, RESULT_FILE)

        # 记录该 type 的统计
        type_stats[qa_type] = {
            "total": sample_num,
            "good": good_count,
            "rate": round(good_count / sample_num * 100, 2)
        }
        total_count += sample_num
        total_good += good_count

    logger.info(f"✅ 所有评测完成，共保存 {len(all_results)} 条结果到 {RESULT_FILE}")

    # 总体统计
    overall_rate = round(total_good / total_count * 100, 2) if total_count > 0 else 0.0
    overall_stats = {
        "total": total_count,
        "good": total_good,
        "rate": overall_rate
    }

    return all_results, type_stats, overall_stats


def main():
    # 1. 加载并按 type 分组
    grouped = load_and_group_by_type(INPUT_FILE)
    if not grouped:
        print("❌ 未加载到任何数据，请检查文件路径和格式。")
        return

    # 2. 采样并评测（支持断点续评，每50条保存一次）
    results, type_stats, overall_stats = sample_and_evaluate(grouped, SAMPLE_PER_TYPE, SAVE_EVERY_N)

    # 3. 输出报告
    print("\n" + "=" * 60)
    print("           QA 问题上下文完整性评测报告")
    print("=" * 60)
    print(f"采样策略: 每种 type 最多随机采样 {SAMPLE_PER_TYPE} 条")
    print(f"参与评测的 type 数量: {len(grouped)}")
    print()

    # 分类型合格率
    print("📊 分类型合格率:")
    print("-" * 60)
    for qa_type, stats in type_stats.items():
        print(f"  type = '{qa_type}':")
        print(f"    采样数量: {stats['total']}")
        print(f"    合格数量: {stats['good']}")
        print(f"    合格率: {stats['rate']}%")
        print()

    # 总体合格率
    print("📈 总体合格率:")
    print("-" * 60)
    print(f"  总采样数量: {overall_stats['total']}")
    print(f"  总合格数量: {overall_stats['good']}")
    print(f"  ✅ 总体合格率: {overall_stats['rate']}%")
    print("=" * 60)

    logger.info("✅ 评测流程全部完成。")


if __name__ == "__main__":
    main()