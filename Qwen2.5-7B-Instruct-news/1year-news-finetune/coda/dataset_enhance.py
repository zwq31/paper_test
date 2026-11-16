import json
import os
import re
import random
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ============================================================================
# 配置变量 - 使用 DeepSeek-V3
# ============================================================================
DATASETS_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset"
AUGMENTATION_COUNT = 9  # 每个QA生成9个增强版本
OUTPUT_JSON_PATH = os.path.join(DATASETS_DIR, 'augmented_qa_simple.json')

# 基于1000 RPM的优化配置
RPM_LIMIT = 10000  # 每分钟请求数
MAX_WORKERS = 200  # 并发线程数
BATCH_SAVE_SIZE = 100  # 批量保存大小
MODEL_NAME = "deepseek-r1-distill-qwen-1.5b"  # 使用 DeepSeek-V3

# ============================================================================
# 设置日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dataset_enhance_v3.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

os.makedirs(DATASETS_DIR, exist_ok=True)

# 创建线程锁
file_lock = threading.Lock()

class SimpleRateLimiter:
    """简化的速率限制器，只考虑RPM限制"""
    
    def __init__(self, rpm_limit):
        self.rpm_limit = rpm_limit
        self.requests_this_minute = 0
        self.minute_start = time.time()
        self.lock = threading.Lock()
        
    def can_make_request(self):
        """检查是否可以发起新请求"""
        with self.lock:
            current_time = time.time()
            # 检查是否进入新的一分钟
            if current_time - self.minute_start >= 60:
                self.requests_this_minute = 0
                self.minute_start = current_time
                return True
                
            # 检查RPM限制
            if self.requests_this_minute >= self.rpm_limit:
                return False
                
            return True
            
    def record_request(self):
        """记录已完成的请求"""
        with self.lock:
            self.requests_this_minute += 1

# 初始化速率限制器
rate_limiter = SimpleRateLimiter(RPM_LIMIT)

def wait_for_rate_limit():
    """等待直到可以发起新请求"""
    while not rate_limiter.can_make_request():
        # 计算需要等待的时间
        current_time = time.time()
        time_since_minute_start = current_time - rate_limiter.minute_start
        wait_time = max(0, 60 - time_since_minute_start + 0.1)  # 加0.1秒缓冲
        
        if wait_time > 0:
            logger.debug(f"达到RPM限制，等待 {wait_time:.1f} 秒")
            time.sleep(wait_time)

def augment_qa_pair_simple(original_qa):
    """
    基于问答对本身进行增强，不需要新闻内容
    """
    question = original_qa["question"]
    answer = original_qa["answer"]
    qa_type = original_qa["type"]
    source_id = original_qa["source_id"]

    prompt = f"""
    你是一个数据增强专家。请为以下问答对生成9个不同的增强版本。每个增强版本要保持原意和事实不变，但在表达方式、句式结构、词汇选择上有所变化。

    原始问答对：
    问题：{question}
    答案：{answer}
    
    增强要求：
    1. 保持答案的事实准确性完全不变
    2. 问题可以重新表述，但核心询问内容必须一致
    3. 答案可以重新组织语言，但信息点必须完整保留
    4. 使用不同的句式结构、同义词替换、语序调整等方式
    5. 确保增强后的问题和答案仍然自然流畅
    6. 避免简单的同义词替换，要进行有意义的句式重构
    
    增强策略示例：
    - 改变问题角度（从"什么"改为"如何"、"为什么"等）
    - 调整句子结构（主动变被动，陈述变疑问等）
    - 使用不同的连接词和表达方式
    - 改变信息的呈现顺序
    - 使用近义词替换（但要确保专业术语准确）
    
    重要原则：
    - 不能改变事实内容
    - 不能添加原始答案中没有的信息
    - 不能删除原始答案中的关键信息
    - 保持专业术语的准确性
    
    请返回JSON格式：
    {{
        "augmentations": [
            {{
                "question": "增强后的问题1",
                "answer": "增强后的答案1"
            }},
            {{
                "question": "增强后的问题2", 
                "answer": "增强后的答案2"
            }},
            ... // 共9个增强版本
        ]
    }}
    """

    try:
        # 速率限制检查
        wait_for_rate_limit()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,  # 使用 DeepSeek-V3
            messages=[
                {"role": "system", "content": "你是一个专业的数据增强专家，擅长在保持原意和事实不变的情况下生成多样化的文本变体。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.7
        )

        # 记录请求
        rate_limiter.record_request()

        result = response.choices[0].message.content.strip()

        # 提取 JSON
        json_match = re.search(r'(\{.*\})', result, re.DOTALL)
        if json_match:
            result = json_match.group(1)

        try:
            augmentation_data = json.loads(result)
            augmentations = augmentation_data.get("augmentations", [])

            enhanced_versions = []

            # 添加原始版本 (variant_id=1)
            enhanced_versions.append({
                "news_id": source_id,
                "qa_id": original_qa.get("qa_id", 0),
                "variant_id": 1,
                "question": question,
                "answer": answer,
                "type": qa_type
            })

            # 添加增强版本 (variant_id 2-10)
            for i, aug_qa in enumerate(augmentations[:AUGMENTATION_COUNT]):
                enhanced_versions.append({
                    "news_id": source_id,
                    "qa_id": original_qa.get("qa_id", 0),
                    "variant_id": i + 2,
                    "question": aug_qa.get("question", ""),
                    "answer": aug_qa.get("answer", ""),
                    "type": qa_type
                })

            # 过滤空内容
            enhanced_versions = [
                v for v in enhanced_versions
                if v.get("question") and v.get("answer")
            ]

            # 补齐到10条（含原始）
            while len(enhanced_versions) < AUGMENTATION_COUNT + 1:
                enhanced_versions.append({
                    "news_id": source_id,
                    "qa_id": original_qa.get("qa_id", 0),
                    "variant_id": len(enhanced_versions) + 1,
                    "question": question,
                    "answer": answer,
                    "type": qa_type
                })

            return enhanced_versions

        except json.JSONDecodeError as e:
            logger.error(f"JSON解码失败: {e}, 响应: {result}")
            return [{
                "news_id": source_id,
                "qa_id": original_qa.get("qa_id", 0),
                "variant_id": 1,
                "question": question,
                "answer": answer,
                "type": qa_type
            }]

    except Exception as e:
        logger.error(f"增强出错: {e}")
        return [{
            "news_id": source_id,
            "qa_id": original_qa.get("qa_id", 0),
            "variant_id": 1,
            "question": question,
            "answer": answer,
            "type": qa_type
        }]


def load_and_prepare_qa_data(qa_data_file):
    """加载所有QA数据"""
    try:
        with open(qa_data_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        logger.info(f"已加载 {len(qa_data)} 条原始QA数据")

        qa_by_source = {}
        for qa in qa_data:
            sid = qa.get("source_id")
            if sid not in qa_by_source:
                qa_by_source[sid] = []
            qa_by_source[sid].append(qa)

        prepared_data = []
        for sid, qa_list in qa_by_source.items():
            for i, qa in enumerate(qa_list):
                qa["qa_id"] = i + 1
                prepared_data.append(qa)

        return prepared_data
    except Exception as e:
        logger.error(f"加载QA失败: {e}")
        return []


def save_incremental_json(new_samples, output_path):
    """线程安全的JSON文件追加写入"""
    with file_lock:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        else:
            existing = []

        existing.extend(new_samples)

        temp_path = output_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)


def process_single_qa(qa_item):
    """处理单个QA项，用于多线程调用"""
    try:
        augmented = augment_qa_pair_simple(qa_item)
        return augmented
    except Exception as e:
        logger.error(f"处理QA失败 {qa_item.get('qa_id', 'unknown')}: {e}")
        return [{
            "news_id": qa_item["source_id"],
            "qa_id": qa_item.get("qa_id", 0),
            "variant_id": 1,
            "question": qa_item["question"],
            "answer": qa_item["answer"],
            "type": qa_item["type"]
        }]


def get_processed_qa_set(output_path):
    """获取已处理的QA集合，用于断点续传"""
    processed_set = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            for item in existing_data:
                key = (item["news_id"], item["qa_id"])
                processed_set.add(key)
                
            logger.info(f"发现 {len(processed_set)} 个已处理的QA对")
        except Exception as e:
            logger.error(f"读取已处理数据失败: {e}")
    
    return processed_set


def process_qa_augmentation_v3(qa_data_file):
    """使用 DeepSeek-V3 的数据增强"""
    qa_data = load_and_prepare_qa_data(qa_data_file)
    if not qa_data:
        logger.error("无QA数据")
        return []

    logger.info(f"开始使用 {MODEL_NAME} 处理所有 {len(qa_data)} 个原始QA")
    logger.info(f"配置: {MAX_WORKERS} 线程, RPM限制: {RPM_LIMIT}")

    # 获取已处理的QA集合
    processed_set = get_processed_qa_set(OUTPUT_JSON_PATH)
    
    # 过滤掉已处理的QA
    qa_to_process = []
    for qa in qa_data:
        key = (qa["source_id"], qa["qa_id"])
        if key not in processed_set:
            qa_to_process.append(qa)
    
    logger.info(f"需要处理 {len(qa_to_process)} 个新QA，跳过 {len(qa_data) - len(qa_to_process)} 个已处理QA")

    if not qa_to_process:
        logger.info("所有QA都已处理完成")
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        return len(existing_data)

    total_written = 0
    completed_count = 0
    start_time = time.time()

    # 批量处理以减少文件IO
    batch_results = []

    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_qa = {executor.submit(process_single_qa, qa_item): qa_item for qa_item in qa_to_process}
        
        # 使用tqdm显示进度
        with tqdm(total=len(qa_to_process), desc=f"{MODEL_NAME}增强", unit="qa") as pbar:
            # 异步处理完成的任务
            for future in as_completed(future_to_qa):
                qa_item = future_to_qa[future]
                try:
                    augmented_data = future.result()
                    batch_results.extend(augmented_data)
                    total_written += len(augmented_data)
                    completed_count += 1
                    
                    # 批量保存
                    if completed_count % BATCH_SAVE_SIZE == 0:
                        save_incremental_json(batch_results, OUTPUT_JSON_PATH)
                        batch_results = []  # 清空批量
                        
                        # 计算处理速度
                        elapsed_time = time.time() - start_time
                        qps = completed_count / elapsed_time if elapsed_time > 0 else 0
                        estimated_remaining = (len(qa_to_process) - completed_count) / qps if qps > 0 else 0
                        
                        logger.info(f"进度: {completed_count}/{len(qa_to_process)} "
                                  f"({completed_count/len(qa_to_process)*100:.1f}%), "
                                  f"速度: {qps:.1f} QA/秒, "
                                  f"预计剩余: {estimated_remaining/60:.1f} 分钟")
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed': completed_count,
                        'total_written': total_written,
                        'qps': f"{completed_count/(time.time()-start_time):.1f}" if time.time() > start_time else "0"
                    })
                        
                except Exception as e:
                    logger.error(f"处理QA失败 {qa_item.get('qa_id', 'unknown')}: {e}")
                    pbar.update(1)

            # 保存剩余的结果
            if batch_results:
                save_incremental_json(batch_results, OUTPUT_JSON_PATH)

    # 读取最终数据量
    with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    total_time = time.time() - start_time
    logger.info(f"✅ {MODEL_NAME} 增强完成！")
    logger.info(f"- 处理 {completed_count} 个新QA")
    logger.info(f"- 总数据量 {len(final_data)} 条")
    logger.info(f"- 总耗时 {total_time/60:.1f} 分钟")
    logger.info(f"- 平均速度 {completed_count/total_time:.1f} QA/秒")
    
    return len(final_data)


def estimate_processing_time(qa_count):
    """估算处理时间（基于RPM限制）"""
    # 基于1000 RPM限制计算最大处理速度
    max_qps = RPM_LIMIT / 60  # 每秒最大请求数 ≈ 16.67
    # 保守估计，使用90%的最大速度
    estimated_qps = max_qps * 0.9
    
    total_seconds = qa_count / estimated_qps
    
    # 转换为小时和分钟
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return hours, minutes, estimated_qps


def analyze_augmentation_quality(augmented_data_file):
    """分析增强数据质量"""
    from collections import defaultdict
    with open(augmented_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = defaultdict(lambda: defaultdict(int))
    for item in data:
        stats[item['news_id']][item['qa_id']] += 1

    variant_counts = [count for news in stats.values() for count in news.values()]
    if not variant_counts:
        logger.warning("无有效数据用于质量分析")
        return {'avg_variants': 0, 'min': 0, 'max': 0}

    avg = sum(variant_counts) / len(variant_counts)
    min_variants = min(variant_counts)
    max_variants = max(variant_counts)
    
    logger.info(f"增强质量分析:")
    logger.info(f"- 平均变体数: {avg:.1f}/原始QA")
    logger.info(f"- 最少变体数: {min_variants}")
    logger.info(f"- 最多变体数: {max_variants}")
    
    # 统计变体分布
    distribution = {}
    for count in variant_counts:
        distribution[count] = distribution.get(count, 0) + 1
    
    logger.info(f"- 变体分布: {distribution}")

    return {
        'avg_variants': avg, 
        'min': min_variants, 
        'max': max_variants,
        'distribution': distribution
    }


def create_simple_splits(augmented_data_file, output_dir=None):
    """创建训练/验证/测试集分割"""
    if output_dir is None:
        output_dir = DATASETS_DIR

    with open(augmented_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        logger.warning("无数据用于划分训练/验证/测试集")
        return {}

    random.shuffle(data)
    n = len(data)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)

    splits = {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }

    for name, d in splits.items():
        path = os.path.join(output_dir, f'{name}_data.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        logger.info(f"{name}集: {len(d)} 条")

    return splits


def main():
    """主函数"""
    qa_data_file = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/combined_qa.json"
    
    # 加载数据以估算处理时间
    qa_data = load_and_prepare_qa_data(qa_data_file)
    if not qa_data:
        logger.error("无法加载QA数据")
        return
    
    processed_set = get_processed_qa_set(OUTPUT_JSON_PATH)
    remaining_qa = len([qa for qa in qa_data if (qa["source_id"], qa["qa_id"]) not in processed_set])
    
    hours, minutes, estimated_qps = estimate_processing_time(remaining_qa)
    
    print(f"\n{'='*60}")
    print(f"🚀 DeepSeek-V3 数据增强启动")
    print(f"{'='*60}")
    print(f"模型: {MODEL_NAME}")
    print(f"总原始QA数量: {len(qa_data)}")
    print(f"已处理QA数量: {len(processed_set)}")
    print(f"待处理QA数量: {remaining_qa}")
    print(f"API限制: {RPM_LIMIT} RPM")
    print(f"并发线程数: {MAX_WORKERS}")
    print(f"预计处理速度: {estimated_qps:.1f} QA/秒")
    print(f"预计处理时间: {hours}小时{minutes}分钟")
    print(f"{'='*60}")
    
    logger.info(f"🚀 启动 {MODEL_NAME} 增强")
    start_time = time.time()
    
    total_count = process_qa_augmentation_v3(qa_data_file)
    
    end_time = time.time()
    processing_time = end_time - start_time
    hours = int(processing_time // 3600)
    minutes = int((processing_time % 3600) // 60)
    actual_qps = remaining_qa / processing_time if processing_time > 0 else 0
    
    if total_count == 0:
        logger.error("未生成任何数据")
        return

    quality = analyze_augmentation_quality(OUTPUT_JSON_PATH)
    splits = create_simple_splits(OUTPUT_JSON_PATH)

    print(f"\n{'='*60}")
    print(f"✅ {MODEL_NAME} 数据增强完成！")
    print(f"{'='*60}")
    print(f"总处理时间: {hours}小时{minutes}分钟")
    print(f"实际处理速度: {actual_qps:.1f} QA/秒")
    print(f"总增强样本数: {total_count}")
    print(f"平均变体数: {quality['avg_variants']:.1f}")
    print(f"训练集大小: {len(splits['train']) if splits else 0}")
    print(f"验证集大小: {len(splits['val']) if splits else 0}")
    print(f"测试集大小: {len(splits['test']) if splits else 0}")
    print(f"输出文件: {OUTPUT_JSON_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()