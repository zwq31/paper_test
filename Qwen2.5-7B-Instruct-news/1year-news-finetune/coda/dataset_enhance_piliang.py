import json
import os
import re
import logging
import time
import sys
from openai import OpenAI

# ============================================================================
# 配置变量 - 28000条完整数据处理
# ============================================================================
DATASETS_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected"
AUGMENTATION_COUNT = 9
OUTPUT_JSON_PATH = os.path.join(DATASETS_DIR, 'augmented_qa_full_28000.json')

# 批量推理配置
MODEL_NAME = "deepseek-v3"
BATCH_SIZE = 5000  # 每批5000条，共6批（28000÷5000≈5.6，取6批）
MAX_WAIT_HOURS = 72  # 更长的等待时间，因为总数据量大

# ============================================================================
# 设置日志配置
# ============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('full_28000_processing.log', mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

os.makedirs(DATASETS_DIR, exist_ok=True)

def log_progress(message):
    logger.info(message)
    print(message)

def load_all_data(qa_data_file):
    """加载所有QA数据"""
    try:
        with open(qa_data_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        log_progress(f"✅ 已加载 {len(qa_data)} 条原始QA数据")
        
        # 按source_id和type分组分配qa_id
        qa_by_source_and_type = {}
        for qa in qa_data:
            key = (qa["source_id"], qa["type"])
            if key not in qa_by_source_and_type:
                qa_by_source_and_type[key] = []
            qa_by_source_and_type[key].append(qa)
        
        # 分配qa_id
        prepared_data = []
        source_stats = {}
        
        for (sid, qa_type), qa_list in qa_by_source_and_type.items():
            if sid not in source_stats:
                source_stats[sid] = {"conceptual": 0, "situational": 0}
            source_stats[sid][qa_type] = len(qa_list)
            
            for i, qa in enumerate(qa_list):
                qa["qa_id"] = i + 1
                prepared_data.append(qa)
        
        # 输出统计信息
        log_progress(f"📊 数据统计:")
        log_progress(f"  总source_id数量: {len(source_stats)}")
        log_progress(f"  总QA对数量: {len(prepared_data)}")
        
        # 显示前几个source_id的分布
        for sid, counts in list(source_stats.items())[:5]:
            log_progress(f"  source_id {sid}: conceptual={counts['conceptual']}, situational={counts['situational']}")
        
        if len(source_stats) > 5:
            log_progress(f"  ... 还有 {len(source_stats) - 5} 个source_id")
        
        return prepared_data
        
    except Exception as e:
        log_progress(f"❌ 加载数据失败: {e}")
        return []

def create_batch_file(qa_batch, batch_num):
    """创建批量推理输入文件"""
    input_file_path = os.path.join(DATASETS_DIR, f"batch_input_{batch_num}.jsonl")
    
    with open(input_file_path, 'w', encoding='utf-8') as f:
        for qa in qa_batch:
            custom_id = f"{qa['source_id']}_{qa['type']}_{qa['qa_id']}"
            
            prompt = f"""
你是一个数据增强专家。请为以下问答对生成9个不同的增强版本。每个增强版本要保持原意和事实不变，但在表达方式、句式结构、词汇选择上有所变化。

原始问答对：
问题：{qa["question"]}
答案：{qa["answer"]}

增强要求：
1. 保持答案的事实准确性完全不变
2. 问题可以重新表述，但核心询问内容必须一致
3. 答案可以重新组织语言，但信息点必须完整保留
4. 使用不同的句式结构、同义词替换、语序调整等方式
5. 确保增强后的问题和答案仍然自然流畅

请返回严格的JSON格式：
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
            batch_request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "你是一个专业的数据增强专家，擅长在保持原意和事实不变的情况下生成多样化的文本变体。请严格按照要求的JSON格式返回结果。"
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            }
            
            f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
    
    log_progress(f"📄 创建批次 {batch_num} 输入文件: {len(qa_batch)} 个请求")
    return input_file_path

def submit_batch_job(input_file_path, batch_num):
    """提交批量任务"""
    try:
        with open(input_file_path, 'rb') as file:
            uploaded_file = client.files.create(file=file, purpose="batch")
        
        batch_job = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window=f"{MAX_WAIT_HOURS}h"
        )
        
        log_progress(f"✅ 批次 {batch_num} 提交成功")
        log_progress(f"🆔 任务ID: {batch_job.id}")
        return batch_job.id
        
    except Exception as e:
        log_progress(f"❌ 批次 {batch_num} 提交失败: {e}")
        return None

def monitor_batch_progress(batch_id, batch_num):
    """监控批量任务进度"""
    log_progress(f"🔍 开始监控批次 {batch_num} 进度: {batch_id}")
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            batch_status = client.batches.retrieve(batch_id)
            current_status = batch_status.status
            
            if current_status != last_status:
                elapsed_hours = (time.time() - start_time) / 3600
                log_progress(f"🔄 批次 {batch_num} 状态: {current_status} (已等待 {elapsed_hours:.1f} 小时)")
                last_status = current_status
            
            if current_status == "completed":
                log_progress(f"✅ 批次 {batch_num} 处理完成!")
                return batch_status.output_file_id
                
            elif current_status in ["failed", "expired", "cancelled"]:
                log_progress(f"❌ 批次 {batch_num} 失败: {current_status}")
                if hasattr(batch_status, 'errors'):
                    log_progress(f"错误详情: {batch_status.errors}")
                return None
                
            else:
                # 还在处理中，等待5分钟后再次检查
                time.sleep(300)
                
        except Exception as e:
            log_progress(f"⚠️ 检查批次 {batch_num} 状态失败: {e}")
            time.sleep(300)

def download_and_process_batch_results(output_file_id, original_batch, batch_num):
    """下载并处理批次结果"""
    try:
        # 下载结果
        result_content = client.files.content(output_file_id)
        results = result_content.text.split('\n')
        
        enhanced_data = []
        success_count = 0
        error_count = 0
        
        log_progress(f"📥 下载批次 {batch_num} 结果: {len(results)} 行，开始解析...")
        
        # 构建映射表
        qa_mapping = {}
        for qa in original_batch:
            custom_id = f"{qa['source_id']}_{qa['type']}_{qa['qa_id']}"
            qa_mapping[custom_id] = qa
        
        # 解析每一行结果
        for line_num, line in enumerate(results, 1):
            if not line.strip():
                continue
                
            try:
                result_obj = json.loads(line)
                custom_id = result_obj.get("custom_id")
                response = result_obj.get("response", {})
                
                if response.get("status_code") == 200:
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        
                        # 提取JSON
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                augmentation_data = json.loads(json_match.group(0))
                                augmentations = augmentation_data.get("augmentations", [])
                                
                                if custom_id in qa_mapping:
                                    original_qa = qa_mapping[custom_id]
                                    
                                    # 添加原始版本 (variant_id=1)
                                    enhanced_data.append({
                                        "news_id": original_qa["source_id"],
                                        "qa_id": original_qa["qa_id"],
                                        "variant_id": 1,
                                        "question": original_qa["question"],
                                        "answer": original_qa["answer"],
                                        "type": original_qa["type"]
                                    })
                                    
                                    # 添加增强版本 (variant_id 2-10)
                                    for i, aug_qa in enumerate(augmentations[:AUGMENTATION_COUNT]):
                                        if aug_qa.get("question") and aug_qa.get("answer"):
                                            enhanced_data.append({
                                                "news_id": original_qa["source_id"],
                                                "qa_id": original_qa["qa_id"],
                                                "variant_id": i + 2,
                                                "question": aug_qa.get("question", ""),
                                                "answer": aug_qa.get("answer", ""),
                                                "type": original_qa["type"]
                                            })
                                    
                                    success_count += 1
                                    if success_count % 100 == 0:  # 每100个显示一次进度
                                        log_progress(f"✅ 批次 {batch_num}: 已成功处理 {success_count} 个QA对")
                                else:
                                    error_count += 1
                                    
                            except json.JSONDecodeError as e:
                                error_count += 1
                                log_progress(f"❌ 批次 {batch_num} JSON解析失败: {e}")
                        else:
                            error_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
        
        log_progress(f"📊 批次 {batch_num} 解析完成: 成功 {success_count}, 失败 {error_count}")
        return enhanced_data
        
    except Exception as e:
        log_progress(f"❌ 下载或解析批次 {batch_num} 结果失败: {e}")
        return []

def save_incremental_results(enhanced_data, output_path):
    """增量保存结果"""
    # 如果文件已存在，读取现有数据
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    # 合并数据
    existing_data.extend(enhanced_data)
    
    # 保存到临时文件
    temp_path = output_path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    # 替换原文件
    os.replace(temp_path, output_path)
    
    log_progress(f"💾 增量保存完成: 总数据量 {len(existing_data)} 条")

def get_processed_batches():
    """获取已处理的批次信息"""
    processed_file = os.path.join(DATASETS_DIR, "processed_batches.json")
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_processed_batch(batch_num):
    """保存已处理的批次信息"""
    processed_file = os.path.join(DATASETS_DIR, "processed_batches.json")
    processed_batches = get_processed_batches()
    processed_batches.add(batch_num)
    
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(list(processed_batches), f, indent=2)
    
    log_progress(f"📝 记录批次 {batch_num} 为已处理")

def cleanup_batch_files(batch_num):
    """清理批次临时文件"""
    input_file = os.path.join(DATASETS_DIR, f"batch_input_{batch_num}.jsonl")
    
    if os.path.exists(input_file):
        try:
            os.remove(input_file)
            log_progress(f"🧹 已清理批次 {batch_num} 临时文件")
        except:
            log_progress(f"⚠️ 清理批次 {batch_num} 临时文件失败")

def analyze_final_results(output_path):
    """分析最终结果"""
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计变体数量
        variant_stats = {}
        for item in data:
            key = (item["news_id"], item["type"], item["qa_id"])
            if key not in variant_stats:
                variant_stats[key] = 0
            variant_stats[key] += 1
        
        variant_counts = list(variant_stats.values())
        avg_variants = sum(variant_counts) / len(variant_counts) if variant_counts else 0
        
        log_progress(f"📈 最终结果分析:")
        log_progress(f"  总数据量: {len(data)} 条")
        log_progress(f"  处理的QA对: {len(variant_stats)} 个")
        log_progress(f"  平均变体数: {avg_variants:.1f} 个/QA对")
        log_progress(f"  预期数据量: {28000 * 10} 条")
        log_progress(f"  完成度: {len(data)/(28000 * 10)*100:.1f}%")
        
        return len(data)
        
    except Exception as e:
        log_progress(f"❌ 分析最终结果失败: {e}")
        return 0

def main():
    """主函数 - 处理28000条完整数据"""
    qa_data_file = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/train.json"
    
    print("\n" + "="*70)
    print("🚀 批量推理完整数据处理 - 28000条样本")
    print("="*70)
    print(f"模型: {MODEL_NAME}")
    print(f"总样本数: 28000")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"预计批次: {28000 // BATCH_SIZE + 1}")
    print(f"预期输出: 28000 × 10 = {28000 * 10} 条")
    print(f"最大等待时间: {MAX_WAIT_HOURS} 小时")
    print(f"成本: 实时推理的50%")
    print(f"输出文件: {OUTPUT_JSON_PATH}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # 1. 加载所有数据
        log_progress("📥 步骤1: 加载所有数据...")
        all_data = load_all_data(qa_data_file)
        if not all_data:
            return
        
        # 2. 分批处理
        total_batches = (len(all_data) + BATCH_SIZE - 1) // BATCH_SIZE
        processed_batches = get_processed_batches()
        
        log_progress(f"📦 数据分 {total_batches} 批处理")
        log_progress(f"📋 已处理批次: {processed_batches}")
        
        completed_batches = 0
        total_enhanced = 0
        
        for batch_num in range(1, total_batches + 1):
            if batch_num in processed_batches:
                log_progress(f"⏭️  跳过已处理的批次 {batch_num}")
                continue
            
            batch_start = (batch_num - 1) * BATCH_SIZE
            batch_end = batch_num * BATCH_SIZE
            batch_data = all_data[batch_start:batch_end]
            
            log_progress(f"\n🔄 开始处理批次 {batch_num}/{total_batches}")
            log_progress(f"📊 批次数据: {len(batch_data)} 条")
            
            # 2.1 创建批量文件
            input_file = create_batch_file(batch_data, batch_num)
            
            # 2.2 提交批量任务
            batch_id = submit_batch_job(input_file, batch_num)
            if not batch_id:
                log_progress(f"❌ 批次 {batch_num} 提交失败，跳过")
                continue
            
            # 2.3 监控进度
            output_file_id = monitor_batch_progress(batch_id, batch_num)
            if not output_file_id:
                log_progress(f"❌ 批次 {batch_num} 处理失败，跳过")
                continue
            
            # 2.4 下载和处理结果
            enhanced_batch = download_and_process_batch_results(output_file_id, batch_data, batch_num)
            
            if enhanced_batch:
                # 2.5 增量保存结果
                save_incremental_results(enhanced_batch, OUTPUT_JSON_PATH)
                
                # 2.6 记录为已处理
                save_processed_batch(batch_num)
                
                # 2.7 清理临时文件
                cleanup_batch_files(batch_num)
                
                completed_batches += 1
                total_enhanced += len(enhanced_batch)
                
                log_progress(f"✅ 批次 {batch_num} 完成!")
                log_progress(f"📈 当前进度: {completed_batches}/{total_batches} 批次")
                log_progress(f"📊 累计生成: {total_enhanced} 条数据")
                
                # 计算预计剩余时间
                elapsed_time = time.time() - start_time
                if completed_batches > 0:
                    avg_time_per_batch = elapsed_time / completed_batches
                    remaining_batches = total_batches - completed_batches
                    estimated_remaining = avg_time_per_batch * remaining_batches
                    
                    hours = int(estimated_remaining // 3600)
                    minutes = int((estimated_remaining % 3600) // 60)
                    
                    log_progress(f"⏳ 预计剩余时间: {hours}小时{minutes}分钟")
            else:
                log_progress(f"❌ 批次 {batch_num} 未生成有效数据")
        
        # 3. 分析最终结果
        log_progress("\n📊 步骤3: 分析最终结果...")
        final_count = analyze_final_results(OUTPUT_JSON_PATH)
        
        # 计算总时间
        total_time = time.time() - start_time
        days = int(total_time // 86400)
        hours = int((total_time % 86400) // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("\n" + "="*70)
        print("🎉 完整数据处理完成!")
        print("="*70)
        print(f"总处理时间: {days}天{hours}小时{minutes}分钟")
        print(f"完成批次: {completed_batches}/{total_batches}")
        print(f"总生成数据量: {final_count} 条")
        print(f"输出文件: {OUTPUT_JSON_PATH}")
        print(f"成本节省: 50% (相比实时推理)")
        print("="*70)
        
    except Exception as e:
        log_progress(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()