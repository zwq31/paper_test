import pandas as pd
import json
import os
import re
import logging
import time
import sys
from openai import OpenAI

# ============================================================================
# 配置变量 - 批量推理版本
# ============================================================================
DATASETS_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/rejected/unfinetuned_qa"
FACTUAL_QA_FILE = os.path.join(DATASETS_DIR, 'unfinetuned_qa_batch.json')

# 批量推理配置
MODEL_NAME = "deepseek-v3"  # 使用支持批量推理的模型
BATCH_SIZE = 4000  # 每批处理的新闻数量改为2000
MAX_WAIT_HOURS = 48  # 最大等待时间

# ============================================================================
# 设置日志配置
# ============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('factual_qa_batch.log', mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="__SET_VIA_ENVIRONMENT__",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

os.makedirs(DATASETS_DIR, exist_ok=True)

def log_progress(message):
    """统一的进度日志记录"""
    logger.info(message)
    print(message)

def load_news_data():
    """加载新闻数据"""
    try:
        news_df = pd.read_json('/mnt/project/Qwen2.5-7B-Instruct-news/1year-news-finetune/data/datasets/enhance_dataset/1years-news-unfinetuned.json')
        log_progress(f"✅ 已加载 {len(news_df)} 条新闻文章")
        
        # 过滤掉空内容
        valid_news = []
        for idx, row in news_df.iterrows():
            if row['content'] and len(str(row['content']).strip()) >= 50:
                valid_news.append({
                    'id': row['id'],
                    'title': row['title'],
                    'content': row['content'],
                    'pub_date': row['pub_date']
                })
        
        log_progress(f"📊 有效新闻数量: {len(valid_news)} (已过滤空内容)")
        return valid_news
        
    except Exception as e:
        log_progress(f"❌ 加载新闻数据时出错: {e}")
        return []

def create_batch_input_file(news_batch, batch_num):
    """创建批量推理输入文件"""
    input_file_path = os.path.join(DATASETS_DIR, f"factual_batch_input_{batch_num}.jsonl")
    
    with open(input_file_path, 'w', encoding='utf-8') as f:
        for news in news_batch:
            custom_id = str(news['id'])  # 直接使用新闻ID作为custom_id
            
            prompt = f"""
基于以下新闻内容、标题、发表时间，生成2个事实性问题及对应答案：

标题：{news['title']}
内容：{news['content']}
新闻发表时间：{news['pub_date']}

严格要求：
1. 每个问题必须具备完整的上下文信息，即使脱离原文也能独立理解，且问题侧重于新闻的重点事实。
2. 问题中但凡涉及具体的时间、地点、人物、活动、政策、事件等实体时，必须明确指出其完整信息。
3. 活动、会议、项目、政策等等必须在问题中使用完整名称，如"金华市第八次党代会"
4. 严禁使用模糊指代词语，如"此次"、"这个"、"该活动"、"相关负责人"。所有提及的人物、地点、事件名称等必须具体指出，不能有歧义。
5. 所有问题必须直接来源于新闻内容，不得自行推断或添加未提及的信息，尽可能多的覆盖新闻的内容。
6. 答案必须在新闻中能找到明确依据，不能含糊其辞或泛泛而谈。
7. 避免出现任何模糊或需要上下文才能理解的表述，比如"永嘉县委书记娄绍光在会议上强调了哪些关于干部管理的要求？"这种问题不能出现，必须指明是什么时间在什么地点举行的什么会议
8. 如果新闻中提到某个活动、会议、政策等，问题中必须包含该活动的完整信息，包括时间、地点、具体名称。例如：不能问"座谈会上提到什么"，而应该问"[具体时间]在[具体地点]举行的[具体会议名称]座谈会上提到什么"
9. 对于任何涉及人物、地点、事件的问题，都必须明确指出具体的时间背景和事件背景，确保问题没有歧义
10. 新闻中没有出现时间的情况，根据新闻发表时间和新闻内容中的与时间相关的词推断出时间
11. 每个问答对尽可能多包含新闻内容，问答对内容之间可以有重复的内容
12. 问题必须是事实性问题，即可以通过新闻内容直接回答的客观问题，而不是主观性或概念性问题

请返回有效的JSON格式：
[
    {{"question": "问题1", "answer": "答案1"}},
    {{"question": "问题2", "answer": "答案2"}}
]

只返回JSON数组，不要有额外说明。
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
                            "content": "你是一个有用的助手，用于生成基于新闻内容的事实性问题和答案。请严格按照要求的JSON格式返回结果。"
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
    
    log_progress(f"📄 创建批次 {batch_num} 输入文件: {len(news_batch)} 个新闻")
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

def download_and_process_batch_results(output_file_id, news_batch, batch_num):
    """下载并处理批次结果"""
    try:
        # 下载结果
        result_content = client.files.content(output_file_id)
        results = result_content.text.split('\n')
        
        all_qa_pairs = []
        success_count = 0
        error_count = 0
        
        log_progress(f"📥 下载批次 {batch_num} 结果: {len(results)} 行，开始解析...")
        
        # 构建新闻ID映射
        news_mapping = {str(news['id']): news for news in news_batch}
        
        # 解析每一行结果
        for line in results:
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
                        json_match = re.search(r'\[.*\]', content, re.DOTALL)
                        if json_match:
                            try:
                                qa_pairs = json.loads(json_match.group(0))
                                
                                if custom_id in news_mapping:
                                    news_item = news_mapping[custom_id]
                                    
                                    # 为每个问答对添加source_id
                                    for qa in qa_pairs:
                                        qa["source_id"] = news_item['id']
                                        qa["type"] = "factual"
                                    
                                    all_qa_pairs.extend(qa_pairs)
                                    success_count += 1
                                    log_progress(f"✅ 批次 {batch_num}: 成功处理新闻 {news_item['id']}，生成 {len(qa_pairs)} 个问答对")
                                else:
                                    error_count += 1
                                    log_progress(f"⚠️ 未找到新闻数据: {custom_id}")
                                    
                            except json.JSONDecodeError as e:
                                error_count += 1
                                log_progress(f"❌ 批次 {batch_num} JSON解析失败: {e}")
                        else:
                            error_count += 1
                            log_progress(f"❌ 批次 {batch_num} 未找到JSON内容")
                    else:
                        error_count += 1
                        log_progress(f"❌ 批次 {batch_num} 无响应内容")
                else:
                    error_count += 1
                    log_progress(f"❌ 批次 {batch_num} 请求失败")
                    
            except Exception as e:
                error_count += 1
                log_progress(f"❌ 批次 {batch_num} 解析行失败: {e}")
        
        log_progress(f"📊 批次 {batch_num} 解析完成: 成功 {success_count}, 失败 {error_count}")
        return all_qa_pairs
        
    except Exception as e:
        log_progress(f"❌ 下载或解析批次 {batch_num} 结果失败: {e}")
        return []

def save_incremental_results(qa_pairs, output_path):
    """增量保存结果"""
    # 如果文件已存在，读取现有数据
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    # 合并数据
    existing_data.extend(qa_pairs)
    
    # 保存到临时文件
    temp_path = output_path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    # 替换原文件
    os.replace(temp_path, output_path)
    
    log_progress(f"💾 增量保存完成: 总数据量 {len(existing_data)} 条问答对")

def get_processed_batches():
    """获取已处理的批次信息"""
    processed_file = os.path.join(DATASETS_DIR, "factual_processed_batches.json")
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_processed_batch(batch_num):
    """保存已处理的批次信息"""
    processed_file = os.path.join(DATASETS_DIR, "factual_processed_batches.json")
    processed_batches = get_processed_batches()
    processed_batches.add(batch_num)
    
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(list(processed_batches), f, indent=2)
    
    log_progress(f"📝 记录批次 {batch_num} 为已处理")

def cleanup_batch_files(batch_num):
    """清理批次临时文件"""
    input_file = os.path.join(DATASETS_DIR, f"factual_batch_input_{batch_num}.jsonl")
    
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
        
        # 统计每个新闻生成的问答对数量
        news_stats = {}
        for item in data:
            news_id = item["source_id"]
            if news_id not in news_stats:
                news_stats[news_id] = 0
            news_stats[news_id] += 1
        
        avg_qa_per_news = sum(news_stats.values()) / len(news_stats) if news_stats else 0
        
        log_progress(f"📈 最终结果分析:")
        log_progress(f"  总问答对数量: {len(data)} 条")
        log_progress(f"  处理的新闻数量: {len(news_stats)} 个")
        log_progress(f"  平均每个新闻生成: {avg_qa_per_news:.1f} 个问答对")
        log_progress(f"  预期数据量: {len(news_stats) * 2} 条")
        log_progress(f"  完成度: {len(data)/(len(news_stats) * 2)*100:.1f}%")
        
        return len(data)
        
    except Exception as e:
        log_progress(f"❌ 分析最终结果失败: {e}")
        return 0

def main():
    """主函数 - 批量处理事实问答对生成"""
    print("\n" + "="*70)
    print("🚀 事实问答对批量生成 - 1600+条新闻")
    print("="*70)
    print(f"模型: {MODEL_NAME}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"预期输出: 1600 × 2 = 3200+ 条问答对")
    print(f"最大等待时间: {MAX_WAIT_HOURS} 小时")
    print(f"成本: 实时推理的50%")
    print(f"输出文件: {FACTUAL_QA_FILE}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # 1. 加载新闻数据
        log_progress("📥 步骤1: 加载新闻数据...")
        all_news = load_news_data()
        if not all_news:
            return
        
        # 2. 分批处理
        total_batches = (len(all_news) + BATCH_SIZE - 1) // BATCH_SIZE
        processed_batches = get_processed_batches()
        
        log_progress(f"📦 新闻数据分 {total_batches} 批处理")
        log_progress(f"📋 已处理批次: {processed_batches}")
        
        completed_batches = 0
        total_qa_pairs = 0
        
        for batch_num in range(1, total_batches + 1):
            if batch_num in processed_batches:
                log_progress(f"⏭️  跳过已处理的批次 {batch_num}")
                continue
            
            batch_start = (batch_num - 1) * BATCH_SIZE
            batch_end = batch_num * BATCH_SIZE
            batch_news = all_news[batch_start:batch_end]
            
            log_progress(f"\n🔄 开始处理批次 {batch_num}/{total_batches}")
            log_progress(f"📊 批次数据: {len(batch_news)} 条新闻")
            
            # 2.1 创建批量文件
            input_file = create_batch_input_file(batch_news, batch_num)
            
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
            qa_batch = download_and_process_batch_results(output_file_id, batch_news, batch_num)
            
            if qa_batch:
                # 2.5 增量保存结果
                save_incremental_results(qa_batch, FACTUAL_QA_FILE)
                
                # 2.6 记录为已处理
                save_processed_batch(batch_num)
                
                # 2.7 清理临时文件
                cleanup_batch_files(batch_num)
                
                completed_batches += 1
                total_qa_pairs += len(qa_batch)
                
                log_progress(f"✅ 批次 {batch_num} 完成!")
                log_progress(f"📈 当前进度: {completed_batches}/{total_batches} 批次")
                log_progress(f"📊 累计生成: {total_qa_pairs} 条问答对")
                
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
        final_count = analyze_final_results(FACTUAL_QA_FILE)
        
        # 计算总时间
        total_time = time.time() - start_time
        days = int(total_time // 86400)
        hours = int((total_time % 86400) // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("\n" + "="*70)
        print("🎉 事实问答对批量生成完成!")
        print("="*70)
        print(f"总处理时间: {days}天{hours}小时{minutes}分钟")
        print(f"完成批次: {completed_batches}/{total_batches}")
        print(f"总生成问答对: {final_count} 条")
        print(f"输出文件: {FACTUAL_QA_FILE}")
        print(f"成本节省: 50% (相比实时推理)")
        print("="*70)
        
    except Exception as e:
        log_progress(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



