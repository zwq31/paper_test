# """
# 处理所有数据的实体关系提取
# 文件名: extract_all_entity_relations_complete.py
# """
# import os
# import json
# import time
# import re
# import logging
# import sys
# from neo4j import GraphDatabase
# from openai import OpenAI
# from tqdm import tqdm

# # 配置信息
# NEO4J_URI = "bolt://122.226.69.234:3045"
# NEO4J_USER = "neo4j"
# NEO4J_PASSWORD = "neo4j123456"

# API_KEY = "__SET_VIA_ENVIRONMENT__"
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# # 输出目录配置
# OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data"
# PROCESSED_DOCS_FILE = os.path.join(OUTPUT_DIR, "processed_docs.json")
# RESULTS_FILE = os.path.join(OUTPUT_DIR, "extracted_relations.json")
# ERRORS_FILE = os.path.join(OUTPUT_DIR, "errors.json")
# STATS_FILE = os.path.join(OUTPUT_DIR, "statistics.json")

# # 创建输出目录
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 设置日志配置
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename=os.path.join(OUTPUT_DIR, 'extraction.log'),
#     filemode='w',
#     encoding='utf-8'
# )
# logger = logging.getLogger(__name__)

# # 添加控制台输出
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# logger.addHandler(console_handler)

# def log_progress(message):
#     """记录进度信息"""
#     logger.info(message)
#     print(message)

# class CompleteRelationExtractor:
#     def __init__(self, neo4j_uri, neo4j_user, neo4j_password, api_key, base_url):
#         self.neo4j_uri = neo4j_uri
#         self.neo4j_user = neo4j_user
#         self.neo4j_password = neo4j_password
#         self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
#         self.driver = None
#         self.stats = {
#             "total_docs": 0,
#             "processed_docs": 0,
#             "successful_docs": 0,
#             "failed_docs": 0,
#             "total_relations": 0,
#             "start_time": None,
#             "end_time": None
#         }

#     def connect_neo4j(self):
#         """连接Neo4j数据库"""
#         try:
#             self.driver = GraphDatabase.driver(
#                 self.neo4j_uri,
#                 auth=(self.neo4j_user, self.neo4j_password)
#             )
#             self.driver.verify_connectivity()
#             log_progress("✅ 成功连接到 Neo4j 数据库")
#             return True
#         except Exception as e:
#             log_progress(f"❌ 连接 Neo4j 失败: {e}")
#             return False

#     def get_total_document_count(self):
#         """获取文档总数"""
#         try:
#             with self.driver.session() as session:
#                 query = """
#                 MATCH (n:NewsArticle)
#                 WHERE n.content IS NOT NULL AND n.content <> ''
#                 RETURN count(n) as total_count
#                 """
#                 result = session.run(query)
#                 record = result.single()
#                 return record['total_count'] if record else 0
#         except Exception as e:
#             log_progress(f"❌ 获取文档总数失败: {e}")
#             return 0

#     def get_all_documents(self):
#         """获取所有文档及其实体"""
#         documents = []
#         try:
#             with self.driver.session() as session:
#                 query = """
#                 MATCH (n:NewsArticle)
#                 WHERE n.content IS NOT NULL AND n.content <> ''
#                 WITH n
#                 ORDER BY n.title
#                 OPTIONAL MATCH (n)-[rel]->(e)
#                 WHERE e.name IS NOT NULL AND e.name <> ''
#                 WITH n, 
#                      collect(DISTINCT {
#                         name: e.name,
#                         type: labels(e)[0]
#                      }) as all_entities
#                 RETURN 
#                     n.title as title,
#                     n.content as content,
#                     all_entities,
#                     size(all_entities) as entity_count
#                 """
#                 result = session.run(query)
                
#                 for record in tqdm(result, desc="加载文档", unit="个"):
#                     entities = record['all_entities'] if record['all_entities'] else []
#                     documents.append({
#                         'title': record['title'],
#                         'content': record['content'],
#                         'entities': entities,
#                         'entity_count': len(entities)
#                     })
#         except Exception as e:
#             log_progress(f"❌ 获取文档失败: {e}")
        
#         return documents

#     def load_processed_docs(self):
#         """加载已处理的文档"""
#         if os.path.exists(PROCESSED_DOCS_FILE):
#             try:
#                 with open(PROCESSED_DOCS_FILE, 'r', encoding='utf-8') as f:
#                     return set(json.load(f))
#             except Exception as e:
#                 log_progress(f"⚠️ 加载已处理文档失败: {e}")
#                 return set()
#         return set()

#     def save_processed_docs(self, processed_docs):
#         """保存已处理的文档"""
#         try:
#             with open(PROCESSED_DOCS_FILE, 'w', encoding='utf-8') as f:
#                 json.dump(list(processed_docs), f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             log_progress(f"⚠️ 保存已处理文档失败: {e}")

#     def clean_relation_type(self, relation):
#         """清理关系类型，使其适合作为Neo4j关系标签"""
#         # 移除特殊字符，只保留中文、英文、数字、下划线
#         cleaned = re.sub(r'[^\w\u4e00-\u9fa5]', '_', relation)
#         # 移除多余的下划线
#         cleaned = re.sub(r'_+', '_', cleaned)
#         # 移除首尾下划线
#         cleaned = cleaned.strip('_')
#         # 如果为空，使用默认值
#         if not cleaned:
#             cleaned = "相关"
#         return cleaned

#     def extract_relations_from_document(self, document, retries=3):
#         """提取单个文档的关系"""
#         if not document['entities'] or document['entity_count'] == 0:
#             return {
#                 'title': document['title'],
#                 'relations': [],
#                 'entity_count': 0,
#                 'status': 'no_entities'
#             }

#         # 准备实体列表
#         entities_text = "文档中已有的实体列表：\n"
#         entity_map = {}
#         for i, entity in enumerate(document['entities'][:30]):  # 限制显示数量
#             entity_name = entity['name']
#             entity_type = entity['type']
#             entities_text += f"{i+1}. [{entity_type}] {entity_name}\n"
#             entity_map[entity_name] = entity_type
        
#         # 如果实体超过30个，添加提示
#         if len(document['entities']) > 30:
#             entities_text += f"...以及 {len(document['entities']) - 30} 个其他实体\n"
#             # 添加其他实体到映射
#             for entity in document['entities'][30:]:
#                 entity_map[entity['name']] = entity['type']

#         # 截取文档内容
#         content_preview = document['content']
#         if len(content_preview) > 3000:
#             content_preview = content_preview[:1500] + "...[中间部分省略]..." + content_preview[-1500:]

#         prompt = f"""
# 你是一个知识图谱专家，需要从新闻文档中提取所有实体之间的关系。

# # 文档信息
# **标题：** {document['title']}

# **文档内容：**
# {content_preview}

# # 已有的实体：
# {entities_text}

# # 任务要求
# 请分析上述文档中这些实体之间的关系，提取实体之间的关系三元组（头实体-关系-尾实体）。

# # 提取规则
# 1. 只提取文档中明确提到或强烈暗示的关系
# 2. 关系方向要准确：明确是A→B的关系方向
# 3. 关系要具体：使用具体的动词短语，如"任职于"、"位于"、"参与"、"投资"、"领导"等

# # 输出格式
# 请以JSON数组格式输出，每个关系包含以下5个字段：
# ```json
# [
#   {{
#     "head_entity": "实体A名称",
#     "head_type": "实体A的类型",
#     "relation": "关系类型（使用动词短语）",
#     "tail_entity": "实体B名称", 
#     "tail_type": "实体B的类型"
#   }}
# ]
# 注意：如果没有可靠关系，返回空数组 []。
# """

#         for attempt in range(retries):
#             try:
#                 response = self.openai_client.chat.completions.create(
#                     model="deepseek-v3",
#                     messages=[
#                         {"role": "system", "content": "你是一个专业的知识图谱构建专家，严格基于文档内容提取实体间的关系。输出规范的JSON格式。"},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.7,
#                     max_tokens=10000,
#                     response_format={"type": "json_object"}
#                 )

#                 content = response.choices[0].message.content

#                 # 清理JSON
#                 json_str = content.strip()
#                 if "```json" in json_str:
#                     json_str = json_str.split("```json")[1].split("```")[0].strip()
#                 elif "```" in json_str:
#                     json_str = json_str.split("```")[1].split("```")[0].strip()

#                 result = json.loads(json_str)

#                 # 处理响应格式
#                 relations = []
#                 if isinstance(result, dict):
#                     for key in ["relations", "data", "triples"]:
#                         if key in result and isinstance(result[key], list):
#                             relations = result[key]
#                             break
#                 elif isinstance(result, list):
#                     relations = result

#                 # 验证关系
#                 validated_relations = []
#                 for rel in relations:
#                     if not isinstance(rel, dict):
#                         continue

#                     head_entity = rel.get("head_entity") or rel.get("head") or rel.get("source")
#                     relation = rel.get("relation") or rel.get("relationship")
#                     tail_entity = rel.get("tail_entity") or rel.get("tail") or rel.get("target")

#                     if not all([head_entity, relation, tail_entity]):
#                         continue

#                     # 获取实体类型
#                     head_type = rel.get("head_type", entity_map.get(head_entity, "未知"))
#                     tail_type = rel.get("tail_type", entity_map.get(tail_entity, "未知"))

#                     # 清理关系类型
#                     clean_relation = self.clean_relation_type(relation)

#                     validated_relations.append({
#                         "head_entity": head_entity,
#                         "head_type": head_type,
#                         "relation": clean_relation,
#                         "tail_entity": tail_entity,
#                         "tail_type": tail_type
#                     })

#                 return {
#                     'title': document['title'],
#                     'relations': validated_relations,
#                     'entity_count': document['entity_count'],
#                     'status': 'success',
#                     'relations_count': len(validated_relations)
#                 }

#             except json.JSONDecodeError:
#                 if attempt < retries - 1:
#                     time.sleep(2)
#                     continue
#                 else:
#                     return {
#                         'title': document['title'],
#                         'relations': [],
#                         'entity_count': document['entity_count'],
#                         'status': 'json_error'
#                     }
#             except Exception as e:
#                 log_progress(f"⚠️ 文档 '{document['title'][:30]}...' API调用失败: {str(e)[:100]}，尝试 {attempt+1}/{retries}")
#                 if attempt < retries - 1:
#                     time.sleep(3)
#                     continue

#         return {
#             'title': document['title'],
#             'relations': [],
#             'entity_count': document['entity_count'],
#             'status': 'failed'
#         }

#     def save_relations_to_neo4j(self, results):
#         """将提取的关系保存到Neo4j"""
#         if not self.driver:
#             log_progress("❌ 数据库未连接")
#             return 0, 0

#         total_created = 0
#         total_skipped = 0

#         log_progress("💾 开始保存关系到Neo4j...")

#         with self.driver.session() as session:
#             for result in tqdm(results, desc="保存关系", unit="个"):
#                 if not result.get('relations'):
#                     total_skipped += 1
#                     continue

#                 title = result['title']
#                 for rel in result['relations']:
#                     try:
#                         # 动态创建关系，关系类型就是提取的关系
#                         relation_type = rel['relation']
                        
#                         query = f"""
#                         // 查找头实体节点
#                         MATCH (head {{name: $head_name}})
#                         WHERE any(label IN labels(head) WHERE label = $head_type)
                        
#                         // 查找尾实体节点
#                         MATCH (tail {{name: $tail_name}})
#                         WHERE any(label IN labels(tail) WHERE label = $tail_type)
                        
#                         // 创建关系，关系类型是动态的
#                         MERGE (head)-[r:`{relation_type}` {{
#                             source_document: $title,
#                             extracted_at: timestamp()
#                         }}]->(tail)
                        
#                         RETURN count(r) as created
#                         """
                        
#                         query_result = session.run(query, {
#                             'head_name': rel['head_entity'],
#                             'head_type': rel['head_type'],
#                             'tail_name': rel['tail_entity'],
#                             'tail_type': rel['tail_type'],
#                             'title': title
#                         })
                        
#                         record = query_result.single()
#                         if record and record['created'] > 0:
#                             total_created += 1
#                         else:
#                             total_skipped += 1
                            
#                     except Exception as e:
#                         # 备用方案：只按名称匹配
#                         try:
#                             simple_query = f"""
#                             MATCH (head {{name: $head_name}})
#                             MATCH (tail {{name: $tail_name}})
#                             MERGE (head)-[r:`{rel['relation']}` {{
#                                 source: $title
#                             }}]->(tail)
#                             """
                            
#                             session.run(simple_query, {
#                                 'head_name': rel['head_entity'],
#                                 'tail_name': rel['tail_entity'],
#                                 'title': title
#                             })
#                             total_created += 1
#                         except Exception:
#                             total_skipped += 1

#         log_progress(f"✅ 关系保存完成: 成功创建 {total_created} 个关系，跳过 {total_skipped} 个")
#         return total_created, total_skipped

#     def save_results_to_file(self, results, errors):
#         """保存提取结果到文件"""
#         try:
#             # 保存关系结果
#             output_data = {
#                 "metadata": {
#                     "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
#                     "total_documents": len(results),
#                     "total_relations": sum(len(r.get('relations', [])) for r in results),
#                 },
#                 "documents": results
#             }
            
#             with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
#                 json.dump(output_data, f, ensure_ascii=False, indent=2)
            
#             log_progress(f"💾 关系结果已保存到 {RESULTS_FILE}")
            
#             # 保存错误信息
#             if errors:
#                 with open(ERRORS_FILE, 'w', encoding='utf-8') as f:
#                     json.dump(errors, f, ensure_ascii=False, indent=2)
#                 log_progress(f"💾 错误信息已保存到 {ERRORS_FILE}")
            
#         except Exception as e:
#             log_progress(f"❌ 保存结果失败: {e}")

#     def analyze_results(self, results):
#         """分析提取结果"""
#         log_progress("\n" + "="*60)
#         log_progress("📊 关系提取结果分析")
#         log_progress("="*60)

#         stats = {
#             "total_docs": len(results),
#             "successful_docs": sum(1 for r in results if r.get('status') == 'success'),
#             "no_entities_docs": sum(1 for r in results if r.get('status') == 'no_entities'),
#             "failed_docs": sum(1 for r in results if r.get('status') not in ['success', 'no_entities']),
#             "total_relations": sum(len(r.get('relations', [])) for r in results),
#             "avg_relations_per_doc": 0,
#             "relation_types": {}
#         }
        
#         if stats['total_docs'] > 0:
#             stats['avg_relations_per_doc'] = stats['total_relations'] / stats['total_docs']
        
#         # 关系类型统计
#         for result in results:
#             if result.get('relations'):
#                 for rel in result['relations']:
#                     rel_type = rel['relation']
#                     stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
        
#         # 显示统计信息
#         log_progress(f"📄 总处理文档: {stats['total_docs']}")
        
#         if stats['total_docs'] > 0:
#             log_progress(f"✅ 成功文档: {stats['successful_docs']} ({stats['successful_docs']/stats['total_docs']*100:.1f}%)")
#             log_progress(f"⚠️ 无实体文档: {stats['no_entities_docs']} ({stats['no_entities_docs']/stats['total_docs']*100:.1f}%)")
#             log_progress(f"❌ 失败文档: {stats['failed_docs']} ({stats['failed_docs']/stats['total_docs']*100:.1f}%)")
        
#         log_progress(f"🔗 总关系数: {stats['total_relations']}")
#         log_progress(f"📈 平均关系数/文档: {stats['avg_relations_per_doc']:.2f}")
        
#         # 关系类型统计
#         if stats['relation_types']:
#             log_progress(f"\n📋 关系类型分布 (共 {len(stats['relation_types'])} 种):")
#             top_relations = sorted(stats['relation_types'].items(), key=lambda x: x[1], reverse=True)[:15]
#             for rel_type, count in top_relations:
#                 percentage = count / stats['total_relations'] * 100 if stats['total_relations'] > 0 else 0
#                 log_progress(f"  {rel_type}: {count} ({percentage:.1f}%)")
        
#         return stats

#     def save_statistics(self, stats):
#         """保存统计信息"""
#         try:
#             with open(STATS_FILE, 'w', encoding='utf-8') as f:
#                 json.dump(stats, f, ensure_ascii=False, indent=2)
            
#             log_progress(f"📈 统计信息已保存到 {STATS_FILE}")
#         except Exception as e:
#             log_progress(f"❌ 保存统计信息失败: {e}")

#     def run_complete_extraction(self, save_to_db=True):
#         """运行完整的提取流程，处理所有数据"""
#         log_progress("🚀 开始处理所有文档的实体关系提取")
#         log_progress("="*70)
        
#         self.stats['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
#         start_time = time.time()
        
#         # 1. 连接数据库
#         if not self.connect_neo4j():
#             return
        
#         try:
#             # 2. 获取文档总数
#             total_docs = self.get_total_document_count()
#             log_progress(f"📊 知识图谱中共有 {total_docs} 个文档")
            
#             if total_docs == 0:
#                 log_progress("❌ 没有找到文档，流程终止")
#                 return
            
#             # 3. 加载已处理的文档
#             processed_docs = self.load_processed_docs()
#             log_progress(f"📋 已处理文档数: {len(processed_docs)}")
            
#             # 4. 获取所有文档
#             log_progress("\n📥 获取所有文档及其实体...")
#             all_documents = self.get_all_documents()
            
#             if not all_documents:
#                 log_progress("❌ 未能获取文档")
#                 return
            
#             # 过滤已处理的文档
#             documents_to_process = [doc for doc in all_documents if doc['title'] not in processed_docs]
            
#             if not documents_to_process:
#                 log_progress("🎉 所有文档都已处理过，无需处理")
#                 return
            
#             log_progress(f"🎯 需要处理 {len(documents_to_process)} 个新文档")
            
#             # 5. 提取关系
#             log_progress("\n🔍 开始提取文档关系...")
#             all_results = []
#             errors = []
            
#             for i, doc in enumerate(tqdm(documents_to_process, desc="提取关系", unit="个")):
#                 try:
#                     log_progress(f"\n[{i+1}/{len(documents_to_process)}] 处理文档: '{doc['title'][:50]}...'")
#                     log_progress(f"  实体数量: {doc['entity_count']}")
                    
#                     result = self.extract_relations_from_document(doc)
#                     all_results.append(result)
                    
#                     # 记录处理状态
#                     if result['status'] == 'success':
#                         log_progress(f"  ✅ 提取到 {len(result['relations'])} 个关系")
#                     elif result['status'] == 'no_entities':
#                         log_progress(f"  ⚠️  文档没有实体")
#                     else:
#                         log_progress(f"  ❌ 提取失败")
                    
#                     # 添加处理成功的文档到已处理列表
#                     processed_docs.add(doc['title'])
                    
#                     # 定期保存进度
#                     if (i + 1) % 10 == 0:
#                         self.save_processed_docs(processed_docs)
#                         log_progress(f"💾 已保存 {i+1} 个文档的处理进度")
                    
#                     # 避免API限制
#                     if i < len(documents_to_process) - 1:
#                         time.sleep(1)  # 每秒处理1个文档
                        
#                 except Exception as e:
#                     error_msg = f"处理文档 '{doc['title']}' 时出错: {str(e)}"
#                     log_progress(f"❌ {error_msg}")
#                     errors.append({
#                         'title': doc['title'],
#                         'error': error_msg,
#                         'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
#                     })
            
#             # 6. 保存已处理文档列表
#             self.save_processed_docs(processed_docs)
            
#             # 7. 分析结果
#             stats = self.analyze_results(all_results)
            
#             # 8. 保存结果到文件
#             log_progress("\n💾 保存提取结果到文件...")
#             self.save_results_to_file(all_results, errors)
            
#             # 9. 保存到Neo4j
#             if save_to_db and all_results:
#                 log_progress("\n🗄️ 保存关系到Neo4j...")
#                 created, skipped = self.save_relations_to_neo4j(all_results)
#                 log_progress(f"✅ 成功创建 {created} 个关系，跳过 {skipped} 个")
            
#             # 10. 最终统计
#             end_time = time.time()
#             total_duration = end_time - start_time
#             self.stats['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
#             log_progress("\n" + "="*70)
#             log_progress("📊 最终统计结果")
#             log_progress("="*70)
            
#             log_progress(f"⏱️ 总耗时: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
#             log_progress(f"⚡ 平均速度: {len(documents_to_process)/total_duration*60:.1f} 文档/分钟")
            
#             # 更新统计信息并保存
#             stats.update({
#                 'start_time': self.stats['start_time'],
#                 'end_time': self.stats['end_time'],
#                 'duration_seconds': total_duration,
#                 'processed_documents': len(documents_to_process)
#             })
            
#             self.save_statistics(stats)
            
#             log_progress("\n" + "="*70)
#             log_progress("🎉 所有文档的关系提取流程完成！")
#             log_progress("="*70)
            
#             log_progress(f"\n📁 输出文件:")
#             log_progress(f"  • {RESULTS_FILE}: 提取的关系结果")
#             log_progress(f"  • {ERRORS_FILE}: 错误信息")
#             log_progress(f"  • {PROCESSED_DOCS_FILE}: 已处理文档列表")
#             log_progress(f"  • {STATS_FILE}: 统计信息")
            
#             return all_results
            
#         except Exception as e:
#             log_progress(f"❌ 流程执行出错: {e}")
#             import traceback
#             traceback.print_exc()
        
#         finally:
#             if self.driver:
#                 self.driver.close()
#                 log_progress("\n🔌 已关闭数据库连接")

# def main():
#     """主函数"""
#     print("\n" + "="*70)
#     print("实体关系提取系统 - 处理所有数据")
#     print("="*70)
    
#     print("\n📊 配置信息:")
#     print(f"  • Neo4j服务器: {NEO4J_URI}")
#     print(f"  • 大模型: deepseek-v3")
#     print(f"  • 输出目录: {OUTPUT_DIR}")
#     print("="*70)
    
#     # 确认开始
#     print("\n⚠️  注意：这将处理知识图谱中的所有文档，可能需要较长时间。")
#     print("   系统会记录处理进度，如果中断可以从中断点继续。")
    
#     # 创建提取器
#     extractor = CompleteRelationExtractor(
#         neo4j_uri=NEO4J_URI,
#         neo4j_user=NEO4J_USER,
#         neo4j_password=NEO4J_PASSWORD,
#         api_key=API_KEY,
#         base_url=BASE_URL
#     )
    
#     # 开始处理
#     extractor.run_complete_extraction(save_to_db=True)

# if __name__ == "__main__":
#     main()
"""
批量实体关系提取系统 - 处理所有知识图谱数据
文件名: batch_extract_entity_relations.py
"""
import os
import json
import time
import re
import logging
import sys
from typing import List, Dict, Any
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
from datetime import datetime

# ============================================================================
# 配置参数 - 批量处理
# ============================================================================

# Neo4j配置
NEO4J_URI = "bolt://122.226.69.234:3045"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j123456"

# 大模型配置
API_KEY = "__SET_VIA_ENVIRONMENT__"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3"

# 批量处理配置
BATCH_SIZE = 50  # 每批处理的文档数
MAX_CONCURRENT = 3  # 最大并发数（避免API限制）
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试延迟（秒）

# 输出目录配置
OUTPUT_DIR = "/mnt/project/Qwen2.5-7B-Instruct-news/Knowledge_Boundary_Awareness/multihop_qa/data/batch_relation_extraction"
PROCESSED_DOCS_FILE = os.path.join(OUTPUT_DIR, "processed_docs.json")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.jsonl")
ERRORS_FILE = os.path.join(OUTPUT_DIR, "errors.jsonl")
STATS_FILE = os.path.join(OUTPUT_DIR, "statistics.json")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 设置日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(OUTPUT_DIR, 'batch_extraction.log'),
    filemode='w',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 添加控制台输出
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def log_progress(message):
    """记录进度信息"""
    logger.info(message)
    print(message)

class BatchRelationExtractor:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, api_key, base_url):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
        self.driver = None
        self.stats = {
            "total_docs": 0,
            "processed_docs": 0,
            "successful_docs": 0,
            "failed_docs": 0,
            "total_relations": 0,
            "start_time": None,
            "end_time": None
        }

    def connect_neo4j(self):
        """连接Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            self.driver.verify_connectivity()
            log_progress("✅ 成功连接到 Neo4j 数据库")
            return True
        except Exception as e:
            log_progress(f"❌ 连接 Neo4j 失败: {e}")
            return False

    def get_total_document_count(self):
        """获取文档总数"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:NewsArticle)
                WHERE n.content IS NOT NULL AND n.content <> ''
                RETURN count(n) as total_count
                """
                result = session.run(query)
                record = result.single()
                return record['total_count'] if record else 0
        except Exception as e:
            log_progress(f"❌ 获取文档总数失败: {e}")
            return 0

    def get_documents_batch(self, skip=0, limit=100):
        """批量获取文档"""
        documents = []
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n:NewsArticle)
                WHERE n.content IS NOT NULL AND n.content <> ''
                WITH n
                ORDER BY n.title
                SKIP $skip LIMIT $limit
                OPTIONAL MATCH (n)-[rel]->(e)
                WHERE e.name IS NOT NULL AND e.name <> ''
                WITH n, 
                     collect(DISTINCT {
                        name: e.name,
                        type: labels(e)[0]
                     }) as all_entities
                RETURN 
                    n.title as title,
                    n.content as content,
                    all_entities,
                    size(all_entities) as entity_count
                """
                result = session.run(query, skip=skip, limit=limit)
                
                for record in result:
                    entities = record['all_entities'] if record['all_entities'] else []
                    documents.append({
                        'title': record['title'],
                        'content': record['content'],
                        'entities': entities,
                        'entity_count': len(entities)
                    })
        except Exception as e:
            log_progress(f"❌ 获取文档批次失败: {e}")
        
        return documents

    def load_processed_docs(self):
        """加载已处理的文档"""
        if os.path.exists(PROCESSED_DOCS_FILE):
            try:
                with open(PROCESSED_DOCS_FILE, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                log_progress(f"⚠️ 加载已处理文档失败: {e}")
                return set()
        return set()

    def save_processed_doc(self, doc_title):
        """保存已处理的文档"""
        processed_docs = self.load_processed_docs()
        processed_docs.add(doc_title)
        
        try:
            with open(PROCESSED_DOCS_FILE, 'w', encoding='utf-8') as f:
                json.dump(list(processed_docs), f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_progress(f"⚠️ 保存已处理文档失败: {e}")

    def clean_relation_type(self, relation):
        """清理关系类型，使其适合作为Neo4j关系标签"""
        # 移除特殊字符，只保留中文、英文、数字、下划线
        cleaned = re.sub(r'[^\w\u4e00-\u9fa5]', '_', relation)
        # 移除多余的下划线
        cleaned = re.sub(r'_+', '_', cleaned)
        # 移除首尾下划线
        cleaned = cleaned.strip('_')
        # 如果为空，使用默认值
        if not cleaned:
            cleaned = "相关"
        return cleaned

    def extract_relations_from_document_batch(self, documents):
        """批量提取文档关系"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            # 提交所有任务
            future_to_doc = {
                executor.submit(self._extract_single_document, doc): doc 
                for doc in documents
            }
            
            # 处理结果
            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log_progress(f"❌ 处理文档失败 '{doc['title'][:30]}...': {e}")
                    results.append({
                        'title': doc['title'],
                        'relations': [],
                        'status': 'failed',
                        'error': str(e)
                    })
        
        return results

    def _extract_single_document(self, document):
        """提取单个文档的关系"""
        if not document['entities'] or document['entity_count'] == 0:
            return {
                'title': document['title'],
                'relations': [],
                'entity_count': 0,
                'status': 'no_entities'
            }

        # 准备实体列表
        entities_text = "文档中已有的实体列表：\n"
        entity_map = {}
        for i, entity in enumerate(document['entities']):
            entity_name = entity['name']
            entity_type = entity['type']
            entities_text += f"{i+1}. [{entity_type}] {entity_name}\n"
            entity_map[entity_name] = entity_type

        # 截取文档内容
        content_preview = document['content']
        if len(content_preview) > 4800:
            content_preview = content_preview[:2400] + "...[中间部分省略]..." + content_preview[-2400:]

        prompt = f"""
你是一个知识图谱专家，需要从新闻文档中提取所有实体之间的关系。

# 文档信息
**标题：** {document['title']}

**文档内容：**
{content_preview}

# 已有的实体：
{entities_text}

# 任务要求
请分析上述文档中这些实体之间的关系，提取实体之间的关系三元组（头实体-关系-尾实体）。

# 提取规则
1. 只提取文档中明确提到或强烈暗示的关系
2. 关系方向要准确：明确是A→B的关系方向
3. 关系要具体：使用具体的动词短语，如"任职于"、"位于"、"参与"、"投资"、"领导"等

# 输出格式
请以JSON数组格式输出，每个关系包含以下5个字段：
```json
[
  {{
    "head_entity": "实体A名称",
    "head_type": "实体A的类型",
    "relation": "关系类型",
    "tail_entity": "实体B名称", 
    "tail_type": "实体B的类型"
  }}
]
注意：如果没有可靠关系，返回空数组 []。
"""

        for attempt in range(MAX_RETRIES):
            try:
                response = self.openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一个专业的知识图谱构建专家，严格基于文档内容提取实体间的关系。输出规范的JSON格式。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content

                # 清理JSON
                json_str = content.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                result = json.loads(json_str)

                # 处理响应格式
                relations = []
                if isinstance(result, dict):
                    for key in ["relations", "data", "triples"]:
                        if key in result and isinstance(result[key], list):
                            relations = result[key]
                            break
                elif isinstance(result, list):
                    relations = result

                # 验证关系
                validated_relations = []
                for rel in relations:
                    if not isinstance(rel, dict):
                        continue

                    head_entity = rel.get("head_entity") or rel.get("head") or rel.get("source")
                    relation = rel.get("relation") or rel.get("relationship")
                    tail_entity = rel.get("tail_entity") or rel.get("tail") or rel.get("target")

                    if not all([head_entity, relation, tail_entity]):
                        continue

                    # 获取实体类型
                    head_type = rel.get("head_type", entity_map.get(head_entity, "未知"))
                    tail_type = rel.get("tail_type", entity_map.get(tail_entity, "未知"))

                    # 清理关系类型
                    clean_relation = self.clean_relation_type(relation)

                    validated_relations.append({
                        "head_entity": head_entity,
                        "head_type": head_type,
                        "relation": clean_relation,
                        "tail_entity": tail_entity,
                        "tail_type": tail_type
                    })

                return {
                    'title': document['title'],
                    'relations': validated_relations,
                    'entity_count': document['entity_count'],
                    'status': 'success',
                    'relations_count': len(validated_relations)
                }

            except json.JSONDecodeError:
                log_progress(f"⚠️ 文档 '{document['title'][:30]}...' JSON解析失败，尝试 {attempt+1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return {
                        'title': document['title'],
                        'relations': [],
                        'entity_count': document['entity_count'],
                        'status': 'json_error'
                    }
            except Exception as e:
                log_progress(f"⚠️ 文档 '{document['title'][:30]}...' API调用失败: {str(e)[:100]}，尝试 {attempt+1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue

        return {
            'title': document['title'],
            'relations': [],
            'entity_count': document['entity_count'],
            'status': 'failed'
        }

    def save_results_to_file(self, results, mode='append'):
        """保存结果到文件"""
        try:
            if mode == 'append':
                file_mode = 'a'
            else:
                file_mode = 'w'
            
            with open(RESULTS_FILE, file_mode, encoding='utf-8') as f:
                for result in results:
                    # 添加时间戳
                    result['extracted_at'] = datetime.now().isoformat()
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            log_progress(f"💾 结果已保存到 {RESULTS_FILE}")
        except Exception as e:
            log_progress(f"❌ 保存结果失败: {e}")

    def save_errors_to_file(self, errors):
        """保存错误信息到文件"""
        try:
            with open(ERRORS_FILE, 'a', encoding='utf-8') as f:
                for error in errors:
                    error['timestamp'] = datetime.now().isoformat()
                    f.write(json.dumps(error, ensure_ascii=False) + '\n')
        except Exception as e:
            log_progress(f"❌ 保存错误信息失败: {e}")

    def save_relations_to_neo4j_batch(self, results):
        """批量保存关系到Neo4j"""
        if not self.driver:
            log_progress("❌ 数据库未连接")
            return 0, 0

        total_created = 0
        total_skipped = 0

        log_progress("💾 开始批量保存关系到Neo4j...")

        try:
            with self.driver.session() as session:
                for result in tqdm(results, desc="保存关系到Neo4j"):
                    if not result.get('relations'):
                        total_skipped += 1
                        continue

                    title = result['title']
                    relations_to_save = []
                    
                    # 收集要保存的关系
                    for rel in result['relations']:
                        relations_to_save.append({
                            'head_name': rel['head_entity'],
                            'head_type': rel['head_type'],
                            'tail_name': rel['tail_entity'],
                            'tail_type': rel['tail_type'],
                            'relation_type': rel['relation'],
                            'title': title
                        })
                    
                    # 批量创建关系
                    if relations_to_save:
                        query = """
                        UNWIND $relations as rel
                        MATCH (head {name: rel.head_name})
                        WHERE any(label IN labels(head) WHERE label = rel.head_type)
                        MATCH (tail {name: rel.tail_name})
                        WHERE any(label IN labels(tail) WHERE label = rel.tail_type)
                        MERGE (head)-[r:`${rel.relation_type}` {
                            source_document: rel.title,
                            created_at: timestamp()
                        }]->(tail)
                        """
                        
                        try:
                            # 注意：Neo4j不支持关系类型参数化，这里需要特殊处理
                            # 这里我们按关系类型分组处理
                            grouped_relations = {}
                            for rel in relations_to_save:
                                rel_type = rel['relation_type']
                                if rel_type not in grouped_relations:
                                    grouped_relations[rel_type] = []
                                grouped_relations[rel_type].append(rel)
                            
                            # 按关系类型分别处理
                            for rel_type, rels in grouped_relations.items():
                                # 动态构建查询
                                dynamic_query = f"""
                                UNWIND $relations as rel
                                MATCH (head {{name: rel.head_name}})
                                WHERE any(label IN labels(head) WHERE label = rel.head_type)
                                MATCH (tail {{name: rel.tail_name}})
                                WHERE any(label IN labels(tail) WHERE label = rel.tail_type)
                                MERGE (head)-[r:`{rel_type}` {{
                                    source_document: rel.title,
                                    created_at: timestamp()
                                }}]->(tail)
                                RETURN count(r) as created
                                """
                                
                                result = session.run(dynamic_query, relations=rels)
                                record = result.single()
                                if record:
                                    total_created += record['created']
                                    
                        except Exception as e:
                            log_progress(f"⚠️ 批量保存关系失败: {e}")
                            total_skipped += len(relations_to_save)

        except Exception as e:
            log_progress(f"❌ 批量保存关系出错: {e}")

        log_progress(f"✅ 关系保存完成: 成功创建 {total_created} 个关系，跳过 {total_skipped} 个")
        return total_created, total_skipped

    def analyze_batch_results(self, results):
        """分析批量结果"""
        log_progress("\n📊 分析批量提取结果...")
        
        stats = {
            "total_docs": len(results),
            "successful_docs": sum(1 for r in results if r.get('status') == 'success'),
            "no_entities_docs": sum(1 for r in results if r.get('status') == 'no_entities'),
            "failed_docs": sum(1 for r in results if r.get('status') not in ['success', 'no_entities']),
            "total_relations": sum(len(r.get('relations', [])) for r in results),
            "avg_relations_per_doc": 0,
            "relation_types": {}
        }
        
        if stats['total_docs'] > 0:
            stats['avg_relations_per_doc'] = stats['total_relations'] / stats['total_docs']
        
        # 关系类型统计
        for result in results:
            if result.get('relations'):
                for rel in result['relations']:
                    rel_type = rel['relation']
                    stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
        
        return stats

    def save_statistics(self, stats):
        """保存统计信息"""
        try:
            stats['end_time'] = datetime.now().isoformat()
            stats['duration_seconds'] = (datetime.fromisoformat(stats['end_time']) - 
                                       datetime.fromisoformat(stats['start_time'])).total_seconds()
            
            with open(STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            log_progress(f"📈 统计信息已保存到 {STATS_FILE}")
        except Exception as e:
            log_progress(f"❌ 保存统计信息失败: {e}")

    def run_batch_extraction(self, total_docs_to_process=None, save_to_db=True):
        """运行批量提取流程"""
        log_progress("🚀 开始批量实体关系提取流程")
        log_progress("="*70)
        
        self.stats['start_time'] = datetime.now().isoformat()
        
        # 1. 连接数据库
        if not self.connect_neo4j():
            return
        
        try:
            # 2. 获取文档总数
            total_docs = self.get_total_document_count()
            log_progress(f"📊 知识图谱中共有 {total_docs} 个文档")
            
            if total_docs == 0:
                log_progress("❌ 没有找到文档，流程终止")
                return
            
            # 3. 加载已处理的文档
            processed_docs = self.load_processed_docs()
            log_progress(f"📋 已处理文档数: {len(processed_docs)}")
            
            # 4. 确定要处理的文档数
            if total_docs_to_process:
                docs_to_process = min(total_docs_to_process, total_docs - len(processed_docs))
            else:
                docs_to_process = total_docs - len(processed_docs)
            
            log_progress(f"🎯 计划处理 {docs_to_process} 个文档")
            
            # 5. 分批处理
            batches = (docs_to_process + BATCH_SIZE - 1) // BATCH_SIZE
            all_results = []
            
            for batch_num in range(batches):
                log_progress(f"\n🔄 处理批次 {batch_num+1}/{batches}")
                
                # 获取当前批次的文档
                skip = len(processed_docs) + (batch_num * BATCH_SIZE)
                limit = min(BATCH_SIZE, docs_to_process - (batch_num * BATCH_SIZE))
                
                log_progress(f"📥 获取文档: skip={skip}, limit={limit}")
                documents = self.get_documents_batch(skip=skip, limit=limit)
                
                if not documents:
                    log_progress("⚠️ 批次没有获取到文档，跳过")
                    continue
                
                # 过滤已处理的文档（以防万一）
                documents = [doc for doc in documents if doc['title'] not in processed_docs]
                
                if not documents:
                    log_progress("⚠️ 所有文档都已处理过，跳过")
                    continue
                
                log_progress(f"🔍 开始提取 {len(documents)} 个文档的关系...")
                
                # 批量提取关系
                batch_start_time = time.time()
                batch_results = self.extract_relations_from_document_batch(documents)
                batch_duration = time.time() - batch_start_time
                
                log_progress(f"✅ 批次提取完成，耗时 {batch_duration:.1f} 秒")
                
                # 分析批次结果
                batch_stats = self.analyze_batch_results(batch_results)
                log_progress(f"📊 批次统计: {batch_stats['successful_docs']}成功, "
                           f"{batch_stats['no_entities_docs']}无实体, "
                           f"{batch_stats['failed_docs']}失败, "
                           f"{batch_stats['total_relations']}关系")
                
                # 保存结果
                self.save_results_to_file(batch_results, mode='append' if batch_num > 0 else 'w')
                
                # 保存到Neo4j
                if save_to_db and batch_results:
                    created, skipped = self.save_relations_to_neo4j_batch(batch_results)
                    log_progress(f"🗄️ 保存到数据库: {created}创建, {skipped}跳过")
                
                # 更新已处理文档
                for result in batch_results:
                    self.save_processed_doc(result['title'])
                
                all_results.extend(batch_results)
                
                # 更新统计
                self.stats['processed_docs'] += len(documents)
                self.stats['successful_docs'] += batch_stats['successful_docs']
                self.stats['failed_docs'] += batch_stats['failed_docs']
                self.stats['total_relations'] += batch_stats['total_relations']
                
                # 显示进度
                progress = (batch_num + 1) / batches * 100
                log_progress(f"📈 总进度: {progress:.1f}% ({self.stats['processed_docs']}/{docs_to_process})")
                
                # 避免API限制，批次间延迟
                if batch_num < batches - 1:
                    delay = 10  # 批次间延迟10秒
                    log_progress(f"⏳ 等待 {delay} 秒后处理下一批次...")
                    time.sleep(delay)
            
            # 6. 最终分析和保存
            log_progress("\n" + "="*70)
            log_progress("📊 最终统计结果")
            log_progress("="*70)
            
            self.stats['end_time'] = datetime.now().isoformat()
            total_duration = (datetime.fromisoformat(self.stats['end_time']) - 
                            datetime.fromisoformat(self.stats['start_time'])).total_seconds()
            
            # 最终统计
            final_stats = self.analyze_batch_results(all_results)
            
            log_progress(f"📄 总处理文档: {final_stats['total_docs']}")
            log_progress(f"✅ 成功文档: {final_stats['successful_docs']} ({final_stats['successful_docs']/final_stats['total_docs']*100:.1f}%)")
            log_progress(f"⚠️ 无实体文档: {final_stats['no_entities_docs']} ({final_stats['no_entities_docs']/final_stats['total_docs']*100:.1f}%)")
            log_progress(f"❌ 失败文档: {final_stats['failed_docs']} ({final_stats['failed_docs']/final_stats['total_docs']*100:.1f}%)")
            log_progress(f"🔗 总关系数: {final_stats['total_relations']}")
            log_progress(f"📈 平均关系数/文档: {final_stats['avg_relations_per_doc']:.2f}")
            log_progress(f"⏱️ 总耗时: {total_duration:.1f} 秒 ({total_duration/3600:.2f} 小时)")
            log_progress(f"⚡ 平均速度: {final_stats['total_docs']/total_duration*60:.1f} 文档/分钟")
            
            # 关系类型统计
            if final_stats['relation_types']:
                log_progress(f"\n📋 关系类型分布 (共 {len(final_stats['relation_types'])} 种):")
                top_relations = sorted(final_stats['relation_types'].items(), key=lambda x: x[1], reverse=True)[:10]
                for rel_type, count in top_relations:
                    percentage = count / final_stats['total_relations'] * 100
                    log_progress(f"  {rel_type}: {count} ({percentage:.1f}%)")
            
            # 保存统计信息
            self.save_statistics(final_stats)
            
            log_progress("\n" + "="*70)
            log_progress("🎉 批量关系提取流程完成！")
            log_progress("="*70)
            
            log_progress(f"\n📁 输出文件:")
            log_progress(f"  • {RESULTS_FILE}: 提取结果")
            log_progress(f"  • {ERRORS_FILE}: 错误信息")
            log_progress(f"  • {PROCESSED_DOCS_FILE}: 已处理文档")
            log_progress(f"  • {STATS_FILE}: 统计信息")
            
        except Exception as e:
            log_progress(f"❌ 批量处理出错: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.driver:
                self.driver.close()
                log_progress("\n🔌 已关闭数据库连接")

def main():
    """主函数"""
    print("\n" + "="*70)
    print("批量实体关系提取系统")
    print("="*70)
    
    # 用户配置
    print("\n🔧 请选择处理模式:")
    print("1. 处理所有未处理文档")
    print("2. 处理指定数量的文档")
    print("3. 只处理测试批次")
    
    choice = 1
    
    total_docs_to_process = None
    save_to_db = True
    
    if choice == "1":
        log_progress("🎯 模式: 处理所有未处理文档")
    elif choice == "2":
        try:
            total_docs_to_process = int(input("请输入要处理的文档数量: "))
            log_progress(f"🎯 模式: 处理 {total_docs_to_process} 个文档")
        except ValueError:
            log_progress("❌ 输入无效，使用默认模式")
    elif choice == "3":
        total_docs_to_process = 10
        save_to_db = False  # 测试模式不保存到数据库
        log_progress("🎯 模式: 测试模式 (处理10个文档，不保存到数据库)")
    else:
        log_progress("❌ 选择无效，使用默认模式1")
    
    print("\n📊 配置信息:")
    print(f"  • Neo4j服务器: {NEO4J_URI}")
    print(f"  • 大模型: {MODEL_NAME}")
    print(f"  • 批次大小: {BATCH_SIZE}")
    print(f"  • 最大并发: {MAX_CONCURRENT}")
    print(f"  • 保存到数据库: {'是' if save_to_db else '否'}")
    print("="*70)
    
    # 创建提取器
    extractor = BatchRelationExtractor(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        api_key=API_KEY,
        base_url=BASE_URL
    )
    
    # 开始批量处理
    extractor.run_batch_extraction(
        total_docs_to_process=total_docs_to_process,
        save_to_db=save_to_db
    )

if __name__ == "__main__":
    main()