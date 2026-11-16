# 新闻问答系统评估体系设计方案

## 一、评估体系概述

### 1.1 设计原则
- **全面性**：覆盖答案质量、系统性能、模块效果等多个维度
- **可操作性**：所有指标都可以通过现有数据计算得出
- **对比性**：支持不同配置间的横向对比
- **可解释性**：每个指标都有明确的业务含义

### 1.2 评估维度
```
评估体系
├── 答案质量指标 (Answer Quality)
├── 系统性能指标 (System Performance)  
├── 决策模块指标 (Decision Module)
├── 检索模块指标 (Retrieval Module)
└── RAG效果指标 (RAG Impact)
```

---

## 二、五大评估维度详细设计

### 2.1 答案质量指标 (Answer Quality Metrics)

#### 核心指标
| 指标名称 | 计算方法 | 业务意义 |
|---------|---------|---------|
| **准确率 (Accuracy)** | 正确回答数 / 总问题数 | 整体答案质量 |
| **幻觉率 (Hallucination Rate)** | 幻觉回答数 / 总回答数 | 错误信息比例 |
| **拒答率 (Rejection Rate)** | 拒答数 / 总问题数 | 系统谨慎程度 |
| **拒答准确率 (Rejection Precision)** | 正确拒答数 / 总拒答数 | 拒答决策质量 |
| **误拒率 (False Rejection)** | 错误拒答数 / 可答问题数 | 过度保守程度 |

#### 细分指标
- **直接回答准确率**：不使用RAG时的准确率
- **RAG回答准确率**：使用RAG后的准确率
- **不同问题类型的准确率**：
  - 时间敏感问题
  - 事实性问题
  - 推理性问题

#### 实现方案
```python
def evaluate_answer_quality(results):
    """
    输入：测试结果列表
    输出：答案质量指标字典
    
    需要的标注：
    1. is_correct: 答案是否正确
    2. is_hallucination: 是否为幻觉
    3. should_reject: 是否应该拒答
    """
    metrics = {
        "total": len(results),
        "correct": 0,
        "hallucination": 0,
        "rejection": 0,
        "correct_rejection": 0,
        "false_rejection": 0,
    }
    
    # 遍历统计
    # 计算各项指标
    
    return metrics
```

---

### 2.2 系统性能指标 (System Performance Metrics)

#### 延迟指标
| 指标名称 | 计算方法 | 目标值 |
|---------|---------|-------|
| **平均响应时间** | mean(response_time) | < 5秒 |
| **P50延迟** | 50th percentile | < 3秒 |
| **P95延迟** | 95th percentile | < 10秒 |
| **P99延迟** | 99th percentile | < 15秒 |

#### 吞吐量指标
- **QPS** (Queries Per Second)：每秒处理请求数
- **每小时处理量**：系统处理能力

#### 延迟分解
```
总延迟 = 置信度生成 + 初始答案生成 + [检索时间] + [RAG答案生成]

需要分析：
1. 各阶段耗时占比
2. 直接回答 vs RAG回答的延迟对比
3. 检索带来的额外开销
```

#### 实现方案
```python
def evaluate_performance(results):
    """
    利用已有的 time_stats 数据
    计算延迟分布和吞吐量
    """
    
    # 提取所有样本的处理时间
    times = [r["retrieval_decision"]["time_stats"]["total_processing_time"] 
             for r in results]
    
    # 计算延迟指标
    metrics = {
        "avg_latency": np.mean(times),
        "p50_latency": np.percentile(times, 50),
        "p95_latency": np.percentile(times, 95),
        "p99_latency": np.percentile(times, 99),
        "qps": 1.0 / np.mean(times),
    }
    
    # 分解延迟组成
    # 对比直接回答和RAG回答
    
    return metrics
```

---

### 2.3 决策模块指标 (Decision Module Metrics)

#### 决策质量指标
| 指标名称 | 计算方法 | 业务意义 |
|---------|---------|---------|
| **RAG触发率** | RAG样本数 / 总样本数 | 系统对RAG的依赖程度 |
| **决策准确率** | 正确决策数 / 总决策数 | 决策模块质量 |
| **过度依赖率** | 不必要的RAG触发 / 总RAG触发 | 资源浪费程度 |
| **漏检率** | 应用RAG但未触发 / 应用RAG总数 | 决策遗漏程度 |

#### 阈值敏感性分析
**目标**：找到最优阈值组合

**实验设计**：
```python
# 阈值网格搜索
ppl_thresholds = [0.8, 1.0, 1.1, 1.3, 1.5]
confidence_thresholds = [0.3, 0.4, 0.45, 0.5, 0.6]
min_prob_thresholds = [0.2, 0.3, 0.36, 0.4, 0.5]

# 对每组阈值重新计算决策
# 统计准确率、RAG触发率、延迟等
# 绘制阈值-准确率曲线
```

**分析维度**：
1. 单阈值变化对准确率的影响
2. 阈值组合的交互效应
3. 准确率-延迟的权衡曲线

#### 决策一致性分析
```python
# 分析决策依据与结果的关系
cases = {
    "高置信度 + 直接回答 → 正确": count,
    "高置信度 + 直接回答 → 错误": count,
    "低置信度 + RAG → 正确": count,
    "低置信度 + RAG → 错误": count,
}
```

#### 实现方案
```python
def evaluate_decision_module(results):
    """
    评估决策模块性能
    """
    # 1. 基础统计
    direct_answers = [r for r in results if not r["use_retrieval"]]
    rag_answers = [r for r in results if r["use_retrieval"]]
    
    # 2. 计算决策指标
    metrics = {
        "rag_trigger_rate": len(rag_answers) / len(results),
        "direct_accuracy": calculate_accuracy(direct_answers),
        "rag_accuracy": calculate_accuracy(rag_answers),
    }
    
    # 3. 决策正确性分析
    # 需要判断：该用RAG时用了，不该用时没用
    
    return metrics

def analyze_threshold_sensitivity(results):
    """
    阈值敏感性分析
    """
    # 对不同阈值组合重新计算决策
    # 生成阈值-准确率曲线数据
    pass
```

---

### 2.4 检索模块指标 (Retrieval Module Metrics)

#### 检索质量指标
| 指标名称 | 计算方法 | 业务意义 |
|---------|---------|---------|
| **Recall@K** | 包含答案的文档数 / K | 检索召回能力 |
| **Precision@K** | 相关文档数 / K | 检索精确度 |
| **MRR** | 1 / 第一个相关文档排名 | 排序质量 |
| **答案覆盖率** | 包含答案的问题数 / 使用RAG的问题数 | 检索有效性 |

#### 上下文长度影响
**实验设计**：
```python
# 对比不同K值的效果
experiments = [
    {"top_k": 3},
    {"top_k": 5},   # 当前配置
    {"top_k": 10},
    {"top_k": 20},
]

# 对每个配置评估：
# 1. 准确率
# 2. 平均延迟
# 3. 检索质量指标
```

**分析目标**：
- 找到准确率-延迟的最佳平衡点
- 确定边际收益递减的拐点

#### 检索策略对比
```python
strategies = {
    "vector_only": "仅向量检索",
    "keyword_only": "仅关键词检索", 
    "hybrid_no_rerank": "混合检索不重排",
    "hybrid_with_rerank": "混合检索+重排序",  # 当前配置
}

# 对比各策略的：
# - 检索质量
# - 最终准确率
# - 延迟开销
```

#### 过滤阈值分析
**当前设置**：`final_score > 0.1`

**实验**：测试不同过滤阈值的影响
```python
filter_thresholds = [0.0, 0.05, 0.1, 0.15, 0.2]

# 分析：
# 1. 过滤掉多少文档
# 2. 对准确率的影响
# 3. 是否引入新的错误
```

#### 实现方案
```python
def evaluate_retrieval_module(results):
    """
    评估检索模块性能
    需要人工标注或自动判断：检索文档是否包含答案
    """
    rag_results = [r for r in results if r["use_retrieval"]]
    
    recall_scores = []
    precision_scores = []
    has_answer_count = 0
    
    for item in rag_results:
        docs = item["retrieved_docs_preview"]
        
        # 判断文档是否包含答案
        relevant_docs = check_docs_relevance(docs, item["standard_answer"])
        
        # 计算指标
        recall = len(relevant_docs) / len(docs)
        # ...
    
    return metrics

def analyze_context_length(results, k_values):
    """
    上下文长度影响分析
    """
    # 模拟不同K值下的效果
    # 需要重新从检索结果中筛选top-k
    pass
```

---

### 2.5 RAG效果指标 (RAG Impact Metrics)

#### 提升幅度分析
| 对比维度 | 计算方法 | 期望结果 |
|---------|---------|---------|
| **整体准确率提升** | RAG准确率 - 直接回答准确率 | > 10% |
| **困难问题提升** | 低置信度问题的准确率差异 | > 20% |
| **时间敏感问题提升** | 有时间过滤的问题准确率差异 | > 15% |

#### 成本收益分析
```python
# 计算ROI
improvement = (rag_accuracy - direct_accuracy) / direct_accuracy * 100
latency_cost = (rag_latency - direct_latency) / direct_latency * 100

roi = improvement / latency_cost  # 每增加1%延迟带来的准确率提升
```

#### 失败案例分析
**RAG失败的原因分类**：
1. 检索失败：未检索到相关文档
2. 检索成功但生成失败：文档有答案但模型未正确利用
3. 检索引入噪声：检索到错误信息导致幻觉

#### 实现方案
```python
def evaluate_rag_impact(results):
    """
    评估RAG的整体效果
    """
    direct = [r for r in results if not r["use_retrieval"]]
    rag = [r for r in results if r["use_retrieval"]]
    
    metrics = {
        "direct_accuracy": calculate_accuracy(direct),
        "rag_accuracy": calculate_accuracy(rag),
        "improvement": rag_accuracy - direct_accuracy,
        "direct_avg_latency": calculate_latency(direct),
        "rag_avg_latency": calculate_latency(rag),
        "latency_overhead": rag_latency - direct_latency,
        "roi": improvement / latency_overhead,
    }
    
    # 细分场景分析
    # 失败案例分类
    
    return metrics
```

---

## 三、实施路线图

### Phase 1: 数据补全 (第1-2天)
**目标**：补充评估所需的标注数据

**任务清单**：
- [ ] 人工标注或自动评估答案正确性
- [ ] 标注是否为幻觉
- [ ] 判断是否应该拒答
- [ ] 标注检索文档是否包含答案
- [ ] 标注问题类型（时间敏感/事实性/推理性）

**工具**：可以开发简单的标注界面，或使用Excel批量标注

---

### Phase 2: 实现评估脚本 (第3-4天)
**目标**：自动化计算所有评估指标

**文件结构**：
```
evaluation/
├── evaluate_answer_quality.py      # 答案质量评估
├── evaluate_performance.py         # 性能评估
├── evaluate_decision.py            # 决策模块评估
├── evaluate_retrieval.py           # 检索模块评估
├── evaluate_rag_impact.py          # RAG效果评估
├── threshold_analysis.py           # 阈值敏感性分析
├── context_length_analysis.py      # 上下文长度分析
└── generate_report.py              # 生成完整报告
```

**关键函数**：
```python
# 主评估脚本
def evaluate_system(result_file, annotation_file):
    """
    完整的系统评估
    """
    # 加载结果和标注
    results = load_results(result_file)
    annotations = load_annotations(annotation_file)
    
    # 合并数据
    results = merge_with_annotations(results, annotations)
    
    # 执行各维度评估
    metrics = {
        "answer_quality": evaluate_answer_quality(results),
        "performance": evaluate_performance(results),
        "decision": evaluate_decision_module(results),
        "retrieval": evaluate_retrieval_module(results),
        "rag_impact": evaluate_rag_impact(results),
    }
    
    return metrics
```

---

### Phase 3: 对比实验 (第5-7天)
**目标**：测试不同配置对系统的影响

#### 实验1：阈值敏感性
```python
# 修改代码中的阈值常量
experiments = [
    {"name": "loose", "ppl": 1.5, "conf": 0.3, "prob": 0.2},
    {"name": "current", "ppl": 1.1, "conf": 0.445, "prob": 0.366},
    {"name": "strict", "ppl": 0.8, "conf": 0.6, "prob": 0.5},
]

# 对每组阈值：
# 1. 重新运行推理
# 2. 评估所有指标
# 3. 对比差异
```

#### 实验2：检索配置
```python
experiments = [
    {"top_k": 3, "rerank": True},
    {"top_k": 5, "rerank": True},   # 当前
    {"top_k": 10, "rerank": True},
    {"top_k": 5, "rerank": False},   # 不重排
]
```

#### 实验3：过滤阈值
```python
filter_experiments = [
    {"score_threshold": 0.0},   # 不过滤
    {"score_threshold": 0.1},   # 当前
    {"score_threshold": 0.2},
]
```

---

### Phase 4: 可视化与报告 (第8-9天)
**目标**：生成直观的可视化图表和完整报告

#### 关键图表
1. **答案质量对比图**
   - 柱状图：不同配置的准确率对比
   - 饼图：答案分布（正确/幻觉/拒答）

2. **性能分析图**
   - 延迟分布直方图
   - 延迟组成堆叠图
   - 准确率-延迟散点图

3. **阈值敏感性曲线**
   - 折线图：不同阈值下的准确率
   - 热力图：阈值组合效果矩阵

4. **检索效果分析**
   - 折线图：top-K对准确率的影响
   - 对比图：不同检索策略效果

5. **ROI分析**
   - 散点图：准确率提升 vs 延迟增加
   - 柱状图：不同场景下的RAG效果

#### 报告结构
```markdown
# 系统评估报告

## 1. 执行摘要
- 核心发现
- 主要问题
- 优化建议

## 2. 答案质量评估
- 整体指标
- 细分场景分析
- 典型案例

## 3. 系统性能评估
- 延迟分析
- 吞吐量分析
- 性能瓶颈

## 4. 决策模块评估
- 决策质量
- 阈值敏感性
- 优化建议

## 5. 检索模块评估
- 检索质量
- 配置影响
- 改进方向

## 6. RAG效果评估
- 整体提升
- 成本收益
- 适用场景

## 7. 综合建议
- 最优配置
- 改进优先级
- 后续工作
```

---

### Phase 5: 持续优化 (第10天+)
**目标**：基于评估结果迭代改进系统

**优化方向**：
1. 调整阈值到最优点
2. 优化检索配置
3. 改进prompt设计
4. 优化模型推理
5. 增加缓存机制

---

## 四、评估结果示例

### 4.1 完整评估报告结构
```json
{
  "evaluation_date": "2025-01-15",
  "dataset": "test_450.json",
  "total_samples": 450,
  
  "answer_quality": {
    "accuracy": 0.78,
    "hallucination_rate": 0.12,
    "rejection_rate": 0.10,
    "rejection_precision": 0.85,
    "false_rejection_rate": 0.08
  },
  
  "performance": {
    "avg_latency_seconds": 4.5,
    "p50_latency": 3.2,
    "p95_latency": 8.9,
    "p99_latency": 12.3,
    "qps": 0.22,
    "latency_breakdown": {
      "confidence_generation": {"seconds": 0.8, "percentage": 17.8},
      "answer_generation": {"seconds": 1.5, "percentage": 33.3},
      "retrieval": {"seconds": 1.2, "percentage": 26.7},
      "rag_answer": {"seconds": 1.0, "percentage": 22.2}
    }
  },
  
  "decision_module": {
    "rag_trigger_rate": 0.45,
    "direct_answer_accuracy": 0.85,
    "rag_answer_accuracy": 0.72,
    "decision_accuracy": 0.80,
    "threshold_analysis": {
      "optimal_ppl": 1.1,
      "optimal_confidence": 0.45,
      "optimal_min_prob": 0.36
    }
  },
  
  "retrieval_module": {
    "avg_recall@5": 0.68,
    "avg_precision@5": 0.72,
    "answer_coverage_rate": 0.75,
    "avg_retrieval_time": 1.2,
    "rerank_improvement": 0.08,
    "optimal_top_k": 5
  },
  
  "rag_impact": {
    "overall_improvement": 0.05,
    "hard_questions_improvement": 0.18,
    "time_sensitive_improvement": 0.12,
    "latency_overhead_seconds": 2.2,
    "roi": 2.27,
    "failure_reasons": {
      "retrieval_failure": 0.15,
      "generation_failure": 0.08,
      "noise_introduction": 0.02
    }
  }
}
```

---

## 五、关键要点总结

### 5.1 评估体系的核心价值
1. **全面性**：从5个维度评估系统
2. **可对比性**：支持不同配置的横向对比
3. **可优化性**：指出明确的改进方向
4. **可解释性**：每个指标都有业务含义

### 5.2 重点关注指标
| 优先级 | 指标 | 原因 |
|-------|------|------|
| P0 | 准确率、幻觉率 | 核心质量指标 |
| P0 | 平均延迟、P95延迟 | 用户体验关键 |
| P1 | RAG触发率、决策准确率 | 系统设计核心 |
| P1 | Recall@5、答案覆盖率 | 检索有效性 |
| P2 | ROI、成本收益比 | 系统价值证明 |

### 5.3 实施建议
1. **先快速实现基础指标**：准确率、延迟、RAG触发率
2. **再深入分析关键模块**：阈值敏感性、检索配置
3. **最后优化系统**：基于评估结果调整配置

### 5.4 预期输出
- **定量报告**：所有指标的具体数值
- **可视化图表**：10-15张关键图表
- **优化建议**：3-5条具体改进措施
- **最优配置**：系统参数的推荐值

---

## 六、常见问题

**Q1: 哪些指标最重要？**
A: 根据应用场景决定。如果追求质量，关注准确率和幻觉率；如果追求体验，关注延迟；如果论证RAG价值，关注ROI。

**Q2: 标注工作量大吗？**
A: 可以先标注100-200个样本，计算关键指标，然后决定是否需要全量标注。部分指标（如性能指标）不需要标注。

**Q3: 如何处理阈值敏感性分析的计算量？**
A: 不需要重新运行推理，只需要基于已有的PPL、置信度等数据，重新模拟决策过程即可。

**Q4: 评估结果如何呈现给老师？**
A: 准备三份材料：
1. 完整的技术报告（详细数据+图表）
2. PPT摘要（核心发现+可视化）
3. 可交互的Jupyter Notebook（可复现评估过程）

---

## 七、后续工作建议

完成评估后，可以进一步：

1. **A/B测试框架**：在真实场景中对比不同配置
2. **在线监控系统**：实时追踪关键指标
3. **自动调优工具**：根据反馈自动调整阈值
4. **用户研究**：评估用户对不同配置的满意度

---

**祝评估顺利！如有问题随时交流。**