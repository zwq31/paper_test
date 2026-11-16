# Agentic RAG 新闻问答系统

## 🎯 核心改进

相比原系统,Agentic RAG实现了以下关键改进:

### 1. **智能查询处理**
- ✅ **查询分析**: 自动识别问题类型(事实/时间/比较/因果)
- ✅ **查询改写**: 生成3个改写版本,从不同角度检索
- ✅ **查询分解**: 将复杂问题分解为多个子问题
- ✅ **HyDE增强**: 生成假设性文档提升检索召回

### 2. **迭代检索策略**
- ✅ **多轮检索**: 最多3轮迭代,逐步收集相关信息
- ✅ **增量去重**: 避免重复检索相同文档
- ✅ **上下文反馈**: 基于已检索文档调整后续查询

### 3. **高级重排序**
- ✅ **CrossEncoder**: 使用BGE-reranker-large精确打分
- ✅ **多样性排序**: MMR算法平衡相关性和多样性
- ✅ **时间加权**: 对时间敏感问题给予新闻时效性加权
- ✅ **综合排序**: 融合多种信号得到最优排序

### 4. **答案验证与自我反思**
- ✅ **支持度验证**: 检查答案是否被文档充分支持
- ✅ **置信度评估**: 计算0-1的答案置信度
- ✅ **缺失信息检测**: 识别回答所需但缺失的信息
- ✅ **二次检索**: 置信度低时针对缺失信息再检索

## 📊 预期性能提升

| 指标 | 原系统 | Agentic RAG | 提升 |
|------|--------|-------------|------|
| 召回率@10 | ~70% | **90%+** | +20% |
| 答案准确率 | ~65% | **85%+** | +20% |
| 置信度准确性 | 中等 | **高** | 显著 |
| 处理复杂问题 | 较弱 | **强** | 显著 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-community sentence-transformers
pip install pymilvus transformers torch pydantic
```

### 2. 目录结构

```
project/
├── generate_rag_agentic.py      # 主程序
├── agentic_rag_system.py        # Agentic RAG核心
├── advanced_reranker.py         # 高级重排序
├── generate_rag_time.py         # 原检索管理器(保留)
└── utils/
    ├── keywords_extractor.py
    ├── time_extractor.py
    ├── rag_utils.py
    └── model_utils.py
```

### 3. 运行

```bash
python generate_rag_agentic.py
```

## ⚙️ 配置说明

### 重要参数

```python
# 在 agentic_rag_system.py 中

# 迭代检索
max_iterations = 3          # 最大迭代轮数
docs_per_iteration = 10     # 每轮检索文档数

# 重排序
use_cross_encoder = True    # 使用CrossEncoder
use_diversity = True        # 使用多样性排序
use_temporal = True         # 使用时间加权

# 答案验证
confidence_threshold = 0.7  # 置信度阈值
```

### 查询改写示例

**原问题**: "2023年中国GDP增长率是多少"

**改写版本**:
1. "中国 2023 GDP 经济增长 数据"
2. "2023年中国国内生产总值增速"  
3. "去年中国经济增长情况统计"

### HyDE示例

**问题**: "华为Mate60发布时间"

**假设性文档**:
```
华为公司于2023年8月29日正式发布了新一代旗舰手机Mate60系列。
该系列包括Mate60、Mate60 Pro等多款机型,搭载麒麟9000S处理器...
```

## 🔧 核心工作流程

```mermaid
graph TD
    A[输入问题] --> B[查询分析]
    B --> C{需要分解?}
    C -->|是| D[分解为子问题]
    C -->|否| E[直接处理]
    D --> F[子问题1]
    D --> G[子问题2]
    E --> H[查询改写]
    F --> H
    G --> H
    H --> I[迭代检索-轮1]
    I --> J[去重合并]
    J --> K{文档够?}
    K -->|否| L[迭代检索-轮2]
    K -->|是| M[高级重排序]
    L --> M
    M --> N[CrossEncoder]
    N --> O[多样性排序]
    O --> P[时间加权]
    P --> Q[生成答案]
    Q --> R[答案验证]
    R --> S{置信度>0.7?}
    S -->|否| T[补充检索]
    S -->|是| U[返回答案]
    T --> Q
```

## 📈 效果对比示例

### 示例1: 事实性问题

**问题**: "2024年巴黎奥运会中国金牌数"

**原系统**:
- 检索方法: 单次混合检索
- 召回文档: 5个,相关性一般
- 答案: "中国在2024巴黎奥运会获得40枚金牌" (实际可能不准)
- 置信度: 0.45

**Agentic RAG**:
- 查询改写: 3个版本
- 迭代检索: 2轮,共15个文档
- 重排序: CrossEncoder + 时间加权
- 答案: "根据最新统计,中国代表团在2024年巴黎奥运会上获得40枚金牌,位居金牌榜第一"
- 置信度: 0.92 ✅
- 来源: 提供具体新闻来源和日期

### 示例2: 复杂比较问题

**问题**: "比较2023年和2024年中国新能源汽车销量"

**原系统**:
- 无法分解问题
- 只检索到2024年数据
- 答案不完整

**Agentic RAG**:
- 问题分解:
  - 子问题1: "2023年中国新能源汽车销量"
  - 子问题2: "2024年中国新能源汽车销量"
- 分别检索并合并
- 答案: 完整比较两年数据,包括增长率和趋势分析
- 置信度: 0.88 ✅

## 🎛️ 调优建议

### 1. 针对不同数据集

```python
# 新闻量大、质量高 -> 减少迭代
max_iterations = 2

# 新闻稀疏、难检索 -> 增加迭代和改写
max_iterations = 4
rewrite_versions = 5
```

### 2. 速度优化

```python
# 快速模式(牺牲一点准确率)
use_cross_encoder = False  # 跳过CrossEncoder
max_iterations = 1         # 只检索一轮

# 平衡模式(推荐)
use_cross_encoder = True
max_iterations = 2

# 精确模式(最高准确率)
use_cross_encoder = True
max_iterations = 3
use_diversity = True
```

### 3. 针对不同问题类型

```python
# 事实性问题 -> 重视准确性
use_cross_encoder = True
confidence_threshold = 0.8

# 观点性问题 -> 重视多样性
use_diversity = True
diversity_weight = 0.4
```

## 🐛 常见问题

### Q1: 处理速度太慢?
**A**: 
- 减少迭代次数: `max_iterations = 1`
- 关闭CrossEncoder: `use_cross_encoder = False`
- 使用GPU加速重排序模型

### Q2: 答案置信度普遍偏低?
**A**:
- 检查检索文档质量
- 调低置信度阈值: `confidence_threshold = 0.6`
- 增加迭代轮数收集更多文档

### Q3: 检索到的文档不相关?
**A**:
- 检查关键词提取质量
- 启用HyDE: `needs_hyde = True`
- 增加查询改写版本数

### Q4: 内存占用过高?
**A**:
- 减少每轮检索文档数
- 使用4bit量化模型
- 批量处理时清理缓存

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request!

## 📧 联系

如有问题,请提交Issue或联系维护者。