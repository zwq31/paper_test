# 面向区域新闻可信问答系统的幻觉抑制方法研究

本仓库保存毕业设计已有实验的代码、文档、小型数据和评测结果，是从实验工作区导出的脱敏快照。

## 项目导航

| 目录 | 内容 |
|---|---|
| `Qwen2.5-7B-Instruct-news/` | 新闻数据构建、持续预训练、MOL-CPT、SFT、知识边界、拒答及 RAG |
| `H-Neurous/` | CETT 神经元特征与幻觉检测方法 |
| `semantic_entropy/` | 语义熵、离散语义熵与阈值分析 |
| `Qwen3-8B-news/` | Qwen3 新闻训练及生成实验 |
| `unknow_test/` | TriviaQA 与模型知识边界实验 |
| `Qwen2.5-1.5B-lora/`、`deepseek-r1-distill-llama-8b-qlora/`、`llama_grpo/` | 早期微调与强化学习实验 |
| `metrics/`、`unsloth-offline/` | 原工作区保留的离线指标和训练框架源码 |
| `paper_test/md/` | 项目全景、逐文件用途与实验结果汇总 |
| `paper_test/analysis/` | 已有结果复核脚本和指标台账 |
| `github-export/` | 本次导出的保留／排除清单与脱敏报告 |

建议先读：[已有实验分块实现与结果汇总](paper_test/md/已有实验分块实现与结果汇总.md)，再读[毕业设计项目结构与实验全景分析](paper_test/md/毕业设计项目结构与实验全景分析.md)。

## 本次上传范围

保留实验代码、说明文档、小于 50 MiB 的数据及结果、图表、所保留实验的配置。以下内容没有上传，也没有从原工作区删除：

- 整个 `milvus/` 目录。
- `model/`、`models/`、预训练 BERT 资源、模型快照和 checkpoint 目录；这些目录内附带的配置与结果也随目录排除，可在排除清单中定位。
- 模型／适配器／分类器权重，以及 pickle 等序列化模型文件。
- W&B／SwanLab 追踪目录、生成缓存、日志、临时文件、安装包和压缩归档。
- 单个文件大于或等于 50 MiB 的非模型数据，包括 CETT 大型特征文件。
- 无法作为有效 JSON 解析、需要后续修复的 notebook。
- 本地密钥配置及凭据文件；发现的内嵌凭据已在上传副本中替换为 `__SET_VIA_ENVIRONMENT__`。

逐文件范围以 [included_files.csv](github-export/included_files.csv) 和 [excluded_files.csv](github-export/excluded_files.csv) 为准；前者记录源文件大小，notebook 清理后的上传大小可能更小。新增的本 README、根目录忽略规则和导出报告不属于原工作区文件清单。

## 使用与复现

1. 本仓库不含大模型权重与完整的大型训练数据，不是一键运行环境；请按具体实验准备底座、adapter 和依赖。
2. 原脚本保留历史绝对路径，例如 `/mnt/project/...`，需要结合自己的工作区调整。
3. 凭据占位符必须在运行前改为安全的本地配置或环境变量读取。现有脚本尚未统一接入环境变量，不能仅创建 `.env` 就假设所有脚本会自动读取。
4. notebook 的输出和执行计数已清除；结构无效的 notebook 未纳入本次上传。第三方目录遗留的 Git LFS 过滤规则已在上传副本中移除，本次采用普通 Git 保存已筛选的小文件。
5. 部分脚本含多个历史入口；文档已记录标签、数据划分、评价口径和原型完整性等限制。已有分数不代表重新运行得到的结果。
6. 逐文件索引描述的是原工作区快照，因此指向已排除模型、Milvus 或大数据的链接在 GitHub 上可能不存在。

导出工具见 [prepare_github_upload.py](paper_test/tools/prepare_github_upload.py)：默认仅预检，`--write` 写入独立、空的目标目录，拒绝覆盖非空目录。不会删除或脱敏原实验文件。重新导出前先运行 `--help` 并明确选择源和目标目录。

第三方代码与资源保留其已有许可说明；本仓库未替原有第三方内容另行声明许可。
