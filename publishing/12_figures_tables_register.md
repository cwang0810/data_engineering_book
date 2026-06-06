# 图表台账

本文件用于统一管理全书插图、流程图、架构图、表格的编号、状态、来源与负责人。

## 一、图表编号规则

- 图编号：`图 章节号-序号`
- 表编号：`表 章节号-序号`
- 附录图表：`图 A-1`、`表 B-1`
- 同一图表如果从线上素材改写，需标记“改绘”。

## 二、图表状态定义

- 待规划：还未决定是否保留
- 待绘制：已经确定内容，尚未出草图
- 草图完成：结构可评审，未定稿
- 可出版：编号、标题、来源和清晰度都达标

## 三、全书图表示例台账

| 编号 | 类型 | 标题 | 所属章节 | 来源 | 负责人 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 图 1-1 | 图 | 大模型时代数据工程职责重构图 | 第 1 章 | 本书自绘：`docs/images/part1/data_engineering_roles_1775830393574.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 1-2 | 图 | 全书十四篇制生命周期地图 | 第 1 章 | 本书自绘：`docs/images/part1/data_lifecycle_map_1775830407042.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 1-1 | 表 | DeepMind 旧范式模型与新范式模型数据资源对比 | 第 1 章 | 本书整理，依据 Gopher / Chinchilla 论文 | B | 可出版 | 需在最终参考文献中保留对应论文 |
| 表 1-2 | 表 | 大模型数据工程的三阶魔方：质量、规模与多样性成本约束基准矩阵 | 第 1 章 | 本书整理 | B | 可出版 | 已做出版风格降噪 |
| 表 1-3 | 表 | 传统经验 AI 机器学习研发生命线 vs 大语言模型原生数据体系 | 第 1 章 | 本书整理 | B | 可出版 | 已做出版风格降噪 |
| 表 1-4 | 表 | 六大 LLM 项目核心角色与数据接口职责定义表 | 第 1 章 | 本书整理 | B | 可出版 | SLA 数值为示例口径 |
| 表 1-5 | 表 | LLM 数据工程师 vs 传统 ML 数据工程师能力边界对照表 | 第 1 章 | 本书整理 | B | 可出版 | 已修复重复表头 |
| 表 1-6 | 表 | 各类型读者的章节优先级权重建议 | 第 1 章 | 本书整理 | B | 可出版 | 已同步 14 篇结构 |
| 表 2-1 | 表 | LLM 数据四阶段质量目标演变矩阵 | 第 2 章 | 本书整理 | B | 可出版 | 需最终统一表题样式 |
| 图 2-1 | 图 | 生命周期视角下的多维度质量分层架构 | 第 2 章 | 本书自绘：`docs/images/part1/data_quality_hierarchy_1775835516841.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 2-2 | 图 | 大模型数据缺陷与质量指标交叉映射图 | 第 2 章 | 本书自绘：`docs/images/part1/defect_metric_radar_1775835533937.png` | B / F | 可出版 | 已在正文补来源说明 |
| 图 2-3 | 图 | 数据评分卡驱动的自动截断与治理流 | 第 2 章 | 本书自绘：`docs/images/part1/data_quality_gates_1775835548587.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 3-1 | 图 | AI 原生数据栈五层架构 | 第 3 章 | 本书自绘：`docs/images/part1/ai_data_stack_architecture.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 3-2 | 图 | 训练数据成本治理闭环图 | 第 3 章 | 本书自绘：`docs/images/part1/cost_governance_loop.png` | B / F | 可出版 | 已在正文补来源与 alt text；价格口径需随正文同步 |
| 表 3-1 | 表 | Apache Spark vs Ray Data 核心特性对比 | 第 3 章 | 本书整理，依据 Spark / Ray 论文与工程实践 | B | 可出版 | 需最终统一表题样式 |
| 表 3-2 | 表 | 数据湖表格式选型对比 | 第 3 章 | 本书整理 | B | 可出版 | 技术生态可能变化，交付前复核 |
| 表 3-3 | 表 | 三类团队数据栈选型速查矩阵 | 第 3 章 | 本书整理 | B | 可出版 | 团队规模与周期为建议口径 |
| 图 4-1 | 图 | 预训练数据源分层地图 | 第 4 章 | 本书自绘：`docs/images/part2/pretrain_data_source_map.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 4-1 | 表 | 数据源类型、许可与风险矩阵 | 第 4 章 | 本书整理 | B | 可出版 | 权属判断需随最终参考文献与法务口径复核 |
| 表 4-2 | 表 | 数据配比策略与业务目标对应矩阵 | 第 4 章 | 本书整理 | B | 可出版 | 比例为工程示例口径 |
| 图 4-2 | 图 | 数据采集与权属存证流程图 | 第 4 章 | 本书自绘：`docs/images/part2/data_ingestion_provenance_chain.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 5-1 | 图 | 清洗与去污染全景流程图 | 第 5 章 | 本书自绘：`docs/images/part2/cleaning_pipeline_overview.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 5-2 | 图 | 质量过滤漏斗与抽检闭环图 | 第 5 章 | 本书自绘：`docs/images/part2/quality_filter_funnel_loop.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 5-1 | 表 | 常见缺陷、检测方法与代价表 | 第 5 章 | 本书整理 | B | 可出版 | 需最终统一表题样式 |
| 表 5-2 | 表 | 清洗动作对训练效果影响对照 | 第 5 章 | 本书整理 | B | 可出版 | 提升幅度已标注截至 2026-06 的示例口径 |
| 表 5-3 | 表 | 轻量级清洗方案最小可行组合 | 第 5 章 | 本书整理 | B | 可出版 | 团队规模与数据量为建议口径 |
| 图 6-1 | 图 | LLM 训练输入管道分层架构 | 第 6 章 | 本书自绘：`docs/images/part2/training_input_pipeline_layers.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 6-2 | 图 | 吞吐瓶颈诊断流程图 | 第 6 章 | 本书自绘：`docs/images/part2/io_bottleneck_diagnosis_flow.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 6-1 | 表 | 数据格式、压缩与访问模式对照表 | 第 6 章 | 本书整理 | B | 可出版 | 技术生态可能变化，交付前复核 |
| 表 6-2 | 表 | 采样与混采策略收益对照表 | 第 6 章 | 本书整理 | B | 可出版 | 吞吐与成本数字已按估算示例处理 |
| 图 7-1 | 图 | 数据运营飞轮图 | 第 7 章 | 本书自绘：`docs/images/part2/data_operations_flywheel.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 7-2 | 图 | 数据评估闭环图 | 第 7 章 | 本书自绘：`docs/images/part2/data_evaluation_loop.png` | B / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 7-1 | 表 | 评估指标与治理动作映射表 | 第 7 章 | 本书整理 | B | 可出版 | 需最终统一表题样式 |
| 表 7-2 | 表 | 版本迭代记录模板表 | 第 7 章 | 本书整理 | B | 可出版 | 模板字段为建议口径 |
| 图 8-1 | 图 | 多模态图文数据工程全景图 | 第 8 章 | 本书自绘：`docs/images/part3/multimodal_data_panorama.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 8-2 | 图 | 图像语义对齐与过滤流程图 | 第 8 章 | 本书自绘：`docs/images/part3/image_semantic_alignment_flow.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 8-3 | 图 | AnyRes 动态多分辨率切割算法原理图 | 第 8 章 | 本书自绘：`docs/images/part3/anyres_dynamic_patching.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 8-1 | 表 | 图文样本类型、特征与适用任务表 | 第 8 章 | 本书整理 | C | 可出版 | Pair/Interleaved/Document Grounded 范式对照 |
| 表 8-2 | 表 | 图像清洗策略与代价对照表 | 第 8 章 | 本书整理 | C | 可出版 | 阈值与成本需随终稿模型版本复核 |
| 图 9-1 | 图 | 重标注与 OCR 双流线增强图 | 第 9 章 | 本书自绘：`docs/images/part3/recaptioning_ocr_pipeline.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 9-2 | 图 | 文档结构 Layout-to-Token 映射图 | 第 9 章 | 本书自绘：`docs/images/part3/document_structure_sample.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 9-1 | 表 | 重描述自动化生产梯队对比与优劣表 | 第 9 章 | 本书整理 | C | 可出版 | 成本与吞吐已标注截至 2026-06 的估算示例 |
| 表 9-2 | 表 | 跨模态及高级文档识别 OCR 核心错误归因与修复阵列矩阵 | 第 9 章 | 本书整理 | C | 可出版 | 需最终统一表题样式 |
| 图 10-1 | 图 | 音视频对齐分布式管线图 | 第 10 章 | 本书自绘：`docs/images/part3/av_sample_pipeline.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 10-2 | 图 | 自适应镜头边界检测与语义防泄漏架构图 | 第 10 章 | 本书自绘：`docs/images/part3/av_shot_boundary_hsv.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 10-3 | 图 | 大规模 ASR 提取与时间轴动态校准对比图 | 第 10 章 | 本书自绘：`docs/images/part3/asr_whisperx_comparison.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 10-4 | 图 | 跨模态时序校准与几何对齐架构图 | 第 10 章 | 本书自绘：`docs/images/part3/av_alignment_diagram.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 10-1 | 表 | 时序音视频数据缺陷类型与多层检测处置策略表 | 第 10 章 | 本书整理 | C | 可出版 | 需最终统一表题样式 |
| 表 10-2 | 表 | 长时序音视频处理成本模型与降本策略 | 第 10 章 | 本书整理 | C | 可出版 | 成本占比已标注截至 2026-06 的估算示例 |
| 表 10-3 | 表 | 音视频管线高频错误类型与修复策略 | 第 10 章 | 本书整理 | C | 可出版 | 错误日志为匿名化示例 |
| 图 11-1 | 图 | 跨模态对齐的三级金字塔架构 | 第 11 章 | 本书自绘：`docs/images/part3/cross_modal_alignment_hierarchy.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 图 11-2 | 图 | 多模态融合样本设计图 | 第 11 章 | 本书自绘：`docs/images/part3/fusion_training_sample_design.png` | C / F | 可出版 | 已在正文补来源与 alt text；交付前复核 300dpi/字号 |
| 表 11-1 | 表 | 三层异构对齐策略、成本特征与适用任务 | 第 11 章 | 本书整理 | C | 可出版 | 成本特征为工程示例口径 |
| 表 11-2 | 表 | 五种难负样本挖掘策略对比 | 第 11 章 | 本书整理 | C | 可出版 | 需最终统一表题样式 |
| 表 11-3 | 表 | 核心评价指标、误差来源与治理动作映射表 | 第 11 章 | 本书整理 | C | 可出版 | 需最终统一表题样式 |
| 表 11-4 | 表 | 跨模态对齐高频错误类型与修复策略 | 第 11 章 | 本书整理 | C | 可出版 | 错误日志为匿名化示例 |
| 图 12-1 | 图 | Synthetic Data Factory 流程图 | 第 12 章 | 新增 | D / F | 待绘制 | 必做 |
| 图 13-1 | 图 | 偏好信号流转图 | 第 13 章 | 新增 | D / F | 待绘制 | 必做 |
| 图 14-1 | 图 | 标注平台工作流图 | 第 14 章 | 新增 | D / F | 待绘制 | 必做 |
| 图 15-1 | 图 | RAG 数据处理全景图升级版 | 第 15 章 | 改绘自 `docs/images/part5/图12_3` | E / F | 待绘制 | 样章核心图 |
| 表 16-1 | 表 | 传统 RAG vs 视觉检索 RAG 对比表 | 第 16 章 | 新增 | E | 待绘制 | 必做 |
| 图 17-1 | 图 | 线上反馈闭环图 | 第 17 章 | 新增 | E / F | 待绘制 | 必做 |
| 图 18-1 | 图 | Mini-C4 案例流程图 | 第 18 章 | 改绘自 `docs/images/part6/图1_构建Mini_C4预训练集数据流水线图.png` | F | 待绘制 | |
| 图 19-1 | 图 | 领域 SFT 数据工厂流程图 | 第 19 章 | 改绘自 `docs/images/part6/图2_构建垂直领域专家SFT数据流水线图.png` | F | 待绘制 | |
| 图 20-1 | 图 | 财报多模态 RAG 总架构图 | 第 20 章 | 改绘自 `docs/images/part6/图6_多模态RAG企业财报助手数据流水线图.png` | F | 待绘制 | |

## 四、第一篇图表交付清单

| 图号 | 标题 | 文件路径 | 正文首次引用位置 | 来源 / 权限 | Alt text | 是否需高清源文件 |
| --- | --- | --- | --- | --- | --- | --- |
| 图 1-1 | 大模型时代数据工程职责重构图 | `docs/images/part1/data_engineering_roles_1775830393574.png` | `docs/zh/part1/ch01_data_change.md` §1.3.1 | 本书自绘；可随本书出版使用 | 大模型时代数据工程职责重构图，展示平台、数据、算法、标注、产品与合规角色之间的闭环接口 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 1-2 | 全书十四篇制生命周期地图 | `docs/images/part1/data_lifecycle_map_1775830407042.png` | `docs/zh/part1/ch01_data_change.md` §1.4 | 本书自绘；可随本书出版使用 | 全书十四篇制生命周期地图，展示从总论、预训练、多模态、对齐、应用、平台、合规到项目实战的知识结构 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 2-1 | 生命周期视角下的多维度质量分层架构 | `docs/images/part1/data_quality_hierarchy_1775835516841.png` | `docs/zh/part1/ch02_quality_framework.md` §2.2 | 本书自绘；可随本书出版使用 | 生命周期视角下的多维度质量分层架构，展示不同阶段质量指标权重从规模、多样性转向真实性、帮助性 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 2-2 | 大模型数据缺陷与质量指标交叉映射图 | `docs/images/part1/defect_metric_radar_1775835533937.png` | `docs/zh/part1/ch02_quality_framework.md` §2.3 | 本书自绘；可随本书出版使用 | 大模型数据缺陷与质量指标交叉映射图，展示六类缺陷与准确度、一致性、多样性、覆盖度和可追溯性之间的关系 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 2-3 | 数据评分卡驱动的自动截断与治理流 | `docs/images/part1/data_quality_gates_1775835548587.png` | `docs/zh/part1/ch02_quality_framework.md` §2.4 | 本书自绘；可随本书出版使用 | 数据评分卡驱动的自动截断与治理流，展示硬闸门、软闸门、人工复核和回滚动作 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 3-1 | AI 原生数据栈五层架构 | `docs/images/part1/ai_data_stack_architecture.png` | `docs/zh/part1/ch03_data_stack.md` §3.2 | 本书自绘；可随本书出版使用 | AI 原生数据栈五层架构，展示采集接入、处理编排、存储索引、评测运营和治理安全层之间的数据流 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 3-2 | 训练数据成本治理闭环图 | `docs/images/part1/cost_governance_loop.png` | `docs/zh/part1/ch03_data_stack.md` §3.3.3 | 本书自绘；可随本书出版使用 | 训练数据成本治理闭环图，展示预算规划、成本监控、ROI 评估、优化决策和预算复盘的循环 | 是；终稿前复核 300dpi、字号和矢量源 |

## 五、第二篇图表交付清单

| 图号 / 表号 | 标题 | 文件路径 | 正文首次引用位置 | 来源 / 权限 | Alt text | 是否需高清源文件 |
| --- | --- | --- | --- | --- | --- | --- |
| 图 4-1 | 预训练数据源分层地图 | `docs/images/part2/pretrain_data_source_map.png` | `docs/zh/part2/ch04_data_sources.md` §4.2 | 本书自绘；可随本书出版使用 | 预训练数据源分层地图，展示开放网页、论坛问答、百科、代码、学术论文、书籍、企业内部数据和用户反馈数据的质量与合规位置 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 4-2 | 数据采集与权属存证流程图 | `docs/images/part2/data_ingestion_provenance_chain.png` | `docs/zh/part2/ch04_data_sources.md` §4.3.3 | 本书自绘；可随本书出版使用 | 数据采集与权属存证流程图，展示来源触达、采集、解析、清洗、入库和审计记录之间的链路 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 4-1 | 数据源类型、许可与风险矩阵 | 正文表格 | `docs/zh/part2/ch04_data_sources.md` §4.2.2 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 4-2 | 数据配比策略与业务目标对应矩阵 | 正文表格 | `docs/zh/part2/ch04_data_sources.md` §4.2.3 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 图 5-1 | 清洗与去污染全景流程图 | `docs/images/part2/cleaning_pipeline_overview.png` | `docs/zh/part2/ch05_cleaning_dedup.md` §5.2 | 本书自绘；可随本书出版使用 | 清洗与去污染全景流程图，展示规则过滤、模型评分、去重、PII 脱敏、去污染和人工抽检的顺序关系 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 5-2 | 质量过滤漏斗与抽检闭环图 | `docs/images/part2/quality_filter_funnel_loop.png` | `docs/zh/part2/ch05_cleaning_dedup.md` §5.6.2 | 本书自绘；可随本书出版使用 | 质量过滤漏斗与抽检闭环图，展示规则过滤、模型评分、去重、人工抽检和规则回写之间的循环关系 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 5-1 | 常见缺陷、检测方法与代价表 | 正文表格 | `docs/zh/part2/ch05_cleaning_dedup.md` §5.7 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 5-2 | 清洗动作对训练效果影响对照 | 正文表格 | `docs/zh/part2/ch05_cleaning_dedup.md` §5.7 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 5-3 | 轻量级清洗方案最小可行组合 | 正文表格 | `docs/zh/part2/ch05_cleaning_dedup.md` §5.9.1 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 图 6-1 | LLM 训练输入管道分层架构 | `docs/images/part2/training_input_pipeline_layers.png` | `docs/zh/part2/ch06_tokenization_loading.md` §6.5 | 本书自绘；可随本书出版使用 | 训练输入管道分层图，展示分词、序列化、混采、Packing、DataLoader 和 GPU 馈送之间的顺序关系 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 6-2 | 吞吐瓶颈诊断流程图 | `docs/images/part2/io_bottleneck_diagnosis_flow.png` | `docs/zh/part2/ch06_tokenization_loading.md` §6.4.2 | 本书自绘；可随本书出版使用 | 吞吐瓶颈诊断流程图，展示从 GPU 利用率异常到磁盘 I/O、CPU 预处理和 PCIe 传输排查的决策路径 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 6-1 | 数据格式、压缩与访问模式对照表 | 正文表格 | `docs/zh/part2/ch06_tokenization_loading.md` §6.2.3 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 6-2 | 采样与混采策略收益对照表 | 正文表格 | `docs/zh/part2/ch06_tokenization_loading.md` §6.3.2 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 图 7-1 | 数据运营飞轮图 | `docs/images/part2/data_operations_flywheel.png` | `docs/zh/part2/ch07_data_operations.md` §7.1.3 | 本书自绘；可随本书出版使用 | 数据运营飞轮图，展示数据生产、模型评估、根因分析、规则回写和资产复用之间的循环关系 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 7-2 | 数据评估闭环图 | `docs/images/part2/data_evaluation_loop.png` | `docs/zh/part2/ch07_data_operations.md` §7.4.1 | 本书自绘；可随本书出版使用 | 数据评估闭环图，展示抽样评估、指标异常、根因排查、治理动作和规则更新之间的闭环 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 7-1 | 评估指标与治理动作映射表 | 正文表格 | `docs/zh/part2/ch07_data_operations.md` §7.2.4 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 7-2 | 版本迭代记录模板表 | 正文表格 | `docs/zh/part2/ch07_data_operations.md` §7.3.4 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |

## 六、第三篇图表交付清单

| 图号 / 表号 | 标题 | 文件路径 | 正文首次引用位置 | 来源 / 权限 | Alt text | 是否需高清源文件 |
| --- | --- | --- | --- | --- | --- | --- |
| 图 8-1 | 多模态图文数据工程全景图 | `docs/images/part3/multimodal_data_panorama.png` | `docs/zh/part3/ch08_multimodal_image.md` §8.1.3 | 本书自绘；可随本书出版使用 | 图文数据工程全景图，展示 DOM 抽取、图片下载、格式解析、过滤、语义对齐、重标注和序列拼装之间的流程 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 8-2 | 图像语义对齐与过滤流程图 | `docs/images/part3/image_semantic_alignment_flow.png` | `docs/zh/part3/ch08_multimodal_image.md` §8.4.2 | 本书自绘；可随本书出版使用 | 图像语义对齐与过滤流程图，展示质量过滤、CLIP 打分、重标注、动态切分和训练池入库之间的路径 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 8-3 | AnyRes 动态多分辨率切割算法原理图 | `docs/images/part3/anyres_dynamic_patching.png` | `docs/zh/part3/ch08_multimodal_image.md` §8.5.1 | 本书自绘；可随本书出版使用 | AnyRes 动态多分辨率切割算法原理图，展示全景图被切成局部块并与全局缩略图共同输入视觉编码器 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 8-1 | 图文样本类型、特征与适用任务表 | 正文表格 | `docs/zh/part3/ch08_multimodal_image.md` §8.2.3 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 8-2 | 图像清洗策略与代价对照表 | 正文表格 | `docs/zh/part3/ch08_multimodal_image.md` §8.5.2 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 图 9-1 | 重标注与 OCR 双流线增强图 | `docs/images/part3/recaptioning_ocr_pipeline.png` | `docs/zh/part3/ch09_recaptioning_ocr.md` §9.2.2 | 本书自绘；可随本书出版使用 | 重标注与 OCR 双流线增强图，展示视觉重描述、OCR 结构提取、BBox 注入和混合监督格式之间的关系 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 9-2 | 文档结构 Layout-to-Token 映射图 | `docs/images/part3/document_structure_sample.png` | `docs/zh/part3/ch09_recaptioning_ocr.md` §9.3.1 | 本书自绘；可随本书出版使用 | 文档结构 Layout-to-Token 映射图，展示文档页面被版面检测、OCR、公式解析和坐标标注转换为层级文本序列 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 9-1 | 重描述自动化生产梯队对比与优劣表 | 正文表格 | `docs/zh/part3/ch09_recaptioning_ocr.md` §9.2.1 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 9-2 | 跨模态及高级文档识别 OCR 核心错误归因与修复阵列矩阵 | 正文表格 | `docs/zh/part3/ch09_recaptioning_ocr.md` §9.4.2 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 图 10-1 | 音视频对齐分布式管线图 | `docs/images/part3/av_sample_pipeline.png` | `docs/zh/part3/ch10_video_audio.md` §10.2 | 本书自绘；可随本书出版使用 | 音视频对齐分布式管线图，展示原始视频被拆分为视觉轨、音频轨和文本轨，并通过时间对齐引擎生成 JSONL 样本 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 10-2 | 自适应镜头边界检测与语义防泄漏架构图 | `docs/images/part3/av_shot_boundary_hsv.png` | `docs/zh/part3/ch10_video_audio.md` §10.2.1 | 本书自绘；可随本书出版使用 | 自适应镜头边界检测图，展示 HSV 差分、光流差分和双阈值路由如何共同判断镜头切分点 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 10-3 | 大规模 ASR 提取与时间轴动态校准对比图 | `docs/images/part3/asr_whisperx_comparison.png` | `docs/zh/part3/ch10_video_audio.md` §10.2.2 | 本书自绘；可随本书出版使用 | ASR 提取与时间轴校准对比图，展示传统 ASR 漂移、WhisperX 校准和词级时间戳对齐结果 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 10-4 | 跨模态时序校准与几何对齐架构图 | `docs/images/part3/av_alignment_diagram.png` | `docs/zh/part3/ch10_video_audio.md` §10.2.3 | 本书自绘；可随本书出版使用 | 跨模态时序校准图，展示视觉帧、音频波形和文本 Token 如何通过同一时间轴锚点绑定 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 10-1 | 时序音视频数据缺陷类型与多层检测处置策略表 | 正文表格 | `docs/zh/part3/ch10_video_audio.md` §10.3.2 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 10-2 | 长时序音视频处理成本模型与降本策略 | 正文表格 | `docs/zh/part3/ch10_video_audio.md` §10.4.3 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 10-3 | 音视频管线高频错误类型与修复策略 | 正文表格 | `docs/zh/part3/ch10_video_audio.md` §10.6.6 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 图 11-1 | 跨模态对齐的三级金字塔架构 | `docs/images/part3/cross_modal_alignment_hierarchy.png` | `docs/zh/part3/ch11_cross_modal_alignment.md` §11.2.2 | 本书自绘；可随本书出版使用 | 跨模态对齐三级金字塔，展示对象级、片段级和文档级对齐之间的层级关系 | 是；终稿前复核 300dpi、字号和矢量源 |
| 图 11-2 | 多模态融合样本设计图 | `docs/images/part3/fusion_training_sample_design.png` | `docs/zh/part3/ch11_cross_modal_alignment.md` §11.3.1 | 本书自绘；可随本书出版使用 | 多模态融合样本设计图，展示图片、音频、文本池如何通过 JSONL 和 Placeholder 映射为统一训练样本 | 是；终稿前复核 300dpi、字号和矢量源 |
| 表 11-1 | 三层异构对齐策略、成本特征与适用任务 | 正文表格 | `docs/zh/part3/ch11_cross_modal_alignment.md` §11.2.2 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 11-2 | 五种难负样本挖掘策略对比 | 正文表格 | `docs/zh/part3/ch11_cross_modal_alignment.md` §11.3.3 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 11-3 | 核心评价指标、误差来源与治理动作映射表 | 正文表格 | `docs/zh/part3/ch11_cross_modal_alignment.md` §11.4.1 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |
| 表 11-4 | 跨模态对齐高频错误类型与修复策略 | 正文表格 | `docs/zh/part3/ch11_cross_modal_alignment.md` §11.6.6 | 本书整理；可随本书出版使用 | 不适用（正文表格） | 否 |

## 七、图表交付要求

- 每张图都要有唯一编号、标题和一句结论。
- 每张图都要标明来源：新增、改绘、复用。
- 复用线上图时，检查分辨率、字号和纸书排版可读性。
- 表格标题要能独立表达结论，不依赖正文解释。

## 八、维护规则

- 图表编号由资料编辑统一维护。
- 章节作者不能自行跳号。
- 图表状态每周例会前更新一次。
