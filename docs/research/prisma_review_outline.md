# 综述论文提纲：可解释 AI 在肺炎胸片诊断中的应用（PRISMA 2020）

- 起草日期：2026-08-03
- 目标期刊：中国医学影像技术 / 中国生物医学工程学报（综述栏目），亦可按英文综述扩展
- 计划周期：3 个月出初稿

## 1. 暂定标题

《深度学习在肺炎胸部 X 线诊断中的应用与可解释性研究进展：系统综述》

英文：*Explainable and trustworthy deep learning for pneumonia detection on chest radiographs: a systematic review*

## 2. 引言（拟写内容）

- 背景：肺炎流行病学负担、胸片作为一线影像、影像医生短缺与 AI 辅助初筛需求；
- 现状：深度学习肺炎检测论文大量涌现（2020–2026），但存在数据碎片化、单中心、可解释性不足、泛化能力差四大痛点（Siddiqi2024 等）；
- 目的：系统检索并评价"深度学习肺炎胸片检测"研究中可解释性（XAI）、不确定性量化（UQ）、外部验证的采用情况，回答：
  1. 已发表的肺炎胸片 DL 研究中有多大比例报告了外部验证？
  2. XAI/UQ 方法的使用与验证方式如何？
  3. 报告质量（TRIPOD/CLAIM 依从）如何？

## 3. 方法（PRISMA 2020）

### 3.1 注册
- PROSPERO 注册（若选英文）或预先登记检索方案（中文期刊可附"检索策略附录"）。

### 3.2 检索数据库与时间

| 数据库 | 覆盖 | 语言 |
|---|---|---|
| PubMed / Embase / Web of Science | 英文 | EN |
| CNKI / 万方 / 维普 | 中文核心 | ZH |
| arXiv / medRxiv（预印本，可选） | 补充 | EN |

时间：2015-01-01 至 2026-08-31；语言：中英文。

### 3.3 检索式草案

英文（PubMed 风格）：
```
("pneumonia"[MeSH] OR pneumonia[tiab]) AND ("radiograph*"[tiab] OR "chest X-ray"[tiab] OR "chest radiograph*"[tiab] OR CXR[tiab]) AND ("deep learning"[tiab] OR "convolutional neural network*"[tiab] OR "neural network*"[tiab] OR transformer[tiab]) AND (explainab*[tiab] OR interpretab*[tiab] OR "uncertainty"[tiab] OR calibrat*[tiab] OR saliency[tiab] OR Grad-CAM[tiab] OR "external validation"[tiab] OR generaliz*[tiab])
```

中文（CNKI/万方风格）：
```
肺炎 AND (胸片 OR 胸部X线 OR 胸部X光) AND (深度学习 OR 卷积神经网络 OR 神经网络) AND (可解释 OR 不确定性 OR 校准 OR 泛化 OR 外部验证 OR 可视化)
```

### 3.4 纳入排除标准（草案）

纳入：
- 研究对象：胸部 X 线（非 CT）肺炎/肺炎相关征象检测或分类；
- 干预：深度学习模型（CNN/Transformer/集成/迁移学习）；
- 报告任一：可解释性、不确定性、校准、外部验证、泛化评估；
- 研究类型：原始研究（回顾性/前瞻性）、公开发表的中英文文献。

排除：
- 仅 CT/MRI 等其他模态；非肺炎任务（如仅结核）；无模型实验的纯观点文；
- 非中英文、无全文、综述（综述作为背景引用而非纳入数据）；
- 数据泄漏或无明确训练/测试划分的研究（在质量评价中标记）。

### 3.5 筛选与数据提取

- 双人独立筛选（标题/摘要 → 全文），分歧第三仲裁；**AI 辅助初筛，人工复核**；
- 数据提取表字段：数据集（名称/规模/中心数）、任务（二分类/多分类/定位）、模型架构、XAI 方法、UQ 方法、校准报告、外部验证类型（内部/时序/地理/跨人群）、主要指标（AUC/ACC/敏感度/特异度/ECE）、报告规范依从；
- 工具：nature-literature-pipeline（粗筛）→ nature-paper-card（关键文献精读）→ 手工表格复核。

### 3.6 质量评价

- 诊断性研究：QUADAS-2（偏倚与适用性）；
- AI 报告规范：CLAIM 或 TRIPOD 条目核查表（按研究类型选择）。

### 3.7 综合方法

- 叙事综合为主（因异质性高），辅以描述性统计（比例、中位数、范围）；
- 若亚组同质性允许，meta 分析外部验证 AUC（随机效应模型，I²）；
- 亚组：任务类型、模型架构、是否外部验证、是否报告 UQ/XAI。

## 4. 预期结果章节结构

1. 文献筛选流程（PRISMA 流程图，记录各库命中数与排除原因）；
2. 研究特征总览（年份趋势、数据集、中心数、任务）；
3. 可解释性方法使用情况（XAI 类型、验证方式、Grad-CAM 占比）；
4. 不确定性/校准报告情况（UQ 方法、ECE 报告比例）；
5. 外部验证情况（外部验证比例、类型、性能衰减幅度）；
6. 报告质量评价（QUADAS-2 / CLAIM 得分分布）。

## 5. 讨论与结论

- 主要发现：内部验证主导、外部验证稀缺、XAI 多为"展示"而非"验证"、UQ 报告几乎缺失；
- 对肺炎 AI 研究者的建议：外部验证 + 校准 + 拒识策略的报告模板；
- 局限：检索语言与库覆盖、发表偏倚、异质性高难以定量合并。

## 6. 时间表（3 个月）

| 周次 | 任务 |
|---|---|
| 1–2 | 检索策略定稿 + 数据库检索 + 去重（Zotero/手工） |
| 3–4 | 标题/摘要双筛（AI 辅助） |
| 5–8 | 全文筛选 + 数据提取（表格化） |
| 9–10 | 质量评价 + 综合 |
| 11–12 | 图表 + 初稿 + 引用核验 |

## 7. 工具映射（已装 skills）

| 阶段 | Skill |
|---|---|
| 检索 | nature-academic-search / nature-literature-pipeline |
| 精读 | nature-paper-card（关键文献，PDF 到位后） |
| 写作 | paper-spine / scientific-writing / nature-writing |
| 图表 | figure-planner / nature-figure / publication-chart-skill |
| 引用核验 | citation-verifier / nature-ref-verifier / nature-citation |
| 统计 | results-analysis / stats-reporting-audit |
| 投稿前 | submission-audit |
