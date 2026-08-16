# 肺炎胸片深度学习 — 第一轮文献扫描清单

- 日期：2026-08-03
- 来源：OpenAlex（nature-academic-search 无 MCP 回退）+ 中文网络检索
- 说明：本清单为粗筛结果；核心文献按 A–E 分类，全文精读将在下一轮按 note-template 规范逐篇建档。BibTeX 见 `refs_round1.bib`。

## A. 综述与背景（12 篇）

| 标题（缩写） | 期刊 | 年 | 引用 | DOI |
|---|---|---|---|---|
| Deep learning for chest X-ray analysis: A survey | Medical Image Analysis | 2021 | 489 | 10.1016/j.media.2021.102125 |
| Explainable AI in deep learning-based medical image analysis | Medical Image Analysis | 2022 | 1241 | 10.1016/j.media.2022.102470 |
| Deep Learning for Pneumonia Detection in Chest X-ray Images: A Comprehensive Survey | Journal of Imaging | 2024 | 92 | 10.3390/jimaging10080176 |
| A review of uncertainty quantification in medical image analysis | Medical Image Analysis | 2024 | 98 | 10.1016/j.media.2024.103223 |
| Trustworthy clinical AI solutions: unified review of UQ | Artificial Intelligence in Medicine | 2024 | 218 | 10.1016/j.artmed.2024.102830 |
| Machine Learning Augmented Interpretation of Chest X-rays: SR | Diagnostics | 2023 | 43 | 10.3390/diagnostics13040743 |
| A systematic review of generalization research in medical image classification | Computers in Biology and Medicine | 2024 | 51 | 10.1016/j.compbiomed.2024.109256 |
| Diagnostic accuracy of deep learning in medical imaging: SR & meta-analysis | npj Digital Medicine | 2021 | 907 | 10.1038/s41746-021-00438-z |
| ML for medical imaging: methodological failures and recommendations | npj Digital Medicine | 2022 | 614 | 10.1038/s41746-022-00592-y |
| Common pitfalls for ML to detect/prognosticate COVID-19 | Nature Machine Intelligence | 2021 | 968 | 10.1038/s42256-021-00307-0 |
| A survey of uncertainty in deep neural networks | Artificial Intelligence Review | 2023 | 1231 | 10.1007/s10462-023-10562-9 |
| Redefining Radiology: A Review of AI Integration in Medical Imaging | Diagnostics | 2023 | 733 | 10.3390/diagnostics13172760 |

## B. 数据集与基准（3 篇）

| 标题（缩写） | 期刊 | 年 | 引用 | DOI |
|---|---|---|---|---|
| VinDr-CXR: open dataset with radiologist annotations | Scientific Data | 2022 | 387 | 10.1038/s41597-022-01498-w |
| REFLACX: reports and eye-tracking data | Scientific Data | 2022 | 51 | 10.1038/s41597-022-01441-z |
| Expert-level detection from unannotated CXRs via self-supervised learning | Nature Biomedical Engineering | 2022 | 415 | 10.1038/s41551-022-00936-9 |

## C. 不确定性与校准（5 篇）

| 标题（缩写） | 期刊 | 年 | 引用 | DOI |
|---|---|---|---|---|
| Second opinion needed: communicating uncertainty in medical ML | npj Digital Medicine | 2020 | 455 | 10.1038/s41746-020-00367-3 |
| Objective evaluation of deep uncertainty predictions for COVID-19 | Scientific Reports | 2022 | 69 | 10.1038/s41598-022-05052-x |
| Improving Uncertainty Estimation with Semi-Supervised DL for COVID-19 | IEEE Access | 2021 | 46 | 10.1109/access.2021.3085418 |
| Uncertainty quantification in multi-class image classification using CXR | Frontiers in AI | 2024 | 12 | 10.3389/frai.2024.1410841 |
| A survey of uncertainty in deep neural networks | Artificial Intelligence Review | 2023 | 1231 | 10.1007/s10462-023-10562-9 |

## D. 泛化、外部验证与偏倚（11 篇）

| 标题（缩写） | 期刊 | 年 | 引用 | DOI |
|---|---|---|---|---|
| Underdiagnosis bias of AI on chest radiographs in under-served populations | Nature Medicine | 2021 | 821 | 10.1038/s41591-021-01595-0 |
| AI for radiographic COVID-19 detection selects shortcuts over signal | Nature Machine Intelligence | 2021 | 495 | 10.1038/s42256-021-00338-7 |
| The importance of being external: external validation methods | Computer Methods and Programs in Biomedicine | 2021 | 266 | 10.1016/j.cmpb.2021.106288 |
| Shortcut learning in medical AI hinders generalization | npj Digital Medicine | 2024 | 71 | 10.1038/s41746-024-01118-4 |
| Addressing cross-population domain shift in chest X-ray classification | Scientific Reports | 2025 | 25 | 10.1038/s41598-025-95390-3 |
| Shifting ML for healthcare from development to deployment | Nature Biomedical Engineering | 2022 | 408 | 10.1038/s41551-022-00898-y |
| Benchmarking saliency methods for chest X-ray interpretation | Nature Machine Intelligence | 2022 | 226 | 10.1038/s42256-022-00536-x |
| Generalization challenges in DR-TB detection from chest X-rays | Diagnostics | 2022 | 28 | 10.3390/diagnostics12010188 |
| Using AI to stratify normal vs abnormal CXRs: external validation | Diagnostics | 2023 | 17 | 10.3390/diagnostics13223408 |
| Deep learning improves physician accuracy on chest X-rays | Scientific Reports | 2024 | 54 | 10.1038/s41598-024-76608-2 |
| Leakage and the reproducibility crisis in ML-based science | Patterns | 2023 | 758 | 10.1016/j.patter.2023.100804 |

## E. 中文期刊代表工作（5 篇，DOI 待补）

| 标题 | 期刊/来源 | 年 | 特点 |
|---|---|---|---|
| 基于深度学习的儿童肺炎检测模型建立及应用 | 维普（中华医学会系列） | 2023 | ResNet-50 + Kaggle 5856 张，仅内部验证 |
| 后疫情时代人工智能肺炎辅助诊断系统的临床应用场景探索 | 放射学实践 | 2024 | 3 家医院 1049 例 CT，多中心 |
| 基于多约束潜在表征学习的双向特征映射分类模型：肺炎鉴别诊断 | 南方医科大学学报 | 2026 | 2913 例 CXR + 影像组学 + 可解释性 |
| 基于 Transformer-ResNet50 混合架构的儿童肺炎识别与分类模型 | 医疗卫生装备 | 2026 | Kaggle 数据 + Transformer |
| 基于证据深度神经网络的医学影像三支决策 | 维普 | 近年 | 证据理论 + 三支决策（不确定病例分诊） |

## 检索日志

- 工具：`nature-academic-search/scripts/academic_search.py`（OpenAlex，Python 3.14）
- 首轮 4 组查询（按引用重排，2021+）：共 41 篇去重
- 二轮 4 组查询（按相关度，2022+）：补充 VinDr-CXR、泛化综述、域偏移等
- 中文 3 组查询（网络）：确认中文核心期刊单中心现状
- 下一步：Top 10 全文精读建档 → MeSH 策略扩展检索 → PRISMA 综述提纲
