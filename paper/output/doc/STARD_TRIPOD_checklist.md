# Reporting Checklists — STARD 2015 & TRIPOD 2015

> Manuscript: *Uncertainty-guided referral for pneumonia detection on chest radiographs*
> Status: 2026-08-12, manuscript v0.6（总评后修订：双外部/术语/定义/3-seed 补表）。Location of items: `paper/output/doc/manuscript.md`.
> Legend: ✓ present / (n/a) not applicable to this study design / — absent (planned or in progress)

---

## STARD 2015 (Bossuyt et al., BMJ 2015)

| # | Item | Location | Status |
|---|------|----------|--------|
| 1 | Identify as a study of diagnostic accuracy | Abstract, Methods (Study design) | ✓ |
| 2 | Structured summary: background, methods, results, conclusions | Abstract | ✓ |
| 3 | Scientific and clinical background, study rationale | Introduction ¶1–3 | ✓ |
| 4 | Study objectives and hypotheses | Introduction ¶4 | ✓ |
| 5 | Study population: inclusion/exclusion criteria, setting, recruitment | Methods (Internal dataset; External dataset — RSNA + NIH ChestXray-14) | ✓ |
| 6 | Participant sampling: consecutive or random | Methods (patient-level re-split) | ✓ |
| 7 | Data collection: prospective vs retrospective | Methods (Study design) | ✓ |
| 8 | Rationale for sample size | n/a (secondary analysis of public data) | ✓ |
| 9 | Blinding of readers of the index test | n/a (automated classifier) | ✓ |
| 10 | Reference standard and its rationale | Methods (Reference standard) | ✓ |
| 11 | Rationale for choosing target condition | Methods (Reference standard) | ✓ |
| 12 | Index test, including cutoff/threshold, and how executed | Methods (Model, Uncertainty, Referral) | ✓ |
| 13 | Blinding of those assessing reference standard to index test | n/a (independent public labels) | ✓ |
| 14 | Clinical and demographic characteristics of participants | Results (Dataset characteristics), Table 1 | ✓ |
| 15 | Distribution of severity of the target condition | n/a (binary public labels) | — |
| 16 | Cross-tabulation: index test × reference standard | Results (Internal/External), confusion matrices | ✓ |
| 17 | Diagnostic accuracy estimates with CIs (sensitivity/specificity) | Results, Table 2 | ✓ |
| 18 | AUC / ROC, if reported | Results, Table 2, Fig 2 | ✓ |
| 19 | Adverse events from index test | n/a (no patient contact) | ✓ |
| 20 | Limitations | Discussion | ✓ |
| 21 | Implications for practice | Discussion | ✓ |
| 22 | Funding sources | Ethics / Funding | ✓ |
| 23 | Registration | n/a (retrospective, no protocol registry) | ✓ |
| 24 | Protocol/analysis code availability | Ethics and data availability | ✓ |
| 25 | Flow diagram | Fig 1 | ✓ |

---

## TRIPOD 2015 (Collins et al., Ann Intern Med 2015) — items applicable to this diagnostic prediction study

| # | Item | Location | Status |
|---|------|----------|--------|
| 1 | Title identifies the study as a prediction model | Title | ✓ |
| 2 | Structured abstract: objectives, setting, participants, model, performance | Abstract | ✓ |
| 3a | Background and rationale | Introduction ¶1–3 | ✓ |
| 4a | Study objectives | Introduction ¶4 | ✓ |
| 5a | Source of data (e.g., cohort) | Methods (Internal/External dataset) | ✓ |
| 6 | Eligibility criteria | Methods (Internal/External dataset) | ✓ |
| 7 | Outcome definition and assessment | Methods (Reference standard) | ✓ |
| 8 | Predictors: how measured, including their definition | Methods (Model development) | ✓ |
| 9 | Sample size considerations | n/a (secondary analysis) | ✓ |
| 10a | Missing data handling | n/a (no exclusions; labels complete) | ✓ |
| 10d | Handling of overlap / leakage (same-patient split) | Methods (Internal dataset), Fig 1 | ✓ |
| 11 | Model development: method of estimation, complexity | Methods (Model development) | ✓ |
| 12 | Model performance evaluation: internal validation | Results (Internal test) | ✓ |
| 13 | External validation | Results (External RSNA; second external cohort NIH ChestXray-14) | ✓ |
| 14a | Full model (all predictors) | Methods (Model development) | ✓ |
| 15 | Performance measures with CIs | Results, Table 2 | ✓ |
| 16 | Calibration | Results, Fig 3 | ✓ |
| 17 | Discrimination | Results, Fig 2 | ✓ |
| 18 | Limitations | Discussion | ✓ |
| 19 | Interpretation of results | Discussion | ✓ |
| 20 | Implications | Discussion | ✓ |
| 21 | Supplementary information (code/data) | Ethics and data availability | ✓ |
| 22 | Funding | Ethics / Funding | ✓ |

---

## 备注 / 待办

- **缺失项 15 (STARD)**：内部数据集无严重度分层信息（公共二分类标签）——方法中已说明，可不补。
- **项 22 (STARD) / 22 (TRIPOD)（Funding）**：已补 "No external funding" 声明（2026-08-06）。
- **项 23 (STARD)（Registration）**：回顾性公共数据研究，注明未注册即可。
- **数据可用性**：GitHub 仓库地址待创建后回填（见 data availability 声明）；已准备 `CITATION.cff` 模板。
- **第二个外部队列（NIH ChestXray-14）已覆盖（2026-08-12）**：STARD #5、TRIPOD #13 已更新为两个外部队列；NIH 队列的标签来源（NLP 提取，异于 RSNA 专家共识）在 Methods (External dataset)、Limitations 中均有说明（对应 TRIPOD #7 结局定义）。
- **3-seed 转诊稳定性**：新增 `supplementary.md` 表 S1（3-seed × 10%/25% 转诊率），正文 Run-to-run stability 引用。
- **CLAIM 清单缺失**：论文声明遵循 CLAIM [@claim2020]，但项目内仅维护 STARD/TRIPOD 合并清单；若目标期刊（BMC Medical Imaging）投稿系统要求 CLAIM 逐项核对表，需单独补一份 CLAIM checklist 文件。
