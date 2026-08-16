# Figure Manifest

| Figure | Path | Type | Tool | Critic | Rounds | Description |
|---|---|---|---|---|---|---|
| Fig 1 | paper/figures/fig1_flow.png / .pdf | Flow diagram | matplotlib (simplified STARD) | done (2026-08-12) | 3 | 三列研究流程：内部数据→患者级切分→训练→内部评估（AUC 0.992）；两个独立外部队列 RSNA（n=26,684，AUC 0.807）与 NIH ChestXray-14（n=9,103，AUC 0.658），各含参考标准、冻结模型应用 |
| Fig 2 | paper/figures/fig2_roc.png / .pdf | ROC curve | matplotlib | done (2026-08-07) | 3 | 三队列 ROC：内部 test（AUC 0.992）、外部 RSNA（AUC 0.807）、外部 NIH（AUC 0.658）；600 dpi 线图 |
| Fig 3 | paper/figures/fig3_calibration.png / .pdf | Calibration plot | matplotlib | done (2026-08-07) | 3 | 三面板校准图（15-bin raw ECE）：内部 0.034/Brier 0.032、RSNA 0.369/0.355、NIH 0.265/0.315，底部预测概率直方图 |
| Fig 4 | paper/figures/fig4_referral.png / .pdf | Referral curve | matplotlib | done (2026-08-07) | 3 | 三面板拒识-转诊曲线：内部 MC vs 置信度 vs 随机（P 标注 10%/25%）、外部 RSNA/NIH 三者重叠 |
| Fig 5 | paper/figures/fig5_subgroup.png / .pdf | Bar chart | matplotlib | done (2026-08-06) | 3 | RSNA 亚组平均预测概率 + 样本量标注 + 0.5 阈值线 |

## Critic notes

- **2026-08-06 v3 重做**（make-figures 投稿级标准）：
  - 全部数据图 300→**600 dpi**（线图标准），单栏/双栏期刊尺寸
  - 统一字号：轴标签 9pt、刻度 8pt、图例 8pt，Arial/Helvetica
  - Fig 3 改为双面板 + 预测概率直方图 + ECE/Brier 文本框
  - Fig 4 改为双面板，内部标注 χ² 显著（P=0.002 / P<0.001）
  - Fig 5 加样本量标注，术语统一（No LO / Not Normal）
  - Wong 色盲安全色板 + 线型/标记冗余编码
- 数据来源：显式权重 seed_42 权威重跑（`outputs/verify_seed_42`、
  `outputs/triage_verify/`、`outputs/external_rsna_verify/`）
- Fig 1 为简化流程（非完整 STARD）；checklist 见 `STARD_TRIPOD_checklist.md`

## Critic notes

- **2026-08-06 修正**（配合自我审查 + 评估管线 bug 修复）：
  - 数据来源全部改为显式权重 seed_42 的权威重跑：
    内部 `outputs/verify_seed_42` + `outputs/triage_verify/triage_report.json`；
    外部 `outputs/external_rsna_verify/rsna_external_report.json`。
  - Fig 3 ECE 改为 15-bin 按样本量加权（`count_weighted_ece`），统一 raw 口径：
    内部 0.0335、外部 0.3692，与正文一致。
  - Fig 4 增加内部随机拒识基线（此前缺失）；外部 0% 基线用 `base_error` 0.3929。
  - Fig 5 亚组标签统一为 "Lung Opacity / No LO / Not Normal / Normal"（正文同）。
- 色板：Wong 色盲安全（fig4_referral 中内部随机 = 随机色，外部随机 = 淡蓝）。
- Fig 1 为简化流程（非完整 STARD flow diagram）；STARD 25 项 checklist 见
  `paper/output/doc/STARD_TRIPOD_checklist.md`。
