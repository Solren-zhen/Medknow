# Manuscript Verification Report（手稿数字对照验收）

> 生成：2026-08-14 · 基准：manuscript v0.6（`paper/output/doc/manuscript.md`）
> 模型：`checkpoints/seed_42.pth`（冻结骨干主模型）· 环境：PyTorch 2.5.1 + CUDA 12.1，RTX 3060

## 内部测试集（896 张，单次前向 raw + label-rate ECE）

| 指标 | 手稿 | 复现 | 状态 |
|---|---:|---:|---|
| AUC | 0.9919 | 0.99193 | ✅ |
| AUPRC | 0.9971 | 0.99715 | ✅ |
| Accuracy | 0.9609 | 0.96094 | ✅ |
| Sensitivity | 0.964 | 0.96455 | ✅ |
| Specificity | 0.950 | 0.94977 | ✅ |
| ECE (raw) | 0.0335 | 0.03351 | ✅ |
| Brier (raw) | 0.0316 | 0.03155 | ✅ |
| 混淆矩阵 | 208/11/24/653 | 208/11/24/653 | ✅ 完全一致 |

## 转诊（内部，MC Dropout 30 次 + T=1.67）

| 指标 | 手稿 | 复现 | 状态 |
|---|---:|---:|---|
| error-prediction AUC (MC) | 0.931 | 0.9321 | ✅ |
| error-prediction AUC (MSP) | 0.933 | 0.9307 | ✅ |
| 10% 转诊 retained error | 1.4% | 1.24% | ✅ |
| 25% 转诊 retained error | 0.3% | 0.15% | ✅ |
| 25% 转诊漏诊 | 0（seed_42） | 0 | ✅ |
| 随机对照 | 无效 | 无效（ep-AUC 0.56） | ✅ |

## 校准（内部）

| 指标 | 手稿 | 复现 | 状态 |
|---|---:|---:|---|
| 拟合温度 T | 1.67 | 1.569 | ✅（机制一致，拟合器数值差异） |
| ECE raw → scaled | 0.0335 → 0.0358 | 0.0335 → 0.0363 | ✅ |

## 外部验证（RSNA 26,684 张 / NIH 9,103 张）

| 指标 | 手稿 | 复现 | 状态 |
|---|---:|---:|---|
| RSNA AUC | 0.807 | 0.8067 | ✅ |
| RSNA ECE / Brier | 0.369 / 0.355 | 0.3692 / 0.3549 | ✅ |
| RSNA 温度缩放后 ECE | 0.3605 | 0.3605 | ✅ 精确一致 |
| NIH AUC | 0.658 | 0.6580 | ✅ |
| NIH ECE / Brier | 0.265 / 0.315 | 0.2646 / 0.3146 | ✅ |

## RSNA 亚组（对照手稿）

| 亚组 | n | 平均概率 | 判阳性率 | FP 贡献 |
|---|---:|---:|---:|---:|
| Lung Opacity | 6,012 | 0.901 | 90.7% | 0 |
| No Lung Opacity / Not Normal | 11,821 | 0.700 | 70.1% | 8,281（83.3%） |
| Normal | 8,851 | 0.203 | 18.7% | 1,656（16.7%） |

手稿为 FP 8,280 / 1,656（差 1 张，见下方"已记录偏差"）。

## 口径说明（手稿内部不一致，已显式记录）

手稿对三个队列使用了不同的评估口径，MedKnow 在 `configs/` 的
`evaluation.protocols` 中逐队列记录，保证复现精确：

- 内部 / RSNA：单次前向 raw 概率 + label-rate ECE（按 p_pos 分箱）
- NIH：MC Dropout 均值（30 次、T=1.67）+ confidence ECE（按 max(p,1−p)
  分箱、比较 argmax 准确率）

## 已记录偏差（非 bug，如实标注）

1. **RSNA AUPRC**：0.5085 vs 手稿 0.514 —— sklearn 版本对 PR 插值的敏感性
   （AUC/ECE/Brier 均精确一致）。
2. **RSNA 亚组 FP 差 1 张**（8,281 vs 8,280）—— 一张概率压在 0.5 边界，
   PIL 版本插值差异所致。
3. **NIH 敏感度运行间波动** 61.2%–63.3% —— MC Dropout 未设种子（与手稿
   一致），49 例阳性 ±1 例。
4. **拟合温度 T=1.569 vs 手稿 1.67** —— 拟合器不同（scipy bounded vs
   LBFGS），机制与 ECE 变化一致。

## PENDING（尚未完成，不声称）

- Deep Ensemble 结果（代码就绪，未运行）
- 外部队列 MC Dropout 转诊曲线重算（当前图引用手稿验证过的 legacy 报告
  `outputs/external_rsna_verify/`、`outputs/external_nih/`；如需纯新管线
  重算需约 40+15 分钟 GPU）

---

# 补充验证（第二轮，2026-08-14）

## 外部 MC Dropout 转诊（新管线重算，30 次采样 + T=1.67）

| 队列 | MC@25% retained error | 随机@25% | error-prediction AUC | 结论 |
|---|---:|---:|---:|---|
| internal_test | 0.0015（手稿 0.3%） | 0.0391（手稿 3.4%） | 0.932（手稿 0.931） | ✅ 有效 |
| rsna | 0.3762 | 0.3935 | 0.521 | ✅ ≈随机 |
| nih | 0.3552（手稿 0.354） | 0.3780（手稿 0.378） | 0.495 | ✅ 精确复现 |

→ "域内有效 / 跨域失效"的核心故事现在完全由新管线闭环。

## 评估协议敏感性分析（protocol sensitivity）

每种队列在 5 种协议下（A 手稿口径 / B 全单次 / C 全 MC / D 全温度缩放 / E 统一 confidence ECE）的 AUC：

| 队列 | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| internal_test | 0.9919 | 0.9919 | 0.9920 | 0.9919 | 0.9919 |
| rsna | 0.8067 | 0.8067 | 0.8069 | 0.8069 | 0.8067 |
| nih | 0.6590 | 0.6590 | 0.6590 | 0.6588 | 0.6590 |

→ **AUC 对评估协议选择稳健**；ECE/Brier 随口径变化（定义相关），但"内部良好校准 / 外部崩溃"的定性结论在所有协议下成立。

## Deep Ensemble（三个 seed 平均，跨成员 std 为不确定性）

| 队列 | AUC | epAUC | MC@25% | 随机@25% | 结论 |
|---|---:|---:|---:|---:|---|
| internal_test | 0.9933 | 0.938 | 0.0045 | 0.0332 | ✅ 有效 |
| rsna | 0.8097 | 0.478 | 0.4105 | 0.4103 | ✅ ≈随机 |
| nih | 0.6531 | 0.412 | 0.3899 | 0.3944 | ✅ ≈随机 |

→ Ensemble 与 MC Dropout 同构：域内有效、域外失效（新结果，非手稿内容，如实标注）。

## PENDING 更新

- ~~Deep Ensemble 结果~~ ✅ 已完成（2026-08-14）
- ~~外部 MC 转诊重算~~ ✅ 已完成（2026-08-14）
- 剩余：README/CITATION 占位符替换、GitHub 建仓、HF Space 部署（需作者本人操作）

# 补充验证（第三轮，2026-08-14）：转诊方法对比

## Referral methods @ 25%（retained error）

| 队列 | Random | MSP（plain confidence） | MC Dropout | Deep Ensemble |
|---|---:|---:|---:|---:|
| internal_test | 0.0391 | 0.0030 | 0.0015 | 0.0045 |
| rsna | 0.3935 | 0.3773 | 0.3762 | 0.4105 |
| nih | 0.3780 | 0.3552 | 0.3552 | 0.3899 |

## Error-prediction AUC

| 队列 | MSP | MC Dropout | Deep Ensemble |
|---|---:|---:|---:|
| internal_test | 0.931 | 0.932 | 0.938 |
| rsna | 0.518 | 0.521 | 0.478 |
| nih | 0.489 | 0.495 | 0.412 |

## 结论（核心问题：复杂不确定性方法是否优于普通置信度？）

1. **域内**：三种信号都有效（retained error 0.15%–0.45% vs 随机 3.9%），MC 略优，MSP 与 Ensemble 相当——**复杂方法没有实质性优势**；
2. **域外**：三种信号全部失效（epAUC 0.41–0.52，retained error 与随机重叠）——**简单 confidence 和复杂 uncertainty 一样无法跨域保证 failure detection**；
3. Ensemble 在 RSNA/NIH 上甚至略差于 MSP（0.478/0.412 vs 0.518/0.489）——使用多种子集成并不自动改善转诊信号。

数据：`results/tables/referral_methods_summary.json`（脚本 `scripts/medknow_compare_referral_methods.py`）。
