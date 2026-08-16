# MedKnow — Project Overview（定稿）

> 用途：GitHub README 首页 / Space model card / 申请材料。
> 数字口径：全部来自 manuscript v0.6；内部 seed_42 数字已由
> `scripts/medknow_evaluate.py` 实测复现（AUC 0.99193，混淆矩阵 208/11/24/653）。
> 修订：2026-08-14 —— 采用"域内/域外反差"定位，落实三处限定
> （in-domain 措辞、外部队列限定、"目标是提供"而非"已提供"管线）。

## 一句话（首页第一行 / 面试开场）

**When should medical AI say "I don't know"?** — an open-source benchmark
for uncertainty-aware referral and selective prediction in medical imaging,
showing that the signal works in-domain but fails under domain shift.

## English（README 主版 Project Overview）

MedKnow is an open-source benchmark for uncertainty-aware referral and
selective prediction in medical imaging. The research question is simple: when
should a medical AI say "I don't know"? We trained a ResNet-18 pneumonia
classifier on 5,856 chest X-rays (patient-level split, three seeds) and asked
whether predictive uncertainty can identify unreliable predictions and route
them to human review.

The results show a sharp in-domain / out-of-domain contrast. Under the
internal data distribution, referring the 25% most uncertain cases reduced
retained-case missed pneumonias to zero in the primary seed-42 analysis (≤0.2%
across seeds; internal AUC 0.992). Deployed unchanged to two independent
external cohorts, discrimination dropped (RSNA AUC 0.807; NIH ChestXray-14
AUC 0.658, 95% CI 0.585–0.735 on 49 positive cases), calibration collapsed
(ECE 0.034 → 0.369), and the uncertainty-driven referral strategy became no
better than random referral on either external cohort.

Core conclusion: a model can be confident and wrong under domain shift, so
confidence- or uncertainty-based triage in medical AI cannot be trusted on the
strength of internal validation alone — discrimination, calibration, and the
referral signal must be re-validated in each new deployment environment.

MedKnow aims to provide a reproducible research pipeline — patient-level data
splitting, model training, uncertainty estimation, selective prediction /
referral, calibration analysis, and external validation with domain-shift
assessment. *(Repository under construction; the pipeline is being verified
before this claim is finalized.)*

## 中文（README 中文版 / 组会 / 申请材料）

**MedKnow** 是一个开源的医学影像不确定性与选择性预测基准项目，核心研究
问题是：**医学 AI 什么时候应该"承认自己不知道"？**

我们以 5,856 张胸部 X 线为内部数据集，在患者级切分下训练 ResNet-18 肺炎
分类器，并使用 3 个随机种子评估预测不确定性是否能够识别不可靠的预测并
将其转交人工复核。

**结果呈现出明显的域内/域外反差：**在内部数据分布下，将最不确定的 25%
病例转诊后，保留病例中的漏诊降至 0（主模型 seed 42；3 个种子最差不超过
0.2%；内部 AUC 0.992）；但当同一模型不做任何调整地迁移到两个独立外部
队列时，判别能力明显下降（RSNA AUC 0.807；NIH ChestXray-14 AUC 0.658，
95% CI 0.585–0.735，阳性仅 49 例），概率校准显著恶化（ECE 0.034 → 0.369），
而不确定性驱动的转诊策略在两个外部队列上都不再优于随机转诊。

**核心结论：模型在域漂移下可能"自信且错误"。** 因此，基于模型置信度或
不确定性的医学 AI 分诊机制不能仅凭内部验证建立信任，而应在新的部署环境
中重新验证其判别能力、校准和转诊信号。

MedKnow 的目标是提供从**患者级数据切分、模型训练、不确定性估计、选择性
预测/转诊、校准分析到外部验证与域漂移评估**的可复现研究管线。
*（仓库正在建设中，管线验证完成前不声称已提供。）*

## 数字安全（限定语记录）

1. **"漏诊归零"限定为 seed_42 主分析**：seed_2026 在 25% 转诊率下保留集
   仍有 1 例漏诊（0.2%），概述写 "zero in the primary seed-42 analysis
   (≤0.2% across seeds)"。
2. **NIH AUC 带 CI 与阳性例数**：仅 49 例阳性（9,103 张，0.5%），
   95% CI 0.585–0.735。
3. **"在内部数据分布下有效"而非"临床有效"**：实验是公开数据集验证，
   不是医院部署，措辞与 manuscript 的 deployment institution / in-domain
   限定一致。
4. **转诊失效限定为外部队列**：内部 MC Dropout 转诊显著优于随机，
   外部才失效——反差本身是卖点，不能写反。
5. **"目标是提供可复现管线"**：仓库尚未完成，不提前声称已提供。
6. **ECE 口径**：0.034 / 0.369 均为 15 bins raw ECE。
7. **环境锚定**：数字来自 PyTorch 2.5 + RTX 3060 + AMP + seed 42，
   重跑允许 ±0.001 级别差异。
