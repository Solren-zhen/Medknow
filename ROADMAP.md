# MedKnow — Roadmap

> 定位：**Teaching Medical AI When Not to Know** —— 医学影像不确定性 / 选择性
> 预测评估框架（chest X-ray 是 V1 的第一个 benchmark 任务，不是项目终点）。

## 版本状态

### V1.0（2026-08-14 完成）✅

- 可复现管线：数据切分 → 训练（3 seeds）→ uncertainty → calibration → referral
  → 外部验证 → 可视化 → 测试 → CI → 文档
- 手稿全部数字已由新管线再生成并验证（`results/tables/manuscript_verification.md`）
- 外部 MC 转诊 + Deep Ensemble + 协议敏感性 + 转诊方法对比（MSP/MC/Ensemble/Random）
- 测试 74 个通过，lint 清零；发布文件夹 `medknow_github` 已 git init + 首次提交
- **HF Space 在线 Demo 发布包**（`hf_space/`，自包含 app.py + model card）
- **Colab 一键演示**（`notebooks/medknow_colab.ipynb`：推理 + 不确定性 + 转诊曲线）
- **仓库基建**：CONTRIBUTING / CODE_OF_CONDUCT / CHANGELOG / SECURITY /
  issue+PR 模板（含诚实清单）/ dependabot / PyPI 发布工作流
- **README 增强**（EN/ZH）：Live Demo + Colab + PyPI 徽章、30 秒快速开始、
  Leaderboard（`12，可提交方法）

### V1.1（计划）

- **实际部署**：创建 HF Space（上传 `model.pth` + `temperature.txt`）、发布
  GitHub Release v1.0.0（3 个 seed 权重）、配 PyPI token 并打 tag
- 教学包：教学病例库（正常/肺炎/边界/模型误判四类）、2-4 小时讲义、
  零代码使用指南（`docs/teaching/`）
- 更多不确定性方法：**TTA（Test-Time Augmentation）**、**Conformal Prediction**

### V1.2+（研究方向，按价值排序）

1. **视图（AP/PA）混杂分析**：NIH CSV 含 View Position 列（当前按文件夹加载
   丢失了该元数据）——域漂移失效可能部分由视图差异解释，是论文级补充分析
2. **转诊/risk-coverage 的协议敏感性**（当前 protocol sensitivity 只覆盖
   判别/校准指标，可扩展到转诊信号）
3. **第二个医学影像任务**（如气胸/骨折/皮肤科），验证框架可移植性
4. **更多模型**（DenseNet/EfficientNet）：仅在需要回答"结论是否架构依赖"时
   添加，不作为主路线
5. **与放射科医生对比**（转诊机制 vs 人工阅片的临床价值主张）

## 原则

- 不为了"看起来高级"堆模型/堆依赖；科学清晰 > 工程复杂度
- 数字必须可复现、可追溯；"已验证"与"代码就绪但未运行"严格区分
- 数据/影像/权重不上传 GitHub（权重走 HF Space / GitHub Release）；研究用途，非医疗器械
