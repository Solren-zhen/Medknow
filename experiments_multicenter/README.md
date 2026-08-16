# 多中心外部验证实验管线（CheXpert 训练 → RSNA / VinDr 外部测试）

对应评估协议：`docs/research/evaluation_protocol.md`（见项目 docs/research/ 目录）。
**全部实验在 3060（Windows + CUDA）上运行，不要在本机 Mac 上跑训练/评估。**

## 1. 环境准备（3060，一次性）

```bash
conda create -n pneumonia python=3.10 -y
conda activate pneumonia
pip install -r experiments_multicenter/requirements.txt
```

## 2. 数据布局（3060）

```
D:/datasets/
  chexpert/                      # 含 train.csv、valid.csv、train/（或预处理后的图）
  rsna/
    stage_1_train_images/*.dcm
    stage_1_train_labels.csv
    png/                         # 转换后生成
    rsna_index.csv               # 转换后生成
  vindr/
    train.csv                    # VinDr 标注 CSV
    png/                         # 图片（00000000_00000000.png 形式）
```

修改 `experiments_multicenter/config_multicenter.yaml` 中的路径与模型设置。

## 3. 运行顺序

### 3.1 RSNA DICOM 转 PNG（只需一次）

```bash
python experiments_multicenter/convert_rsna_dicom.py --rsna-dir D:/datasets/rsna
```

### 3.2 训练（CheXpert，患者级 85/15 划分）

```bash
python experiments_multicenter/train_chexpert.py
# 换基线模型：
python experiments_multicenter/train_chexpert.py --model resnet18
```

产出（`outputs_multicenter/`）：`{model}_chexpert_best.pth`、`training_history.json`、`training_curves.png`、`chexpert_val_patients.csv`。

### 3.3 内部验证评估 + 温度缩放

```bash
# 1) 评估内部验证集（保存预测 CSV）
python experiments_multicenter/evaluate_external.py \
  --checkpoint outputs_multicenter/efficientnet_b0_chexpert_best.pth \
  --dataset chexpert_val

# 2) 拟合温度（用上一步的预测 CSV）
python experiments_multicenter/temperature_scaling.py \
  --predictions outputs_multicenter/predictions_CheXpert-val.csv
```

### 3.4 外部测试（RSNA / VinDr）

```bash
# RSNA（患者级去重）
python experiments_multicenter/evaluate_external.py \
  --checkpoint outputs_multicenter/efficientnet_b0_chexpert_best.pth \
  --dataset rsna --patient-level --temperature outputs_multicenter/temperature.json

# VinDr-CXR
python experiments_multicenter/evaluate_external.py \
  --checkpoint outputs_multicenter/efficientnet_b0_chexpert_best.pth \
  --dataset vindr --patient-level --temperature outputs_multicenter/temperature.json
```

产出：`report_{dataset}.json`（AUC 95%CI、敏感度@95%特异度、ECE、Brier、漏诊率）、`roc_*.png`、`predictions_*.csv`。

### 3.5 不确定性引导分诊（核心贡献）

```bash
# 外部集开 MC Dropout（30 次采样）重新评估
python experiments_multicenter/evaluate_external.py \
  --checkpoint outputs_multicenter/efficientnet_b0_chexpert_best.pth \
  --dataset vindr --patient-level --mc-samples 30 --temperature outputs_multicenter/temperature.json

# 分诊权衡曲线 + H2 假设检验
python experiments_multicenter/triage_analysis.py \
  --predictions outputs_multicenter/predictions_VinDr-CXR.csv
```

产出：`triage_*.json`（转诊率-敏感度曲线 + 推荐操作点）、`triage_*.png`。

## 4. 关键设计决策（投稿前不要改）

| 决策 | 选择 | 原因 |
|---|---|---|
| 划分 | 患者级（GroupShuffleSplit, seed=42） | 杜绝同一患者跨集泄漏 |
| CheXpert 不确定标签 | 默认 ignore | 最干净的标签口径，报告里写明 |
| 外部测试 | 患者级去重（一人一图） | 避免相关性高估 |
| 校准 | 温度缩放（内部验证集拟合） | 外部集不可见原则 |
| 主模型 | EfficientNet-B0 | 你项目已实测最优；ResNet18 作基线 |
| 不确定性 | MC Dropout ×30 | 无需重训，复用现有 dropout 层 |

## 5. 注意事项

- Windows 路径用反斜杠或正斜杠均可，yaml 里建议 `D:/datasets/...`；
- `use_amp` 仅在 CUDA 上生效，MPS/CPU 自动关闭；
- 评估时 BatchNorm 保持 eval、仅 Dropout 开启（`uncertainty_mc.enable_dropout`）；
- 结果出问题先看 `outputs_multicenter/` 的 JSON，不要改协议。
