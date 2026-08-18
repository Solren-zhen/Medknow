# MedKnow — GitHub 发布清单

> 生成：2026-08-14（第四次同步，发布增强版）· 来源：pneumonia_classifier（本地工作目录，不随仓库发布）

> 总文件数：171 · 新增：`hf_space/`、`notebooks/`、CONTRIBUTING、CODE_OF_CONDUCT、
> CHANGELOG、SECURITY、issue/PR 模板、dependabot、PyPI 发布工作流

## 顶层内容

| 项目 | 类型 | 文件数 |
|---|---|---:|
| .github/ | 目录 | 7 |
| api/ | 目录 | 4 |
| configs/ | 目录 | 3 |
| data/ | 目录 | 1 |
| docs/ | 目录 | 8 |
| examples/ | 目录 | 4 |
| experiments_multicenter/ | 目录 | 10 |
| hf_space/ | 目录 | 3 |
| models/ | 目录 | 2 |
| notebooks/ | 目录 | 1 |
| paper/ | 目录 | 31 |
| results/ | 目录 | 20 |
| scripts/ | 目录 | 10 |
| src/ | 目录 | 27 |
| tests/ | 目录 | 16 |
| utils/ | 目录 | 3 |
| 根级文件 | — | 20 |

## 新增（第四次同步，发布增强）

- `hf_space/`：**Hugging Face Space 自包含发布包**（`app.py` 零依赖 + `requirements.txt` + model-card `README.md`），部署步骤见 `HF_SPACE.md`
- `notebooks/medknow_colab.ipynb`：**一键 Colab 演示**（加载权重 → MC Dropout 推理 → 转诊三队列曲线）
- `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `CHANGELOG.md` / `SECURITY.md`
- `.github/ISSUE_TEMPLATE/`（bug 表单 + feature 表单 + config）、`PULL_REQUEST_TEMPLATE.md`（含"诚实清单"）、`dependabot.yml`、`workflows/publish.yml`（打 tag 自动发 PyPI）
- README（EN/ZH）：Live Demo / Colab / PyPI 徽章、30 秒快速开始、Leaderboard（`12）

## 已排除（不上传）

- `data/`（影像，除 README）、`checkpoints/`、`outputs/`、`*.pth`、`*.db`、`qc/`、个人笔记、缓存
- **模型权重不进 Git 仓库**，通过 HF Space / GitHub Release 分发（Space 需手动上传 `model.pth`）

## 发布前必做（checklist）

1. **发布信息核对**：仓库中的 GitHub 链接、作者字段和 Space 链接已替换为当前项目地址；发布前仍应由实际维护者确认作者署名和模型权重下载地址。
2. **HF Space 部署**（免费，约 5 分钟）：新建 Space → 推 `hf_space/` 内文件 → 上传 `outputs/pneumonia_model.pth`（重命名 `model.pth`）+ `outputs/experiments/resnet18_20260801_180344/temperature.txt`。
3. **GitHub Release v1.0.0**：附 3 个 seed 权重（`checkpoints/seed_42.pth`、`seed_2024.pth`、`seed_2026.pth`）+ `temperature.txt`，供 Colab 与外部复现下载。
4. **推送建仓**：`git add -A && git commit && git push` → 建 GitHub 仓库（V1.0 含 HF Demo）。
5. **PyPI 发布**：Settings → Secrets 配 `PYPI_TOKEN`，打 `v1.0.0` tag 触发 `publish.yml`。
6. **arXiv**：上传手稿（`paper/output/doc/manuscript.md`），回填 README 的 arXiv 徽章与引用。
