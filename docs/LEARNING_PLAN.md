# 🗺️ 肺炎分类器项目 · 6 周学习课程规划

> 目标学员：Python 基础入门（会变量/循环/函数，没接触过 PyTorch 和深度学习）
> 学习目标：**看懂这个项目 + 在自己电脑上跑通**
> 时间投入：每周 10+ 小时，共 6 周（约 60-70 小时）
> 配套工具：遇到看不懂的地方，随时可以让 Claude 逐行讲给你听
> ⚠️ **2026-08 更新（MedKnow 迁移）**：命令行入口已统一为 `scripts/medknow_*.py`
> （训练 `medknow_train.py` / 评估 `medknow_evaluate.py` / 切分 `medknow_split_patient.py` /
> 转诊 `medknow_run_referral.py` / 校准 `medknow_run_calibration.py` / 外部验证 `medknow_evaluate_external.py` /
> 作图 `medknow_make_figures.py`）。下文提到的旧脚本名（train.py、evaluate.py、split_patient.py 等）
> 均为历史名称，学习时按上面的对应关系替换；`models/`、`config.yaml`、`utils/` 等模块文件保持不变。

---

## 一、先建立全局地图：这个项目到底在干什么

一句话：**用深度学习给胸片（X光）自动判断"有没有肺炎"，并且让模型"敢说不知道、能解释依据"。**

它由 5 大块组成，学的时候永远记住这张图：

```
   数据                     模型                     输出
┌─────────┐   ┌──────────┐   ┌─────────┐
│ 数据集    │ → │ 训练/评估  │ → │ 分类结果  │ → Web应用/API
│ chest_xray│   │ train.py  │   │ 肺炎/正常 │    (Streamlit/FastAPI)
│ + RSNA   │   │ evaluate.py│   │          │
└─────────┘   └──────────┘   └────┬─────┘
                                  │ 附加能力
                          ┌───────┼──────────────┐
                          ▼       ▼              ▼
                       可解释性   不确定性         校准
                      "我看了哪片" "我没把握"     "90%就是90%"
                       explain.py uncertainty  temperature scaling
```

### 文件速查表（学完第一周你应该能看懂这张表）

| 文件 | 一句话作用 | 行数 | 优先级 |
|:---|:---|:---:|:---|
| `README.md` | 项目说明书，最好的入口 | 390 | ⭐⭐⭐ |
| `config.yaml` | 全局配置，所有参数都在这 | 81 | ⭐⭐⭐ |
| `config.py` | 把 yaml 读成 Python 对象的加载器 | 123 | ⭐⭐ |
| `scripts/train.py` | **单模型训练**（最核心的学习文件） | 545 | ⭐⭐⭐ |
| `scripts/evaluate.py` | **评估**，产出所有指标和图 | 465 | ⭐⭐⭐ |
| `models/model_factory.py` | 模型工厂，统一创建三种网络 | 268 | ⭐⭐⭐ |
| `scripts/split_patient.py` | 患者级切分（修复数据泄漏的关键） | 160 | ⭐⭐⭐ |
| `scripts/explain.py` | 可解释性：Grad-CAM 热力图等 | 332 | ⭐⭐ |
| `scripts/uncertainty.py` | 不确定性量化（MC Dropout） | 329 | ⭐⭐ |
| `scripts/triage_curve.py` | 不确定性"转人工复核"曲线 | 174 | ⭐⭐ |
| `scripts/download_data.py` | 下载数据集 | — | ⭐ |
| `scripts/prepare_data.py` | 数据统计与分析 | 130 | ⭐⭐ |
| `app.py` | Streamlit Web 前端 | 485 | ⭐⭐ |
| `api/main.py` | FastAPI 后端接口 | 280 | ⭐ |
| `api/schemas.py` | API 数据格式定义 | 66 | ⭐ |
| `api/database.py` | SQLite 预测历史数据库 | 134 | ⭐ |
| `utils/dicom_utils.py` | 读取医院真实格式 DICOM | 103 | ⭐ |
| `experiments_multicenter/` | 多中心外部验证（RSNA 等） | — | ⭐⭐ |
| `paper/` | 论文手稿、研究笔记、图 | — | ⭐⭐ |
| `outputs/` | 所有结果：模型权重、图、报告 | — | ⭐ |
| `data/` | 数据集存放处 | — | ⭐ |

> **学习方法论**：永远"**先跑起来，再拆开看**"。一个文件先完整跑一次看到输出，再一行行读代码。

---

## 二、每周详细计划

---

### 📅 第 1 周：搭建环境 + 把项目跑起来

**目标**：环境 OK；训练/评估/Web 三个入口都能跑；建立全局地图。
**核心原则**：第一周不求理解原理，只求"亲眼看到它工作"。

**🧠 概念课（约 2h）**
- 什么是命令行、什么是虚拟环境（venv/conda）、pip 是干什么的
- PyTorch 是什么：一句话 = 科学计算库 + 神经网络库
- 什么叫"训练一个模型"（概念级：拿数据教它，不用懂细节）

**📖 精读（约 3h）**
- `README.md` 全文（特别是「快速开始」「核心技术详解」两节）
- `config.yaml` 扫一遍，不求懂每个参数
- `config.py` 看它怎么把 yaml 读进来

**🛠️ 动手任务（约 5h）**
1. 安装 Python 3.10+，确认 `python --version`
2. 建虚拟环境 → `pip install -r requirements.txt`（首次较久，耐心）
3. 下载数据：`python scripts/download_data.py`（约 1.1GB，慢就挂代理/找镜像）
4. **先跑一次小训练**：把 `config.yaml` 里 `epochs` 改成 2-3，跑 `python scripts/train.py`，确认能出权重文件
5. 跑评估：`python scripts/evaluate.py`
6. 启动 Web：`streamlit run app.py`，上传一张胸片图，看诊断结果

**✅ 验收标准**
- [ ] 能独立启动 Web 应用并在浏览器看到诊断结果
- [ ] 能画出/说出项目模块地图：数据 → 训练 → 评估 → 解释 → Web
- [ ] 自己在终端跑过至少 3 个 python 脚本
- [ ] 读懂了 `README.md` 里的「面试/汇报叙事」一节

---

### 📅 第 2 周：数据从哪来、怎么切（数据管线 + 泄漏的教训）

**目标**：彻底搞懂数据组织方式；理解医学 AI 最关键的"患者级切分"。

**🧠 概念课（约 2h）**
- Python 文件操作：`os` / `shutil` / `pathlib`，遍历文件夹
- 数据集的三种集合：train（训练）/ val（验证调参）/ test（测试定结论）
- 什么是**数据泄漏**（data leakage）：信息从测试集漏进训练集 = 成绩造假

**📖 精读（约 3h）**
- `scripts/download_data.py`、`scripts/prepare_data.py`（130行）
- **`scripts/split_patient.py`（160行）** ← 本周最重要，读 3 遍
- `utils/dicom_utils.py`（103行）：DICOM vs JPEG 的区别
- `config.yaml` 的 `paths` 和 `data` 段

**🛠️ 动手任务（约 5h）**
1. 写几行 Python 统计 `data/chest_xray` 下 train/val/test 各有多少张、两类各多少
2. 读 `split_patient.py`，在纸上画出"同一个患者怎样保证只出现在一个集合"
3. 对比 `outputs/evaluation`（修正后）和 `outputs/evaluation_leak_on_patient`（泄漏版），亲眼看看泄漏让数字虚高了多少

**✅ 验收标准**
- [ ] 能解释：图片级切分 vs 患者级切分的区别
- [ ] 能说出"340 个患者跨集合"为什么是严重 bug（关系到论文结论）
- [ ] 知道每个子文件夹放什么数据
- [ ] 能说清 JPEG 和 DICOM 的区别

> 💡 **本项目最重要的一个故事**：项目曾经用"图片级切分"，发现 340 个患者同时出现在训练和测试集（占全部 3174 患者的 10.7%），这是造假级的 bug。改成患者级切分后数字大幅变化。**医学 AI 里，切分方式就是研究设计。** 这一课比任何一行代码都重要。

---

### 📅 第 3 周：模型怎么学（训练全流程）

**目标**：读懂 `train.py`，理解"训练"这个魔法背后的循环。

**🧠 概念课（约 3h）**
- 神经网络是什么、CNN 卷积网络直觉（卷积=找局部特征，池化=压缩）
- **迁移学习**：用 ImageNet 预训练权重起步，`freeze_backbone` 冻结骨干
- batch（一次喂几张）、epoch（把所有数据过一遍）、学习率（步子大小）
- 训练循环五步：**前向 → 算损失 → 清零梯度 → 反向传播 → 更新权重**
- 损失函数（CrossEntropy）、优化器（Adam）、早停（EarlyStopping）

**📖 精读（约 4h）**
- `models/model_factory.py`（268行）← 重点，看三种网络怎么统一创建
- `config.yaml` 的 `training` 段：每个参数都要会解释
- `scripts/train.py`（545行）← 分 3 次读完，用 Claude 帮你逐段翻译

**🛠️ 动手任务（约 5h）**
1. 改 `config.yaml` 里 3 个参数（epochs / learning_rate / batch_size）各训一次小模型，对比 loss 曲线
2. 在 `train.py` 里加一行 `print` 看训练进度
3. 尝试 `freeze_backbone: true` vs `false`，观察差异
4. 用 TensorBoard（`tensorboard --logdir outputs/logs`）看训练曲线

**✅ 验收标准**
- [ ] 能逐条解释 `config.yaml` training 段里每个参数的作用
- [ ] 能画出训练循环的流程图（前向→损失→反向→更新）
- [ ] 能独立训出一个模型，并找到权重文件在哪
- [ ] 能用一句话解释"迁移学习为什么能省时间省数据"

---

### 📅 第 4 周：模型好不好（评估与指标）

**目标**：读懂 `evaluate.py`；能解读全部指标；理解"内部好、外部差"的含义。

**🧠 概念课（约 3h）**
- 混淆矩阵：TP/FP/FN/TN 四个格子
- 准确率 / 精确率 / **召回率** / F1 —— 召回率=不漏诊，医疗最看重
- ROC 曲线 & AUC；PR 曲线（不平衡数据用 PR 更准）
- **ECE 校准误差**："模型说 90% 置信度，真实正确率是不是 90%"
- **域漂移（domain shift）**：换医院/设备/人群，性能掉多少

**📖 精读（约 4h）**
- `scripts/evaluate.py`（465行）← 本周重点，对照概念课边读边对
- `experiments_multicenter/evaluate_external.py`（外部验证怎么做的）
- `outputs/verify_seed_42/evaluation/` 里的报告和四张图

**🛠️ 动手任务（约 5h）**
1. 跑 `evaluate.py`，把 `evaluation_report.json` 里每个指标用自己的话写一遍解释
2. 打开 4 张输出图：混淆矩阵 / ROC+PR / 校准曲线 / 错误分析，逐一读图
3. 对比内部 AUC 0.992 和外部 RSNA AUC 0.807，写下 3 条你想到的解释

**✅ 验收标准**
- [ ] 能独立解读任意一份评估报告
- [ ] 能解释"召回率 96.4% 为什么比准确率 96.1% 更重要"
- [ ] 能用大白话解释 ECE（校准）
- [ ] 能说出"为什么内部 0.992、外部 0.807"至少 2 条原因（域漂移、No Lung Opacity 亚组误判等）

> 💡 本项目核心叙事：**内部测试不确定性转诊很有效（漏诊归零），但跨机构一塌糊涂**。这个"好-坏"反差就是论文的卖点，也是医学 AI 泛化的真实边界。

---

### 📅 第 5 周：模型怎么解释自己 + 怎么知道自己不懂（XAI 与不确定性）

**目标**：读懂 `explain.py` / `uncertainty.py` / `triage_curve.py`，理解三个"信任机制"。

**🧠 概念课（约 3h）**
- 可解释性（XAI）为什么对医疗重要（医生不敢用黑盒）
- **Grad-CAM**：热力图定位"模型看了哪片肺"
- **Integrated Gradients**：像素级归因
- **Occlusion Sensitivity**：遮住一块看预测变不变
- **MC Dropout**：推理时开 Dropout 跑 30 次，方差=不确定性
- **Temperature Scaling**：一个温度系数让概率变"诚实"
- **转诊/拒识曲线**：把高不确定的交给人工，剩余集错误率下降

**📖 精读（约 4h）**
- `scripts/explain.py`（332行）
- `scripts/uncertainty.py`（329行）
- `scripts/triage_curve.py`（174行）
- `README.md`「核心技术详解」段落

**🛠️ 动手任务（约 5h）**
1. 跑 `explain.py` 生成热力图，对比一张正常图和一张肺炎图
2. 跑 `uncertainty.py`，看哪些样本被标为"不确定"
3. 复现转诊结论：转诊 25% → 剩余集漏诊归零（对比随机拒识无效）

**✅ 验收标准**
- [ ] 能对比说明三种可解释方法的原理和区别
- [ ] 能讲清楚 MC Dropout 怎样"算出"不确定性（均值=预测，标准差=不确定）
- [ ] 能解释为什么转诊高不确定病例能降低错误率、而随机转诊不能
- [ ] 能说清"校准"和"不确定性"是两码事

---

### 📅 第 6 周：工程化 + 串讲成故事

**目标**：读懂 `app.py` 和 `api/`；能把整个项目给别人讲明白。

**🧠 概念课（约 2h）**
- 什么是前后端、REST API、SQLite 数据库
- Streamlit（Python 写网页）vs FastAPI（后端接口）各自角色

**📖 精读（约 3h）**
- `app.py`（485行）← 看"上传→诊断→展示"的数据流
- `api/main.py`（280行）、`api/schemas.py`、`api/database.py`
- `Dockerfile` / `docker-compose.yml`（了解即可）

**🛠️ 动手任务（约 5h）**
1. 在 `app.py` 加一个小功能（例如显示预测的 MC Dropout 标准差）
2. 启动 FastAPI，用浏览器打开 `/docs` 页面，调一次 `/predict` 接口
3. （可选）`docker-compose up -d` 一键启动
4. **终极验收**：用「面试/汇报叙事」口吻，写一段 3 分钟项目介绍，讲给朋友/同事听

**✅ 验收标准**
- [ ] 能讲清完整数据流：前端上传 → API → 模型 → 返回结果
- [ ] 3 分钟能把项目讲明白（问题→方案→创新→结果）
- [ ] 能说出项目至少 3 个"坑"（泄漏、评估 bug、外部泛化失败）

---

## 三、学习资源清单（概念不会的来这里补）

| 需要补的 | 免费资源 | 建议 |
|:---|:---|:---|
| Python 基础 | 廖雪峰 Python 教程 | 补到"函数、文件读写"即可 |
| NumPy / PyTorch 基础 | PyTorch 官方《60 分钟入门》 | 必看，本项目的骨骼 |
| 深度学习概念 | 李沐《动手学深度学习》(d2l.ai 中文版) | 只读 CNN、训练章节 |
| Grad-CAM / 可解释性 | 搜索"Grad-CAM 原理 医学影像"科普 | 看 2-3 篇博客即可 |
| MC Dropout 不确定性 | 搜"MC Dropout 不确定性量化 原理" | 看懂"采样求方差"即可 |
| 医学影像基础知识 | 了解 X 光、胸片读片基本概念 | 够用即可 |

**省时间原则**：你目标是"看懂+跑通"，**不要**一头扎进数学推导。看到公式，问 Claude"用大白话讲"，理解直觉后跳过。

---

## 四、常见坑提醒

1. **环境装不上** → 用 `python -m venv .venv` 建独立环境，别污染系统 Python
2. **数据下载慢/失败** → 手动下载 Kaggle chest_xray 后放到 `data/` 对应位置
3. **显存不够** → 把 `config.yaml` 的 `batch_size` 改小（如 16→8），或用 CPU（会慢）
4. **训练太久** → 验证时把 `epochs` 改 2-3；正式复现再改回 25
5. **别乱改 `outputs/` 里的权威结果文件** → 那是论文要用的证据（`paper/notes/project_truth.md` 里叫"稳定约束"）
6. **评估必须显式传权重** → 这是项目踩过的 bug（D8），`load_trained_model` 无参时会加载错模型，学会看 `--weights` 参数

---

## 五、学习心态建议

- **费曼学习法**：每周学完，用大白话把本周内容讲给一个"不懂技术的人"听，讲不通的地方就是没懂
- **带着问题读代码**：每次读文件前先问"这个文件想解决什么问题"，再读代码
- **让 Claude 当陪练**：随时问它"这行代码在干嘛""为什么这么写""train 和 eval 模式区别"，把它当免费一对一老师
- **验收是硬指标**：每周的复选框都打勾再进入下周，别赶进度

祝学习顺利！学完第 6 周，你就是能给别人讲清楚医学影像 AI 的人了 🩻
