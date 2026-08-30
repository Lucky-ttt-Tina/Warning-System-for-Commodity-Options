# 期权波动率风险智能预警系统（官方数据口径）

---

## 1. 项目简介

本项目针对期权市场的波动率尾部风险，构建了一套"特征—预警—解释—量化"四位一体的智能预警系统。

- **问题 1（必选）**：从 IV 曲面中提取 6 维结构特征 + 5 维订单流压力特征，构建 **L0–L3 四级预警**体系，并配套可视化看板。
- **问题 2（加分）**：以知识图谱组织"触发因子—市场形态—历史相似—宏观环境—推理结论"五段式归因，支持人工可审计解释。
- **问题 3（加分）**：采用 **`BC+τ*`** 自适应预警子系统，在外生事件口径下相对固定阈值 **F1 提升 +83%**。
- **尾部风险量化**：双口径并列——竞赛原文口径（多空双向）与方法学口径（H-日去敞口）均实现 **CVaR(95%) 改善 >10%**。

所有官方数据脚本置于 `data_pipeline/official/`，与既有自建程序物理隔离，一键复现。

---

## 2. 环境与安装

### 2.1 Python 环境

建议使用 **Python 3.11+**，并新建虚拟环境：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

> 注意：`polars` 使用 Rust 引擎读写 Parquet。运行过程中**请勿**调用 `polars.to_pandas()`，以避免被 Windows 策略按文件哈希拦截。

### 2.2 数据准备

本系统使用宁证期货提供的交易所官方期权 archive（SF/INE/DF/GF 四所）：

- 官方数据目录：`官方data/`（位于项目根下，与所有脚本同源；脚本已改为按自身位置相对推导根路径，**换机器无需改任何路径常量**）。
- 若官方数据放在别处，可用符号链接将 `官方data/` 指到项目根下，或把整个项目文件夹整体移动到任意路径。

数据规模约 6,866 万行（2,798 个 Parquet 文件，约 1.73 GB）。

---

## 3. 一键复现

在项目根目录下执行：

```bash
python data_pipeline/official/run_official_pipeline.py
```

该编排器会按顺序执行 **A0 → E16 共 19 个脚本**：

- A0 数据诊断
- A1 官方特征重建
- A2 质量对齐
- A3 融合事件
- B6 规则预警 + 系统级双轨回测
- B7 DRL 自适应 + 调优扫描
- B8 真 PPO ablation
- B9/B9c CVaR（方法学口径 + 竞赛原文口径）
- C10 知识图谱
- C11 解释文本抽检样本
- D12 可视化看板
- D13 技术报告
- E16 稳健性对照

运行完成后，所有产物落入 `data/clean/official/` 与 `data/clean/warnings/official/`。

Windows 用户也可直接双击：

```text
run_official_pipeline.bat
```

---

## 4. Notebook 分章节验证

打开 `data_pipeline/official/run_all.ipynb`，使用 JupyterLab：

```bash
jupyter lab data_pipeline/official/run_all.ipynb
```

- 8 章 29 cells，每章 `%run` 调用现成 `.py` 脚本。
- 10 段轻量章节已填充真实输出，8 段重量级章节标注耗时与产物位置。
- 点击 **Run All** 即可重跑验证全链路可复现性（预计总耗时约 20 分钟，重量级章节可单独执行）。

---

## 5. 核心产物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| 官方特征 | `data/clean/features/15m_official/_features_combined.parquet` | 162,239 行，11 维 + variety |
| Greeks 特征 | `data/clean/official/greeks/_greeks_combined.parquet` | 162,374 行，6 品种 ATM Greeks + HHI |
| 规则预警 | `data/clean/warnings/official/15m/_warnings_15m.parquet` | L0–L3 分级 |
| DRL 预警 | `data/clean/warnings/official/drl/drl_15m_alert.parquet` | BC+τ* 双确认 |
| CVaR（方法学） | `data/clean/official/cvar/cvar_official.md(.json)` | H-日去敞口口径 |
| CVaR（竞赛） | `data/clean/official/cvar/cvar_competition.md(.json)` | 多空双向原文口径 |
| 知识图谱 | `data/clean/warnings/official/kg/` | 48,135 节点 / 257,311 边 |
| 解释样本 | `data/clean/warnings/official/kg/kg_sample_for_review.md` | 27 条分层抽样 |
| 可视化看板 | `data/clean/warnings/official/dashboard/index.html` | 离线自包含 HTML |
| 技术文档 | `data/clean/official/report/技术文档_期权波动率风险智能预警系统.docx` | 图文报告 |
| 精益画布 | `精益画布_新兴一代.pptx` | 附件 3 模板填充 |
| 技术报告 Markdown | `data/clean/official/report/final_report.md` | docx 源文件 |

---

## 6. 目录结构

```text
.
├── data_pipeline/official/          # 官方数据管线（物理隔离，可独立复现）
│   ├── build_official_features.py
│   ├── build_greeks_features.py      # Greeks 截面特征
│   ├── data_profiling.py             # 数据诊断
│   ├── build_warnings_15m.py
│   ├── backtest_systemic.py
│   ├── train_drl_15m.py
│   ├── cvar_official.py
│   ├── cvar_competition.py          
│   ├── build_knowledge_graph.py
│   ├── build_dashboard.py
│   ├── build_report_official.py
│   ├── run_official_pipeline.py      # 19 脚本一键编排
│   ├── run_all.ipynb                 # 分章节复现 notebook
│   └── ...
├── data/
│   ├── clean/official/               # 产物
│   └── clean/warnings/official/      # 预警/看板/KG 产物
├── 官方data/                         
├── build_lean_canvas.py
├── build_docx_report.py
├── build_figures.py
├── requirements.txt
└── README.md

```

---

## 7. 复现注意事项

1. **路径（已自动化）**：所有脚本的 `ROOT` 均按脚本自身位置相对推导，**换机器/换工作目录均无需修改任何路径常量**。只需保证官方数据目录 `官方data/` 与项目根同源（或软链到根下）。
2. **内存（重要）**：C10 知识图谱在内存紧张时可能 OOM 中断。若中断，已完成的 A0–B9 产物保留，单独运行 `run_remaining.py` 即可补齐 C11–E16；或分批执行各阶段脚本规避累积内存。
3. **时间**：一键编排首次全量运行约 20–30 分钟（A1 特征重建 + C10 KG 为耗时大户）。
4. **环境**：所有脚本在 Python 3.13 + polars 1.43 环境下验证通过。
4. **诚实口径**：技术报告中已诚实说明规则全量 L1 精确率天花板、DCE 命名事件稀疏、PPO 未超越 BC+τ* 等局限。

