# -*- coding: utf-8 -*-
"""
生成 run_all.ipynb —— 官方管线一键复现 notebook（Day 3，用户要求2）
==============================================================
- 单文件、分 8 章，每章 markdown 说明 + code cell（%run 调用现成 .py）
- 代码单一来源：调用 data_pipeline/official/ 下已验证脚本，避免双份维护
- 每段脚本已有 print，%run 后输出作为 cell output，用户本地重跑即可验证可复现
- 原 .py 保留为源码库交付（题目要求"全部源代码"）
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}
}
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 宁证期货赛题 · 官方数据管线一键复现（团队：新兴一代）

**使用方法**：在 JupyterLab 中打开本文件，菜单 *Run All* 即可重跑全部 8 章，每段输出（print/统计表）会内嵌显示，用于验证可复现性。

**章节结构**：
1. 01 数据诊断与清洗
2. 02 Greeks 截面特征
3. 03 分级预警
4. 04 回测与指标
5. 05 CVaR 竞赛口径
6. 06 DRL 自适应
7. 07 知识图谱与解释
8. 08 看板与报告

**耗时提示**：轻量章节（01 诊断、02 Greeks、05 CVaR、08 看板）秒级；重量级（01 特征构建 ~12min、03 预警、06 DRL、07 KG）数分钟。预计总耗时 ~20min。

**环境**：Python 3.12 + polars（Rust 引擎，规避 pyarrow 哈希拦截）。依赖见 `requirements.txt`。""")

md("## 00 环境检查")
code("""import polars as pl, numpy as np, sys, os
print("polars", pl.__version__, "| numpy", np.__version__, "| python", sys.version.split()[0])
print("工作目录:", os.getcwd())
# 官方管线根目录（脚本内 ROOT 硬编码绝对路径，与 notebook 位置无关）
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("项目根:", ROOT, "| 存在:", os.path.isdir(ROOT))""")

md("""## 01 数据诊断与清洗
**官方 archive 全量诊断**（68,662,686 行 / 2798 文件 / 1.73GB）：24 列字段字典、缺失率、IV 越界、Greeks 覆盖、清洗规则与保留率。产物 `data/clean/official/data_profiling.md`。
- Greeks 列缺失率 0，数据完整度高
- IV 越界率 10-18%（深虚值/近到期 BS 反推失真），A1 用 (0.01,2.5) 清洗保留 82-90%""")
code("""# 1a 全量诊断（约 7 秒）
%run data_profiling.py""")
code("""# 1b 特征构建 build_official_features（A1，约 12 分钟）
# 产出 11 维特征（6 IV 曲面 + 5 订单流）×6 品种 = 162,239 行
# 重量级：重跑会覆盖 data/clean/features/15m_official/_features_combined.parquet
%run build_official_features.py""")

md("""## 02 Greeks 截面特征（题目点名的 Gamma/Vega 截面集中度）
读官方 archive 的 delta/gamma/vega/theta 列，产出：①近月 ATM Greeks 时序 ②Gamma/Vega 截面集中度 HHI ③最新数据齐全时刻 IV 曲面快照。约 52 秒。
- delta_atm 均值 0.4992≈0.5（验证正确）
- gamma_conc 中位 0.01-0.07（风险分散时段）""")
code("%run build_greeks_features.py")

md("""## 03 分级预警（问题1 必选）
基于 11 维特征 + 滚动分位构建 L0-L3 分级预警，输出可审计警报。""")
code("%run build_warnings_15m.py")

md("""## 04 回测与指标（Track A 命名事件 + Track B 外生标签）
- Track A：18 命名事件召回率 au/cu=1.0, sc=0.80, SR=0.83
- Track B：外生 rv/px 标签综合召回 0.68-0.81
- 中位提前量 au 4762 / cu 7050 / sc 1350 min ≫ 30min""")
code("%run backtest_systemic.py")

md("""## 05 CVaR(95%) 竞赛口径（核心排名指标，多空双向）
严格按题目原文：多空双向各1手、L≥2 触发、下一交易日开盘减半、持有到底、仅首次预警、多空改善率算术平均。
- 结果（全达标）：rule 全期 50.0% / drl 全期 49.53%（平均改善率 45-50% ≫ 10%）
- 下方 cvar_official 为方法学对照（H-日去敞口口径，rule 51.4%/drl 23.9%）""")
code("# 5a 竞赛口径（约 10 秒）\n%run cvar_competition.py")
code("# 5b 方法学对照口径\n%run cvar_official.py")

md("""## 06 DRL 自适应（问题3 加分项）
- BC+τ* 部署系统（F1=0.544）+ 真 PPO 公平 ablation（PPO+τ* F1=0.034，小样本弱信号崩溃）
- DRL vs 固定阈值 F1 +83%（外生事件口径）
- 诚实记录：PPO 作 ablation，部署保留 BC+τ*""")
code("# 6a DRL 双确认预警\n%run train_drl_15m.py")
code("# 6b 真 PPO 训练 + 公平 ablation\n%run train_ppo_15m.py")
code("# 6c DRL 双确认扫描\n%run scan_drl_ivthr.py")
code("# 6d 精确率调优\n%run tune_precision.py")

md("""## 07 知识图谱与解释（问题2 加分项）
- KG：48,135 节点 / 257,311 边 / 48,109 预警实例，五段式归因（触发因子→形态→历史相似→宏观→结论）
- C11 抽检样本：27 条分层抽样，27/27 五段齐全，待人工评分 ≥4/5""")
code("# 7a KG 构建\n%run build_knowledge_graph.py")
code("# 7b 解释抽检样本\n%run prepare_explanation_review.py")

md("""## 08 看板与报告
- A3 融合事件去重（42 条：ours18 + official24 − 2合并 + DCE m/p 2）
- D12 看板：ECharts 离线自包含，6 品种切换，Greeks 曲线 + 曲面热力图 + 42 场考卷
- D13 报告：终稿 + 英文摘要
- E16 稳健性：阈值敏感性不敏感 + 时间分段波动""")
code("# 8a 融合事件去重\n%run build_fused_events.py")
code("# 8b 看板生成\n%run build_dashboard.py")
code("# 8c 技术报告\n%run build_report_official.py")
code("# 8d 稳健性检查\n%run robustness_check.py")

md("""---
## 完成核对清单
执行完以上 8 章后，以下产物应全部存在（路径相对项目根）：
- `data/clean/official/data_profiling.md` — 数据诊断
- `data/clean/features/15m_official/_features_combined.parquet` — 11 维特征
- `data/clean/features/15m_official/_greeks_combined.parquet` — Greeks 特征
- `data/clean/warnings/official/15m/_warnings_15m.parquet` — 预警
- `data/clean/official/cvar/cvar_competition.md` — CVaR 竞赛口径
- `data/clean/warnings/official/drl/drl_15m_alert.parquet` — DRL 信号
- `data/clean/warnings/official/kg/` — KG 节点/边/解释
- `data/clean/warnings/official/dashboard/index.html` — 看板
- `data/clean/official/report/final_report.md` — 技术报告
- `data/clean/official/robustness_official.md` — 稳健性""")

nb['cells'] = cells

OUT = os.path.join(ROOT, "data_pipeline/official/run_all.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"OK -> {OUT} | cells={len(cells)}")
