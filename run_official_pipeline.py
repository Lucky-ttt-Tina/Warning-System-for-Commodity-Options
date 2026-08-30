# -*- coding: utf-8 -*-
"""
官方数据管线 · 一键编排（可复现入口）
====================================
按依赖顺序运行 data_pipeline/official/ 下所有独立并行脚本，
与 data_pipeline/{clean,features,warning,events,report} 及 docs/ 下的
【自建程序完全隔离】——自建脚本一律不读、不改、不依赖。

运行：
    python data_pipeline/official/run_official_pipeline.py

说明：
    - 须使用已安装 polars 的 Python 解释器运行（polars 用 Rust 引擎读写 parquet，禁 to_pandas）。
    - 任一阶段失败即中断并打印到错误位置；成功阶段产出不回滚，可重跑补齐。
    - 所有阶段（A0→E16）已落地，共 19 个脚本按依赖顺序执行。
"""
import subprocess, sys, os, time

# ===== 根路径按脚本自身位置相对推导（换机器/换 cwd 均可复现，无需改路径）=====
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFF  = f"{ROOT}/data_pipeline/official"
PY   = sys.executable  # 用运行本编排器的解释器（须为已安装 polars 的 Python）

STEPS = [
    # (阶段, 脚本, 说明)
    ("A0", "data_profiling.py",          "官方 archive 全量诊断 (6866万行 24列字段字典/缺失率/IV越界/清洗保留率)"),
    ("A1", "build_official_features.py",  "官方特征重建：11维IV曲面+订单流分位"),
    ("A2", "align_quality.py",            "11维质量对齐 + 官方vs自建 atm_iv 相关性"),
    ("A1g","build_greeks_features.py",    "Greeks 截面特征 (题目点名 Gamma/Vega 集中度 HHI + 近月ATM Greeks + IV曲面快照)"),
    ("A3", "build_fused_events.py",       "融合事件 42 条 (ours18+official24 去重合并 2 对双重确认 + DCE m/p 命名事件)"),
    ("B6", "build_warnings_15m.py",       "官方规则预警 L0-L3 (6 品种 au/cu/sc + DCE m/c/p)"),
    ("B6", "backtest_systemic.py",        "系统级双轨回测 Track A/B"),
    ("B7", "train_drl_15m.py",            "DRL 自适应预警 BC+τ* + 双确认合并"),
    ("B7", "tune_precision.py",           "规则精确率调优扫描 (可选，约0.5min)"),
    ("B7", "scan_drl_ivthr.py",           "DRL 双确认 iv_thr 扫描 (可选，约0.5min)"),
    ("B8", "train_ppo_15m.py",            "真 PPO 深度强化学习 + BC+τ* ablation"),
    # ---- 已落地，纳入一键编排 ----
    ("B9", "cvar_official.py",               "CVaR(95%) 改善率 (方法学对照口径: H-日去敞口)"),
    ("B9c","cvar_competition.py",            "CVaR(95%) 竞赛口径 (多空双向/L≥2/次日减半/持有到底/仅首次)"),
    ("C10","build_knowledge_graph.py",        "知识图谱重建 (官方预警)"),
    ("C11","prepare_explanation_review.py",   "解释文本分层抽样 + 评分量规 (人工抽检样本)"),
    ("D12","build_dashboard.py",              "官方预警系统可视化看板 (自包含 HTML, 含 Greeks曲线+曲面热力图)"),
    ("D13","build_report_official.py",        "技术报告终稿 + 英文执行摘要 (可复现)"),
    # ---- E15 跨品种扩展（已落地：A1/A3/B6/B7/B9/C10/D12/D13 品种列表扩至 6 品种，无独立脚本）----
    # ---- E16 跨样本稳健性对照 ----
    ("E16", "robustness_check.py",          "跨样本稳健性对照 (时间分段 + 阈值敏感性)"),
]

def run(stage, script, desc):
    path = f"{OFF}/{script}"
    if not os.path.exists(path):
        print(f"  [SKIP] {stage} {script} 不存在（待落地）")
        return True
    t0 = time.time()
    print(f"\n>>> [{stage}] {script} — {desc}")
    try:
        rc = subprocess.run([PY, path], cwd=ROOT, check=True,
                            stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {stage} {script} 退出码 {e.returncode}")
        return False
    print(f"  [OK] {stage} {script} 用时 {time.time()-t0:.1f}s")
    return True

def main():
    print("=" * 64)
    print("官方数据管线一键编排 (与自建程序物理隔离)")
    print(f"ROOT={ROOT}\nPY={PY}")
    print("=" * 64)
    ok = 0
    for stage, script, desc in STEPS:
        if not run(stage, script, desc):
            print("\n编排中断：上一阶段失败，请修复后重跑（已完成阶段不回滚）。")
            sys.exit(1)
        ok += 1
    print("\n" + "=" * 64)
    print(f"完成 {ok}/{len(STEPS)} 个阶段（PENDING 段为待落地脚本）")
    print("产物位于 data/clean/features/15m_official/ 与 data/clean/official/")
    print("=" * 64)

if __name__ == "__main__":
    main()
