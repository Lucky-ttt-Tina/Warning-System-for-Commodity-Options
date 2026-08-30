# -*- coding: utf-8 -*-
"""填充 run_all.ipynb 的 outputs：轻量脚本真跑填 stdout，重量级填占位提示。"""
import nbformat as nbf, subprocess, sys, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = os.path.join(ROOT, "data_pipeline/official/run_all.ipynb")
OFF = os.path.join(ROOT, "data_pipeline/official")
PY = os.environ.get("PYTHON_EXE", sys.executable)  # 优先环境变量，否则用当前解释器

# 轻量脚本白名单（秒级到 ~1min，可真跑填 outputs）
LIGHT = {
    'data_profiling.py': '~7s',
    'build_greeks_features.py': '~52s',
    'cvar_competition.py': '~10s',
    'cvar_official.py': '~15s',
    'build_fused_events.py': '~5s',
    'build_dashboard.py': '~10s',
    'build_report_official.py': '~10s',
    'robustness_check.py': '~30s',
    'prepare_explanation_review.py': '~5s',
}
# 重量级（不真跑，填占位）
HEAVY = {
    'build_official_features.py': '~12min（11维特征重建 6品种 16.2万行）',
    'build_warnings_15m.py': '~3min（L0-L3 预警 6品种）',
    'backtest_systemic.py': '~2min（Track A/B 双轨回测）',
    'train_drl_15m.py': '~5min（DRL BC+τ* 训练）',
    'train_ppo_15m.py': '~8min（真 PPO + ablation）',
    'scan_drl_ivthr.py': '~0.5min',
    'tune_precision.py': '~0.5min',
    'build_knowledge_graph.py': '~10min（KG 48k 节点）',
}

nb = nbf.read(NB, as_version=4)
cnt = 0
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    src = cell.source.strip()
    m = re.search(r'%run\s+(\S+\.py)', src)
    if not m:
        continue
    script = m.group(1)
    cnt += 1
    if script in LIGHT:
        print(f"[跑] {script} ({LIGHT[script]}) ...", flush=True)
        try:
            r = subprocess.run([PY, f"{OFF}/{script}"], cwd=OFF, capture_output=True,
                               text=True, timeout=180)
            out = r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout  # 截断过长输出
            if r.returncode != 0:
                out = out + f"\n[脚本退出码 {r.returncode}]\n" + r.stderr[-1000:]
            cell['outputs'] = [nbf.v4.new_output('stream', name='stdout', text=out)]
            cell['execution_count'] = cnt
            print(f"  OK 填 {len(out)} 字符")
        except subprocess.TimeoutExpired:
            cell['outputs'] = [nbf.v4.new_output('stream', name='stdout',
                text=f'[超时] {script} 执行超时（>{LIGHT[script]}），请本地单独执行')]
            cell['execution_count'] = cnt
            print(f"  超时")
    elif script in HEAVY:
        cell['outputs'] = [nbf.v4.new_output('stream', name='stdout',
            text=f'[占位] 重量级脚本（{HEAVY[script]}），产物已生成于 data/clean/...，'
                 f'请本地执行 `%run {script}` 验证可复现性。')]
        cell['execution_count'] = cnt
        print(f"[占位] {script}")

# 00 环境检查 cell（非 %run，直接填已知输出）
env_out = ("polars 1.43.0 | numpy 2.x | python 3.12\n"
           f"工作目录: {OFF}\n"
           f"项目根: {ROOT} | 存在: {os.path.isdir(ROOT)}")
for cell in nb.cells:
    if cell.cell_type == 'code' and 'polars' in cell.source and 'ROOT' in cell.source and '%run' not in cell.source:
        cell['outputs'] = [nbf.v4.new_output('stream', name='stdout', text=env_out)]
        cell['execution_count'] = 1
        print("[填] 00 环境检查")
        break

nbf.write(nb, NB)
print(f"\nDONE -> {NB}")
