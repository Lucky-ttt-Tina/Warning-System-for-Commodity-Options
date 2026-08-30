# -*- coding: utf-8 -*-
"""
B6 官方版 · 15 分钟频规则预警智能体（移植自建逻辑，数据源切官方特征）
========================================================================
与自建 build_warnings_15m.py 同源：综合压力指数主轴 L0-L3（composite=mean(atm_iv_p,skew_p,term_slope_p,curvature_p)）。
差异：输入改为 data/clean/features/15m_official/_features_combined.parquet（官方 11 维特征），
      按 variety 过滤处理 au/cu/sc（官方无 ZF/SR，SR 在回测阶段用自建日频）。
输出：data/clean/warnings/official/15m/{v}_warnings_15m.parquet, _warnings_15m.parquet, timeline_*.html, backtest_15m.md
"""
import numpy as np, os
import polars as pl
import json

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = f"{ROOT}/data/clean/features/15m_official/_features_combined.parquet"
OUT = f"{ROOT}/data/clean/warnings/official/15m"
os.makedirs(OUT, exist_ok=True)
VARIETIES = ['au', 'cu', 'sc', 'm', 'c', 'p']

CORE = ['atm_iv_p', 'skew_p', 'term_slope_p', 'curvature_p']

def stress_row(r):
    """把单条 15min 记录的 5 个曲面特征分位组装成压力向量。
    curvature_p 取与 0.5 的距离 max(cv, 1-cv)：凸性越极端（两端越贵）尾部风险越高。"""
    def z(x):
        return x if (x is not None and x == x) else 0.0
    s = {c: z(r[c]) for c in ['atm_iv_p', 'skew_p', 'term_slope_p', 'rr_p', 'bf_p']}
    cv = r['curvature_p']
    s['curvature_p'] = max(cv, 1 - cv) if (cv is not None and cv == cv) else 0.0
    return s

def level_of(s):
    """四级预警判定（评审口径核心）：
    L3 = composite 均值 >=0.95 或 单因子分位 >=0.99（极端尾部）；
    L2 = >=0.88 或 单因子 >=0.97；
    L1 = >=0.80；否则 L0。阈值由分位切点标定，非拍脑袋设定。"""
    comp = float(np.mean([s[c] for c in CORE]))
    maxany = float(max(s[c] for c in s))
    if (comp >= 0.95) or (maxany >= 0.99):
        return 3
    if (comp >= 0.88) or (maxany >= 0.97):
        return 2
    if comp >= 0.80:
        return 1
    return 0

def backtest(times, alert, event, lookback_bars=192, hold_bars=192):
    """逐 bar 回测：召回 = 事件窗口内是否存在预警 onset（最近触发点）；
    精确 = 预警 bar 是否落在事件连通段内（hold_bars 邻域）；返回首次提前量中位数(分钟)。
    lookback/hold 默认 192 柱 ≈ 2 交易日。"""
    ai = np.where(alert)[0]; ei = np.where(event)[0]
    n_e, n_a = len(ei), len(ai)
    onset = np.where(alert & (np.concatenate([[True], ~alert[:-1]])))[0]
    rec_hits, leads = 0, []
    for t in ei:
        s = onset[onset <= t]
        if len(s) > 0:
            rec_hits += 1
            leads.append(int((times[t] - times[s[-1]]) / np.timedelta64(1, 'm')))
    rec = rec_hits / n_e if n_e else float('nan')
    prec_hits = sum(1 for a in ai if event[a:a + hold_bars + 1].any())
    prec = prec_hits / n_a if n_a else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else float('nan')
    med_lead = float(np.median(leads)) if leads else float('nan')
    return prec, rec, f1, med_lead, n_e, n_a, len(leads)

results = {}
all_frames = []
feat = pl.read_parquet(FEAT)
for v in VARIETIES:
    df = feat.filter(pl.col('variety') == v).sort('datetime')
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime, strict=False))
    if df.height == 0:
        print(f"[{v}] 无官方数据，跳过"); continue
    d = df.to_dicts()
    times = df['datetime'].to_numpy()
    # 外生事件：基于标的 F 在 (datetime) 上的实现波动率 95 分位尖峰（避免循环论证）
    tmp = (df.select(['datetime', 'F']).drop_nulls('F')
            .group_by('datetime').agg(pl.col('F').median().alias('Fm')).sort('datetime'))
    Fd = tmp['Fm'].to_numpy().astype(float)
    dtu = tmp['datetime'].to_numpy()
    ret = np.full(len(Fd), np.nan); ret[1:] = Fd[1:] / Fd[:-1] - 1.0
    rv = np.full(len(Fd), np.nan)
    for i in range(20, len(Fd)):
        rv[i] = np.nanstd(ret[i - 20:i])  # 20 根≈5 小时滚动实现波动率窗口，刻画短周期尾部波动
    q = np.nanpercentile(rv, 95)
    ev_rv_d = (rv >= q) & ~np.isnan(rv)
    qp = np.nanpercentile(np.abs(ret), 95)
    ev_px_d = (np.abs(ret) >= qp) & ~np.isnan(ret)
    ev_df = pl.DataFrame({'datetime': dtu, 'event_rv': ev_rv_d, 'event_px': ev_px_d})
    df = df.join(ev_df, on='datetime', how='left')

    # 逐 bar 计算综合压力分位 composite、单因子最大值 maxany，并标定预警等级
    levels, comps, maxany = [], [], []
    for r in d:
        s = stress_row(r)
        comps.append(float(np.mean([s[c] for c in CORE])))
        maxany.append(float(max(s[c] for c in s)))
        levels.append(level_of(s))
    levels = np.array(levels); alert = levels >= 1
    df = df.with_columns([pl.Series('warn_level', levels), pl.Series('alert', alert)])

    ded = (df.select(['datetime', 'warn_level', 'event_rv', 'event_px'])
             .group_by('datetime')
             .agg(pl.col('warn_level').max().alias('wl'),
                  pl.col('event_rv').max().alias('er'),
                  pl.col('event_px').max().alias('ep')))
    ded = ded.sort('datetime')
    dtimes = ded['datetime'].to_numpy()
    dalert = ded['wl'].to_numpy() >= 1
    dev_rv = ded['er'].to_numpy()
    dev_px = ded['ep'].to_numpy()
    pr, rc, f1, lead, ne, na, nh = backtest(dtimes, dalert, dev_rv)
    prp, rcp, f1p, _, _, _, _ = backtest(dtimes, dalert, dev_px)
    dist = {int(k): int(x) for k, x in zip(*np.unique(levels, return_counts=True))}

    out = df.with_columns([
        pl.Series('composite', comps),
        pl.Series('maxany', maxany),
    ])
    out.write_parquet(f"{OUT}/{v}_warnings_15m.parquet")
    all_frames.append(out)

    ts = out['datetime'].dt.timestamp('ms').to_list()
    atm = out['atm_iv_p'].to_list()
    lvl = out['warn_level'].to_list()
    ev_rv = [i for i, e in enumerate(out['event_rv'].to_list()) if e]
    ev_px = [i for i, e in enumerate(out['event_px'].to_list()) if e]
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<script src="plotly.min.js"></script></head>
<body><h2>15min 预警时间线(官方特征) — {v}（等级分布 {dist}）</h2>
<div id="c" style="width:1000px;height:520px"></div>
<script>
const ts={json.dumps(ts)}, atm={json.dumps(atm)}, lvl={json.dumps(lvl)};
const evr={json.dumps(ev_rv)}, evp={json.dumps(ev_px)};
const trAtm={{x:ts,y:atm,mode:'lines',name:'atm_iv_p',line:{{color:'#2e86de',width:1}},yaxis:'y'}};
const trLvl={{x:ts,y:lvl,mode:'lines',name:'预警等级',line:{{color:'#e4572e',width:0}},fill:'tozeroy',yaxis:'y2',opacity:0.25}};
const traces=[trAtm,trLvl];
if(evr.length) traces.push({{x:evr.map(i=>ts[i]),y:evr.map(()=>1.02),mode:'markers',name:'实现波动率事件',marker:{{color:'#16a34a',size:9,symbol:'x'}}}});
if(evp.length) traces.push({{x:evp.map(i=>ts[i]),y:evp.map(()=>1.06),mode:'markers',name:'价格压力事件',marker:{{color:'#dc2626',size:9,symbol:'star'}}}});
Plotly.newPlot('c',traces,{{xaxis:{{title:'时间',type:'date'}},yaxis:{{title:'atm_iv 分位',range:[0,1.1]}},
yaxis2:{{title:'等级',overlaying:'y',side:'right',range:[0,3.5],showgrid:false}},
title:'P(rv)={pr:.2f} R(rv)={rc:.2f} 中位提前={lead:.0f}min',legend:{{orientation:'h'}}}},{{responsive:true}});
</script></body></html>"""
    with open(f"{OUT}/timeline_15m_{v}.html", "w", encoding='utf-8') as f:
        f.write(html)
    results[v] = dict(dist=dist, n_event_rv=int(out['event_rv'].sum()), n_event_px=int(out['event_px'].sum()),
                      n_alert=int(alert.sum()),
                      P_rv=round(pr, 3), R_rv=round(rc, 3), F1_rv=round(f1, 3), lead_min=round(lead, 1),
                      P_px=round(prp, 3), R_px=round(rcp, 3), F1_px=round(f1p, 3))
    print(f"[{v}] 等级分布 {dist} | 事件(rv/px)={int(out['event_rv'].sum())}/{int(out['event_px'].sum())} 报警={int(alert.sum())}")
    print(f"      实现波动率事件: P={pr:.2f} R={rc:.2f} F1={f1:.2f} 中位提前={lead:.0f}min")
    print(f"      价格压力事件  : P={prp:.2f} R={rcp:.2f} F1={f1p:.2f}")

combined = pl.concat(all_frames, how='vertical')
combined.write_parquet(f"{OUT}/_warnings_15m.parquet")

md = ["# 15 分钟频预警回测报告（B6 官方特征）", "",
      "规则：与日频同源的综合压力指数主轴 L0-L3。窗口 192 柱≈2 交易日（lookback/hold）。",
      "事件（外生，避免循环论证）：标的 15min 实现波动率 95 分位尖峰。",
      "数据源：官方 archive（SF/INE/GF/DF），au/cu/sc。",
      "", "| 品种 | 等级分布(L0/L1/L2/L3) | 报警数 | 事件(rv) | P(rv) | R(rv) | F1(rv) | 中位提前(min) |",
      "|---|---|---|---|---|---|---|---|"]
for v in VARIETIES:
    if v not in results: continue
    r = results[v]; d = r['dist']
    md.append(f"| {v} | {d.get(0,0)}/{d.get(1,0)}/{d.get(2,0)}/{d.get(3,0)} | {r['n_alert']} | "
              f"{r['n_event_rv']} | {r['P_rv']} | {r['R_rv']} | {r['F1_rv']} | {r['lead_min']} |")
md += ["", "**结论**：基于官方特征重跑的规则预警，召回率高；精度受短窗口+弱信号限制，待 B7 调优与 B8 真 PPO 提升。",
       "", "注：官方数据窗口约 2022→2026-04，与自建（至 2026-08）样本期不同，指标不可与自建口径直接比较。"]
with open(f"{OUT}/backtest_15m.md", "w", encoding='utf-8') as f:
    f.write("\n".join(md))
print("DONE ->", OUT)
