# -*- coding: utf-8 -*-
"""
B7 精确率调优扫描：在官方特征上搜索规则预警最优配置，使 rule 口径精确率 P>=0.50 且召回率 R>=0.60。
- 模式A: 仅调 composite L1 阈值 t
- 模式B: composite>=0.80 且 订单流分位最大值 >= k（协同确认，过滤纯 IV 尖峰误报）
- 模式C: composite>=t 且 订单流分位最大值 >= 0.90
正样本 = fused 命名事件 ∪ 数据驱动 IV 压力连通段(>=0.85 连续 32 根, 含前 1 日提前区)。
"""
import numpy as np
import polars as pl

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = f"{ROOT}/data/clean/features/15m_official/_features_combined.parquet"
EV = f"{ROOT}/data/clean/events/fused_events_official.parquet"
DAY = np.timedelta64(1, "D")
CORE = ['atm_iv_p', 'skew_p', 'term_slope_p', 'curvature_p']
OF = ['vpin_p', 'oi_flow_p', 'vol_p', 'amihud_p', 'jump_p']

feat = pl.read_parquet(FEAT)
events = pl.read_parquet(EV)

def stress_row(r):
    def z(x):
        return x if (x is not None and x == x) else 0.0
    s = {c: z(r[c]) for c in ['atm_iv_p', 'skew_p', 'term_slope_p', 'rr_p', 'bf_p']}
    cv = r['curvature_p']
    s['curvature_p'] = max(cv, 1 - cv) if (cv is not None and cv == cv) else 0.0
    return s

def iv_runs(aiv, times, thr, min_bars):
    flag = aiv >= thr; runs = []; ss = None
    for i in range(len(flag)):
        if flag[i] and ss is None: ss = i
        if (not flag[i] or i == len(flag) - 1) and ss is not None:
            e = i if not flag[i] else i
            if e - ss + 1 >= min_bars: runs.append((times[ss], times[e]))
            ss = None
    return runs

def pos_windows(v, times, aiv):
    evs = events.filter((pl.col('variety') == v) & (pl.col('freq') == '15m')).to_dicts()
    wins = [(np.datetime64(e['start']) - DAY, np.datetime64(e['end'])) for e in evs]
    wins += iv_runs(aiv, times, 0.85, 32)
    pos = np.zeros(len(times), bool)
    for vs, ve in wins: pos |= (times >= vs) & (times <= ve)
    return pos

def pr(alert, pos):
    alert = np.asarray(alert, bool)
    tp = int((alert & pos).sum()); fp = int((alert & ~pos).sum()); fn = int((~alert & pos).sum())
    P = tp / (tp + fp) if (tp + fp) else float('nan')
    R = tp / (tp + fn) if (tp + fn) else float('nan')
    return P, R

def eval_config(t, k=None):
    Ps = []; Rs = []; Na = []
    for v in ['au', 'cu', 'sc']:
        d = feat.filter(pl.col('variety') == v).sort('datetime')
        times = d['datetime'].to_numpy(); aiv = d['atm_iv_p'].to_numpy().astype(float)
        ofmax = d.select([pl.col(c) for c in OF]).to_numpy().astype(float).max(axis=1)
        alerts = []
        for i, r in enumerate(d.to_dicts()):
            s = stress_row(r); comp = float(np.mean([s[c] for c in CORE]))
            maxany = float(max(s[c] for c in s))
            lvl = 0
            if (comp >= 0.95) or (maxany >= 0.99): lvl = 3
            elif (comp >= 0.88) or (maxany >= 0.97): lvl = 2
            elif comp >= t: lvl = 1
            if lvl >= 1 and k is not None and not (ofmax[i] >= k): lvl = 0
            alerts.append(lvl >= 1)
        alert = np.array(alerts)
        pos = pos_windows(v, times, aiv)
        P, R = pr(alert, pos)
        Ps.append(P); Rs.append(R); Na.append(int(alert.sum()))
    return dict(P=round(float(np.nanmean(Ps)), 3), R=round(float(np.nanmean(Rs)), 3), N=sum(Na))

print("=== 模式A: 仅 composite L1 阈值 t ===")
for t in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90]:
    r = eval_config(t=t)
    flag = "达标" if (r['P'] >= 0.50 and r['R'] >= 0.60) else ""
    print(f"  t={t}: P={r['P']} R={r['R']} 报警={r['N']} {flag}")

print("=== 模式B: composite>=0.80 且 订单流max>=k (协同) ===")
for k in [0.85, 0.90, 0.95]:
    r = eval_config(t=0.80, k=k)
    flag = "达标" if (r['P'] >= 0.50 and r['R'] >= 0.60) else ""
    print(f"  k={k}: P={r['P']} R={r['R']} 报警={r['N']} {flag}")

print("=== 模式C: composite>=t 且 订单流max>=0.90 ===")
for t in [0.82, 0.84, 0.86, 0.88]:
    r = eval_config(t=t, k=0.90)
    flag = "达标" if (r['P'] >= 0.50 and r['R'] >= 0.60) else ""
    print(f"  t={t},k=0.90: P={r['P']} R={r['R']} 报警={r['N']} {flag}")
