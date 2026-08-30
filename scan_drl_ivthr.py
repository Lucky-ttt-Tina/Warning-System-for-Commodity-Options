# -*- coding: utf-8 -*-
"""B7: 扫描 DRL 合并的 IV 确认阈值 iv_thr，找 P>=0.50 & R>=0.60 平衡点（不重训，加载已训策略）。"""
import numpy as np
import polars as pl

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = f"{ROOT}/data/clean/features/15m_official/_features_combined.parquet"
WARN = f"{ROOT}/data/clean/warnings/official/15m/_warnings_15m.parquet"
EV = f"{ROOT}/data/clean/events/fused_events_official.parquet"
POL = f"{ROOT}/data/clean/warnings/official/drl/drl_15m_policy.npz"
DAY = np.timedelta64(1, "D")
FEATS = ['atm_iv_p', 'skew_p', 'term_slope_p', 'curvature_p', 'rr_p', 'bf_p',
         'vpin_p', 'oi_flow_p', 'vol_p', 'amihud_p', 'jump_p']
feat = pl.read_parquet(FEAT)
warn = pl.read_parquet(WARN).sort(['variety', 'datetime'])
events = pl.read_parquet(EV)
npz = np.load(POL)
W_px, b_px, tau_px = npz['W_px'], npz['b_px'], npz['tau_px']
joined = feat.join(warn.select(['variety', 'datetime', 'expiry_month', 'event_px']),
                  on=['variety', 'datetime', 'expiry_month'], how='left').filter(pl.col('atm_iv_p').is_not_null())

def make_state(r):
    return np.array([(r[c] if (r[c] is not None and r[c] == r[c]) else 0.0) for c in FEATS], float)

def iv_runs(aiv, times, thr, mb):
    flag = aiv >= thr; runs = []; s = None
    for i in range(len(flag)):
        if flag[i] and s is None: s = i
        if (not flag[i] or i == len(flag) - 1) and s is not None:
            e = i if not flag[i] else i
            if e - s + 1 >= mb: runs.append((times[s], times[e])); s = None
    return runs

def event_runs(flag, times, mb):
    runs = []; s = None
    for i in range(len(flag)):
        if flag[i] and s is None: s = i
        if (not flag[i] or i == len(flag) - 1) and s is not None:
            e = i if not flag[i] else i
            if e - s + 1 >= mb: runs.append((times[s], times[e])); s = None
    return runs

def prob(S, W, b):
    return 1 / (1 + np.exp(-(S @ W + b)))

def pr(alert, pos):
    alert = np.asarray(alert, bool)
    tp = int((alert & pos).sum()); fp = int((alert & ~pos).sum()); fn = int((~alert & pos).sum())
    P = tp / (tp + fp) if (tp + fp) else float('nan')
    R = tp / (tp + fn) if (tp + fn) else float('nan')
    return P, R

for iv_thr in [0.72, 0.75, 0.78, 0.80, 0.82, 0.85]:
    Ps = []; Rs = []; Na = []
    for v in ['au', 'cu', 'sc']:
        sub = [r for r in joined.to_dicts() if r['variety'] == v]
        S = np.array([make_state(r) for r in sub])
        iv_p = np.array([(r['atm_iv_p'] if r['atm_iv_p'] == r['atm_iv_p'] else 0.0) for r in sub])
        evpx = np.array([bool(r['event_px']) for r in sub])
        times = np.array([r['datetime'] for r in sub], dtype='datetime64[ns]')
        alert_px = prob(S, W_px, b_px) >= tau_px
        a_comb = alert_px & (iv_p >= iv_thr)
        evs = events.filter((pl.col('variety') == v) & (pl.col('freq') == '15m')).to_dicts()
        wins = [(np.datetime64(e['start']) - DAY, np.datetime64(e['end'])) for e in evs]
        wins += iv_runs(iv_p, times, 0.85, 32)
        wins += event_runs(evpx, times, 32)
        pos = np.zeros(len(times), bool)
        for vs, ve in wins: pos |= (times >= vs) & (times <= ve)
        P, R = pr(a_comb, pos)
        Ps.append(P); Rs.append(R); Na.append(int(a_comb.sum()))
    Pm, Rm = float(np.nanmean(Ps)), float(np.nanmean(Rs))
    print(f"iv_thr={iv_thr}: P={Pm:.3f} R={Rm:.3f} 报警={sum(Na)} {'P>=0.50&R>=0.60' if (Pm >= 0.50 and Rm >= 0.60) else ''}")
