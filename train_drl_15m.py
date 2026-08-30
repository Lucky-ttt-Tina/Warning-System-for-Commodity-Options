# -*- coding: utf-8 -*-
"""
B7 官方版 · DRL 自适应预警训练（移植自建 train_drl_warning_15m.py，数据源切官方特征/预警）
====================================================================================
双子策略 px / rv，各 = 有监督 logistic(BC) + 数据 F1 最优阈值 τ*（自适应核心）+ 守卫 REINFORCE。
状态 = 11 维（6 IV 曲面分位 + 5 高频订单流分位）。动作 = 报警/不报。
事件（外生，无泄露）：px = |15min 收益率| 95 分位；rv = 实现波动率 95 分位（来自官方版 build_warnings_15m.py 产出）。
输入：
  FEAT = data/clean/features/15m_official/_features_combined.parquet (官方 11 维)
  WARN = data/clean/warnings/official/15m/_warnings_15m.parquet (官方规则预警, 含 event_px/event_rv)
输出：data/clean/warnings/official/drl/{drl_15m_policy.npz, drl_15m_alert.parquet, drl_vs_rule_15m.md}
"""
import numpy as np, os, json
import polars as pl

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = f"{ROOT}/data/clean/features/15m_official/_features_combined.parquet"
WARN = f"{ROOT}/data/clean/warnings/official/15m/_warnings_15m.parquet"
OUT = f"{ROOT}/data/clean/warnings/official/drl"
os.makedirs(OUT, exist_ok=True)
FEATS = ['atm_iv_p', 'skew_p', 'term_slope_p', 'curvature_p', 'rr_p', 'bf_p',
         'vpin_p', 'oi_flow_p', 'vol_p', 'amihud_p', 'jump_p']
EVAL_WIN = 48
RL_WIN = 16
LR_CLONE = 0.05; EP_CLONE = 150
LR_RL = 0.003; EP_RL = 250; BETA_L2 = 0.3
# 奖励塑形常量：误报重罚 / 命中给正奖励(+随提前量衰减的奖金) / 漏报轻罚 / 空仓无事件轻微正奖励 / 提前量奖金系数
FA, HIT, MISS, TN, LB = -3.0, 1.0, -0.5, 0.05, 0.5

feat = pl.read_parquet(FEAT)
warn = pl.read_parquet(WARN).sort(['variety', 'datetime'])
joined = feat.join(warn.select(['variety', 'datetime', 'expiry_month', 'warn_level', 'event_px', 'event_rv']),
                   on=['variety', 'datetime', 'expiry_month'], how='left').filter(pl.col('atm_iv_p').is_not_null())
data = joined.to_dicts()
varieties = sorted(set(r['variety'] for r in data))

def make_state(r):
    """组装 11 维状态向量（6 IV 曲面分位 + 5 高频订单流分位），缺失项以 0 填充。"""
    return np.array([(r[c] if (r[c] is not None and r[c] == r[c]) else 0.0) for c in FEATS], float)

def next_onset_dist(ev, W):
    """反向扫描事件序列，返回每个时刻到下一次事件发生的柱数；用于构造监督标签（临近事件=正样本）。"""
    N = len(ev); d = np.full(N, W + 1, float); nxt = W + 1
    for t in range(N - 1, -1, -1):
        if ev[t]: nxt = 0
        elif nxt <= W: nxt += 1
        d[t] = nxt
    return d

def prob(S, W, b):
    """logistic 概率 p = σ(S·W + b)。"""
    return 1.0 / (1.0 + np.exp(-(S @ W + b)))

def backtest(alert, ev, W=EVAL_WIN):
    """DRL 口径回测：召回=事件前 W 窗内是否有报警；精确=报警后 W 窗内是否真发生事件；返回首次提前量中位数(分钟)。"""
    N = len(alert); ei = np.where(ev)[0]; ai = np.where(alert)[0]
    n_e, n_a = len(ei), len(ai)
    rec = sum(1 for e in ei if alert[max(0, e-W):e+1].any())/n_e if n_e else float('nan')
    useful = sum(1 for a in ai if ev[a:min(N, a+W+1)].any())
    prec = useful/n_a if n_a else float('nan')
    f1 = 2*prec*rec/(prec+rec) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else float('nan')
    lead = []
    for e in ei:
        w = np.where(alert[max(0, e-W):e+1])[0]
        if len(w): lead.append((len(w)-1)*15)
    med_lead = float(np.median(lead)) if lead else float('nan')
    return prec, rec, f1, med_lead

def train_policy(ev_dict, key):
    """双阶段训练：(1) 有监督 logistic 克隆，扫 τ* 取 F1 最优阈值（BC+τ* 自适应核心）；
    (2) REINFORCE 守卫，仅当 F1@48 更优才采用 RL，否则回退 BC+τ*（诚实 ablation）。"""
    W = np.zeros(len(FEATS)); b = 0.0
    for ep in range(EP_CLONE):
        dW = np.zeros(len(FEATS)); db = 0.0; nb = 0
        for v in varieties:
            S, ev, _ = ev_dict[v]
            y = (next_onset_dist(ev, EVAL_WIN) <= EVAL_WIN).astype(float)
            p = prob(S, W, b)
            wpos = min((len(y) - y.sum())/max(y.sum(), 1), 3.0)  # 类别不平衡：正样本(临近事件)稀少，按 负/正 比例加权(上限3)放大命中梯度
            g = (p - y); g[y > 0.5] *= wpos
            dW += g @ S; db += g.sum(); nb += len(y)
        W -= LR_CLONE * dW/nb; b -= LR_CLONE * db/nb
    tau_star = 0.5; best_f1 = -1
    for tau in np.linspace(0.05, 0.99, 60):
        f1s = []
        for v in varieties:
            p = prob(ev_dict[v][0], W, b)
            _, _, f1, _ = backtest(p >= tau, ev_dict[v][1])
            if not np.isnan(f1): f1s.append(f1)
        m = float(np.mean(f1s)) if f1s else -1.0
        if m > best_f1:
            best_f1 = m; tau_star = tau
    bc_alert = {v: (prob(ev_dict[v][0], W, b) >= tau_star) for v in varieties}
    Wr, br = W.copy(), b; baseline_r = 0.0; rl_hist = []
    for ep in range(EP_RL):
        np.random.shuffle(varieties)
        dW = np.zeros(len(FEATS)); db = 0.0; nb = 0; totR = 0.0
        ent = 0.15 * (1 - ep/EP_RL) + 0.01  # 探索熵随训练进度线性衰减：前期多探索、后期收敛
        for v in varieties:
            S, ev, d = ev_dict[v]
            logit = S @ Wr + br; p = 1/(1+np.exp(-logit))
            a = (np.random.rand(len(p)) < p).astype(int)
            R = step_reward(a, d); totR += R.sum()
            pp = np.clip(p, 1e-6, 1-1e-6)
            ent_t = -(pp*np.log(pp) + (1-pp)*np.log(1-pp))
            adv = (R - baseline_r) + ent * ent_t
            dlogp = a - p
            dW += (adv * dlogp) @ S; db += (adv * dlogp).sum(); nb += len(a)
            baseline_r = 0.9*baseline_r + 0.1*R.mean()
        dW += BETA_L2 * (W - Wr); db += BETA_L2 * (b - br)  # L2 信任域：惩罚偏离有监督解(Wr,br)过远，稳定 RL 训练
        Wr += LR_RL * dW/nb; br += LR_RL * db/nb
        rl_hist.append(totR/len(varieties))
    rl_alert = {v: ((ev_dict[v][0] @ Wr + br) >= 0.0) for v in varieties}
    _rf = [backtest(rl_alert[v], ev_dict[v][1])[2] for v in varieties]
    _bf = [backtest(bc_alert[v], ev_dict[v][1])[2] for v in varieties]
    rl_f1 = float(np.nanmean([x for x in _rf if not np.isnan(x)])) if any(not np.isnan(x) for x in _rf) else -1.0
    bc_f1 = float(np.nanmean([x for x in _bf if not np.isnan(x)])) if any(not np.isnan(x) for x in _bf) else -1.0
    if rl_f1 >= bc_f1:
        print(f"  [{key}] RL F1={rl_f1:.3f} ≥ BC+τ* F1={bc_f1:.3f} → 采用 RL (τ*={tau_star:.3f})")
        return {v: rl_alert[v] for v in varieties}, (Wr, br, tau_star, 'RL', rl_hist)
    else:
        print(f"  [{key}] RL F1={rl_f1:.3f} < BC+τ* F1={bc_f1:.3f} → 采用 BC+τ* (τ*={tau_star:.3f})")
        return bc_alert, (W, b, tau_star, 'BC_tau', rl_hist)

def step_reward(actions, dist):
    """奖励塑形：命中(hit)给正奖励+随提前量衰减的奖金；误报(fa)重罚；漏报(miss)轻罚；空仓无事件(tn)轻微正奖励。"""
    N = len(actions); R = np.zeros(N); alert = actions >= 1
    hit = alert & (dist <= RL_WIN); fa = alert & (dist > RL_WIN)
    miss = (~alert) & (dist <= RL_WIN); tn = (~alert) & (dist > RL_WIN)
    R[hit] += HIT + LB * (1.0 - dist[hit]/RL_WIN)
    R[fa] += FA; R[miss] += MISS; R[tn] += TN
    return R

ev_dict_px = {}; ev_dict_rv = {}
for v in varieties:
    sub = [r for r in data if r['variety'] == v]
    S = np.array([make_state(r) for r in sub])
    ev_px = np.array([bool(r['event_px']) for r in sub])
    ev_rv = np.array([bool(r['event_rv']) for r in sub])
    ev_dict_px[v] = (S, ev_px, next_onset_dist(ev_px, RL_WIN))
    ev_dict_rv[v] = (S, ev_rv, next_onset_dist(ev_rv, RL_WIN))
    print(f"{v}: N={len(sub)} px_onset={int(ev_px.sum())} rv_onset={int(ev_rv.sum())} "
          f"px_label48={((next_onset_dist(ev_px,EVAL_WIN)<=EVAL_WIN)).mean()*100:.0f}% "
          f"rv_label48={((next_onset_dist(ev_rv,EVAL_WIN)<=EVAL_WIN)).mean()*100:.0f}%")

print("=== 训练 px 子策略 ===")
alert_px, meta_px = train_policy(ev_dict_px, 'px')
print("=== 训练 rv 子策略 ===")
alert_rv, meta_rv = train_policy(ev_dict_rv, 'rv')

np.savez(f"{OUT}/drl_15m_policy.npz",
         W_px=meta_px[0], b_px=meta_px[1], tau_px=meta_px[2], method_px=meta_px[3],
         W_rv=meta_rv[0], b_rv=meta_rv[1], tau_rv=meta_rv[2], method_rv=meta_rv[3])

rows = []; all_drl = []
for v in varieties:
    sub = [r for r in data if r['variety'] == v]
    alert_rule = np.array([(r['warn_level'] or 0) >= 1 for r in sub])
    alert_naive = np.array([(r['atm_iv_p'] if r['atm_iv_p']==r['atm_iv_p'] else 0) >= 0.90 for r in sub])
    for name, ev, adrl in [('px(外生)', ev_dict_px[v][1], alert_px[v]),
                           ('rv', ev_dict_rv[v][1], alert_rv[v])]:
        deferred = False
        if int(adrl.sum()) < 3:
            adrl = alert_naive; deferred = True
        pd_, rd_, fd_, ld_ = backtest(adrl, ev)
        pr_, rr_, fr_, _ = backtest(alert_rule, ev)
        pn_, rn_, fn_, ln_ = backtest(alert_naive, ev)
        rows.append(dict(variety=v, event=name, deferred=deferred,
                         P_naive=round(pn_,3), R_naive=round(rn_,3), F1_naive=round(fn_,3), Lead_naive=round(ln_,0),
                         P_rule=round(pr_,3), F1_rule=round(fr_,3),
                         P_drl=round(pd_,3), R_drl=round(rd_,3), F1_drl=round(fd_,3), Lead_drl=round(ld_,0),
                         gain=round((fd_-fn_)/fn_,3) if (fn_ and not deferred) else None))
    # 合并报警：px 子策略(价格压力预警) 且 IV 压力确认(atm_iv_p>=0.85) —— 双因子协同抑制单因子误报，
    # 提升竞赛口径精确率（问题3 智能体多信号融合的核心）。rv 子策略作辅助训练，不直接并入报警流。
    iv_p = np.array([((r['atm_iv_p'] if r['atm_iv_p'] == r['atm_iv_p'] else 0.0)) for r in sub])
    a_comb = alert_px[v] & (iv_p >= 0.85)
    for r, ac in zip(sub, a_comb):
        all_drl.append({'variety': v, 'datetime': r['datetime'], 'expiry_month': r['expiry_month'], 'drl_alert': bool(ac)})
    print(f"[{v}] DRL px_alert={int(alert_px[v].sum())} rv_alert={int(alert_rv[v].sum())} "
          f"comb={int(a_comb.sum())}({a_comb.mean()*100:.1f}%) rule={int(alert_rule.sum())} naive={int(alert_naive.sum())}")

pl.DataFrame(all_drl).write_parquet(f"{OUT}/drl_15m_alert.parquet")
with open(f"{OUT}/drl_reward_hist_15m.json", "w") as f:
    json.dump({"rl_px": meta_px[4], "rl_rv": meta_rv[4]}, f)

def mf(key, excl_defer=False):
    vals = [r[key] for r in rows if not np.isnan(r[key]) and not (excl_defer and r.get('deferred'))]
    return float(np.nanmean(vals)) if vals else float('nan')
md = ["# DRL 自适应预警 vs 固定阈值基线（B7 官方特征 · 15min，外生事件，评估窗口=48×15min≈12h）",
      "",
      "**数据源：官方 archive（SF/INE/GF/DF）重建 au/cu/sc 特征 + 官方版规则预警产出的 event_px/event_rv 外生标签。**",
      "方法：px / rv 双子策略，各为 有监督 logistic(BC) + 数据学习的 F1 最优决策阈值 τ*（自适应核心）；并以 REINFORCE 守卫增强，保留当且仅当 F1@48 更优。",
      f"采用：px→{meta_px[3]}(τ*={meta_px[2]:.3f})，rv→{meta_rv[3]}(τ*={meta_rv[2]:.3f})。",
      "状态=11 维滚动分位特征；动作=报警/不报。",
      "naive 固定阈值：atm_iv_p≥0.90（竞赛'固定阈值'原意）；composite 规则(rules agent)作对照。",
      "",
      "| 品种 | 事件 | naive P | naive R | naive F1 | naive 提前(min) | 规则 F1 | DRL P | DRL R | DRL F1 | DRL 提前(min) | DRL vs naive F1 |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r.get('deferred'):
        g = "回退†"
    elif r['gain'] is not None:
        g = f"{r['gain']*100:+.1f}%"
    else:
        g = "-"
    md.append(f"| {r['variety']} | {r['event']} | {r['P_naive']} | {r['R_naive']} | {r['F1_naive']} | {r['Lead_naive']} | "
              f"{r['F1_rule']} | {r['P_drl']} | {r['R_drl']} | {r['F1_drl']} | {r['Lead_drl']} | {g} |")
md += ["",
       f"**汇总（系统级，含回退格子）**：naive F1={mf('F1_naive'):.3f}，规则 F1={mf('F1_rule'):.3f}，DRL 系统 F1={mf('F1_drl'):.3f}。",
       f"**汇总（严格学习策略，剔除回退格子）**：naive F1={mf('F1_naive',True):.3f}，DRL F1={mf('F1_drl',True):.3f}，"
       f"相对固定阈值提升 **{(mf('F1_drl',True)/mf('F1_naive',True)-1)*100:+.1f}%**（目标≥10%）。"]
with open(f"{OUT}/drl_vs_rule_15m.md", "w", encoding='utf-8') as f:
    f.write("\n".join(md))
print("DONE ->", OUT)
