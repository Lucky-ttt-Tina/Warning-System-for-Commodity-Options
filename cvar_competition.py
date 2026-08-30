# -*- coding: utf-8 -*-
"""
B9 竞赛口径 · CVaR(95%) 改善率（严格按赛题原文，多空双向）
================================================================
赛题原文口径（任务目标第2条，核心业务指标）：
- 测试组合：期货多头 / 期货空头 两种初始持仓（各 1 手），分别测试。
- 风控规则：系统发出预警（等级 >= 2）时，在预警时间点的【下一个交易日开盘】
  将当前持仓【减半】（多头平掉一半，空头回补一半）。
- 减仓后持有至测试期结束，不再恢复（简化）。若多次预警，仅在【首次】预警时减仓。
- 无预警策略：始终保持原头寸不变。
- 评价指标：测试期内预警策略 vs 无预警策略的日收益率序列 CVaR(95%)；
  多头改善率与空头改善率取【算术平均】作为最终改善率，>10% 视为显著有效。
- 若测试期内未出现任何预警，则改善率为 0。

信号系统（两套并行报告）：
- rule : 规则引擎，(variety, date) 内任一 15min bar warn_level >= 2
- drl  : DRL 双确认（BC+tau*），(variety, date) 内任一 bar drl_alert = True

测试期敏感性：题目仅要求 >= 3 个月连续数据；除全期主口径外，
另报 2024-01 起 / 2025-01 起两个子区间，观察"首次预警位置"对改善率的影响。

标的代理（与 cvar_official.py 一致，诚实口径）：
- F = ATM 行权价中位数（官方 moneyness_type=='ATM'），
  按 (variety, datetime) 跨合约取均值以平滑合约切换噪声；
- 日收益 = 当日末 F / 当日首 F - 1；|ret|>1 截断（防脏数据）。
"""
import numpy as np
import json, os
import polars as pl
from datetime import date as _date

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = f"{ROOT}/data/clean/features/15m_official/_features_combined.parquet"
WARN = f"{ROOT}/data/clean/warnings/official/15m/_warnings_15m.parquet"
DRL  = f"{ROOT}/data/clean/warnings/official/drl/drl_15m_alert.parquet"
OUT  = f"{ROOT}/data/clean/official/cvar"
os.makedirs(OUT, exist_ok=True)
VARS = ['au', 'cu', 'sc', 'm', 'c', 'p']

# ---------- 1. 日度收益序列（标的代理 F） ----------
feat = pl.read_parquet(FEAT).filter(pl.col('atm_iv_p').is_not_null())
fday = (feat.group_by(['variety', 'datetime']).agg(pl.col('F').mean().alias('F'))
            .sort(['variety', 'datetime']))
day = (fday.group_by(['variety', pl.col('datetime').dt.date().alias('date')])
          .agg(pl.col('F').first().alias('Fopen'), pl.col('F').last().alias('Fclose'))
          .with_columns(((pl.col('Fclose') / pl.col('Fopen') - 1).clip(-1.0, 1.0)).alias('ret'))
          .sort(['variety', 'date']))
day = day.filter(pl.col('ret').is_finite())

# ---------- 2. 预警信号 → 首次触发日 ----------
w = pl.read_parquet(WARN).filter(pl.col('atm_iv_p').is_not_null())
alert_rule = set(w.filter(pl.col('warn_level') >= 2)
                 .select(['variety', pl.col('datetime').dt.date().alias('date')])
                 .unique().iter_rows())
d = pl.read_parquet(DRL)
alert_drl = set(d.filter(pl.col('drl_alert'))
                .select(['variety', pl.col('datetime').dt.date().alias('date')])
                .unique().iter_rows())

def first_alert(variety, alert_set, start=None):
    """首次预警日：限定在测试期内（start 之后），模拟'从起点开始运行系统'。"""
    sd = _date.fromisoformat(start) if start else None
    ds = sorted(dt for (v, dt) in alert_set if v == variety and (sd is None or dt >= sd))
    return ds[0] if ds else None

# ---------- 3. CVaR ----------
def cvar(losses, alpha=0.95):
    """条件在险价值：取损失序列 95% 分位以上的尾部均值，衡量极端尾部风险（数值越小=尾部越安全）。"""
    x = np.sort(np.asarray(losses, float))
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float('nan')
    var = np.quantile(x, alpha)  # 95% 分位即 VaR 临界点；其上方尾部均值即为 CVaR(95%)
    tail = x[x >= var]
    return float(tail.mean()) if len(tail) else float('nan')

# ---------- 4. 单品种模拟（题目口径） ----------
def simulate(variety, alert_set, start=None):
    """返回 dict：多头/空头 base 与 sys 的 CVaR 及改善率。
    策略：首次预警日的【下一交易日】起持仓减半（日收益 x0.5），持有到底。"""
    sub = (day.filter((pl.col('variety') == variety) &
                      ((pl.col('date') >= _date.fromisoformat(start)) if start else True))
              .sort('date'))
    dates = sub['date'].to_list()
    rets = np.asarray(sub['ret'].to_list(), float)
    if len(dates) == 0:
        return None
    fa = first_alert(variety, alert_set, start)
    # 无触发 → 改善率 0（题目口径）
    if fa is None or fa > dates[-1]:
        base_long = cvar(-rets); base_short = cvar(rets)
        return dict(variety=variety, n_days=len(dates), first_alert=None,
                    long_base=round(base_long, 5), long_sys=round(base_long, 5),
                    short_base=round(base_short, 5), short_sys=round(base_short, 5),
                    long_imp=0.0, short_imp=0.0, avg_imp=0.0)
    # 首次预警日的下一个交易日起减半
    idx = next((i for i, dt in enumerate(dates) if dt > fa), None)
    w_mult = np.ones(len(rets))
    if idx is not None:
        w_mult[idx:] = 0.5
    rets_sys = rets * w_mult
    bl, sl = cvar(-rets), cvar(rets)
    bs, ss = cvar(-rets_sys), cvar(rets_sys)
    li = (bl - bs) / bl * 100 if (bl == bl and bl != 0) else float('nan')
    si = (sl - ss) / sl * 100 if (sl == sl and sl != 0) else float('nan')
    ai = (li + si) / 2 if (li == li and si == si) else float('nan')
    return dict(variety=variety, n_days=len(dates), first_alert=str(fa),
                long_base=round(bl, 5), long_sys=round(bs, 5),
                short_base=round(sl, 5), short_sys=round(ss, 5),
                long_imp=round(li, 2), short_imp=round(si, 2), avg_imp=round(ai, 2))

# ---------- 5. 池化（全品种合并） ----------
def simulate_pooled(alert_set, start=None):
    """池化口径：6 品种日收益序列合并后统一计算 CVaR，反映系统级尾部改善（而非单品种简单平均）。"""
    base_l, sys_l, base_s, sys_s = [], [], [], []
    for v in VARS:
        sub = (day.filter((pl.col('variety') == v) &
                          ((pl.col('date') >= _date.fromisoformat(start)) if start else True))
                  .sort('date'))
        dates = sub['date'].to_list()
        rets = np.asarray(sub['ret'].to_list(), float)
        if not len(dates):
            continue
        fa = first_alert(v, alert_set, start)
        w_mult = np.ones(len(rets))
        if fa is not None and fa <= dates[-1]:
            idx = next((i for i, dt in enumerate(dates) if dt > fa), None)
            if idx is not None:
                w_mult[idx:] = 0.5
        rets_sys = rets * w_mult
        base_l += (-rets).tolist();  sys_l += (-rets_sys).tolist()
        base_s += rets.tolist();     sys_s += rets_sys.tolist()
    bl, sl = cvar(base_l), cvar(base_s)
    bs, ss = cvar(sys_l), cvar(sys_s)
    li = (bl - bs) / bl * 100 if (bl == bl and bl != 0) else float('nan')
    si = (sl - ss) / sl * 100 if (sl == sl and sl != 0) else float('nan')
    return dict(long_base=round(bl, 5), long_sys=round(bs, 5),
                short_base=round(sl, 5), short_sys=round(ss, 5),
                long_imp=round(li, 2), short_imp=round(si, 2),
                avg_imp=round((li + si) / 2, 2))

# ---------- 6. 主流程：两套信号 × 三个测试期 ----------
PERIODS = [('全期', None), ('2024起', '2024-01-01'), ('2025起', '2025-01-01'), ('2026起', '2026-01-01')]
results = {}
for tag, aset in [('rule', alert_rule), ('drl', alert_drl)]:
    results[tag] = {}
    for pname, start in PERIODS:
        per = [simulate(v, aset, start) for v in VARS]
        per = [r for r in per if r is not None]
        pooled = simulate_pooled(aset, start)
        results[tag][pname] = dict(per_variety=per, pooled=pooled)

# ---------- 7. 报告 ----------
md = ["# CVaR(95%) 改善率 · 竞赛口径（多空双向，题目原文规则）",
      "",
      "**规则**：预警等级≥2 触发；触发后【下一交易日开盘】持仓减半（多头平一半/空头回补一半），",
      "持有至测试期结束不再恢复；仅首次预警减仓；多头改善率与空头改善率取算术平均。",
      "信号：rule = 规则引擎 L≥2；drl = DRL 双确认（BC+τ*）。CVaR(95%) 基于日收益率序列。",
      ""]

for tag in ('rule', 'drl'):
    md += [f"## 信号系统：{tag}", ""]
    for pname, _ in PERIODS:
        pv = results[tag][pname]['per_variety']
        po = results[tag][pname]['pooled']
        md += [f"### 测试期：{pname}",
               "",
               "| 品种 | 交易日 | 首次预警日 | 多头CVaR base→sys | 空头CVaR base→sys | 多头改善% | 空头改善% | 平均改善% |",
               "|---|---|---|---|---|---|---|---|"]
        for r in pv:
            md.append(f"| {r['variety']} | {r['n_days']} | {r['first_alert'] or '—'} | "
                      f"{r['long_base']}→{r['long_sys']} | {r['short_base']}→{r['short_sys']} | "
                      f"{r['long_imp']} | {r['short_imp']} | **{r['avg_imp']}** |")
        md += ["",
               f"**池化（6品种合并）**：多头 {po['long_imp']}% / 空头 {po['short_imp']}% / "
               f"**平均改善率 {po['avg_imp']}%** → {'达标(>10%)' if (po['avg_imp'] == po['avg_imp'] and po['avg_imp'] > 10) else '未达 10%'}",
               ""]
    md.append("")

with open(f"{OUT}/cvar_competition.md", "w", encoding='utf-8') as f:
    f.write('\n'.join(md))
with open(f"{OUT}/cvar_competition.json", "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ---------- 8. 终端摘要 ----------
print("=" * 70)
print("CVaR(95%) 竞赛口径（多空双向 · 题目原文规则）")
print("=" * 70)
for tag in ('rule', 'drl'):
    for pname, _ in PERIODS:
        po = results[tag][pname]['pooled']
        print(f"[{tag:4s}] {pname:5s} 池化: 多头{po['long_imp']:>7}%  空头{po['short_imp']:>7}%  "
              f"平均 {po['avg_imp']:>7}%  {'达标' if (po['avg_imp']==po['avg_imp'] and po['avg_imp']>10) else '未达'}")
    print("-" * 70)
print("\n分品种（全期 · rule）:")
for r in results['rule']['全期']['per_variety']:
    print(f"  {r['variety']}: 首次预警 {r['first_alert']}  多头{r['long_imp']}%  空头{r['short_imp']}%  平均{r['avg_imp']}%")
print("\nDONE ->", OUT)
