# -*- coding: utf-8 -*-
"""
D14 · Greeks 截面特征提取（官方 archive，补齐题目问题1 点名的 Greeks 维度）
================================================================
背景：官方 parquet 实际含 delta/gamma/theta/vega 列（24 列），A1 特征工程只用了
其中 10 列。本脚本补齐题目点名的 Greeks 风险维度：

  1) 近月 ATM 代表 Greeks 时序（delta/gamma/vega/theta）→ 看板"Greeks 曲线"
  2) Gamma/Vega 截面集中度（HHI，近月合约全部行权价截面）→ 题目原文风险指标
     "Gamma/Vega 截面集中度"；HHI = Σ(x_i)²/(Σx_i)²，∈(0,1]，越接近 1 表示
     风险敞口越集中于个别行权价（做市商对冲压力集中）。
  3) 各品种最新交易日 IV 曲面快照（合约×moneyness 网格）→ 看板"曲面热力图"

口径说明：
- 近月合约 = 该时点 dte 中位数最小的到期月合约（与 A1 term_slope 口径一致）。
- ATM 代表 = moneyness_type=='ATM' 行的均值；若无 ATM 行则取 |moneyness| 最小的行。
- 清洗与 A1 完全一致：dt 解析失败剔除、IV∈(0.01, 2.5)、数值列 cast Float64。
- 与自建程序物理隔离：输出至 data/clean/features/15m_official/。

输出：
  data/clean/features/15m_official/_greeks_combined.parquet  # per (variety, dt)
  data/clean/features/15m_official/_surface_snapshot.json    # 最新日曲面快照
"""
import os, glob, json
import numpy as np
import polars as pl

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH = (f"{ROOT}/官方data/11-基于期权波动率曲面与智能体的量化风控预警系统/"
        f"11-基于期权波动率曲面与智能体的量化风控预警系统/"
        f"11-基于期权波动率曲面与智能体的量化风控预警系统-原始数据/"
        f"11-基于期权波动率曲面与智能体的量化风控预警系统/期权课题数据集/期权课题/archive")
OUT = f"{ROOT}/data/clean/features/15m_official"

EX_MAP = {'au': 'SF', 'cu': 'SF', 'sc': 'INE', 'm': 'DF', 'c': 'DF', 'p': 'DF'}
VARIETIES = ['au', 'cu', 'sc', 'm', 'c', 'p']
IV_LO, IV_HI = 0.01, 2.5
READ_COLS = ['timestamp', 'contract', 'option_type', 'strike', 'iv',
             'delta', 'gamma', 'vega', 'theta', 'moneyness', 'moneyness_type', 'dte']


def load_variety(v):
    """读一个品种全部合约的月度文件 → 清洗 → 单表（含到期月 expiry 提取）。"""
    base = f"{ARCH}/{EX_MAP[v]}"
    contracts = sorted([d for d in os.listdir(base) if d.startswith(v)])
    frames = []
    for c in contracts:
        for p in glob.glob(f"{base}/{c}/options/**/options_*.parquet", recursive=True):
            try:
                d = pl.read_parquet(p, columns=READ_COLS)
            except Exception:
                continue
            if d.height == 0:
                continue
            d = d.with_columns(
                pl.col('timestamp').cast(pl.Utf8).str.strptime(pl.Datetime, '%Y%m%d%H%M%S', strict=False).alias('dt'),
                pl.col('iv').cast(pl.Float64, strict=False),
                pl.col('delta').cast(pl.Float64, strict=False),
                pl.col('gamma').cast(pl.Float64, strict=False),
                pl.col('vega').cast(pl.Float64, strict=False),
                pl.col('theta').cast(pl.Float64, strict=False),
                pl.col('dte').cast(pl.Float64, strict=False),
                pl.col('moneyness').cast(pl.Float64, strict=False),
                # 官方 contract = 完整期权代码（如 au2406P480.SF）；到期月 = 品种码后 4 位
                pl.col('contract').str.extract(r'^[a-z]+(\d{4})', 1).alias('expiry'),
            ).filter(pl.col('dt').is_not_null() & pl.col('iv').is_not_null()
                     & (pl.col('iv') > IV_LO) & (pl.col('iv') < IV_HI)
                     & pl.col('expiry').is_not_null())
            frames.append(d.select(
                ['dt', 'expiry', 'contract', 'option_type', 'strike', 'iv', 'delta', 'gamma',
                 'vega', 'theta', 'moneyness', 'moneyness_type', 'dte']))
    if not frames:
        return None
    return pl.concat(frames, how='vertical_relaxed')


def greeks_for_variety(d):
    """per (dt)：近月到期月 ATM(call) Greeks + 近月截面 gamma/vega HHI。"""
    # 每 (dt, expiry) 的中位 dte → 每 dt 的近月到期月（T 最小；窗口过滤保证正确性）
    per_dc = d.group_by(['dt', 'expiry']).agg(pl.col('dte').median().alias('T'))
    near_e = (per_dc.with_columns(pl.col('T').min().over('dt').alias('Tmin'))
                    .filter(pl.col('T') == pl.col('Tmin'))
                    .unique(subset='dt', keep='first')
                    .select(['dt', 'expiry']).rename({'expiry': 'near_expiry'}))
    d = d.join(near_e, on='dt')
    d = d.with_columns(
        (pl.col('expiry') == pl.col('near_expiry')).alias('near'),
        (pl.col('moneyness_type') == 'ATM').alias('atm'),
        pl.col('gamma').abs().alias('agamma'),
        pl.col('option_type').str.to_uppercase().str.starts_with('C').alias('is_call'),
    )
    # 近月截面 HHI：HHI = Σ(x_i)²/(Σx_i)²，∈(0,1]，越接近 1 风险越集中于个别行权价
    def hhi(col, out):
        s = pl.col(col).filter(pl.col('near'))
        return (s.pow(2).sum() / (s.sum().pow(2) + 1e-12)).alias(out)

    # ATM 代表 Greeks：只取 call 侧（delta ≈ +0.5 有标准解读；gamma/vega/theta 对称）
    call_atm = pl.col('near') & pl.col('atm') & pl.col('is_call')
    agg = [pl.col('delta').filter(call_atm).mean().alias('delta_atm'),
           pl.col('gamma').filter(call_atm).mean().alias('gamma_atm'),
           pl.col('vega').filter(call_atm).mean().alias('vega_atm'),
           pl.col('theta').filter(call_atm).mean().alias('theta_atm'),
           hhi('agamma', 'gamma_conc'), hhi('vega', 'vega_conc'),
           pl.col('dte').filter(pl.col('near')).median().alias('dte_near'),
           pl.col('near_expiry').first().alias('near_expiry')]
    res = d.group_by('dt').agg(agg).sort('dt')

    # ATM(call) 缺失的 dt：fallback 取近月内 |moneyness| 最小的 call 行（窗口过滤保正确性）
    miss_dt = res.filter(pl.col('delta_atm').is_null())['dt'].to_list()
    if miss_dt:
        fb = (d.filter(pl.col('near') & pl.col('dt').is_in(miss_dt) & pl.col('is_call'))
               .with_columns(pl.col('moneyness').abs().alias('md'))
               .filter(pl.col('md') == pl.col('md').min().over('dt'))
               .unique(subset='dt', keep='first')
               .select(['dt', 'delta', 'gamma', 'vega', 'theta'])
               .rename({'delta': 'delta_atm_fb', 'gamma': 'gamma_atm_fb',
                        'vega': 'vega_atm_fb', 'theta': 'theta_atm_fb'}))
        # 仅用 fb 补 miss_dt 行的 null，不覆盖已有值
        res = res.join(fb, on='dt', how='left')
        for c, fb_c in [('delta_atm','delta_atm_fb'),('gamma_atm','gamma_atm_fb'),
                        ('vega_atm','vega_atm_fb'),('theta_atm','theta_atm_fb')]:
            res = res.with_columns(pl.coalesce([pl.col(c), pl.col(fb_c)]).alias(c)).drop(fb_c)
        res = res.sort('dt')
    return res


def surface_snapshot(d, dt_last):
    """数据最齐全时刻的 IV 曲面：到期月(按 T 升序) × moneyness 桶 → IV 均值。
    优先在末尾 60 天内取行数最多时刻；若该时刻到期月 <2，回退到全期行数最多时刻
    （避免末尾只剩近月合约、曲面退化为单列微笑）。"""
    import datetime as _dt

    def best_dt(frame):
        return frame.group_by('dt').agg(pl.len().alias('n')).sort('n', descending=True)['dt'][0]

    dt_snap = best_dt(d.filter(pl.col('dt') >= dt_last - _dt.timedelta(days=60)))
    snap = d.filter(pl.col('dt') == dt_snap).with_columns(((pl.col('moneyness') * 100).round(0) / 100).alias('mb'))
    if snap['expiry'].n_unique() < 3:
        dt_snap = best_dt(d)
        snap = d.filter(pl.col('dt') == dt_snap).with_columns(((pl.col('moneyness') * 100).round(0) / 100).alias('mb'))
    grid = (snap.group_by(['mb', 'expiry'])
               .agg(pl.col('iv').mean().alias('iv'), pl.col('dte').median().alias('T'))
               .sort(['mb', 'T']))
    expiries = sorted(grid['expiry'].unique().to_list(),
                      key=lambda e: grid.filter(pl.col('expiry') == e)['T'][0])
    mbs = sorted(grid['mb'].unique().to_list())
    ivm = [[None] * len(expiries) for _ in mbs]
    for row in grid.to_dicts():
        i = mbs.index(row['mb']); j = expiries.index(row['expiry'])
        ivm[i][j] = round(row['iv'], 4)
    return dict(date=str(dt_snap), expiries=expiries,
                m=[round(m, 2) for m in mbs], iv=ivm)


def main():
    combined, snapshots = [], {}
    for v in VARIETIES:
        print(f"=== {v} 加载中 ...", flush=True)
        d = load_variety(v)
        if d is None or d.height == 0:
            print(f"  {v}: 无有效数据，跳过"); continue
        g = greeks_for_variety(d)
        g = g.with_columns(pl.lit(v).alias('variety'))
        combined.append(g.select(
            ['variety', 'dt', 'delta_atm', 'gamma_atm', 'vega_atm', 'theta_atm',
             'gamma_conc', 'vega_conc', 'dte_near', 'near_expiry']))
        dt_last = g['dt'].max()
        snapshots[v] = surface_snapshot(d, dt_last)
        cov = g['gamma_atm'].is_not_null().mean() * 100
        print(f"  {v}: 行={g.height} | Greeks覆盖={cov:.1f}% | "
              f"gamma_conc中位={g['gamma_conc'].median():.4f} | 最新日={dt_last}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    if combined:
        comb = pl.concat(combined, how='vertical_relaxed')
        comb = comb.rename({'dt': 'datetime'})
        comb.write_parquet(f"{OUT}/_greeks_combined.parquet")
        print(f"\nDONE -> {OUT}/_greeks_combined.parquet | 行={comb.height}")
    with open(f"{OUT}/_surface_snapshot.json", "w", encoding='utf-8') as f:
        json.dump(snapshots, f, ensure_ascii=False)
    print(f"DONE -> {OUT}/_surface_snapshot.json | 品种={list(snapshots.keys())}")


if __name__ == '__main__':
    main()
