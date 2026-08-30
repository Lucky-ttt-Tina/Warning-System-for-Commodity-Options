"""
任务 A1/A2：用官方数据重建 au/cu/sc 的 15 分钟频特征（方案 A：官方 IV 为骨干）
- 6 维 IV 曲面特征（atm_iv/skew/rr/bf/curvature/term_slope）直接用官方预计算 iv；
  F（平值标的价格代理）= moneyness_type=='ATM' 的行权价中位数；m=strike/F，与原口径一致。
- 5 维订单流特征用官方 volume/open_interest 重建：
    vol_p      : 全合约 15min 总成交量 滚动分位（活跃度异常）
    oi_flow_p  : 全合约 总持仓 ΔOI 滚动分位（库存流）
    amihud_p   : |标的收益|/成交量（非流动性），标的收益用平价关系在参考行权价重建 F_t=K0+(C-P)
    jump_p     : 跳跃方差（RV-BV）滚动8根
    vpin_p     : 以 ΔOI 方向作符号代理的成交量失衡
- 官方 volume/OI 有 UInt8(≤255) 精度截断：低流动合约低估，已 cast Float64，并在文档注明。
- IV 离群清洗：[0.01, 2.5]（官方实测最高 488%）。
- 输出：data/clean/features/15m_official/{v}_features.parquet + _features_combined.parquet
       列名与 _15m_features_full 对齐，便于预警/回测脚本直接切换数据源。

【可复现性说明】
- roll_pct 已改为 O(n log n) 向量化（Fenwick 树），并只在「每交易日一行」的近月代表序列上计算，
  避免对 (合约×日期) 数十万行跑 O(n²) 双重循环导致永远跑不完（旧版缺陷）。
- 所有路径在本文件顶部硬编码（ROOT/ARCH/OUT），与自建管线 data/clean/features/ 完全隔离。
"""
import os, glob, numpy as np
import polars as pl

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH = (f"{ROOT}/官方data/11-基于期权波动率曲面与智能体的量化风控预警系统/"
        f"11-基于期权波动率曲面与智能体的量化风控预警系统/"
        f"11-基于期权波动率曲面与智能体的量化风控预警系统-原始数据/"
        f"11-基于期权波动率曲面与智能体的量化风控预警系统/期权课题数据集/期权课题/archive")
OUT = f"{ROOT}/data/clean/features/15m_official"
os.makedirs(OUT, exist_ok=True)

EX_MAP = {'au': 'SF', 'cu': 'SF', 'sc': 'INE', 'm': 'DF', 'c': 'DF', 'p': 'DF'}
VARIETIES = ['au', 'cu', 'sc', 'm', 'c', 'p']

IV_LO, IV_HI = 0.01, 2.5
ROLL_DAYS = 60
MINP = 20


def ninterp(m_arr, iv_arr, m0):
    iv = np.asarray(iv_arr, float); mm = np.asarray(m_arr, float)
    ok = ~np.isnan(iv); mm = mm[ok]; ivv = iv[ok]
    if mm.size == 0:
        return None
    order = np.argsort(mm); mm = mm[order]; ivv = ivv[order]
    if m0 <= mm[0]:
        return float(ivv[0])
    if m0 >= mm[-1]:
        return float(ivv[-1])
    i = int(np.searchsorted(mm, m0)); a, b = mm[i - 1], mm[i]
    if a == b:
        return float(ivv[i - 1])
    w = (m0 - a) / (b - a)
    return float(ivv[i - 1] * (1 - w) + ivv[i] * w)


def to_exp(code):  # au2508 -> 202508
    return "20" + code[2:6]


def roll_pct(times, vals, days=ROLL_DAYS, minp=MINP):
    """向量化滚动分位（Fenwick 树，O(n log n)）：
    对每个 i，返回窗口 [t_i-days, t_i) 内 <= v_i 的样本占比（不含自身）。
    输入 times 为 datetime 数组；输出与输入等长、按原顺序对齐。"""
    t = np.array(times, dtype='datetime64[D]').astype(np.int64)
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return np.full(0, np.nan)
    ord_t = np.argsort(t, kind='mergesort')
    ts = t[ord_t]; vs = v[ord_t]
    # 按值排序得到秩（用于 Fenwick）
    order_v = np.argsort(vs, kind='mergesort')
    ranks = np.empty(n, dtype=np.int64); ranks[order_v] = np.arange(n)
    bit = np.zeros(n + 1, dtype=np.int64)

    def add(idx, delta):
        while idx <= n:
            bit[idx] += delta; idx += idx & (-idx)

    def query(idx):
        s = 0
        while idx > 0:
            s += bit[idx]; idx -= idx & (-idx)
        return s
    out = np.full(n, np.nan)
    L = 0
    for i in range(n):
        while L < i and ts[L] < ts[i] - days:
            add(ranks[L] + 1, -1); L += 1
        if i - 1 >= L:
            add(ranks[i - 1] + 1, +1)
        total = i - L
        if total >= minp:
            out[ord_t[i]] = query(ranks[i] + 1) / total
    return out


def surface_and_of_for_contract(paths, variety, contract):
    """读一个合约的全部月度文件，产出紧凑的 per-(datetime) 行。"""
    frames = []
    for p in paths:
        try:
            d = pl.read_parquet(p, columns=['timestamp', 'option_type', 'strike', 'iv',
                                            'volume', 'open_interest', 'close', 'moneyness_type',
                                            'moneyness', 'dte'])
        except Exception as e:
            continue
        if d.height == 0:
            continue
        d = d.with_columns(
            pl.col('timestamp').cast(pl.Utf8).str.strptime(pl.Datetime, '%Y%m%d%H%M%S', strict=False).alias('dt'),
            pl.col('option_type').str.slice(0, 1).str.to_uppercase().alias('ot'),
            pl.col('volume').cast(pl.Float64, strict=False),
            pl.col('open_interest').cast(pl.Float64, strict=False),
            pl.col('iv').cast(pl.Float64, strict=False),
            pl.col('strike').cast(pl.Float64, strict=False),
            pl.col('close').cast(pl.Float64, strict=False),
            pl.col('dte').cast(pl.Float64, strict=False),
        ).filter(pl.col('dt').is_not_null() & pl.col('iv').is_not_null()
                 & (pl.col('iv') > IV_LO) & (pl.col('iv') < IV_HI))
        frames.append(d)
    if not frames:
        return None
    d = pl.concat(frames, how='vertical_relaxed')
    if d.height == 0:
        return None
    atm = d.filter(pl.col('moneyness_type') == 'ATM')
    if atm.height == 0:
        atm = d.with_columns((pl.col('moneyness').abs()).alias('md')).sort('md')
        if atm.height == 0:
            return None
        k0 = float(atm.select('strike').head(1)['strike'][0])
    else:
        k0 = float(atm['strike'].median())
    rows = []
    for g in d.partition_by('dt', as_dict=False):
        strikes = g['strike'].to_numpy(); iv = g['iv'].to_numpy()
        ot = g['ot'].to_numpy()
        F = k0
        m = strikes / F
        calls = ot == 'C'; puts = ot == 'P'
        atm_iv = ninterp(m, iv, 1.0)
        iv_c105 = ninterp(m[calls], iv[calls], 1.05) if calls.any() else None
        iv_p095 = ninterp(m[puts], iv[puts], 0.95) if puts.any() else None
        iv_k095 = ninterp(m, iv, 0.95); iv_k105 = ninterp(m, iv, 1.05)
        if None in (atm_iv, iv_c105, iv_p095, iv_k095, iv_k105):
            continue
        skew = iv_p095 - iv_c105
        rr = iv_c105 - iv_p095
        bf = (iv_k095 + iv_k105) / 2.0 - atm_iv
        tv = float(g['volume'].sum()) if g['volume'].null_count() < g.height else np.nan
        toi = float(g['open_interest'].sum()) if g['open_interest'].null_count() < g.height else np.nan
        ref = g.filter((pl.col('strike') - k0).abs() <= 1.0)
        c_close = ref.filter(pl.col('ot') == 'C')['close']
        p_close = ref.filter(pl.col('ot') == 'P')['close']
        ft = None
        if len(c_close) and len(p_close):
            ft = k0 + (float(c_close[0]) - float(p_close[0]))
        rows.append({
            'dt': g['dt'][0], 'expiry_month': to_exp(contract), 'contract': contract,
            'atm_iv': atm_iv, 'skew': skew, 'rr': rr, 'bf': bf, 'curvature': bf,
            'T_years': float(g['dte'].median()) / 365.0 if g['dte'].null_count() < g.height else np.nan,
            'total_volume': tv, 'total_oi': toi,
            'Ft': (ft if ft is not None else float('nan')), 'F': k0,
        })
    if not rows:
        return None
    return pl.DataFrame(rows, infer_schema_length=None)


combined = []
for v in VARIETIES:
    ex = EX_MAP[v]
    base = f"{ARCH}/{ex}"
    contracts = sorted([d for d in os.listdir(base) if d.startswith(v)])
    print(f"\n=== {v} ({ex}) 合约数={len(contracts)} ===")
    surf_all = []
    for c in contracts:
        paths = glob.glob(f"{base}/{c}/options/**/options_*.parquet", recursive=True)
        if not paths:
            continue
        r = surface_and_of_for_contract(paths, v, c)
        if r is not None and r.height:
            surf_all.append(r)
    if not surf_all:
        print(f"  {v}: 无有效数据，跳过"); continue
    surf = pl.concat(surf_all, how='vertical_relaxed').sort('dt')
    # term_slope: 每 datetime 近月(atm, 最短T) - 远月(最长T)
    surf = surf.with_columns(
        front=pl.col('atm_iv').sort_by('T_years').first().over('dt'),
        far=pl.col('atm_iv').sort_by('T_years', descending=True).first().over('dt'),
    )
    surf = surf.with_columns((pl.col('front') - pl.col('far')).alias('term_slope')).drop(['front', 'far'])
    # 订单流聚合到 datetime（全合约汇总）
    of = (surf.group_by('dt').agg(
        pl.col('total_volume').sum().alias('vol'),
        pl.col('total_oi').sum().alias('oi'),
        pl.col('Ft').drop_nulls().last().alias('Ft'),
    ).sort('dt'))
    of = of.with_columns(pl.col('Ft').diff().alias('ret')).with_columns(pl.col('ret').fill_null(0.0))
    ret = of['ret'].to_numpy(); vol = of['vol'].to_numpy(); oi = of['oi'].to_numpy()
    absr = np.abs(ret)
    amihud = np.where(vol > 0, absr / np.where(vol <= 0, np.nan, vol), np.nan)
    jump = np.full(len(ret), np.nan)
    for i in range(8, len(ret)):
        rv = np.sum(ret[i - 8:i] ** 2)
        bv = (np.pi / 2) * np.sum(np.abs(ret[i - 8:i]) * np.abs(np.roll(ret[i - 8:i], 1)))
        jump[i] = rv - bv
    oi_diff = np.diff(np.concatenate([[0], oi]))  # 长度= N，行0=首期OI相对0的变化
    vpin = np.where(vol > 0, np.abs(oi_diff) / np.where(vol <= 0, np.nan, vol), np.nan)
    tms = of['dt'].to_numpy()
    for name, arr in [('vol_p', vol), ('oi_flow_p', oi_diff),
                      ('amihud_p', amihud), ('jump_p', jump), ('vpin_p', vpin)]:
        of = of.with_columns(pl.Series(name, roll_pct(tms, arr)))

    # ---- 取每交易日近月(最短T)代表行，仅在此 per-dt 序列上算 IV 曲面滚动分位 ----
    surf_rep = surf.sort('T_years').group_by('dt').first().sort('dt')
    times = surf_rep['dt'].to_numpy()
    for col in ['atm_iv', 'skew', 'rr', 'bf', 'curvature', 'term_slope']:
        surf_rep = surf_rep.with_columns(pl.Series(f"{col}_p", roll_pct(times, surf_rep[col].to_numpy())))

    surf_rep = surf_rep.with_columns(pl.lit(v).alias('variety'))
    merged = surf_rep.join(of, on='dt', how='left').with_columns(pl.col('dt').alias('datetime'))
    if 'Ft' in merged.columns:
        merged = merged.with_columns(pl.col('Ft').alias('F')).drop('Ft')
    cols = ['variety', 'datetime', 'expiry_month', 'atm_iv', 'skew', 'rr', 'bf', 'curvature', 'term_slope', 'F',
            'atm_iv_p', 'skew_p', 'rr_p', 'bf_p', 'curvature_p', 'term_slope_p',
            'vpin_p', 'oi_flow_p', 'vol_p', 'amihud_p', 'jump_p']
    merged = merged.select([c for c in cols if c in merged.columns])
    merged.write_parquet(f"{OUT}/{v}_features.parquet")
    combined.append(merged)
    print(f"  {v}: 行={merged.height} | atm_iv中位={merged['atm_iv'].median():.4f} | "
          f"atm_iv_p有效={merged['atm_iv_p'].is_not_null().mean()*100:.1f}% | "
          f"vol_p有效={merged['vol_p'].is_not_null().mean()*100:.1f}% | "
          f"term_slope_p有效={merged['term_slope_p'].is_not_null().mean()*100:.1f}% | "
          f"日期范围 {merged['datetime'].min()} ~ {merged['datetime'].max()}")

if combined:
    comb = pl.concat(combined, how='vertical_relaxed')
    comb.write_parquet(f"{OUT}/_features_combined.parquet")
    print(f"\nDONE -> {OUT} | combined 行={comb.height}")
else:
    print("\n无数据产出")
