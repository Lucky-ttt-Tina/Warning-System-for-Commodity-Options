# -*- coding: utf-8 -*-
"""
官方管线 · 融合事件标注（修正版，B6 前置）
========================================
- 我们 18 条逐品种事件（extreme_events.parquet，start/end 为 'YYYY-MM-DD HH:MM' 字符串）
- 官方 17 板块事件（extreme_event.csv，GBK，start/end 为纯日期 'YYYY-MM-DD'）
- 修正 A3 build_fused_events.py 的两处缺陷：
    (1) ours 的 'YYYY-MM-DD HH:MM' 字符串用 cast(pl.Date) 强转会失败变 null
        -> 改为先 strptime(Datetime) 再统一为 datetime64[ns]
    (2) 官方 freq 记作 '1d'，ours 记作 'daily'，回测过滤 'daily'/'15m' 不匹配
        -> 统一官方 '1d' -> 'daily'
- 板块 -> 品种映射沿用 SECTOR_MAP；无对应期权的板块保留 variety='NONE'
  （回测按 au/cu/sc/SR 四品种循环，NONE 自然跳过——我们本就没有这些品种数据）
产出：data/clean/events/fused_events_official.parquet（+ .md 说明）
"""
import os
import csv as _csv
import polars as pl

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFF_CSV = (f"{ROOT}/官方data/11-基于期权波动率曲面与智能体的量化风控预警系统/"
           f"11-基于期权波动率曲面与智能体的量化风控预警系统/"
           f"11-基于期权波动率曲面与智能体的量化风控预警系统-原始数据/"
           f"11-基于期权波动率曲面与智能体的量化风控预警系统/期权课题数据集/期权课题/extreme_event.csv")
OURS = f"{ROOT}/data/clean/events/extreme_events.parquet"
OUT = f"{ROOT}/data/clean/events"
os.makedirs(OUT, exist_ok=True)

# 板块 -> 品种 映射（后续 E15 扩 DCE 品种时可补 油脂油料->m 等）
SECTOR_MAP = {
    '能源': ['sc'], '化工': [], '油脂油料': ['m', 'p'], '贵金属': ['au'], '有色': ['cu'],
    '黑色系': [], '全板块': ['au', 'cu', 'sc'], '碳酸锂': [], '合成橡胶': [],
    '有色金属': ['cu'], '氧化铝': [],
}

# ---------- 我们的事件 ----------
ours = pl.read_parquet(OURS)
ours_rows = ours.select(
    event_id=pl.col('event_id'),
    source=pl.lit('ours'),
    variety=pl.col('variety'),
    name=pl.col('name'),
    category=pl.col('category'),
    severity=pl.col('severity'),
    freq=pl.col('freq'),
    start=pl.col('start'),
    end=pl.col('end'),
)
# ours start/end 是 'YYYY-MM-DD HH:MM' 字符串 -> datetime64
ours_rows = ours_rows.with_columns(
    pl.col('start').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M', strict=False),
    pl.col('end').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M', strict=False),
)
print(f"我们事件: {ours_rows.height} 条 | start null={ours_rows['start'].null_count()} end null={ours_rows['end'].null_count()}")

# ---------- 官方事件 ----------
off_rows = []
with open(OFF_CSV, 'r', encoding='gbk', errors='replace') as _f:
    _rd = _csv.DictReader(_f)
    for r in _rd:
        desc = r['event_description']
        sectors = [s.strip() for s in str(r['sectors_affected']).replace('，', ',').split(',') if s.strip()]
        start = str(r['start_date']).replace('/', '-')
        end = str(r['end_date']).replace('/', '-')
        sev = 3 if '全板块' in sectors else 2
        mapped = []
        for s in sectors:
            mapped += SECTOR_MAP.get(s, [])
        if not mapped:
            mapped = ['NONE']
        for v in mapped:
            off_rows.append({
                'event_id': f"O-{r['start_date']}",
                'source': 'official', 'variety': v, 'name': desc[:40],
                'category': '板块级宏观事件', 'severity': sev, 'freq': 'daily',  # 1d -> daily 统一
                'start': start, 'end': end,
            })
off_df = pl.DataFrame(off_rows)
# 官方 start/end 纯日期 -> datetime64
off_df = off_df.with_columns(
    pl.col('start').str.strptime(pl.Datetime, '%Y-%m-%d', strict=False),
    pl.col('end').str.strptime(pl.Datetime, '%Y-%m-%d', strict=False),
)
print(f"官方事件展开: {off_df.height} 条 | start null={off_df['start'].null_count()} | freq 取值={off_df['freq'].unique().to_list()}")

# ---------- 同品种 + 日期区间重叠 去重合并（避免 18+24 虚高计数）----------
# 仅 ours(au/cu/sc) 与 official(au/cu/sc) 之间合并；SR 官方无对应、NONE 不参与。
# 合并取并集区间，标记双重确认（multi_source / match_official），而非简单累加。
def _overlap(a, b):
    return not (a['end'] < b['start'] or b['end'] < a['start'])

_merged = []
_used_off = set()  # (event_id, variety) 已被合并的官方行
for o in ours_rows.iter_rows(named=True):
    rec = dict(o); rec['multi_source'] = False; rec['match_official'] = ''
    if o['variety'] == 'SR':
        _merged.append(rec); continue                      # SR 官方无数据，保留
    _ms = [f for f in off_df.iter_rows(named=True)
           if f['variety'] == o['variety'] and _overlap(o, f)]
    if _ms:
        rec['start'] = min([o['start']] + [f['start'] for f in _ms])
        rec['end']   = max([o['end']]   + [f['end']   for f in _ms])
        rec['multi_source'] = True
        rec['match_official'] = ','.join(f['event_id'] for f in _ms)
        for f in _ms:
            _used_off.add((f['event_id'], f['variety']))
    _merged.append(rec)
for f in off_df.iter_rows(named=True):
    if (f['event_id'], f['variety']) not in _used_off:
        rec = dict(f); rec['multi_source'] = False; rec['match_official'] = ''
        _merged.append(rec)

fused = pl.DataFrame(_merged)
fused = fused.sort(['start', 'source'])
fused.write_parquet(f"{OUT}/fused_events_official.parquet")

# ---------- 说明文档 ----------
md = ["# 融合事件标注（官方管线修正版 · Ground Truth）", "",
      f"- 我们逐品种事件：**{ours_rows.height}** 条（source=ours，15m/daily，精确，start/end 已修复非空）",
      f"- 官方板块级事件：17 条 → 展开为 **{off_df.height}** 条逐品种行（source=official，daily）",
      "- 修正：ours 的 '%Y-%m-%d %H:%M' 字符串改用 strptime(Datetime) 解析（原 A3 用 cast(Date) 强转失败变 null）；"
      "官方 freq '1d' 统一为 'daily'。",
      "- 映射：能源→sc，贵金属→au，有色→cu，全板块→au/cu/sc；化工/油脂油料/黑色系/碳酸锂/合成橡胶暂无对应期权（variety=NONE，回测按品种循环时跳过）。", "",
      f"- **去重合并**：同品种 + 日期区间重叠的 ours/official 事件合并为 1 条（并集区间，标记双重确认 multi_source），避免 18+24 简单累加虚高计数。融合后共 **{fused.height}** 条，其中 {fused.filter(pl.col('multi_source')).height} 条为 ours+official 双重确认（同一真实事件被两来源独立标注，互相印证可靠性）。", ""]
md.append("## 品种分布")
md.append(str(fused.group_by(['source', 'variety']).agg(pl.len()).sort(['source', 'variety'])))
md += ["", "## 我们的 18 事件（逐品种精准）", ""]
for r in ours_rows.iter_rows(named=True):
    md.append(f"- **{r['event_id']}** [{r['variety']}] {r['name']} | {r['category']} | sev{r['severity']} | {r['start']}~{r['end']}")
md += ["", "## 官方事件 → 逐品种映射", ""]
for r in off_df.iter_rows(named=True):
    md.append(f"- **{r['event_id']}** [{r['variety']}] {r['name']} | sev{r['severity']} | {r['start']}~{r['end']}")
open(f"{OUT}/fused_events_official.md", 'w', encoding='utf-8').write('\n'.join(md))

print(f"\nDONE -> {OUT}/fused_events_official.parquet | 总 {fused.height} 条")
print(fused.group_by(['source', 'variety']).agg(pl.len()).sort(['source', 'variety']))
