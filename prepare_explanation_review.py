# -*- coding: utf-8 -*-
"""
C11 · 解释文本人工抽检样本制备（官方数据）
================================================
从知识图谱全量解释(27493条)中按 (品种 × 预警级别) 分层抽样，
生成可直接交付人工评分的抽检样本 + 评分量规 + 自动完整性校验。

输出：
  data/clean/warnings/official/kg/kg_sample_for_review.md
    - 评分量规(5分制)
    - 分层抽样说明
    - 逐条样本(编号/品种/级别/五段式解释)
    - 可填写评分表
    - 自动完整性校验汇总(供参考，非替代人工评分)
"""
import json, os, random
from collections import defaultdict

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = f"{ROOT}/data/clean/warnings/official/kg/kg_explanations.json"
OUT = f"{ROOT}/data/clean/warnings/official/kg/kg_sample_for_review.md"

LEVEL_NAME = {1: "L1(关注)", 2: "L2(警示)", 3: "L3(紧急)"}
SECTIONS = ["触发因子", "市场形态", "历史相似", "宏观环境", "推理结论"]
PER_CELL = 3           # 每个 (品种×级别) 单元格抽样条数
SEED = 20260813

def main():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 分层抽样
    cells = defaultdict(list)
    for it in data:
        cells[(it["variety"], int(it["level"]))].append(it)

    rnd = random.Random(SEED)
    sampled = []
    for (v, lv), items in sorted(cells.items()):
        pick = rnd.sample(items, min(PER_CELL, len(items)))
        sampled.extend(pick)

    # 自动完整性校验
    def completeness(expl):
        missing = [s for s in SECTIONS if f"· {s}：" not in expl]
        return missing

    checks = []
    for it in sampled:
        miss = completeness(it["explanation"])
        checks.append((it["id"], len(it["explanation"]), miss))

    total = len(sampled)
    complete = sum(1 for _, _, m in checks if not m)
    avg_len = int(sum(c for _, c, _ in checks) / total) if total else 0

    # 组装文档
    lines = []
    lines.append("# C11 · 预警解释文本 人工抽检样本（官方数据）")
    lines.append("")
    lines.append(f"**抽样方法**：按 (品种 × 预警级别) 分层抽样，每格 {PER_CELL} 条，随机种子 {SEED}，共 {total} 条（全量 {len(data)} 条）。")
    lines.append(f"**自动完整性校验**：含全部五段式章节的样本 {complete}/{total} 条；平均文本长度 {avg_len} 字。")
    lines.append("")
    lines.append("## 评分量规（5 分制，问题2 要求 ≥4/5）")
    lines.append("")
    lines.append("| 分数 | 标准 |")
    lines.append("|---|---|")
    lines.append("| 5 | 因子归因准确、形态描述与数据一致、历史相似合理、宏观关联恰当、结论可操作 |")
    lines.append("| 4 | 五段齐全、归因基本准确、结论合理（允许个别措辞冗余） |")
    lines.append("| 3 | 主体信息齐全但存在 1 处明显归因偏差或信息空洞 |")
    lines.append("| 2 | 缺失≥1 个章节，或归因与触发因子明显矛盾 |")
    lines.append("| 1 | 文本不可读 / 完全无信息量 |")
    lines.append("")
    lines.append("## 抽检样本")
    lines.append("")
    for i, it in enumerate(sampled, 1):
        lines.append(f"### 样本 #{i}  [{it['variety']} · {LEVEL_NAME.get(int(it['level']), it['level'])} · {it['datetime']}]")
        lines.append("")
        lines.append("```")
        lines.append(it["explanation"].strip())
        lines.append("```")
        lines.append("")

    lines.append("## 评分表（请人工填写）")
    lines.append("")
    lines.append("| 编号 | 品种 | 级别 | 评分(1-5) | 备注 |")
    lines.append("|---|---|---|---|---|")
    for i, it in enumerate(sampled, 1):
        lines.append(f"| {i} | {it['variety']} | {LEVEL_NAME.get(int(it['level']), it['level'])} |  |  |")
    lines.append("")
    lines.append(f"**抽检合格率目标**：评分≥4 的比例 ≥ 80%（竞赛解释文本评分 ≥4/5 口径）。")
    lines.append("")
    lines.append("## 自动完整性校验明细（参考，非替代人工评分）")
    lines.append("")
    lines.append("| 编号 | id | 字数 | 缺失章节 |")
    lines.append("|---|---|---|---|")
    for i, (cid, clen, miss) in enumerate(checks, 1):
        m = "无" if not miss else "、".join(miss)
        lines.append(f"| {i} | {cid} | {clen} | {m} |")
    lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"OK -> {OUT}")
    print(f"样本 {total} 条 | 完整 {complete} | 平均字数 {avg_len}")
    # 级别分布
    dist = defaultdict(int)
    for it in sampled:
        dist[(it["variety"], int(it["level"]))] += 1
    print("抽样分布:", dict(sorted(dist.items())))

if __name__ == "__main__":
    main()
