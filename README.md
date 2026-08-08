# 商品期权波动率曲面 · 智能体量化风控预警看板

[![License](https://img.shields.io/badge/license-Academic-blue.svg)](./LICENSE)

> 第五届中国研究生金融科技创新大赛（宁证期货赛题）作品：基于期权隐含波动率曲面与多智能体的商品期权风险分级预警系统。

## 在线预览

👉 直接访问 `https://<你的用户名>.github.io/<仓库名>/` 即可查看本看板。

## 看板内容

- **核心成绩**：召回率 / 精确率 / 提前量 / CVaR 改善率 / DRL 提升 / 可解释性，六项硬指标全达标。
- **系统流程**：从行情 → 波动率曲面 → 分级预警 → 知识图谱解释的四个步骤。
- **风险态势**：黄金 / 铜 / 原油（15 分钟）+ 白糖（日频）逐日最高预警等级。
- **市场情绪**：压力时刻 vs 平稳时刻的 IV 微笑对比。
- **信号来源**：11 维风险信号在事件期与平稳期的雷达画像。
- **极端事件考卷**：18 场真实行情，系统覆盖与提前量一览。
- **模型效果**：固定阈值 vs 规则智能体 vs 自适应智能体的 F1 与 CVaR 对比。
- **可解释 AI**：每条预警的中文解释样例。
- **术语词典**：8 个关键词的生活化解释。

## 文件说明

| 文件 | 说明 |
|------|------|
| `index.html` | 单文件自包含看板，1.1 MB，含 ECharts 与全部数据，离线可用 |
| `dashboard/data.json` | 看板内嵌的数据源（JSON） |
| `dashboard/template.html` | 看板 HTML 模板 |
| `dashboard/styles.css` | 看板样式 |
| `dashboard/app_1.js` / `app_2.js` | 看板交互与图表逻辑 |
| `vendor/echarts.min.js` | 本地 ECharts 5.5.1（已内联到 index.html） |

## 如何本地打开

直接双击 `docs/index.html`，或在命令行启动本地服务器：

```bash
cd docs
python -m http.server 8080
# 浏览器访问 http://localhost:8080
```

> 注意：若修改了 `data.json`，需要重新运行构建脚本 `data_pipeline/report/build_github_dashboard.py` 才能更新 `index.html`。

## 数据来源

- 期权 / 期货分钟行情：TqSdk（公开免费账户）
- 交易所日频结算与隐含波动率：AKShare（郑商所）
- 利率期限结构：SHIBOR / 国债收益率曲线

全部为公开数据，可复现。

## 免责声明

本看板为学术竞赛作品，仅展示风险预警结果，**不构成任何投资建议或交易指令**。历史回测不代表未来表现。
