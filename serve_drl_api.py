# -*- coding: utf-8 -*-
"""
问题3 交付物 · DRL 自适应预警模型推理 API
=========================================
加载离线训练产出的策略权重 drl_15m_policy.npz，对外提供 HTTP 接口：
接收 11 维状态特征向量，返回 DRL 自适应预警决策（报警 / 不报）与策略概率。

与训练脚本 train_drl_15m.py 的对应关系（保证推理与训练口径一致）：
  - 状态 = 11 维滚动分位特征（FEATS 列表，顺序不可变动）
  - px 子策略概率 p = σ(状态·W_px + b_px)，p ≥ τ*_px 视为价格压力预警
  - rv 子策略概率同理，作为波动率压力参考（辅助，不直接并入报警流）
  - 复合报警（与训练一致）：价格压力预警 且 IV 压力确认(atm_iv_p ≥ 0.85) 才升级，
    以双因子协同抑制单因子误报（问题3 多信号融合核心）
  - 自适应核心：τ*_px / τ*_rv 是训练阶段在验证集上扫 F1 最优所得，不依赖人工拍定阈值
  - 策略类型 policy ∈ {BC_tau, RL}：训练时当且仅当 RL 的 F1@48 优于 BC+τ* 才采用 RL，
    否则诚实回退到 BC+τ*（详见 train_drl_15m.py 的 train_policy 与说明文档）

运行方式（两种，任选其一）：
  A. 推荐（需要 FastAPI + uvicorn）：
       pip install fastapi uvicorn
       python serve_drl_api.py                 # 默认 0.0.0.0:8000
       python serve_drl_api.py --port 8080
     启动后访问 http://localhost:8000/docs 可见交互式接口文档。
  B. 零依赖（未安装 FastAPI 时自动回退，或显式加 --stdlib）：
       python serve_drl_api.py --stdlib        # 使用 Python 标准库 http.server
     访问 http://localhost:8000/ 查看模型信息；POST /predict 提交特征。

请求示例（POST /predict，Content-Type: application/json）：
  {
    "symbol": "au",
    "features": [0.62, 0.10, -0.05, 0.03, 0.08, 0.02, 0.55, -0.20, 0.40, 0.15, 0.07],
    "atm_iv_p": 0.88
  }
响应示例：
  {
    "symbol": "au",
    "prob_px": 0.71, "threshold_px": 0.42,
    "prob_rv": 0.33, "threshold_rv": 0.39,
    "drl_alert": true, "naive_alert": false, "warning_level": 1,
    "policy": "BC_tau",
    "note": "DRL 复合报警：px 价格压力预警且 IV 压力确认(>=0.85)"
  }

说明：本接口只做推理，不触发训练；权重为离线训练产物，已在提交包内
data/clean/warnings/official/drl/drl_15m_policy.npz。完整的 L0–L3 分级预警
由 build_warnings_15m.py（问题1）产出；本接口回答"问题3 的 DRL 模型能否独立
给出自适应预警决策"，二者互补。
"""
import os
import sys
import json
import argparse
import numpy as np

# 基于本文件位置推导项目根（与官方管线其他脚本一致，换机器无需改路径）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = f"{ROOT}/data/clean/warnings/official/drl/drl_15m_policy.npz"

# 11 维状态特征顺序，必须与 train_drl_15m.py 的 FEATS 完全一致
FEATS = ['atm_iv_p', 'skew_p', 'term_slope_p', 'curvature_p', 'rr_p', 'bf_p',
         'vpin_p', 'oi_flow_p', 'vol_p', 'amihud_p', 'jump_p']

# 复合报警的 IV 压力确认阈值（与训练代码 a_comb = alert_px & (iv_p >= 0.85) 对齐）
IV_GATE = 0.85
# 固定阈值基线（与训练代码 alert_naive = atm_iv_p >= 0.90 对齐，用于对照展示）
NAIVE_IV = 0.90
# DRL 自适应升级到 L2 的额外置信余量：prob_px 超出阈值 0.15 以上视为高置信
CONF_MARGIN = 0.15
# DRL 自适应升级到 L3 的极端 IV 阈值
EXTREME_IV = 0.92


def load_policy():
    """加载 npz 权重，返回可直接用于推理的 Python 对象。

    npz 中以 numpy 数组存储；method 字段是 0 维字符串数组，用 .item() 还原为 Python 字符串。
    """
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"未找到权重文件：{WEIGHTS}\n请先运行 train_drl_15m.py 生成 drl_15m_policy.npz")
    d = np.load(WEIGHTS, allow_pickle=False)
    policy = {
        'W_px': d['W_px'].astype(float), 'b_px': float(d['b_px']),
        'tau_px': float(d['tau_px']),
        'method_px': str(np.asarray(d['method_px']).item()),
        'W_rv': d['W_rv'].astype(float), 'b_rv': float(d['b_rv']),
        'tau_rv': float(d['tau_rv']),
        'method_rv': str(np.asarray(d['method_rv']).item()),
    }
    return policy


def _logistic(S, W, b):
    """logistic 概率 p = σ(S·W + b)，与 train_drl_15m.py 的 prob() 完全对齐。"""
    return 1.0 / (1.0 + np.exp(-(float(S @ W) + b)))


def predict(features, atm_iv_p, policy):
    """单次推理：输入 11 维状态特征与当前 atm_iv_p，输出 DRL 自适应预警决策。

    参数
    ----
    features : list[float]，长度 11，顺序须与 FEATS 一致
    atm_iv_p : float，当前平值隐含波动率分位，用于复合报警的 IV 压力确认门
    policy   : dict，load_policy() 返回的权重

    返回
    ----
    dict，含 px/rv 子策略概率、DRL 复合报警、固定阈值对照、预警等级与策略类型
    """
    S = np.asarray(features, dtype=float)
    if S.shape[0] != len(FEATS):
        raise ValueError(f"特征数量须为 {len(FEATS)}，收到 {S.shape[0]}")

    p_px = float(_logistic(S, policy['W_px'], policy['b_px']))
    p_rv = float(_logistic(S, policy['W_rv'], policy['b_rv']))
    alert_px = p_px >= policy['tau_px']
    iv_confirmed = atm_iv_p >= IV_GATE

    # 复合报警：价格压力预警 且 IV 压力确认（双因子协同，抑制单因子误报）
    drl_alert = bool(alert_px and iv_confirmed)
    # 固定阈值基线，用于对照（naive 固定阈值 atm_iv_p >= 0.90）
    naive_alert = bool(atm_iv_p >= NAIVE_IV)

    # DRL 自适应预警等级映射（在二元决策之上做透明、可解释的升级，便于评审理解）：
    #   0 = 无预警；1 = DRL 价格压力预警；2 = DRL 预警 + 高置信；3 = DRL 预警 + 极端 IV
    if drl_alert and atm_iv_p >= EXTREME_IV:
        level = 3
    elif drl_alert and p_px >= policy['tau_px'] + CONF_MARGIN:
        level = 2
    elif drl_alert:
        level = 1
    else:
        level = 0

    # 精确说明未报警的具体原因，避免"或"字歧义
    if drl_alert:
        note = "DRL 复合报警：px 价格压力预警且 IV 压力确认(>=%.2f)" % IV_GATE
    elif not alert_px:
        note = "DRL 未报警：px 概率 %.2f < τ* %.2f（价格压力未触发）" % (p_px, policy['tau_px'])
    elif not iv_confirmed:
        note = "DRL 未报警：px 已预警但 IV 压力未确认(atm_iv_p %.2f < %.2f)" % (atm_iv_p, IV_GATE)
    else:
        note = "DRL 未报警"

    # 所有数值显式转 Python 原生 float，确保 json.dumps 可序列化
    return {
        'prob_px': round(p_px, 4),
        'threshold_px': round(policy['tau_px'], 4),
        'prob_rv': round(p_rv, 4),
        'threshold_rv': round(policy['tau_rv'], 4),
        'drl_alert': drl_alert,
        'naive_alert': naive_alert,
        'warning_level': level,
        'policy': policy['method_px'],
        'note': note,
    }


# ====================== 以下为两种 HTTP 服务实现 ======================

def _build_app(policy):
    """构造 FastAPI 应用（仅当 fastapi 可用时调用）。"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class Req(BaseModel):
        symbol: str = "au"
        features: list
        atm_iv_p: float

    app = FastAPI(title="DRL 自适应预警推理 API", version="1.0")
    info = {
        "model": "DRL 自适应预警（px/rv 双子策略 + BC+τ* / REINFORCE 诚实 ablation）",
        "weights": os.path.relpath(WEIGHTS, ROOT),
        "feature_order": FEATS,
        "policy_px": policy['method_px'],
        "policy_rv": policy['method_rv'],
        "tau_px": policy['tau_px'],
        "tau_rv": policy['tau_rv'],
    }

    @app.get("/")
    def root():
        return info

    @app.post("/predict")
    def do_predict(req: Req):
        try:
            return predict(req.features, req.atm_iv_p, policy)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


def run_fastapi(policy, port):
    """FastAPI + uvicorn 服务（含自动交互文档 /docs）。"""
    import uvicorn
    app = _build_app(policy)
    print(f"[FastAPI] 启动于 http://0.0.0.0:{port}  （交互文档: /docs）")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def run_stdlib(policy, port):
    """零依赖回退：标准库 http.server 实现等价接口，便于无第三方依赖环境快速验证。"""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith('/predict'):
                self._send({"error": "请使用 POST /predict 提交特征"}, 405)
                return
            self._send({
                "model": "DRL 自适应预警（px/rv 双子策略）",
                "weights": os.path.relpath(WEIGHTS, ROOT),
                "feature_order": FEATS,
                "usage": "POST /predict  Body: {\"symbol\":\"au\",\"features\":[11 floats],\"atm_iv_p\":0.88}",
            })

        def do_POST(self):
            if self.path != '/predict':
                self._send({"error": "未知路径，仅支持 POST /predict"}, 404)
                return
            try:
                n = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(n)
                req = json.loads(raw.decode('utf-8'))
                if 'features' not in req or 'atm_iv_p' not in req:
                    self._send({"error": "缺少 features 或 atm_iv_p 字段"}, 400)
                    return
                out = predict(req['features'], float(req['atm_iv_p']), policy)
                out['symbol'] = req.get('symbol', 'au')
                self._send(out)
            except Exception as e:
                self._send({"error": str(e)}, 400)

        def log_message(self, *a):
            pass

    print(f"[stdlib] 启动于 http://0.0.0.0:{port}  （零依赖模式，POST /predict 提交特征）")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description="DRL 自适应预警推理 API")
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--stdlib', action='store_true', help='强制使用标准库回退（不依赖 FastAPI）')
    args = ap.parse_args()

    policy = load_policy()
    print(f"已加载权重：{WEIGHTS}")
    print(f"  策略 px={policy['method_px']}(τ*={policy['tau_px']:.3f})  "
          f"rv={policy['method_rv']}(τ*={policy['tau_rv']:.3f})")

    # 优先 FastAPI（交互文档 /docs），未安装或显式 --stdlib 时回退标准库零依赖模式
    if not args.stdlib and _has_fastapi():
        try:
            run_fastapi(policy, args.port)
            return
        except Exception as e:
            print(f"[warn] FastAPI 启动失败，回退标准库：{e}")
    run_stdlib(policy, args.port)


def _has_fastapi():
    try:
        import fastapi  # noqa
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
