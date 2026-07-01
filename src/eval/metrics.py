"""评测指标：MAE、±tol 一致率、QWK（Quadratic Weighted Kappa）。

QWK 为何比准确率适合 band 打分：band 是**序数**量，QWK 用二次权重惩罚「差得远」的
预测（把 5 判成 8 远重于判成 7），并校正随机一致性；纯准确率/±0.5 把所有错判等同、
看不出偏离幅度，是 AES（自动作文评分）文献的标准指标。
"""
from __future__ import annotations

import numpy as np


def mae(pred: list[float], true: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def within(pred: list[float], true: list[float], tol: float) -> float:
    """|pred - true| <= tol 的比例。"""
    d = np.abs(np.asarray(pred) - np.asarray(true))
    return float(np.mean(d <= tol + 1e-9))


def qwk(pred: list[float], true: list[float],
        lo: float = 0.0, hi: float = 9.0, step: float = 0.5) -> float:
    """Quadratic Weighted Kappa。band 0–9 步进 0.5 → 19 个序数格。"""
    n = int(round((hi - lo) / step)) + 1
    idx = lambda x: int(round((min(max(x, lo), hi) - lo) / step))
    O = np.zeros((n, n))
    for p, t in zip(pred, true):
        O[idx(t), idx(p)] += 1

    w = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2) / (n - 1) ** 2
    act = O.sum(axis=1)
    prd = O.sum(axis=0)
    E = np.outer(act, prd) / O.sum()
    denom = float((w * E).sum())
    if denom == 0:                       # 无变异（全同一格）→ 视为完全一致
        return 1.0
    return float(1 - (w * O).sum() / denom)


def overall_metrics(pred: list[float], true: list[float]) -> dict:
    return {
        "n": len(pred),
        "mae": round(mae(pred, true), 3),
        "within_0.5": round(within(pred, true, 0.5), 3),
        "within_1.0": round(within(pred, true, 1.0), 3),
        "qwk": round(qwk(pred, true), 3),
    }
