"""检索层评测测试（不调 LLM、不调 Ollama）：指标纯函数正确性 + 查询集契约。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.retrieval import QUERIES, hit_at_1, mrr_at_k, recall_at_k


def test_hit_at_1():
    assert hit_at_1([True, False]) == 1.0
    assert hit_at_1([False, True]) == 0.0
    assert hit_at_1([]) == 0.0


def test_recall_at_k_capped():
    flags = [True, False, True, False, False]
    # 相关池 ≥ k：分母截到 k=5 → 2/5
    assert recall_at_k(flags, n_relevant=14, k=5) == pytest.approx(0.4)
    # 相关池 < k：分母是相关总数 → 2/2 满召回
    assert recall_at_k(flags, n_relevant=2, k=5) == pytest.approx(1.0)
    # 空相关池不除零
    assert recall_at_k(flags, n_relevant=0, k=5) == 0.0


def test_mrr_at_k():
    assert mrr_at_k([False, False, True]) == pytest.approx(1 / 3)
    assert mrr_at_k([True]) == 1.0
    assert mrr_at_k([False] * 10) == 0.0
    # 窗口外的相关不计
    assert mrr_at_k([False] * 10 + [True], k=10) == 0.0


def test_query_set_contract():
    """查询集落盘后即 ground truth：字段齐全、qid 唯一、不含 general。"""
    if not QUERIES.exists():
        pytest.skip("查询集未构建（先跑 python -m src.eval.retrieval）")
    import json
    rows = [json.loads(l) for l in QUERIES.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) >= 20
    assert len({r["qid"] for r in rows}) == len(rows)
    for r in rows:
        assert r["task_type"] in (1, 2)
        assert r["topic"] != "general"
        assert len(r["prompt"]) > 30
