"""阶段 2 测试（不调 LLM）：指标正确性 + 泄漏断言两个方向。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.eval import metrics
from src.eval.harness import assert_no_leakage, load_gold_holdout
from src.graph.nodes import _pick_spread, _pick_vector_spread, route_reflection
from src.graph.state import SCORE_NODES
from src.rag import store


def _cand(band, eid, topic="x"):
    return {"document": f"essay{eid}", "metadata": {"band": band, "essay_id": eid, "topic": topic}}


def test_pick_spread():
    cands = [_cand(b, i) for i, b in enumerate([3.0, 5.0, 6.5, 7.5, 9.0])]
    picks = _pick_spread(cands, k=3)
    bands = [p["band"] for p in picks]
    assert bands == [3.0, 6.5, 9.0]                 # 低/中/高，含最高 band 高锚
    # 排除被评本身
    assert all(p["band"] != 9.0 for p in _pick_spread(cands, k=3, exclude_id=4))


def test_pick_vector_spread():
    """池内向量排序选锚：跨 band 铺开 + 每 band 取相似度序里最贴的那篇 + 排除自身。"""
    # ranked 模拟 store.search 输出：已按相似度升序（越靠前越贴本题）。
    ranked = [
        {"document": "close7", "metadata": {"band": 7.0, "essay_id": 1, "topic": "x"}},
        {"document": "far7",   "metadata": {"band": 7.0, "essay_id": 2, "topic": "x"}},
        {"document": "close5", "metadata": {"band": 5.0, "essay_id": 3, "topic": "x"}},
        {"document": "close9", "metadata": {"band": 9.0, "essay_id": 4, "topic": "x"}},
    ]
    picks = _pick_vector_spread(ranked, k=3)
    assert sorted(p["band"] for p in picks) == [5.0, 7.0, 9.0]     # 跨 band 铺开
    # band 7 有两篇 → 取相似度序里首见（更贴）的 close7，而非 far7
    assert next(p["text"] for p in picks if p["band"] == 7.0) == "close7"
    # 排除被评本身（essay_id=4 的 band-9 高锚被剔除）
    assert all(p["band"] != 9.0 for p in _pick_vector_spread(ranked, k=3, exclude_id=4))


def test_route_reflection():
    off = {"run_cfg": {"reflect": False}}
    on = lambda ok, r, mx=2: {"run_cfg": {"reflect": True, "max_retries": mx},
                              "reflection_ok": ok, "retries": r}
    # 收敛 → "aggregate"；回退 → 四维打分节点名列表（fan-out 回四维并行重评）。
    assert route_reflection(off) == "aggregate"            # 不开 reflect → 直接收敛
    assert route_reflection(on(True, 1)) == "aggregate"    # 自洽 → 收敛
    assert route_reflection(on(False, 1)) == SCORE_NODES   # 不自洽且没到上限 → 回退
    assert route_reflection(on(False, 2)) == "aggregate"   # 达到上限 → 收敛（防死循环）


def test_mae_within():
    assert metrics.mae([6, 7], [6, 7]) == 0.0
    assert metrics.mae([6, 8], [6, 6]) == 1.0
    assert metrics.within([6, 7], [6.5, 7], 0.5) == 1.0
    assert metrics.within([6, 8], [6, 6], 0.5) == 0.5


def test_qwk():
    assert metrics.qwk([5, 6, 7, 8], [5, 6, 7, 8]) == 1.0          # 完全一致
    assert metrics.qwk([5, 5], [5, 5]) == 1.0                       # 无变异 → 1.0
    # 越准 QWK 越高（序数性质）
    near = metrics.qwk([5, 6, 7], [5.5, 6, 7])
    far = metrics.qwk([5, 6, 7], [8, 4, 9])
    assert -1.0 <= far < near <= 1.0


@pytest.mark.skipif(not config.CHROMA_DIR.exists(), reason="未构建 chroma，先跑 build_stage0")
def test_no_leakage_holdout():
    """真实 gold holdout 与锚点集零重叠（评测纪律）。"""
    assert_no_leakage(load_gold_holdout())      # 不抛异常即通过


@pytest.mark.skipif(not config.CHROMA_DIR.exists(), reason="未构建 chroma，先跑 build_stage0")
def test_leakage_detected():
    """把一个锚点 essay_id 塞进 holdout，断言必须报错。"""
    coll = store.get_client().get_collection(config.COLL_EXEMPLAR)
    an_exemplar_id = coll.get()["metadatas"][0]["essay_id"]
    with pytest.raises(AssertionError):
        assert_no_leakage([{"id": an_exemplar_id}])
