"""阶段 1 测试。默认不调 LLM（保持 pytest 便宜）：
  - ielts_round 取整规则
  - retrieve_rubric 结构（需已建 chroma）
  - graph 能 compile
全图 smoke 需真调 4 次 flash，仅在设了 RUN_LLM_TESTS=1 且有 key 时跑。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.graph.build import build_grade_graph
from src.graph.nodes import ielts_round, retrieve_rubric
from src.graph.state import CRITERIA


def test_ielts_round():
    assert ielts_round(6.0) == 6.0
    assert ielts_round(6.25) == 6.5     # .25 向上
    assert ielts_round(6.75) == 7.0     # .75 向上
    assert ielts_round(6.1) == 6.0
    assert ielts_round(6.4) == 6.5
    assert ielts_round(0.0) == 0.0 and ielts_round(9.0) == 9.0


def test_graph_compiles():
    build_grade_graph()   # 不抛异常即可


@pytest.mark.skipif(not config.CHROMA_DIR.exists(), reason="未构建 chroma，先跑 build_stage0")
def test_retrieve_rubric_structure():
    out = retrieve_rubric({"task_type": 2})["retrieved_rubric"]
    assert set(out.keys()) == set(CRITERIA)
    for crit in CRITERIA:
        bands = [b["band"] for b in out[crit]]
        assert bands == sorted(bands) and len(bands) == 10   # band 0–9


@pytest.mark.skipif(
    not (os.getenv("RUN_LLM_TESTS") and config.DEEPSEEK_API_KEY),
    reason="需 RUN_LLM_TESTS=1 且有 DEEPSEEK_API_KEY（会真调 flash）")
def test_full_graph_smoke():
    graph = build_grade_graph()
    final = graph.invoke({
        "essay": "Some people think technology makes life better. I agree because "
                 "it saves time and connects people, although it can be distracting.",
        "task_type": 2, "prompt": "Do you agree that technology improves life?",
    })
    assert set(final["dimension_scores"]) == set(CRITERIA)
    assert 0.0 <= final["overall_band"] <= 9.0
