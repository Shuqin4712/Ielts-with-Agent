"""阶段 3 测试。批次 A（LLM-free）：成本路由 / 库 CRUD / schema 迁移。
后续批次的工具与 agent 路由测试需真调 LLM，gated 在 RUN_LLM_TESTS。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.db import library
from src.tools.save import save_to_library

# 需真调 LLM 的用例：设 RUN_LLM_TESTS=1 且有 key 才跑。
_LLM = pytest.mark.skipif(
    not (os.getenv("RUN_LLM_TESTS") and config.DEEPSEEK_API_KEY),
    reason="需 RUN_LLM_TESTS=1 且有 DEEPSEEK_API_KEY")


def test_tier_for_defaults_flash():
    assert config.tier_for("anything") == "flash"
    assert config.tier_for("vocab_upgrade") == "flash"


def test_ensure_schema_idempotent(tmp_path):
    db = tmp_path / "lib.sqlite"
    library.ensure_library_schema(db)
    library.ensure_library_schema(db)          # 再来一次不报错
    from src.db.sqlite import get_conn
    conn = get_conn(db)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(material_library)")}
    finally:
        conn.close()
    assert "source_excerpt" in cols


def test_vocab_crud(tmp_path):
    db = tmp_path / "lib.sqlite"
    vid = library.save_vocab("show", "The chart shows a rise.",
                             [{"word": "illustrate", "trap": "偏举例"}],
                             nuance_note="n", db_path=db)
    rows = library.list_vocab(db_path=db)
    assert len(rows) == 1 and rows[0]["id"] == vid and rows[0]["word"] == "show"
    assert '"illustrate"' in rows[0]["alternatives"]     # JSON 序列化
    assert library.delete_vocab(vid, db_path=db) == 1
    assert library.list_vocab(db_path=db) == []


def test_save_to_library_material(tmp_path):
    db = tmp_path / "lib.sqlite"
    out = save_to_library("material", {
        "type": "sentence_frame", "content": "It is widely argued that X.",
        "source_excerpt": "It is widely argued that remote work boosts productivity.",
        "topic": "work",
    }, db_path=db)
    assert out["target"] == "material"
    rows = library.list_material(db_path=db)
    assert len(rows) == 1
    assert rows[0]["type"] == "sentence_frame"
    assert rows[0]["source_excerpt"].startswith("It is widely argued")


@_LLM
def test_vocab_upgrade_shape():
    from src.tools.vocab import vocab_upgrade
    r = vocab_upgrade("show", "The graph shows a sharp increase in car sales.")
    assert len(r["alternatives"]) >= 1
    assert {"word", "register", "band_hint", "trap"} <= set(r["alternatives"][0])


@_LLM
def test_deconstruct_shape():
    from src.tools.deconstruct import deconstruct_article
    r = deconstruct_article("Some believe X. I agree. Firstly A. In conclusion B.")
    assert {"outline", "sentence_frames", "vocab"} <= set(r)


def test_score_predict_reuses_grade_graph(monkeypatch):
    """护栏：score_predict 必须走 build_grade_graph 的锚定开/reflection 关配置，
    不存在第二条打分路径。用假图拦截，验证配置与复用（LLM-free）。"""
    import src.tools.score as score_mod
    seen = {}

    class FakeGraph:
        def invoke(self, state):
            seen["run_cfg"] = state["run_cfg"]
            return {"dimension_scores": {"TA": {"band": 6.0, "evidence": "x"}},
                    "overall_band": 6.0}

    monkeypatch.setattr(score_mod, "build_grade_graph", lambda: FakeGraph())
    out = score_mod.score_predict("some essay text", 2)
    assert out["overall_band"] == 6.0
    assert seen["run_cfg"]["anchored"] is True and seen["run_cfg"]["reflect"] is False


@_LLM
def test_support_tools_shape():
    from src.tools.grammar import grammar_check
    from src.tools.dictionary import dictionary_lookup
    assert "errors" in grammar_check("He go to school yesterday.")
    d = dictionary_lookup("delineate")
    assert d["definitions"] and d["examples"]


@_LLM
def test_agent_routing():
    """LLM 自主选 tool 的路由（每个核心意图至少一条）。"""
    from langchain_core.messages import HumanMessage
    from src.agent.graph import build_assistant
    agent = build_assistant()

    def first_tool(text: str):
        r = agent.invoke({"messages": [HumanMessage(text)]})
        for m in r["messages"]:
            for tc in (getattr(m, "tool_calls", None) or []):
                return tc["name"]
        return None

    assert first_tool("帮我升级这句里的 show：The graph shows a rise.") == "vocab_upgrade"
    assert first_tool("delineate 什么意思？") == "dictionary_lookup"
    assert first_tool("拆解这篇文章：Some believe X. I agree. In conclusion Y.") == "deconstruct_article"
