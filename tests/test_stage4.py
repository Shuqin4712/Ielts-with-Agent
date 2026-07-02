"""阶段 4 测试：短期 checkpointer + 长期画像 + 个性化反馈。

LLM-free：画像 CRUD / episodic 走势 / weak_criteria 确定性 / checkpointer 线程隔离
/ ★护栏★（打分节点绝不接触 profile；个性化开关真能关掉 LLM 蒸馏）。
需真调 LLM 的（反馈个性化 / 语义蒸馏）gated 在 RUN_LLM_TESTS。
"""
from __future__ import annotations

import operator
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.memory import profile as P

_LLM = pytest.mark.skipif(
    not (os.getenv("RUN_LLM_TESTS") and config.DEEPSEEK_API_KEY),
    reason="需 RUN_LLM_TESTS=1 且有 DEEPSEEK_API_KEY")

_DS = {  # 一份四维打分样本
    "TA": {"band": 6.0, "evidence": "addresses task but underdeveloped"},
    "CC": {"band": 5.0, "evidence": "weak paragraph linking"},
    "LR": {"band": 6.0, "evidence": "adequate range"},
    "GRA": {"band": 6.5, "evidence": "some complex structures"},
}


# ── 长期画像：episodic / semantic / 确定性 ─────────────────────────
def test_load_profile_empty(tmp_path):
    p = P.load_profile("nobody", tmp_path / "p.sqlite")
    assert p["exists"] is False and p["band_history"] == [] and p["recurring_errors"] == []


def test_episodic_append_and_history_rows(tmp_path):
    db = tmp_path / "p.sqlite"
    P.update_after_grading("u", essay_meta={"essay_id": 1, "task_type": 2, "topic": "env"},
                           dimension_scores=_DS, overall_band=6.0, distill=False, db_path=db)
    p = P.update_after_grading("u", essay_meta={"essay_id": 2, "task_type": 2, "topic": "edu"},
                               dimension_scores=_DS, overall_band=6.5, distill=False, db_path=db)
    assert [ep["overall"] for ep in p["band_history"]] == [6.0, 6.5]   # episodic append-only
    from src.db.sqlite import get_conn
    conn = get_conn(db)
    n = conn.execute("SELECT COUNT(*) FROM grading_history WHERE user_id='u'").fetchone()[0]
    conn.close()
    assert n == 2


def test_weak_criteria_deterministic():
    hist = [{"dims": {"TA": 6, "CC": 5, "LR": 6, "GRA": 6}},
            {"dims": {"TA": 6, "CC": 5, "LR": 6.5, "GRA": 7}}]
    assert P._weak_criteria(hist) == ["CC"]          # CC 均分最低
    # 并列弱项（差 ≤0.25）一并算
    hist2 = [{"dims": {"TA": 6, "CC": 5, "LR": 5, "GRA": 7}}]
    assert set(P._weak_criteria(hist2)) == {"CC", "LR"}


def test_profile_summary_shape():
    prof = {"exists": True, "band_history": [{"overall": 6.0, "dims": {}}],
            "weak_criteria": ["CC"], "recurring_errors": [{"criterion": "CC",
            "pattern": "weak linking", "seen": 2}], "vocab_level": "around band 6"}
    s = P.profile_summary(prof)
    assert "CC" in s and "weak linking" in s and s              # 摘要含关键信息
    assert P.profile_summary({"exists": False, "band_history": []}) == ""


# ── 个性化开关：关则不调 LLM 蒸馏 ─────────────────────────────────
def test_personalize_off_skips_llm_distill(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("distill=False 时绝不该调 LLM 蒸馏")
    monkeypatch.setattr(P, "distill_semantic", _boom)
    # distill=False 不触发；distill=True 会触发（此处只验关）
    P.update_after_grading("u", essay_meta={"essay_id": 1, "task_type": 2},
                           dimension_scores=_DS, overall_band=6.0, distill=False,
                           db_path=tmp_path / "p.sqlite")


# ── ★护栏★：打分节点绝不接触 profile ──────────────────────────────
def test_grade_node_never_passes_profile_to_scorer(monkeypatch):
    """session.grade 只能把 essay/task/prompt/run_cfg 喂进内层纯打分图；
    profile / user_id 绝不能流入。用假内层图截获入参断言。"""
    import src.graph.session as S
    captured = {}

    class FakeInner:
        def invoke(self, state):
            captured.update(state)
            return {"dimension_scores": _DS, "overall_band": 6.0}

    monkeypatch.setattr(S, "_inner_grade_graph", lambda: FakeInner())
    S.grade({"user_id": "alice", "essay": "text", "task_type": 2, "prompt": "q",
             "essay_id": 9, "profile": {"recurring_errors": [{"pattern": "SECRET"}]},
             "personalize": True})
    # 内层收到的 key 必须是打分白名单，且绝无 profile / user_id
    assert "profile" not in captured and "user_id" not in captured
    assert set(captured) <= {"essay", "task_type", "prompt", "run_cfg",
                             "essay_id", "anchors", "retries"}


def test_session_grade_cfg_matches_score_predict():
    """外层图默认打分配置 = score_predict 那条（锚定开/reflection 关），不另立打分路径。"""
    import src.graph.session as S
    from src.tools.score import _GRADE_CFG
    assert S._DEFAULT_GRADE_CFG["anchored"] is True
    assert S._DEFAULT_GRADE_CFG["reflect"] is False
    assert _GRADE_CFG["anchored"] is True and _GRADE_CFG["reflect"] is False


def test_harness_configs_have_no_personalization():
    """回归护栏：eval 的配置里不该混入个性化字段（评测走纯打分）。"""
    from src.eval.harness import CONFIGS
    for cfg in CONFIGS.values():
        assert "personalize" not in cfg and "user_id" not in cfg


# ── 短期记忆：checkpointer 线程隔离 ────────────────────────────────
def test_checkpointer_thread_isolation():
    from langgraph.graph import START, END, StateGraph
    from src.memory.checkpoint import get_checkpointer

    class St(TypedDict):
        log: Annotated[list, operator.add]

    g = StateGraph(St)
    g.add_node("s", lambda s: {"log": ["x"]})
    g.add_edge(START, "s"); g.add_edge("s", END)
    app = g.compile(checkpointer=get_checkpointer(persist=False))

    a = {"configurable": {"thread_id": "A"}}
    b = {"configurable": {"thread_id": "B"}}
    app.invoke({"log": ["a1"]}, a)
    app.invoke({"log": ["a2"]}, a)
    app.invoke({"log": ["b1"]}, b)
    la = app.get_state(a).values["log"]
    lb = app.get_state(b).values["log"]
    assert "a1" in la and "a2" in la and "b1" not in la   # 同线程累积、隔离
    assert "b1" in lb and "a1" not in lb


# ── gated：需真调 LLM ──────────────────────────────────────────────
@_LLM
def test_feedback_node_is_text_and_continuous():
    """个性化反馈：给一份含 CC 反复问题的画像，反馈应是非空教练文字并呼应弱项。"""
    from src.graph.session import feedback
    prof = {"exists": True, "band_history": [{"overall": 5.5, "dims": {"CC": 5}}],
            "weak_criteria": ["CC"],
            "recurring_errors": [{"criterion": "CC", "pattern": "weak paragraph linking", "seen": 2}],
            "vocab_level": "around band 6"}
    out = feedback({"dimension_scores": _DS, "overall_band": 6.0, "task_type": 2,
                    "profile": prof, "personalize": True})
    txt = out["feedback"]
    assert isinstance(txt, str) and len(txt) > 20
    assert any(k in txt for k in ("CC", "cohesion", "Cohesion", "衔接", "连贯", "linking"))


@_LLM
def test_distill_semantic_incremental():
    """增量蒸馏：喂已有画像 + 一篇新依据 → 返回受限列表（≤MAX_ERRORS），是合并非清空。"""
    existing = [{"criterion": "GRA", "pattern": "tense errors", "seen": 2}]
    out = P.distill_semantic(existing, _DS, 6.0, 2)
    assert isinstance(out["recurring_errors"], list)
    assert len(out["recurring_errors"]) <= P.MAX_ERRORS
    assert out["recurring_errors"]        # 非空（至少纳入新信号）
