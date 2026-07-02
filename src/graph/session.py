"""批改会话外层图（阶段 4）：在纯打分管道外面套一层「记忆 + 个性化」。

    START → load_profile → grade → memory_write → END
                             │
                             └─ 内部 invoke build_grade_graph()（纯打分，不传 profile）

⚠️ 护栏（最高优先级）：`grade` 节点只把 essay/task/prompt/run_cfg 喂给内层
build_grade_graph——学生画像 profile **物理上进不了打分逻辑**。打分永远是那条被
eval 量化过的唯一管道；本外层图只负责读画像、写画像、（批次 C）个性化反馈文字。

eval harness / score_predict 直接用内层 build_grade_graph，根本不经过这张外层图，
故个性化对评测「结构性关闭」；personalize 开关是双保险。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..data.topic import infer_topic
from ..llm import get_llm, with_backoff
from ..memory import profile as prof_mem
from .build import build_grade_graph
from .state import CRITERIA, SessionState
from .nodes import _criterion_full

# 与阶段 2 验证的默认打分配置一致（锚定开 / reflection 关）——和 score_predict 同一条。
_DEFAULT_GRADE_CFG = {"anchored": True, "reflect": False, "score_tier": "flash"}


@lru_cache(maxsize=1)
def _inner_grade_graph():
    """内层纯打分图，建一次复用（避免每篇重建，见 PROGRESS 遗留项）。"""
    return build_grade_graph()


# ── 节点 ───────────────────────────────────────────────────────────
def load_profile(state: SessionState) -> dict:
    """会话开头取该 user 的长期画像备用（无 user 则空画像）。只读，不影响打分。"""
    uid = state.get("user_id")
    if not uid:
        return {"profile": {"exists": False, "band_history": [], "recurring_errors": [],
                            "weak_criteria": [], "vocab_level": None}}
    return {"profile": prof_mem.load_profile(uid)}


def grade(state: SessionState) -> dict:
    """★护栏点★：只用 essay/task/prompt/run_cfg 调内层纯打分图。绝不传 profile。"""
    final = _inner_grade_graph().invoke({
        "essay": state["essay"],
        "task_type": state["task_type"],
        "prompt": state.get("prompt", "") or "",
        "run_cfg": state.get("run_cfg") or _DEFAULT_GRADE_CFG,
        "essay_id": state.get("essay_id"),
        "anchors": [], "retries": 0,
    })
    return {"dimension_scores": final["dimension_scores"],
            "overall_band": final["overall_band"]}


def feedback(state: SessionState) -> dict:
    """把已定的四维 band + 依据写成教练式反馈文字。个性化时注入画像摘要，
    让反馈呼应历史（「你上次也在 CC 丢分」）。

    ⚠️ 只组织**文字**：band 由上游 grade 节点定死，这里原样引用、绝不改分/重评。
    profile 只进这个节点的 prompt。
    """
    ds, task = state["dimension_scores"], state["task_type"]
    score_lines = "\n".join(
        f"- {_criterion_full(c, task)} ({c}): band {ds[c]['band']} — {ds[c]['evidence']}"
        for c in CRITERIA
    )
    personalize = state.get("personalize", True)
    summary = prof_mem.profile_summary(state.get("profile") or {}) if personalize else ""
    history_block = (
        f"\nThis student's long-term profile (use it to make feedback personal and "
        f"continuous — call back to recurring issues, acknowledge progress; but NEVER "
        f"change the bands above):\n{summary}\n" if summary else ""
    )
    system = (
        "You are a supportive, concrete IELTS writing coach. Given the FIXED per-criterion "
        "bands and evidence, write actionable feedback. You MUST NOT re-score or contradict "
        "the given bands — treat them as final; only explain and advise. For each criterion "
        "give 1-2 sentences (what's holding it back + one concrete fix), then a short overall "
        "note. If a long-term profile is provided, weave in continuity (recurring problems, "
        "progress since last time). Reply in the student's likely language (Chinese for this user). "
        "Keep it under ~220 words. Do NOT restate band numbers as if scoring anew."
    )
    human = (
        f"Task {task}. Overall band: {state['overall_band']}.\n"
        f"Per-criterion (FIXED — do not change):\n{score_lines}\n"
        f"{history_block}\n"
        "Write the personalized coaching feedback now."
    )
    llm = get_llm("flash", temperature=0)
    text = with_backoff(llm.invoke)([SystemMessage(system), HumanMessage(human)]).content
    return {"feedback": text.strip()}


def memory_write(state: SessionState) -> dict:
    """批改结束后更新长期记忆：episodic 确定性 append + semantic LLM 增量蒸馏。

    personalize=False 时跳过 LLM 蒸馏（省钱/测试），但仍记 episodic 流水（确定性、
    零打分影响）。无 user_id 时整体跳过。
    """
    uid = state.get("user_id")
    if not uid:
        return {}
    personalize = state.get("personalize", True)
    meta = {
        "essay_id": state.get("essay_id"),
        "task_type": state["task_type"],
        "topic": infer_topic(state.get("prompt", "") or "", state["essay"]),
    }
    updated = prof_mem.update_after_grading(
        uid, essay_meta=meta, dimension_scores=state["dimension_scores"],
        overall_band=state["overall_band"], feedback=state.get("feedback", ""),
        distill=personalize)
    return {"profile": updated}


# ── 组装 ───────────────────────────────────────────────────────────
def build_grading_session_graph(*, checkpointer=None):
    """批改会话外层图。checkpointer 接短期记忆（thread_id 续跑）。

    流程：load_profile → grade → feedback → memory_write。
    """
    b = StateGraph(SessionState)
    b.add_node("load_profile", load_profile)
    b.add_node("grade", grade)
    b.add_node("feedback", feedback)
    b.add_node("memory_write", memory_write)

    b.add_edge(START, "load_profile")
    b.add_edge("load_profile", "grade")
    b.add_edge("grade", "feedback")
    b.add_edge("feedback", "memory_write")
    b.add_edge("memory_write", END)
    return b.compile(checkpointer=checkpointer)
