"""组装批改 workflow 图。

LangGraph 用法：把 node 注册进 StateGraph、用 edge 连成流程、compile() 得到可 invoke 的图。
本阶段是一条直线，无条件边、无 checkpointer（reflection 回环与短期记忆是后续阶段）。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    aggregate, ingest, reflection, retrieve_exemplars, retrieve_rubric,
    route_reflection, score_dimensions,
)
from .state import GradeState


def build_grade_graph():
    """批改图（阶段 2 全量）：
    ingest → retrieve_rubric → retrieve_exemplars → score_dimensions → reflection
             → [条件边] 不自洽→score_dimensions 重评 ｜ 自洽/达上限→aggregate → END

    retrieve_exemplars / reflection 在 run_cfg 关时是无操作（返回空锚点 / 直接放行），
    故同一张图靠 run_cfg 分档：baseline / anchored / anchored+reflect。
    """
    b = StateGraph(GradeState)
    b.add_node("ingest", ingest)
    b.add_node("retrieve_rubric", retrieve_rubric)
    b.add_node("retrieve_exemplars", retrieve_exemplars)
    b.add_node("score_dimensions", score_dimensions)
    b.add_node("reflection", reflection)
    b.add_node("aggregate", aggregate)

    b.add_edge(START, "ingest")
    b.add_edge("ingest", "retrieve_rubric")
    b.add_edge("retrieve_rubric", "retrieve_exemplars")
    b.add_edge("retrieve_exemplars", "score_dimensions")
    b.add_edge("score_dimensions", "reflection")
    # 条件边：reflection 的判断决定回退重评还是收敛。
    b.add_conditional_edges("reflection", route_reflection,
                            {"retry": "score_dimensions", "done": "aggregate"})
    b.add_edge("aggregate", END)
    return b.compile()
