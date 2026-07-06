"""组装批改 workflow 图。

LangGraph 用法：把 node 注册进 StateGraph、用 edge 连成流程、compile() 得到可 invoke 的图。
四维打分做成 fan-out / fan-in：retrieve_exemplars 之后并行分出四个 score_<crit> 节点，
各自只写自己那一维，dimension_scores 用自定义 reducer（state.merge_scores）合并；四维齐了
再汇入 reflection。reflection 条件边可回退（fan-out 回四维重评）或收敛到 aggregate。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    aggregate, ingest, make_score_node, reflection, retrieve_exemplars,
    retrieve_rubric, route_reflection,
)
from .state import CRITERIA, SCORE_NODES, GradeState


def build_grade_graph():
    """批改图（阶段 2 全量 + 四维并行）：
    ingest → retrieve_rubric → retrieve_exemplars
             → [fan-out] score_TA | score_CC | score_LR | score_GRA [并行]
             → [fan-in] reflection → [条件边] 不自洽→fan-out 回四维重评
                                     ｜ 自洽/达上限→aggregate → END

    retrieve_exemplars / reflection 在 run_cfg 关时是无操作（返回空锚点 / 直接放行），
    故同一张图靠 run_cfg 分档：baseline / anchored / anchored+reflect。
    """
    b = StateGraph(GradeState)
    b.add_node("ingest", ingest)
    b.add_node("retrieve_rubric", retrieve_rubric)
    b.add_node("retrieve_exemplars", retrieve_exemplars)
    for c in CRITERIA:                       # 四个单维打分节点
        b.add_node(f"score_{c}", make_score_node(c))
    b.add_node("reflection", reflection)
    b.add_node("aggregate", aggregate)

    b.add_edge(START, "ingest")
    b.add_edge("ingest", "retrieve_rubric")
    b.add_edge("retrieve_rubric", "retrieve_exemplars")
    # fan-out：retrieve_exemplars 之后并行分出四维；fan-in：四维都写完才进 reflection。
    for c in CRITERIA:
        b.add_edge("retrieve_exemplars", f"score_{c}")
        b.add_edge(f"score_{c}", "reflection")
    # 条件边：reflection 判断决定回退重评（返回四维节点名 → fan-out）还是收敛。
    b.add_conditional_edges("reflection", route_reflection, SCORE_NODES + ["aggregate"])
    b.add_edge("aggregate", END)
    return b.compile()
