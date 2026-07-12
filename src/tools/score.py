"""score_predict：给作文打分——**直接复用阶段 2 那条 eval 验证过的打分管道**。

护栏（最高优先级）：这里绝不写任何新的打分逻辑。凡是给作文打分，一律走
build_grade_graph 的「锚定开 / reflection 关」配置——就是 eval harness 上量化过的那条。
存在第二条打分路径，就会毁掉「评测数字与产品行为一致」这个资产。
"""
from __future__ import annotations

from ..graph.build import build_grade_graph

# 与默认打分配置一致（见 README Evaluation / grade_essay.py）。
# anchor_rank="vector"：v1.4 消融确认池内向量选锚 QWK 优于 band 均匀采样，已设为生产默认。
_GRADE_CFG = {"anchored": True, "reflect": False, "score_tier": "flash", "anchor_rank": "vector"}


def score_predict(essay: str, task_type: int, prompt: str = "") -> dict:
    """返回 {"dimension_scores": {...}, "overall_band": float}。复用批改图，不重写打分。"""
    graph = build_grade_graph()
    final = graph.invoke({
        "essay": essay, "task_type": task_type, "prompt": prompt or "",
        "run_cfg": _GRADE_CFG, "essay_id": None, "anchors": [], "retries": 0,
    })
    return {"dimension_scores": final["dimension_scores"],
            "overall_band": final["overall_band"]}
