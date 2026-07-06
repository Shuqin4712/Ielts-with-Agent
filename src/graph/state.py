"""批改 workflow 的 State schema 与结构化输出 schema。

LangGraph 用法：State 是一个 TypedDict，作为整张图的「共享工作台」。
每个 node 读它、返回一个局部更新 dict，LangGraph 合并进 State 再流向下个 node。
本阶段是最薄竖切，字段只留四维打分必需的。
"""
from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from pydantic import BaseModel, Field

# 官方四维（TA 在 task2 语义上是 Task Response，metadata key 仍统一为 TA）。
CRITERIA = ["TA", "CC", "LR", "GRA"]
# 四维并行打分的节点名（fan-out 目标；build 图与 reflection 回退路由共用，保持一处定义）。
SCORE_NODES = [f"score_{c}" for c in CRITERIA]


def merge_scores(old: dict | None, new: dict | None) -> dict:
    """dimension_scores 的 reducer：四维打分节点并行 fan-out，各自只写自己那一维
    {crit: {...}}，本函数按 criterion 合并成整字典。reflection 回退重评时同名维覆盖。"""
    return {**(old or {}), **(new or {})}


class GradeState(TypedDict):
    essay: str
    task_type: int                 # 1 或 2
    prompt: str
    retrieved_rubric: dict         # {criterion: [{"band": int, "text": str}, ...]}
    # 自定义 reducer：并行四维各写一维，按 criterion 合并（见 merge_scores）。
    dimension_scores: Annotated[dict, merge_scores]  # {criterion: {"band": float, "evidence": str}}
    overall_band: float
    # 阶段 2 新增（可选，用 run_cfg 开关分档；缺省即 stage1 裸基线行为）
    run_cfg: NotRequired[dict]     # {"anchored":bool,"reflect":bool,"score_tier":str,"thinking":bool,"max_retries":int}
    essay_id: NotRequired[int]     # 被评作文 id，供 retrieve_exemplars 排除自身（防泄漏）
    anchors: NotRequired[list]     # 锚定范文 [{"band","text","topic"}]，无锚定时为空
    retries: NotRequired[int]      # reflection 回退计数
    reflection_ok: NotRequired[bool]
    reflection_feedback: NotRequired[dict]  # {criterion: 审查意见}，回退重评时注入


class DimensionScore(BaseModel):
    """约束 LLM 的结构化输出：一个 band + 一句依据。"""
    band: float = Field(description="IELTS band for this criterion, 0-9 in steps of 0.5")
    evidence: str = Field(description="One concise sentence justifying the band")


class SessionState(TypedDict):
    """批改会话外层图（阶段 4）的 State：记忆 + 个性化。

    刻意与 GradeState 分开：GradeState 是纯打分工作台，**不含 profile**——这让
    「学生画像进不了打分」在类型层面就一目了然。外层的 grade 节点调用内层图时，
    只从这里挑 essay/task/prompt 传进 GradeState，profile 留在外层。
    """
    user_id: str
    essay: str
    task_type: int                 # 1 或 2
    prompt: NotRequired[str]
    essay_id: NotRequired[int]
    run_cfg: NotRequired[dict]     # 透传给内层打分图的档位配置
    personalize: NotRequired[bool] # 默认 True；关则跳过 semantic 蒸馏 + 反馈个性化
    profile: NotRequired[dict]     # load_profile 注入的长期画像（只喂 feedback，不喂 grade）
    dimension_scores: NotRequired[dict]
    overall_band: NotRequired[float]
    feedback: NotRequired[str]     # 批次 C 生成
    revision: NotRequired[list]    # v1.2：最弱维度的 1-2 句改写示范 [{original,revised,why}]
