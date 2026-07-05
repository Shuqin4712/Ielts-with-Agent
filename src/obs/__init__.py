"""可观测性（阶段 6）：被动记录每次 LLM 调用的档位 / token / 延迟。

设计原则：**passive observer**——只观察不改行为。挂在唯一的模型收口点
`get_llm` 上（callback），故 grade / chat / 所有 tool 的调用全自动覆盖，
无需改图/节点/打分/记忆逻辑（temp=0 可复现性不受影响）。
"""
from .tracker import LlmUsageCallback, operation, usage_callback, wrap_operation

__all__ = ["LlmUsageCallback", "operation", "usage_callback", "wrap_operation"]
