"""工具共用的「约束 JSON 输出」调用：JSON 模式 + 一次修复重试。

约束输出偶尔仍会返回坏 JSON（尾随说明、代码围栏等），所以 json.loads 失败时
把坏输出回灌给模型、要求只回合法 JSON，重试一次再解析——兜底但不无限重试。
档位走 config.tier_for（默认 flash），复用 get_llm 的超时/退避设置。
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from .. import config
from ..llm import get_llm, with_backoff


def call_json(tool_name: str, system: str, human: str, *, temperature: float = 0) -> dict:
    llm = get_llm(config.tier_for(tool_name), temperature=temperature).bind(
        response_format={"type": "json_object"})
    raw = with_backoff(llm.invoke)([SystemMessage(system), HumanMessage(human)]).content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = with_backoff(llm.invoke)([
            SystemMessage("Return ONLY valid minified JSON — no prose, no code fences."),
            HumanMessage(f"Fix this into valid JSON:\n{raw}"),
        ]).content
        return json.loads(fixed)   # 仍失败则抛出，让调用方感知（不静默吞）
