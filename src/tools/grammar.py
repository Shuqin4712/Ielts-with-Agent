"""grammar_check：定位语法错误 + 解释 + 修改建议。"""
from __future__ import annotations

from ._json import call_json

_SYSTEM = (
    "You are an IELTS grammar checker. Find grammatical, spelling and punctuation errors "
    "in the candidate text. For each error give: span (the exact erroneous fragment), "
    "explanation (what's wrong), suggestion (the corrected form). Do not rewrite the whole "
    "text. Respond with ONLY a JSON object: "
    '{"errors":[{"span":str,"explanation":str,"suggestion":str}]}. Empty list if no errors.'
)


def grammar_check(text: str) -> dict:
    """返回 {"errors":[{span, explanation, suggestion}]}。"""
    return call_json("grammar_check", _SYSTEM,
                     f"Text to check:\n{text}\n\nReturn errors as JSON only.")
