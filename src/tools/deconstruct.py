"""deconstruct_article：拆解用户提供的文章（与 exemplar_provide 方向相反：抽取 vs 生成）。

结构化逆向：每段功能提纲 + 精彩句抽象成可迁移句式模板（带原句 source_excerpt）+ 高级词。
"""
from __future__ import annotations

from ._json import call_json

_SYSTEM = (
    "You are an IELTS writing analyst. Deconstruct the given essay into three parts:\n"
    "1) outline: each paragraph's rhetorical function (intro/position/body-argument/"
    "concession/conclusion, etc.).\n"
    "2) sentence_frames: 3-6 of the most impressive sentences ABSTRACTED into reusable "
    "templates with placeholders (X, Y), each paired with the original source_excerpt.\n"
    "3) vocab: advanced words/collocations worth learning, each with a short usage note.\n"
    "Respond with ONLY a JSON object: "
    '{"outline":[{"para":int,"function":str}],'
    '"sentence_frames":[{"frame":str,"source_excerpt":str}],'
    '"vocab":[{"word":str,"note":str}]}.'
)


def deconstruct_article(article: str) -> dict:
    """返回 {"outline":[...], "sentence_frames":[...], "vocab":[...]}。"""
    human = f"Essay to deconstruct:\n{article}\n\nReturn the deconstruction as JSON only."
    return call_json("deconstruct_article", _SYSTEM, human)
