"""vocab_upgrade：吃整句上下文的词汇升级工具。

关键：必须吃进**整句**，否则退化成同义词词典。要敢标误用陷阱
（illustrate 偏「举例」、delineate 偏「勾勒」），这正是它高于 thesaurus 套壳的地方。
"""
from __future__ import annotations

from ._json import call_json

_SYSTEM = (
    "You are an IELTS vocabulary coach. Given a TARGET word and the FULL sentence it "
    "appears in, propose 3-5 upgrade candidates that genuinely fit THIS sentence's meaning "
    "and context. For each, give: register (formal/neutral/informal), band_hint (e.g. "
    "'band 7+'), and trap (a collocation or misuse warning, else empty). Flag misuse traps "
    "explicitly — e.g. 'illustrate' emphasises giving examples while 'delineate' emphasises "
    "outlining — so the learner does not swap blindly. Respond with ONLY a JSON object: "
    '{"alternatives":[{"word":str,"register":str,"band_hint":str,"trap":str}]}.'
)


def vocab_upgrade(word: str, sentence: str) -> dict:
    """返回 {"alternatives": [{word, register, band_hint, trap}]}（3–5 个）。"""
    human = (f"Target word: {word}\nFull sentence: {sentence}\n"
             f"Propose 3-5 context-appropriate upgrades for '{word}'. JSON only.")
    return call_json("vocab_upgrade", _SYSTEM, human)
