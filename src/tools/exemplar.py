"""exemplar_provide：生成范文 + 思路 + 亮点句 + 高级词。

与 deconstruct_article 方向相反：这是**生成**（给题目造范文），那是**抽取**（拆用户文章）。
"""
from __future__ import annotations

from ._json import call_json

_SYSTEM = (
    "You are an IELTS writing tutor. Generate a model answer for the given topic at roughly "
    "the target band. Also give the planning outline (each paragraph's function), a few "
    "highlight sentences, and advanced vocabulary with usage notes. Respond with ONLY a JSON "
    'object: {"essay":str,"outline":[{"para":int,"function":str}],'
    '"highlight_sentences":[str],"advanced_vocab":[{"word":str,"note":str}]}.'
)


def exemplar_provide(topic: str, task_type: int, band: float = 8.0) -> dict:
    """返回 {"essay","outline","highlight_sentences","advanced_vocab"}。"""
    human = (f"IELTS Writing Task {task_type}. Topic/prompt: {topic}\n"
             f"Target band: {band}. Generate the model answer package. JSON only.")
    return call_json("exemplar_provide", _SYSTEM, human)
