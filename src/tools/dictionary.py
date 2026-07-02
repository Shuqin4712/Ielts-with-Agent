"""dictionary_lookup：查词——释义 + 例句。"""
from __future__ import annotations

from ._json import call_json

_SYSTEM = (
    "You are a learner's dictionary for IELTS students. For the given word, return concise "
    "definitions (one per distinct sense) and natural example sentences. Respond with ONLY "
    'a JSON object: {"word":str,"definitions":[str],"examples":[str]}.'
)


def dictionary_lookup(word: str) -> dict:
    """返回 {"word":..., "definitions":[...], "examples":[...]}。"""
    return call_json("dictionary_lookup", _SYSTEM,
                     f"Word: {word}\nReturn definitions and examples as JSON only.")
