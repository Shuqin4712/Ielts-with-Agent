"""dictionary_lookup：查词——音标 + 词性 + 中英释义 + 双语例句。

v1.1 升级：schema 从贫瘠的 definitions/examples 扩为面向中国考生的生词本字段
（pos/zh_def/en_def/ipa/双语例句），词库卡片直接消费这些字段。
"""
from __future__ import annotations

from ._json import call_json

_SYSTEM = (
    "You are a learner's dictionary for Chinese IELTS students. For the given word return:\n"
    "- ipa: IPA transcription (British), without slashes\n"
    "- pos: part(s) of speech, abbreviated (e.g. 'v.', 'n.', 'adj.'; join multiple with ' / ')\n"
    "- zh_def: concise Chinese definition(s); separate distinct senses with '；'\n"
    "- en_def: concise English definition (one sentence, learner-friendly)\n"
    "- examples: 1-2 natural example sentences, each with a Chinese translation\n"
    "Respond with ONLY a JSON object: "
    '{"word":str,"ipa":str,"pos":str,"zh_def":str,"en_def":str,'
    '"examples":[{"en":str,"zh":str}]}.'
)


def dictionary_lookup(word: str) -> dict:
    """返回 {"word","ipa","pos","zh_def","en_def","examples":[{"en","zh"}]}。"""
    return call_json("dictionary_lookup", _SYSTEM,
                     f"Word: {word}\nReturn the dictionary entry as JSON only.")
