"""agentic 助手图：把工具包成 LangChain tool，交给 create_react_agent 自主选用。

ReAct 循环：LLM 看 user 消息 + 各 tool 的 name/description/参数 schema → 选 tool（function
calling）→ 图执行 → 结果作为 ToolMessage 回灌 → LLM 再决策，直到给最终答复。
**工具描述质量决定选择质量**：DeepSeek 全靠下面 docstring 判断该用哪个，故写清「何时用/吃什么/产出什么」。
"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..db import library
from ..llm import get_llm
from ..tools.deconstruct import deconstruct_article as _deconstruct
from ..tools.dictionary import dictionary_lookup as _dict
from ..tools.exemplar import exemplar_provide as _exemplar
from ..tools.grammar import grammar_check as _grammar
from ..tools.score import score_predict as _score
from ..tools.vocab import vocab_upgrade as _vocab


@tool
def vocab_upgrade(word: str, sentence: str) -> dict:
    """升级/替换句中某个词。当用户想把一句话里的某个具体词换得更高级、更地道时用。
    必须提供该词所在的整句 sentence。返回 3-5 个语境贴合的替换，各带语域/band 暗示/误用陷阱。"""
    return _vocab(word, sentence)


@tool
def deconstruct_article(article: str) -> dict:
    """拆解/分析用户提供的一整篇文章或作文。当用户想「拆解这篇」「分析结构」「学这篇的写法」时用。
    返回逆向提纲（每段功能）+ 可迁移句式模板 + 高级词。注意：这是抽取已有文章，不是生成新范文。"""
    return _deconstruct(article)


@tool
def grammar_check(text: str) -> dict:
    """检查一段文本的语法/拼写/标点错误。当用户想「检查语法」「看看有没有错」时用。
    返回每个错误的定位、解释、修改建议。"""
    return _grammar(text)


@tool
def dictionary_lookup(word: str) -> dict:
    """查一个单词的释义和例句。当用户问「X 什么意思」「查一下 X」「X 怎么用」时用。返回释义列表 + 例句。"""
    return _dict(word)


@tool
def exemplar_provide(topic: str, task_type: int, band: float = 8.0) -> dict:
    """针对一个题目生成范文。当用户想「给我一篇范文」「示范怎么写这题」时用（这是生成新文，与 deconstruct 相反）。
    task_type 是 1 或 2。返回范文 + 提纲 + 亮点句 + 高级词。"""
    return _exemplar(topic, task_type, band)


@tool
def score_predict(essay: str, task_type: int, prompt: str = "") -> dict:
    """给一篇作文打雅思四维分（TA/CC/LR/GRA + overall）。当用户想「打个分」「评估我的作文」时用。
    task_type 是 1 或 2，prompt 是题目（可选）。返回四维 band + 每维依据 + overall。"""
    return _score(essay, task_type, prompt)


@tool
def save_vocab_entry(word: str, context_sentence: str, note: str = "") -> dict:
    """把一个词条存进用户的私有【词库】(SQLite)。当用户说「存到词库」「记下这个词」时用。"""
    return {"saved_id": library.save_vocab(word, context_sentence, [], note), "target": "vocab"}


@tool
def save_material_entry(content: str, kind: str = "exemplar", topic: str = "") -> dict:
    """把一段写作素材（范文/句式模板等）存进用户的私有【素材库】(SQLite)。当用户说「存到素材库」「收藏这篇」时用。
    kind 可为 'exemplar' | 'sentence_frame' | 'vocab'。"""
    return {"saved_id": library.save_material(kind, content, topic=topic), "target": "material"}


TOOLS = [vocab_upgrade, deconstruct_article, grammar_check, dictionary_lookup,
         exemplar_provide, score_predict, save_vocab_entry, save_material_entry]

_SYSTEM = (
    "You are an IELTS writing assistant. You have tools for vocabulary upgrade, article "
    "deconstruction, grammar checking, dictionary lookup, model-essay generation, essay "
    "scoring, and saving to the user's private libraries. For each user message, pick the "
    "single most appropriate tool; if none fits, answer directly. NEVER guess IELTS band "
    "scores yourself — always use the score_predict tool for any scoring. Reply in the "
    "user's language (Chinese if they write Chinese)."
)


def build_assistant():
    """返回编译好的 ReAct agent（tool-calling 对话图）。"""
    # 选 tool 的 LLM：flash + temperature=0，让路由更稳定可复现。
    return create_react_agent(get_llm("flash", temperature=0), TOOLS, prompt=_SYSTEM)
