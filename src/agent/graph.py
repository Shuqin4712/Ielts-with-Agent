"""agentic 助手图：把工具包成 LangChain tool，交给 create_react_agent 自主选用。

ReAct 循环：LLM 看 user 消息 + 各 tool 的 name/description/参数 schema → 选 tool（function
calling）→ 图执行 → 结果作为 ToolMessage 回灌 → LLM 再决策，直到给最终答复。
**工具描述质量决定选择质量**：DeepSeek 全靠下面 docstring 判断该用哪个，故写清「何时用/吃什么/产出什么」。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..db import library
from ..llm import get_llm, with_backoff
from ..obs import operation
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


def _uid(config: RunnableConfig) -> str:
    """从运行时 config 取当前用户（/chat 端点注入；CLI 未注入则回落 default）。"""
    return (config.get("configurable") or {}).get("user_id") or "default"


@tool
def save_vocab_entry(word: str, context_sentence: str, note: str = "",
                     pos: str = "", zh_def: str = "", en_def: str = "",
                     ipa: str = "", *, config: RunnableConfig) -> dict:
    """把一个词条存进用户的私有【词库】(SQLite)。当用户说「存到词库」「记下这个词」时用。
    尽量填全生词本字段：pos 词性（如 'v.'）、zh_def 中文释义、en_def 英文释义、ipa 音标
    ——刚查过词/升级过词时你已经知道这些，不要留空。一次只存一个词；多个词就多次调用。"""
    sid = library.save_vocab(word, context_sentence, [], note,
                             pos=pos or None, zh_def=zh_def or None,
                             en_def=en_def or None, ipa=ipa or None,
                             user_id=_uid(config))
    return {"saved_id": sid, "target": "vocab"}


@tool
def save_material_entry(content: str, kind: str, note: str = "",
                        source_excerpt: str = "", topic: str = "",
                        band: float | None = None, *,
                        config: RunnableConfig) -> dict:
    """把【一条】写作素材存进用户的私有素材库 (SQLite)。当用户说「存到素材库」「收藏这些」时用。

    ⚠️ 粒度规则（必须遵守）：content 只放**单个条目本体**——一个高级词、一个句式模板、
    一条短语、一份提纲或一篇范文正文。讲解/用法放 note，出处原句放 source_excerpt。
    **禁止把你的整段回复原文塞进 content**。回复里有多个素材时，逐条多次调用本工具，
    每条选对 kind：
      advanced_vocab=高级词汇 | synonym=同义替换 | phrase=短语 |
      sentence_frame=句式模板（含 X/Y 占位符）| outline=思路提纲 | exemplar=范文
    """
    return {"saved_id": library.save_material(
        kind, content, note=note or None, source_excerpt=source_excerpt or None,
        topic=topic or None, band=band, user_id=_uid(config)), "target": "material"}


TOOLS = [vocab_upgrade, deconstruct_article, grammar_check, dictionary_lookup,
         exemplar_provide, score_predict, save_vocab_entry, save_material_entry]

_SYSTEM = (
    "You are an IELTS writing assistant. You have tools for vocabulary upgrade, article "
    "deconstruction, grammar checking, dictionary lookup, model-essay generation, essay "
    "scoring, and saving to the user's private libraries. For each user message, use the "
    "appropriate tool(s) — chain multiple tool calls when the request takes several steps "
    "(e.g. look up a word, then save it to the vocab library); if none fits, answer "
    "directly. NEVER guess IELTS band "
    "scores yourself — always use the score_predict tool for any scoring. "
    "When saving materials: split your answer into individual items and call "
    "save_material_entry once per item with the right kind — NEVER dump a whole reply "
    "into one entry. When saving vocabulary you already looked up or upgraded, include "
    "pos/zh_def/ipa so the user's wordbook card is complete. Reply in the "
    "user's language (Chinese if they write Chinese)."
)


# ── 滚动摘要（context engineering）─────────────────────────────────
# 长会话里历史无限膨胀 → 每轮喂给 LLM 的 input token 线性涨、变慢变贵。
# 对策：消息数超阈值时，把**较旧的轮次**压成一条摘要 SystemMessage + 保留最近若干条原文，
# 只改「喂给 LLM 的视图」（llm_input_messages，不落 checkpointer——完整历史仍在）。
_SUMMARY_TRIGGER = 12   # 消息数超过它才开始摘要（短会话零额外成本）
_KEEP_TAIL = 6          # 至少保留最近多少条原文


def _safe_cut(msgs: list, keep_tail: int) -> int:
    """从「保留尾部 keep_tail 条」的位置往前退到最近一条 HumanMessage，作为切点。
    保证 recent 从一个用户回合开头起，不切断 AI tool_calls ↔ ToolMessage 配对
    （否则 create_react_agent 的历史校验会报错）。"""
    i = max(0, len(msgs) - keep_tail)
    while i > 0 and not isinstance(msgs[i], HumanMessage):
        i -= 1
    return i


def _render(m) -> str:
    role = getattr(m, "type", "msg")
    return f"[{role}] {getattr(m, 'content', '')}"


def _summarize(older: list) -> str:
    llm = get_llm("flash", temperature=0)
    transcript = "\n".join(_render(m) for m in older)
    with operation("summarize"):     # obs 里单独归类，SSE 也靠 node 过滤不外泄
        text = with_backoff(llm.invoke)([
            SystemMessage(
                "Summarize this IELTS tutoring conversation so far into a compact Chinese "
                "paragraph: what the student asked, what was provided (words/rewrites/scores), "
                "and anything saved to their library. Under 120 words. No preamble."),
            HumanMessage(transcript)]).content
    return text.strip()


def _summarize_hook(state: dict) -> dict:
    """create_react_agent 的 pre_model_hook：超阈值就摘要旧轮、保留近轮。
    返回 llm_input_messages（只影响这次喂 LLM 的消息，不改 checkpointer 里的历史）。"""
    msgs = state["messages"]
    if len(msgs) <= _SUMMARY_TRIGGER:
        return {}                     # 未超阈值：回退用完整 messages（无额外调用）
    cut = _safe_cut(msgs, _KEEP_TAIL)
    older, recent = msgs[:cut], msgs[cut:]
    if not older:
        return {}
    summary = SystemMessage(f"（对话前情摘要）{_summarize(older)}")
    return {"llm_input_messages": [summary, *recent]}


def build_assistant(*, checkpointer=None):
    """返回编译好的 ReAct agent（tool-calling 对话图）。

    checkpointer：短期记忆（session 内多轮上下文 + 断点续跑）。传入后，调用时
    带 config={"configurable": {"thread_id": ...}} 即按线程隔离并自动接续上文；
    此时每轮只需喂**新消息**，历史由 checkpointer 恢复，无需手动累加。
    None 时退化为无记忆的单轮 agent（阶段 3 行为，测试可用）。

    pre_model_hook=_summarize_hook：长会话滚动摘要，压平 input token 增长。
    """
    # 选 tool 的 LLM：flash + temperature=0，让路由更稳定可复现。
    # max_retries=3：这个 llm 由 create_react_agent 内部 invoke，业务侧没有
    # 插桩点套 with_backoff（其余直调 invoke 的地方都套了）。所以这里破例放开
    # 客户端重试来兜 429——它不会与 with_backoff 叠乘，因为这条路径压根没有
    # 那一层。别把它复制到别处：默认 max_retries=0 是为了避免双层重试相乘。
    return create_react_agent(get_llm("flash", temperature=0, max_retries=3), TOOLS,
                              prompt=_SYSTEM, checkpointer=checkpointer,
                              pre_model_hook=_summarize_hook)
