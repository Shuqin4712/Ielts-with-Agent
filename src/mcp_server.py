"""MCP server：把写作工具层暴露成标准 Model Context Protocol 服务。

这是同一套 `src/tools/` 纯函数的**第三种宿主**——继 CLI REPL（scripts/assistant.py）、
Web（src/api/app.py）之后，再加一层 MCP，让 Claude Desktop / Cursor 等任意 MCP 客户端
都能直接调用本项目的打分/查词/语法/词汇/拆解/范文能力。**没有重写任何智能**：
薄薄一层 FastMCP 适配器包住已有纯函数，正是两层工具架构（纯函数 + 框架适配）可复用性的证据。

边界（刻意为之）：只暴露**无状态分析工具**。写库工具（save_vocab/save_material）需要 user_id
且写用户私有库，不适合 MCP 的匿名宿主，故不在此暴露。

运行（stdio 传输，Claude Desktop 默认）：
    python -m src.mcp_server

Claude Desktop 配置（claude_desktop_config.json）示例见 README。
注意：score_predict / dictionary_lookup 等会调 DeepSeek，故服务进程需能读到 .env 的
DEEPSEEK_API_KEY（config 在 import 期加载 .env）；且 API key 花的是**本服务持有者**的额度。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .tools.deconstruct import deconstruct_article as _deconstruct
from .tools.dictionary import dictionary_lookup as _dict
from .tools.exemplar import exemplar_provide as _exemplar
from .tools.grammar import grammar_check as _grammar
from .tools.score import score_predict as _score
from .tools.vocab import vocab_upgrade as _vocab

mcp = FastMCP("ielts-writing-agent")


@mcp.tool()
def score_predict(essay: str, task_type: int, prompt: str = "") -> dict:
    """给一篇雅思作文打四维分（TA/CC/LR/GRA + overall）。想「打个分」「评估我的作文」时用。
    task_type 是 1 或 2，prompt 是题目（可选）。返回四维 band + 每维依据 + overall。
    复用批改图那条被评测量化过的唯一打分管道（锚定开 / 池内向量选锚）。"""
    return _score(essay, task_type, prompt)


@mcp.tool()
def dictionary_lookup(word: str) -> dict:
    """查一个单词的释义和例句。问「X 什么意思」「查一下 X」「X 怎么用」时用。返回释义列表 + 例句。"""
    return _dict(word)


@mcp.tool()
def grammar_check(text: str) -> dict:
    """检查一段文本的语法/拼写/标点错误。想「检查语法」「看看有没有错」时用。
    返回每个错误的定位、解释、修改建议。"""
    return _grammar(text)


@mcp.tool()
def vocab_upgrade(word: str, sentence: str) -> dict:
    """升级/替换句中某个词，让表达更高级地道。必须提供该词所在的整句 sentence。
    返回 3-5 个语境贴合的替换，各带语域/band 暗示/误用陷阱。"""
    return _vocab(word, sentence)


@mcp.tool()
def deconstruct_article(article: str) -> dict:
    """拆解/分析一整篇文章：逆向提纲（每段功能）+ 可迁移句式模板 + 高级词。
    想「拆解这篇」「分析结构」「学这篇写法」时用。注意：这是抽取已有文章，不是生成新范文。"""
    return _deconstruct(article)


@mcp.tool()
def exemplar_provide(topic: str, task_type: int, band: float = 8.0) -> dict:
    """针对一个题目生成范文（这是生成新文，与 deconstruct 相反）。
    task_type 是 1 或 2。返回范文 + 提纲 + 亮点句 + 高级词。"""
    return _exemplar(topic, task_type, band)


def main() -> None:
    """stdio 传输启动（Claude Desktop / Cursor 等 MCP 客户端默认走 stdio）。"""
    mcp.run()


if __name__ == "__main__":
    main()
