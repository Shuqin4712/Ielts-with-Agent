"""阶段 5 · FastAPI 后端：把现有 LangGraph 图 / 工具 / 库 CRUD 包成 HTTP 端点。

设计原则（护栏）：
- **薄后端胶水**：所有智能都在既有的图/工具/记忆里，这层只做「HTTP ↔ 图/函数」的转接，
  不重写任何打分/对话/记忆逻辑。
- **单一真相源**：批改复用 build_grading_session_graph、对话复用 build_assistant，
  与 CLI 走完全相同的代码路径（不另起炉灶）。
- **API key 只在后端 .env**：前端永远拿不到 key，只拿到结果。

端点：
  POST   /grade       作文 → 批改外层图 → 结构化 band + 反馈
  POST   /chat        (SSE) → assistant 对话图，逐 token 流式 + 本轮工具
  GET    /lookup      查词
  GET    /vocab       列词库    POST /vocab 存词    DELETE /vocab/{id} 删词
  GET    /materials   列素材库  POST /materials 存素材  DELETE /materials/{id} 删

对话/批改的会话状态由 checkpointer 按 thread_id 维持（短期记忆，跨请求接续）。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

from .. import config
from ..agent.graph import build_assistant
from ..db import library
from ..graph.session import build_grading_session_graph
from ..memory.checkpoint import get_checkpointer
from ..obs import operation, wrap_operation
from ..tools.dictionary import dictionary_lookup

# ── 前端静态目录 ──────────────────────────────────────────────────────
FRONTEND_DIR = config.PROJECT_ROOT / "frontend"

app = FastAPI(title="IELTS Writing Agent", version="0.5")

# 开发期放开 CORS（同源提供静态文件时其实用不上；留着方便 file:// 或独立起前端调试）。
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── 进程级单例：图与 checkpointer 建一次复用 ──────────────────────────
# checkpointer 落 data/checkpoints.sqlite，批改与对话共用（靠 thread_id 前缀区分会话）。
_checkpointer = get_checkpointer()
_grading_graph = build_grading_session_graph(checkpointer=_checkpointer)
_assistant = build_assistant(checkpointer=_checkpointer)


# ── 请求体模型 ────────────────────────────────────────────────────────
class GradeReq(BaseModel):
    user_id: str = "demo"
    essay: str
    task_type: int = 2
    prompt: str = ""


class ChatReq(BaseModel):
    user_id: str = "demo"
    conversation_id: str            # = thread_id，前端每个会话生成一个
    message: str


class VocabReq(BaseModel):
    user_id: str = "demo"
    word: str
    context_sentence: str = ""
    alternatives: list = []
    nuance_note: str = ""
    # v1.1 生词本字段（全部可空，旧客户端不传也能存）
    pos: str | None = None
    zh_def: str | None = None
    en_def: str | None = None
    ipa: str | None = None
    examples: list = []


class MaterialReq(BaseModel):
    user_id: str = "demo"
    # v1.1 枚举：advanced_vocab | synonym | phrase | sentence_frame | outline | exemplar
    type: str = "exemplar"
    content: str
    outline: object | None = None
    topic: str | None = None
    band: float | None = None
    tags: object | None = None
    source_excerpt: str | None = None
    note: str | None = None         # v1.1 讲解/用法


# ── 全局异常兜底：坏输入/内部错误都回结构化 JSON，前端好提示、不白屏 ──
@app.exception_handler(Exception)
async def _unhandled(_request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ── 批改 ──────────────────────────────────────────────────────────────
@app.post("/grade")
def grade(req: GradeReq):
    """走批改外层图（load_profile → grade → feedback → memory_write）。

    thread_id = grade:<user_id>：同一用户续跑同一批改线程（与 grade_essay.py 一致）。
    打分逻辑物理隔离在内层纯打分图，profile 只影响反馈措辞、绝不改分。
    """
    if not req.essay.strip():
        raise HTTPException(400, "essay 不能为空")
    with operation("grade"):
        final = _grading_graph.invoke(
            {"user_id": req.user_id, "essay": req.essay, "task_type": req.task_type,
             "prompt": req.prompt or "", "essay_id": None, "personalize": True},
            {"configurable": {"thread_id": f"grade:{req.user_id}"}},
        )
    return {
        "dimension_scores": final["dimension_scores"],   # {crit: {band, evidence}}
        "overall_band": final["overall_band"],
        "feedback": final.get("feedback", ""),
    }


# ── 对话（SSE 流式）─────────────────────────────────────────────────────
def _sse(payload: dict) -> str:
    """一条 SSE 帧：data 行 + 空行分隔。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chat_stream(req: ChatReq):
    """生成器：逐 token 吐助手回复；结束时补一帧本轮调用的工具，再吐 done。

    为何流式：对话是逐 token 生成的慢过程，SSE 让前端「打字机」式增量渲染，
    首字延迟从数秒降到几百毫秒——体验上从「卡住」变「在思考」。
    """
    # user_id 一并注入 config：save_* 工具经 RunnableConfig 取它落库（不进 LLM schema）
    cfg = {"configurable": {"thread_id": req.conversation_id, "user_id": req.user_id}}
    try:
        # stream_mode="messages"：拿到 LLM 的 token 级增量（AIMessageChunk）。
        # ⚠️ 只取 langgraph_node=="agent" 的 token：工具内部也会调 LLM 生成 JSON
        # （如 dictionary_lookup 的 call_json），那些跑在 "tools" 节点，绝不能流给用户。
        # （操作标签由外层 wrap_operation("chat", ...) 按迭代打，不能在生成器体内
        #   用 with operation()——跨 yield 的 contextvar token 会在别的线程上下文失效。）
        for msg, meta in _assistant.stream(
            {"messages": [HumanMessage(req.message)]}, cfg, stream_mode="messages"
        ):
            if (meta.get("langgraph_node") == "agent"
                    and isinstance(msg, AIMessageChunk)
                    and isinstance(msg.content, str) and msg.content):
                yield _sse({"type": "token", "text": msg.content})

        # 本轮调用了哪些工具：从最终 state 里抽（最后一条 Human 之后的 tool_calls）。
        tools = _tools_this_turn(cfg)
        if tools:
            yield _sse({"type": "tools", "tools": tools})
        yield _sse({"type": "done"})
    except Exception as exc:                       # 流已经开头，只能在流里报错
        yield _sse({"type": "error", "message": str(exc)})


def _tools_this_turn(cfg: dict) -> list[str]:
    """从 checkpointer 里的最新消息抽本轮（最后一条 HumanMessage 之后）调用的工具名。"""
    msgs = _assistant.get_state(cfg).values.get("messages", [])
    start = 0
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            start = i
            break
    return [tc["name"] for m in msgs[start:]
            for tc in (getattr(m, "tool_calls", None) or [])]


@app.post("/chat")
def chat(req: ChatReq):
    if not req.message.strip():
        raise HTTPException(400, "message 不能为空")
    return StreamingResponse(wrap_operation("chat", _chat_stream(req)),
                             media_type="text/event-stream")


# ── 查词 ──────────────────────────────────────────────────────────────
@app.get("/lookup")
def lookup(word: str):
    if not word.strip():
        raise HTTPException(400, "word 不能为空")
    return dictionary_lookup(word.strip())


# ── 词库 CRUD ─────────────────────────────────────────────────────────
@app.get("/vocab")
def get_vocab(user_id: str = "demo"):
    return {"items": library.list_vocab(user_id)}


@app.post("/vocab")
def post_vocab(req: VocabReq):
    sid = library.save_vocab(
        req.word, req.context_sentence, req.alternatives, req.nuance_note,
        user_id=req.user_id, pos=req.pos, zh_def=req.zh_def, en_def=req.en_def,
        ipa=req.ipa, examples=req.examples or None)
    return {"saved_id": sid}


@app.delete("/vocab/{row_id}")
def del_vocab(row_id: int):
    return {"deleted": library.delete_vocab(row_id)}


# ── 素材库 CRUD ───────────────────────────────────────────────────────
@app.get("/materials")
def get_materials(user_id: str = "demo"):
    return {"items": library.list_material(user_id)}


@app.post("/materials")
def post_materials(req: MaterialReq):
    sid = library.save_material(
        req.type, req.content, user_id=req.user_id, outline=req.outline,
        topic=req.topic, band=req.band, tags=req.tags,
        source_excerpt=req.source_excerpt, note=req.note)
    return {"saved_id": sid}


@app.delete("/materials/{row_id}")
def del_materials(row_id: int):
    return {"deleted": library.delete_material(row_id)}


# ── 静态前端（放最后，避免吃掉上面的 API 路由）──────────────────────────
if FRONTEND_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
