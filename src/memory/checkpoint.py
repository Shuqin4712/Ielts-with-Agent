"""短期记忆：LangGraph checkpointer 工厂。

概念：checkpointer 在每个 super-step 后把整张 State 存成一个 checkpoint，
以 `thread_id` 为隔离键。同一 thread_id 续跑会自动加载上次 State（多轮上下文、
断点续跑）；不同 thread_id 互不可见（会话隔离）。这是「session 内」的短期记忆，
和长期学生画像（跨 session，落 student_profile 表）正交。

用法模式：把「用哪种 checkpointer」这个决策收口到这里。
  - 生产/CLI：SqliteSaver 落 data/checkpoints.sqlite，跨进程重启也能续。
  - 测试：InMemorySaver，无副作用、无需清理。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .. import config

# 短期记忆的持久化位置，与主库分开（主库放语料/画像，这里只放对话 checkpoint）。
CHECKPOINT_PATH = config.DATA_DIR / "checkpoints.sqlite"


def get_checkpointer(*, persist: bool = True, db_path: str | Path | None = None):
    """返回一个 checkpointer。

    persist=True  → SqliteSaver（默认，落盘、可跨进程续跑）。
    persist=False → InMemorySaver（测试用，进程内有效、无副作用）。

    注意：SqliteSaver 要在多线程/长驻场景用，连接必须 check_same_thread=False；
    首次用要 .setup() 建它自己的 checkpoint 表（幂等）。
    """
    if not persist:
        return InMemorySaver()

    path = Path(db_path) if db_path is not None else CHECKPOINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
