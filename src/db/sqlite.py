"""SQLite 连接与初始化。

约定：连接用 sqlite3.Row 工厂，让查询结果能像 dict 一样按列名取值，
读写代码更清晰。schema 放在同目录的 schema.sql，init_db 一次性执行。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .. import config

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_conn(db_path: str | Path | None = None) -> sqlite3.Connection:
    """打开一个连接。默认用 config.SQLITE_PATH，测试可传临时路径。"""
    path = Path(db_path) if db_path is not None else config.SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # 结果按列名访问
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    """执行 schema.sql 建表（幂等，可重复调用）。

    注意：sqlite3 的 `with conn` 只提交/回滚事务，并不关闭连接；
    Windows 下不关闭会锁住文件。所以这里 try/finally 显式 close。
    """
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_conn(db_path)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()
