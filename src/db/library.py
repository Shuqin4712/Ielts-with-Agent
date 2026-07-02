"""用户私有库（词库 / 素材库）的 schema 迁移 + CRUD。

阶段 0 已建 vocab_library / material_library；本模块补 material 的 source_excerpt 列
（SQLite 无 ADD COLUMN IF NOT EXISTS，故 pragma 查缺再 ALTER），并提供增删查。
list/write 的默认 user_id='default'（单用户 demo，用户系统不在 scope）。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .sqlite import get_conn, init_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonify(v):
    """list/dict → JSON 字符串；其余原样（None/str/数字）。"""
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v


def ensure_library_schema(db_path: str | Path | None = None) -> None:
    """确保库表存在，且 material_library 有 source_excerpt 列（幂等）。"""
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(material_library)")}
        if "source_excerpt" not in cols:
            conn.execute("ALTER TABLE material_library ADD COLUMN source_excerpt TEXT")
            conn.commit()
    finally:
        conn.close()


def save_vocab(word: str, context_sentence: str, alternatives, nuance_note: str = "",
               *, user_id: str = "default", source_essay_id: int | None = None,
               db_path=None) -> int:
    ensure_library_schema(db_path)
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO vocab_library (user_id, word, context_sentence, alternatives, "
            "nuance_note, source_essay_id, created_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, word, context_sentence, _jsonify(alternatives), nuance_note,
             source_essay_id, _now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def save_material(mtype: str, content: str, *, user_id: str = "default",
                  outline=None, topic: str | None = None, band: float | None = None,
                  tags=None, source_excerpt: str | None = None, db_path=None) -> int:
    """mtype: 'exemplar' | 'sentence_frame' | 'vocab' 等。"""
    ensure_library_schema(db_path)
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO material_library (user_id, type, content, outline, topic, band, "
            "tags, source_excerpt, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, mtype, content, _jsonify(outline), topic, band,
             _jsonify(tags), source_excerpt, _now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_vocab(user_id: str = "default", db_path=None) -> list[dict]:
    ensure_library_schema(db_path)
    conn = get_conn(db_path)
    try:
        rows = conn.execute("SELECT * FROM vocab_library WHERE user_id=? ORDER BY id",
                            (user_id,)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def list_material(user_id: str = "default", db_path=None) -> list[dict]:
    ensure_library_schema(db_path)
    conn = get_conn(db_path)
    try:
        rows = conn.execute("SELECT * FROM material_library WHERE user_id=? ORDER BY id",
                            (user_id,)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _delete(table: str, row_id: int, db_path=None) -> int:
    conn = get_conn(db_path)
    try:
        cur = conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete_vocab(row_id: int, db_path=None) -> int:
    return _delete("vocab_library", row_id, db_path)


def delete_material(row_id: int, db_path=None) -> int:
    return _delete("material_library", row_id, db_path)
