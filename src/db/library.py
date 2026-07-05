"""用户私有库（词库 / 素材库）的 schema 迁移 + CRUD。

阶段 0 已建 vocab_library / material_library；本模块补 material 的 source_excerpt 列
（SQLite 无 ADD COLUMN IF NOT EXISTS，故 pragma 查缺再 ALTER），并提供增删查。
list/write 的默认 user_id='default'（单用户 demo，用户系统不在 scope）。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .sqlite import get_conn, init_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonify(v):
    """list/dict → JSON 字符串；其余原样（None/str/数字）。"""
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v


# 运行时迁移声明：表 → 需存在的可空列（SQLite 无 ADD COLUMN IF NOT EXISTS，
# 用 pragma 查缺再 ALTER，幂等；旧数据保留、新列为 NULL，前端优雅降级）。
_MIGRATIONS: dict[str, dict[str, str]] = {
    # note = 讲解/用法说明（v1.1，与 content 条目本体分离）
    "material_library": {"source_excerpt": "TEXT", "note": "TEXT"},
    # v1.1 生词本字段：词性/中英释义/音标/双语例句（JSON）
    "vocab_library": {"pos": "TEXT", "zh_def": "TEXT", "en_def": "TEXT",
                      "ipa": "TEXT", "examples": "TEXT"},
}

# 素材库 type 枚举（v1.1 钉死；旧数据里的历史值保留不迁移，前端显示原文）
MATERIAL_TYPES = ("advanced_vocab", "synonym", "phrase",
                  "sentence_frame", "outline", "exemplar")


def ensure_library_schema(db_path: str | Path | None = None) -> None:
    """确保库表存在且含 _MIGRATIONS 声明的列（幂等）。"""
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        for table, wanted in _MIGRATIONS.items():
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for col, coltype in wanted.items():
                if col not in cols:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                    except sqlite3.OperationalError as e:
                        # 并发调用（agent 并行存多条）会撞「已被别人 ALTER」——幂等忽略
                        if "duplicate column" not in str(e).lower():
                            raise
        conn.commit()
    finally:
        conn.close()


def save_vocab(word: str, context_sentence: str, alternatives, nuance_note: str = "",
               *, user_id: str = "default", source_essay_id: int | None = None,
               pos: str | None = None, zh_def: str | None = None,
               en_def: str | None = None, ipa: str | None = None,
               examples=None, db_path=None) -> int:
    """v1.1 新字段全部 keyword-only 带默认（向后兼容旧调用点）。examples 为
    [{"en","zh"}] 列表，JSON 落库。"""
    ensure_library_schema(db_path)
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO vocab_library (user_id, word, context_sentence, alternatives, "
            "nuance_note, source_essay_id, pos, zh_def, en_def, ipa, examples, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (user_id, word, context_sentence, _jsonify(alternatives), nuance_note,
             source_essay_id, pos, zh_def, en_def, ipa, _jsonify(examples), _now()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def save_material(mtype: str, content: str, *, user_id: str = "default",
                  outline=None, topic: str | None = None, band: float | None = None,
                  tags=None, source_excerpt: str | None = None,
                  note: str | None = None, db_path=None) -> int:
    """mtype 取 MATERIAL_TYPES 之一；content 是**单个条目本体**（一个词/一个句式/
    一篇范文），note 是它的讲解/用法——不要把整段对话回复塞进 content。"""
    ensure_library_schema(db_path)
    conn = get_conn(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO material_library (user_id, type, content, outline, topic, band, "
            "tags, source_excerpt, note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (user_id, mtype, content, _jsonify(outline), topic, band,
             _jsonify(tags), source_excerpt, note, _now()))
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
