"""阶段 0 · 构建两个 ChromaDB 集合：rubric_descriptors + exemplar_essays。

- rubric：官方 band descriptors 的 (criterion, band, task) 块，打分时按维度+band 召回。
- exemplar：SQLite 里 split='exemplar' 的精选范文，按 (task, band, topic) 召回当锚点。
  注意只灌精选子集，不把全部范文塞进向量库（DESIGN §4.4：质量 > 数量）。
"""
from __future__ import annotations

from .. import config
from ..db import sqlite
from . import store
from .rubric import parse_rubric


def build_rubric() -> int:
    chunks = parse_rubric()
    coll = store.reset_collection(config.COLL_RUBRIC)
    ids = [f"rubric-t{c.task_type}-{c.criterion}-b{c.band}" for c in chunks]
    docs = [c.text for c in chunks]
    metas = [{"task_type": c.task_type, "criterion": c.criterion, "band": c.band}
             for c in chunks]
    store.add_documents(coll, ids, docs, metas)
    return len(chunks)


def build_exemplars(db_path=None) -> int:
    conn = sqlite.get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT id, task_type, prompt, body, overall_band, topic, tier, "
            "has_examiner_comment FROM essays WHERE split='exemplar'"
        ).fetchall()
    finally:
        conn.close()

    coll = store.reset_collection(config.COLL_EXEMPLAR)
    ids = [f"essay-{r['id']}" for r in rows]
    # 文档 = 题目 + 正文，让语义召回同时吃到题面与作答。
    docs = [f"{r['prompt']}\n\n{r['body']}" for r in rows]
    metas = [{
        "essay_id": r["id"],
        "task_type": r["task_type"],
        "band": float(r["overall_band"]),
        "topic": r["topic"],
        "tier": r["tier"],
        "has_examiner_comment": r["has_examiner_comment"],
    } for r in rows]
    store.add_documents(coll, ids, docs, metas)
    return len(rows)


if __name__ == "__main__":
    n_rubric = build_rubric()
    print(f"rubric_descriptors: {n_rubric} 块")
    n_ex = build_exemplars()
    print(f"exemplar_essays: {n_ex} 篇")
