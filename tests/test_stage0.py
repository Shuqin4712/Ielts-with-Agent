"""阶段 0 数据地基的断言测试。

依赖已构建好的 SQLite + ChromaDB（先跑 python scripts/build_stage0.py）。
若索引未构建，相关用例自动 skip，不误报。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.db import sqlite
from src.rag import store
from src.rag.rubric import parse_rubric
from src.data.topic import infer_topic
from src.data.gold_extract import extract_book
from scripts.extract_gold import TEXT_LAYER_BOOKS


def test_rubric_parser_complete():
    """rubric 解析出 80 块（band 0-9 × 4 维 × 2 task），无空块。"""
    chunks = parse_rubric()
    assert len(chunks) == 80
    assert all(c.text.strip() for c in chunks)


def test_topic_rule():
    assert infer_topic("Many people believe that protecting the environment "
                       "is the responsibility of governments.") == "environment"


def test_gold_extract_all_books():
    """9 本文本层书各抽出 8 块（4 test × 2 task），sample 必有 band。"""
    for bn, fn in TEXT_LAYER_BOOKS.items():
        pdf = config.RAW_DIR / fn
        if not pdf.exists():
            continue
        recs = extract_book(pdf)
        assert len(recs) == 8, f"剑{bn} 抽出 {len(recs)} 块，期望 8"
        assert all(r.task_type in (1, 2) for r in recs)
        for r in recs:
            if r.answer_type == "sample":
                assert r.band is not None, f"剑{bn} sample 缺 band"


@pytest.mark.skipif(not config.SQLITE_PATH.exists(), reason="未构建 SQLite，先跑 build_stage0")
def test_sqlite_splits_present():
    conn = sqlite.get_conn()
    try:
        rows = dict(conn.execute(
            "SELECT split, COUNT(*) FROM essays GROUP BY split").fetchall())
    finally:
        conn.close()
    assert rows.get("exemplar", 0) > 0
    assert rows.get("holdout", 0) > 0
    assert rows.get("train", 0) > 0


@pytest.mark.skipif(not config.CHROMA_DIR.exists(), reason="未构建 ChromaDB，先跑 build_stage0")
def test_exemplar_filtered_retrieval():
    """阶段 0 验收：按 (task2, band7, environment) 能召回到范文。"""
    res = store.search(
        config.COLL_EXEMPLAR,
        query="protecting the environment from pollution",
        where={"task_type": 2, "band": 7.0, "topic": "environment"},
        n=3,
    )
    assert len(res) >= 1
    for r in res:
        assert r["metadata"]["task_type"] == 2
        assert r["metadata"]["band"] == 7.0
        assert r["metadata"]["topic"] == "environment"
