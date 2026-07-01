"""四项连通性 smoke test：验证四条底层管线各自能跑通。

两种跑法：
  - pytest tests/ -v          （CI / 标准）
  - python tests/test_connectivity.py   （直接跑，打印 ✓/✗ 清单）

每项互相独立：DeepSeek 缺 key 时 skip，不影响其余三项。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# 允许 `python tests/test_connectivity.py` 直接运行时能 import 到 src 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.db import sqlite
from src.rag import embeddings


# ── 1. DeepSeek（无 key 则跳过）────────────────────────────────────
@pytest.mark.skipif(not config.DEEPSEEK_API_KEY, reason="未设置 DEEPSEEK_API_KEY")
def test_deepseek_flash():
    from src.llm import get_llm

    llm = get_llm("flash")
    resp = llm.invoke("Reply with the single word: pong")
    assert resp.content and resp.content.strip(), "DeepSeek 返回了空内容"


# ── 2. 本地 embedding（bge-m3 via Ollama）──────────────────────────
def test_embedding():
    vec = embeddings.embed_text("Climate change is a pressing global issue.")
    assert isinstance(vec, list) and len(vec) > 0, "embedding 为空"
    assert all(isinstance(x, float) for x in vec[:5]), "embedding 不是 float 列表"


# ── 3. ChromaDB 写入 + 检索 ────────────────────────────────────────
def test_chroma_roundtrip():
    import chromadb

    vec = embeddings.embed_text("A band 7 essay about urban transport.")
    # 用内存 client，跑完即弃，不污染真实持久化目录。
    client = chromadb.EphemeralClient()
    coll = client.create_collection("smoke")
    coll.add(ids=["doc1"], embeddings=[vec], documents=["urban transport essay"],
             metadatas=[{"band": 7}])
    res = coll.query(query_embeddings=[vec], n_results=1)
    assert res["ids"][0][0] == "doc1", "未检索回刚写入的文档"


# ── 4. SQLite 建表 + 写读 ──────────────────────────────────────────
def test_sqlite_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "smoke.sqlite"
        sqlite.init_db(db)
        # 显式 close，确保 Windows 下临时目录清理前文件句柄已释放。
        conn = sqlite.get_conn(db)
        try:
            conn.execute(
                "INSERT INTO essays (task_type, prompt, body, tier) VALUES (?,?,?,?)",
                (2, "Some people think...", "In recent years...", "gold"),
            )
            conn.commit()
            row = conn.execute("SELECT task_type, tier FROM essays").fetchone()
        finally:
            conn.close()
        assert row["task_type"] == 2 and row["tier"] == "gold"


# ── 直接运行时的友好清单 ──────────────────────────────────────────
def _run(name: str, fn) -> bool:
    try:
        fn()
    except pytest.skip.Exception as e:  # skipif 抛的异常
        print(f"  ~ {name}: SKIPPED（{e}）")
        return True
    except Exception as e:
        print(f"  x {name}: FAILED — {type(e).__name__}: {e}")
        return False
    print(f"  ✓ {name}: OK")
    return True


if __name__ == "__main__":
    print("连通性 smoke test：")
    ok = True
    ok &= _run("DeepSeek (v4-flash)", test_deepseek_flash)
    ok &= _run(f"Embedding ({config.EMBED_MODEL})", test_embedding)
    ok &= _run("ChromaDB", test_chroma_roundtrip)
    ok &= _run("SQLite", test_sqlite_roundtrip)
    print("\n全部通过 ✅" if ok else "\n有失败项 ❌")
    sys.exit(0 if ok else 1)
