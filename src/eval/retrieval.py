"""检索层评测：范文检索的 Recall@5 / MRR@10，与端到端 QWK 解耦的分环节指标。

动机：锚定消融（QWK +0.065）只证明「检索 + 锚定」端到端有价值，定位不了
检索环节本身的损耗在哪。本评测单独量化「给一道题目，向量召回能不能找对
同话题范文」，并对比三档检索策略，验证结构感知 RAG 里 metadata 过滤的必要性。

评测纪律（与 harness 对齐）：
  - 查询全部取自 split='holdout' 的题目（prompt），与被检索语料
    （split='exemplar' 的范文正文）天然零重叠，无泄漏。
  - 查询集是确定性采样（seed 固定）后落盘的 jsonl，**人工可校对/增删**；
    文件存在即直接加载，保证多次运行可比。
  - 相关性标注 = (task_type, topic) 同为相关。topic 标签来自关键词规则
    （src/data/topic.py），是弱标注——查询文件即标注文件，改它即改 ground truth。

三档检索策略（同一查询集、同一语料、同一 embedding）：
  - vector        纯向量召回，无任何过滤（测 embedding 本身的话题区分度）
  - vector_task   只按 task_type 过滤 + 向量召回
  - filtered      生产路径：task_type + topic 过滤 + 向量召回（结构感知上限）

注意：换 embedding 模型做消融时，必须用新模型**重建索引**到独立集合再跑
（查询侧单边换模型 = 两个向量空间互相检索，结果无效）。

用法：
  python -m src.eval.retrieval                    # 跑三档，打印表格并存档
  python -m src.eval.retrieval --rebuild-queries  # 重新采样查询集（会覆盖人工修改！）
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

from .. import config
from ..db import sqlite
from ..rag import store

QUERIES = config.DATA_DIR / "eval" / "retrieval_queries.jsonl"
RESULTS = config.DATA_DIR / "eval" / "retrieval_results.jsonl"

K = 10             # 每查询召回条数（MRR@10 的窗口）
MIN_POOL = 5       # (topic, task) 的范文库存低于它就不出题（recall@5 没意义）
SEED = 42


# ── 指标（纯函数，可单测）───────────────────────────────────────────
def hit_at_1(flags: list[bool]) -> float:
    """top-1 是否相关。"""
    return 1.0 if flags and flags[0] else 0.0


def recall_at_k(flags: list[bool], n_relevant: int, k: int = 5) -> float:
    """截断召回：top-k 内相关数 / min(相关总数, k)。相关池大于 k 时等价 precision@k。"""
    denom = min(n_relevant, k)
    if denom == 0:
        return 0.0
    return sum(flags[:k]) / denom


def mrr_at_k(flags: list[bool], k: int = 10) -> float:
    """第一条相关结果排名的倒数；top-k 内没有则 0。"""
    for i, f in enumerate(flags[:k]):
        if f:
            return 1.0 / (i + 1)
    return 0.0


# ── 查询集 ─────────────────────────────────────────────────────────
def _exemplar_pool_counts() -> Counter:
    """被检索语料里每个 (task_type, topic) 的范文数——相关池大小。"""
    coll = store.get_client().get_collection(config.COLL_EXEMPLAR)
    metas = coll.get()["metadatas"]
    return Counter((m["task_type"], m["topic"]) for m in metas)


def build_queries() -> list[dict]:
    """从 holdout 题目里确定性采样：每个库存充足的 (task, topic) 取 1 题。"""
    pool = _exemplar_pool_counts()
    conn = sqlite.get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT prompt, topic, task_type FROM essays "
            "WHERE split='holdout' AND topic != 'general' "
            "AND length(prompt) > 30 ORDER BY prompt").fetchall()
    finally:
        conn.close()

    by_cell: dict[tuple, list[str]] = {}
    for r in rows:
        cell = (r["task_type"], r["topic"])
        if pool[cell] >= MIN_POOL:
            by_cell.setdefault(cell, []).append(r["prompt"])

    rng = random.Random(SEED)
    queries = []
    for (task, topic) in sorted(by_cell):
        prompt = rng.choice(by_cell[(task, topic)])
        queries.append({"qid": f"t{task}:{topic}", "task_type": task,
                        "topic": topic, "prompt": prompt})
    return queries


def load_or_build_queries(rebuild: bool = False) -> list[dict]:
    if QUERIES.exists() and not rebuild:
        return [json.loads(l) for l in QUERIES.read_text(encoding="utf-8").splitlines() if l.strip()]
    queries = build_queries()
    QUERIES.parent.mkdir(parents=True, exist_ok=True)
    QUERIES.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queries) + "\n",
                       encoding="utf-8")
    print(f"查询集已写入 {QUERIES}（n={len(queries)}，可人工校对后重跑）")
    return queries


# ── 运行 ───────────────────────────────────────────────────────────
VARIANTS = {
    "vector":      lambda q: None,
    "vector_task": lambda q: {"task_type": q["task_type"]},
    "filtered":    lambda q: {"task_type": q["task_type"], "topic": q["topic"]},
}


def eval_variant(name: str, queries: list[dict], pool: Counter) -> dict:
    per_q = []
    for q in queries:
        hits = store.search(config.COLL_EXEMPLAR, q["prompt"],
                            where=VARIANTS[name](q), n=K)
        flags = [h["metadata"]["task_type"] == q["task_type"]
                 and h["metadata"]["topic"] == q["topic"] for h in hits]
        n_rel = pool[(q["task_type"], q["topic"])]
        per_q.append({"qid": q["qid"],
                      "hit1": hit_at_1(flags),
                      "recall5": recall_at_k(flags, n_rel, 5),
                      "mrr10": mrr_at_k(flags, K)})
    n = len(per_q)
    agg = {m: round(sum(p[m] for p in per_q) / n, 3) for m in ("hit1", "recall5", "mrr10")}
    return {"variant": name, "n": n, **agg, "per_query": per_q}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-queries", action="store_true")
    args = ap.parse_args()

    queries = load_or_build_queries(rebuild=args.rebuild_queries)
    pool = _exemplar_pool_counts()
    print(f"检索评测：n={len(queries)} 查询 × {len(VARIANTS)} 档，"
          f"语料={config.COLL_EXEMPLAR}，embedding={config.EMBED_MODEL}\n")

    results = []
    for name in VARIANTS:
        r = eval_variant(name, queries, pool)
        results.append(r)
        print(f"  {name:<12} hit@1={r['hit1']:.3f}  recall@5={r['recall5']:.3f}  MRR@10={r['mrr10']:.3f}")

    # 错例分析：vector 档 recall@5 最低的 3 个查询（定位 embedding 分不开哪些话题）。
    worst = sorted(results[0]["per_query"], key=lambda p: p["recall5"])[:3]
    print("\n  vector 档最差查询（错例分析入口）：")
    for p in worst:
        print(f"    {p['qid']:<18} recall@5={p['recall5']:.2f}  mrr@10={p['mrr10']:.2f}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with RESULTS.open("a", encoding="utf-8") as f:
        for r in results:
            row = {"ts": ts, "embed_model": config.EMBED_MODEL, "k": K,
                   **{k: r[k] for k in ("variant", "n", "hit1", "recall5", "mrr10")}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n结果已追加到 {RESULTS}")


if __name__ == "__main__":
    main()
