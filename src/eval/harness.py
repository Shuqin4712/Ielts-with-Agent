"""评测 harness：在 gold holdout 上量化打分质量，结果带标签存档、可对比。

评测纪律（硬约束）：
  - overall 指标只在 gold holdout（tier='gold' AND split='holdout'）上算。
  - 跑前断言锚点集与 holdout 零重叠（防泄漏）。
  - 四维小分 gold 没有 → 在 silver holdout 子集上算，明标「辅助·非基准」。

用法：
  python -m src.eval.harness --config baseline_flash
  python -m src.eval.harness --config all --limit 10        # 开发抽样
  python -m src.eval.harness --compare                       # 打印历史对比表
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import config
from ..db import sqlite
from ..graph.build import build_grade_graph
from ..graph.state import CRITERIA
from ..rag import store
from . import metrics

RESULTS = config.DATA_DIR / "eval" / "results.jsonl"
_TRUE_COL = {"TA": "ta_band", "CC": "cc_band", "LR": "lr_band", "GRA": "gra_band"}

# 三档配置 + 消融档（同一张图，run_cfg 分档）。
# 打分一律用 flash（快）；只有 reflection 内部吃 pro+thinking（见 nodes.reflection）。
CONFIGS: dict[str, dict] = {
    "baseline_flash":        {"anchored": False, "reflect": False, "score_tier": "flash", "thinking": False, "max_retries": 0},
    "anchored_flash":        {"anchored": True,  "reflect": False, "score_tier": "flash", "thinking": False, "max_retries": 0},
    # v1.4 消融：锚点选取从「band 均匀采样」换成「池内向量排序 + 跨 band 铺开」，
    # 其余与 anchored_flash 完全一致 → 单变量对照，隔离出向量排序对 QWK 的影响。
    "anchored_vec_flash":    {"anchored": True,  "reflect": False, "score_tier": "flash", "thinking": False, "max_retries": 0, "anchor_rank": "vector"},
    "anchored_reflect_pro":  {"anchored": True,  "reflect": True,  "score_tier": "flash", "thinking": False, "max_retries": 2},
    "noanchor_reflect_pro":  {"anchored": False, "reflect": True,  "score_tier": "flash", "thinking": False, "max_retries": 2},
}
WORKERS = 6   # harness 并发线程数（每篇 essay 一个任务）


# ── 数据加载 ───────────────────────────────────────────────────────
def load_gold_holdout() -> list[dict]:
    conn = sqlite.get_conn()
    try:
        rows = conn.execute(
            "SELECT id, task_type, prompt, body, overall_band FROM essays "
            "WHERE tier='gold' AND split='holdout' ORDER BY id").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def load_silver_fourdim(n: int, seed: int = 42) -> list[dict]:
    conn = sqlite.get_conn()
    try:
        rows = conn.execute(
            "SELECT id, task_type, prompt, body, ta_band, cc_band, lr_band, gra_band "
            "FROM essays WHERE tier='silver' AND split='holdout' AND ta_band IS NOT NULL "
            "ORDER BY id").fetchall()
    finally:
        conn.close()
    rows = [dict(r) for r in rows]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def assert_no_leakage(holdout: list[dict]) -> None:
    """硬断言：gold holdout 的 id 绝不出现在锚点集里。"""
    coll = store.get_client().get_collection(config.COLL_EXEMPLAR)
    exemplar_ids = {m.get("essay_id") for m in coll.get()["metadatas"]}
    overlap = {e["id"] for e in holdout} & exemplar_ids
    assert not overlap, f"泄漏！gold holdout 与锚点集重叠 essay_id={sorted(overlap)}"


# ── 运行 ───────────────────────────────────────────────────────────
def _grade(graph, essay: dict, run_cfg: dict) -> dict:
    return graph.invoke({
        "essay": essay["body"], "task_type": essay["task_type"],
        "prompt": essay["prompt"] or "", "run_cfg": run_cfg,
        "essay_id": essay["id"], "anchors": [], "retries": 0,
    })


def _grade_all(graph, essays: list[dict], run_cfg: dict) -> list[tuple[dict, dict | None]]:
    """线程池并发打分，容忍单篇失败（超时等）→ 失败记 None，不拖垮整批。"""
    print(f"  并发 {WORKERS} 线程，{len(essays)} 篇 …")
    out: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_grade, graph, e, run_cfg): e for e in essays}
        for fut in as_completed(futs):
            e = futs[fut]
            done += 1
            try:
                out[e["id"]] = fut.result()
            except Exception as err:               # 单篇失败（如 timeout）→ 跳过
                out[e["id"]] = None
                print(f"    ⚠ id={e['id']} 失败跳过: {type(err).__name__}: {err}")
            if done % 10 == 0 or done == len(essays):
                print(f"    {done}/{len(essays)}")
    return [(e, out[e["id"]]) for e in essays]


def run_config(label: str, *, limit: int | None = None,
               silver_n: int = 40, do_fourdim: bool = True) -> dict:
    run_cfg = CONFIGS[label]
    graph = build_grade_graph()
    gold = load_gold_holdout()
    assert_no_leakage(gold)
    if limit:
        gold = gold[:limit]

    print(f"[{label}] gold holdout 打分中 …")
    pred, true, n_fail = [], [], 0
    for e, f in _grade_all(graph, gold, run_cfg):
        if f is None:
            n_fail += 1
            continue
        pred.append(f["overall_band"])
        true.append(e["overall_band"])

    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label, "run_cfg": run_cfg, "n_failed": n_fail,
        "overall_gold": metrics.overall_metrics(pred, true),
    }

    if do_fourdim:
        silver = load_silver_fourdim(min(silver_n, limit) if limit else silver_n)
        print(f"[{label}] 四维辅助：silver 打分中 …")
        per = {c: {"pred": [], "true": []} for c in CRITERIA}
        for e, f in _grade_all(graph, silver, run_cfg):
            if f is None:
                continue
            for c in CRITERIA:
                per[c]["pred"].append(f["dimension_scores"][c]["band"])
                per[c]["true"].append(e[_TRUE_COL[c]])
        rec["fourdim_silver_aux"] = {
            "n": len(per["TA"]["pred"]),
            **{c: {"mae": round(metrics.mae(per[c]["pred"], per[c]["true"]), 3)} for c in CRITERIA},
        }

    _append(rec)
    _print_rec(rec)
    return rec


# ── 存档 / 展示 ────────────────────────────────────────────────────
def _append(rec: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _print_rec(rec: dict) -> None:
    o = rec["overall_gold"]
    print(f"\n=== {rec['label']} @ {rec['ts']} ===")
    print(f"  [overall · gold n={o['n']}]  MAE={o['mae']}  ±0.5={o['within_0.5']}  "
          f"±1.0={o['within_1.0']}  QWK={o['qwk']}")
    if "fourdim_silver_aux" in rec:
        fd = rec["fourdim_silver_aux"]
        dims = "  ".join(f"{c}={fd[c]['mae']}" for c in CRITERIA)
        print(f"  [四维 MAE · silver 辅助·非基准 n={fd['n']}]  {dims}")


def compare() -> None:
    if not RESULTS.exists():
        print("还没有结果。"); return
    recs = [json.loads(l) for l in RESULTS.read_text("utf-8").splitlines() if l.strip()]
    print(f"{'label':22s} {'MAE':>6} {'±0.5':>6} {'±1.0':>6} {'QWK':>6}   ts")
    for r in recs:
        o = r["overall_gold"]
        print(f"{r['label']:22s} {o['mae']:>6} {o['within_0.5']:>6} "
              f"{o['within_1.0']:>6} {o['qwk']:>6}   {r['ts']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="配置标签或 all", choices=[*CONFIGS, "all"])
    ap.add_argument("--limit", type=int, help="gold 抽样条数（开发用）")
    ap.add_argument("--silver-n", type=int, default=40, help="四维辅助的 silver 条数")
    ap.add_argument("--no-fourdim", action="store_true", help="跳过四维辅助")
    ap.add_argument("--compare", action="store_true", help="打印历史对比表")
    args = ap.parse_args()

    if args.compare:
        compare(); return
    labels = list(CONFIGS) if args.config == "all" else [args.config]
    for label in labels:
        run_config(label, limit=args.limit, silver_n=args.silver_n,
                   do_fourdim=not args.no_fourdim)
    print()
    compare()


if __name__ == "__main__":
    main()
