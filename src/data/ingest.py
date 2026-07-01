"""阶段 0 · 把归一化后的 essay 灌进 SQLite，并分配 train/holdout/exemplar。

split 策略：
  - holdout：按 (task_type × band) 分层抽样 HOLDOUT_FRAC，留作「临时评测基线」
             （silver，非评测数字；真 gold 到位后替换）。
  - exemplar：从非 holdout 里按 (task_type × band) 每桶精选少量，优先有考官评语 /
             有四维理由的，作为打分锚点候选。质量优先、控制总量。
  - train：其余。
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .. import config
from ..db import sqlite
from .normalize import load_essays

_COLS = [
    "task_type", "prompt", "body", "overall_band",
    "ta_band", "cc_band", "lr_band", "gra_band",
    "examiner_comment", "source", "tier", "topic", "split",
    "has_examiner_comment", "justifications",
]


def _bucket(rec: dict) -> tuple[int, float]:
    """分层用的桶键：(task_type, 四舍五入到 0.5 的 band)。"""
    return rec["task_type"], round(rec["overall_band"] * 2) / 2


def assign_splits(records: list[dict]) -> None:
    """就地给每条记录写入 rec['split']。"""
    rng = random.Random(config.SPLIT_SEED)

    # 1) 分层切 holdout。
    buckets: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for rec in records:
        buckets[_bucket(rec)].append(rec)

    for recs in buckets.values():
        rng.shuffle(recs)
        n_holdout = max(1, int(len(recs) * config.HOLDOUT_FRAC)) if len(recs) >= 8 else 0
        for i, rec in enumerate(recs):
            rec["split"] = "holdout" if i < n_holdout else "train"

    # 2) 从 train 里挑 exemplar：按 (task × band × topic) 分层，保证话题覆盖
    #    （DESIGN §4.4）。每桶优先考官评语 > 有理由 > 其它。
    def priority(rec: dict) -> tuple:
        return (rec["has_examiner_comment"], rec["justifications"] is not None)

    topic_buckets: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        if rec["split"] == "train":
            tb = (rec["task_type"], round(rec["overall_band"]), rec["topic"])
            topic_buckets[tb].append(rec)

    for recs in topic_buckets.values():
        recs.sort(key=priority, reverse=True)
        for rec in recs[: config.EXEMPLAR_PER_BUCKET]:
            rec["split"] = "exemplar"


def build_sqlite(raw_dir: Path | None = None, db_path: Path | None = None) -> dict:
    """读 CSV → 归一化 → 分 split → 重建 essays 表并写入。返回统计摘要。"""
    raw_dir = raw_dir or config.RAW_DIR
    records = load_essays(raw_dir)
    assign_splits(records)

    # 重建 essays 表，保证 schema 最新、可重复执行。
    conn = sqlite.get_conn(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS essays")
    finally:
        conn.close()
    sqlite.init_db(db_path)

    conn = sqlite.get_conn(db_path)
    try:
        conn.executemany(
            f"INSERT INTO essays ({','.join(_COLS)}) VALUES ({','.join('?' * len(_COLS))})",
            [tuple(r.get(c) for c in _COLS) for r in records],
        )
        conn.commit()
    finally:
        conn.close()

    # 统计摘要
    splits = defaultdict(int)
    tiers = defaultdict(int)
    for r in records:
        splits[r["split"]] += 1
        tiers[r["tier"]] += 1
    return {
        "total": len(records),
        "splits": dict(splits),
        "tiers": dict(tiers),
        "with_examiner_comment": sum(r["has_examiner_comment"] for r in records),
        "with_subscores": sum(1 for r in records if r["ta_band"] is not None),
    }


if __name__ == "__main__":
    summary = build_sqlite()
    print("SQLite 入库完成：")
    for k, v in summary.items():
        print(f"  {k}: {v}")
