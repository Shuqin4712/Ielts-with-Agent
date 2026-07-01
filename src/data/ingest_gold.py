"""把人工转写好的剑桥 gold（manifest + bodies）追加进 SQLite essays。

校验闸：只有 stub 里 STATUS: DONE 的记录才入库——人工转写完后显式把 sample 的
STATUS 由 TODO 改成 DONE，确定可靠（乱码率启发式不可靠，故只作二次提醒）。
model 范文印刷体天生干净，stub 默认 DONE，自动入库。

split（防泄漏）：
  - sample（真考生 + 官方 band + 评语）→ 'holdout'：gold 评测集，**不**进锚点。
  - model （剑桥范例，无 band）          → 'exemplar'：按 band≈9 高分锚点。

注意：本步是 APPEND，必须在 build_sqlite（重建 essays 表）之后跑。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .. import config
from ..db import sqlite
from .gold_extract import _is_junk
from .topic import infer_topic

DELIM = "# ====== ESSAY BODY BELOW (此行以下全部视为正文) ======"
MODEL_ANCHOR_BAND = 9.0          # model 范文无官方 band，作高分锚点用
JUNK_GATE = 0.15                 # 正文乱码率高于此 → 判为「未转写」，跳过

_COLS = [
    "task_type", "prompt", "body", "overall_band",
    "ta_band", "cc_band", "lr_band", "gra_band",
    "examiner_comment", "source", "tier", "topic", "split",
    "has_examiner_comment", "justifications",
]


def _read_stub(rid: str) -> tuple[bool, str]:
    """读 stub，返回 (是否 STATUS:DONE, 分隔线之后的正文)。"""
    path = config.GOLD_DIR / "bodies" / f"{rid}.txt"
    if not path.exists():
        return False, ""
    text = path.read_text(encoding="utf-8")
    header, _, body = text.partition(DELIM)
    done = "STATUS: DONE" in header
    return done, body.strip()


def _junk_ratio(text: str) -> float:
    words = text.split()
    return sum(_is_junk(w) for w in words) / len(words) if words else 1.0


def _to_row(rec: dict, body: str) -> dict:
    is_sample = rec["answer_type"] == "sample"
    band = rec.get("band")
    comment = (rec.get("examiner_comment") or "").strip() or None
    return {
        "task_type": rec["task_type"],
        "prompt": (rec.get("prompt") or "").strip(),
        "body": body,
        "overall_band": float(band) if (is_sample and band is not None) else (
            None if is_sample else MODEL_ANCHOR_BAND),
        "ta_band": None, "cc_band": None, "lr_band": None, "gra_band": None,
        "examiner_comment": comment,
        "source": rec.get("source", "cambridge"),
        "tier": "gold",
        "topic": infer_topic(rec.get("prompt", ""), body),
        "split": "holdout" if is_sample else "exemplar",
        "has_examiner_comment": 1 if comment else 0,
        "justifications": None,
    }


def ingest_gold(db_path: Path | None = None) -> dict:
    manifest_path = config.GOLD_DIR / "manifest.json"
    if not manifest_path.exists():
        return {"gold_total": 0, "note": "无 manifest，先跑 scripts/extract_gold.py"}
    manifest = json.loads(manifest_path.read_text("utf-8"))

    rows, skipped, suspect = [], [], []
    for rid, rec in manifest.items():
        done, body = _read_stub(rid)
        # 主闸：STATUS 必须为 DONE；sample 必须有 band；正文非空。
        if not done or len(body) < 50:
            skipped.append(rid)
            continue
        if rec["answer_type"] == "sample" and rec.get("band") is None:
            skipped.append(rid)
            continue
        if _junk_ratio(body) > JUNK_GATE:          # 二次提醒：标了 DONE 但仍像乱码
            suspect.append(rid)
        rows.append(_to_row(rec, body))

    conn = sqlite.get_conn(db_path)
    try:
        conn.execute("DELETE FROM essays WHERE tier='gold'")   # 幂等，不动 silver
        if rows:
            conn.executemany(
                f"INSERT INTO essays ({','.join(_COLS)}) VALUES ({','.join('?' * len(_COLS))})",
                [tuple(r[c] for c in _COLS) for r in rows],
            )
        conn.commit()
    finally:
        conn.close()

    by_split = defaultdict(int)
    for r in rows:
        by_split[r["split"]] += 1
    out = {
        "gold_ingested": len(rows),
        "by_split": dict(by_split),
        "skipped(未转写/缺band)": len(skipped),
        "manifest_total": len(manifest),
    }
    if suspect:
        out["⚠标DONE但疑似乱码"] = suspect
    return out


if __name__ == "__main__":
    print("追加 gold 入库：")
    for k, v in ingest_gold().items():
        print(f"  {k}: {v}")
