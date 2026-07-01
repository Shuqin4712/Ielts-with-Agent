"""把两份异构 CSV 归一化成统一的 essay 记录，并做基础清洗。

两份数据列名/可用字段不同（见勘探结论），这里各写一个映射函数，
统一产出同一份 dict schema，方便后续无差别入库。两份都是网络来源 → tier='silver'。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .topic import infer_topic


def _band(s: str | None) -> float | None:
    """解析 band：'7'/'7.0'→7.0，空→None，越界→None。"""
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if 0.0 <= v <= 9.0 else None


def _clean_text(s: str | None) -> str:
    return (s or "").strip()


def _valid(rec: dict) -> bool:
    """基础清洗规则：task 合法、题目与正文非空、overall 有值。"""
    return (
        rec["task_type"] in (1, 2)
        and len(rec["body"]) >= 50          # 过滤空/截断的垃圾行
        and len(rec["prompt"]) >= 10
        and rec["overall_band"] is not None
    )


def _finalize(rec: dict) -> dict:
    """补 topic 与 has_examiner_comment 派生字段。"""
    rec["topic"] = infer_topic(rec["prompt"], rec["body"])
    rec["has_examiner_comment"] = 1 if _clean_text(rec.get("examiner_comment")) else 0
    return rec


def _from_dataset0(row: dict) -> dict:
    """dataset.csv：四维小分齐全 + 四维 LLM 评分理由。"""
    just = {
        "TR": _clean_text(row.get("justification_TR")),
        "CC": _clean_text(row.get("justification_CC")),
        "LR": _clean_text(row.get("justification_LR")),
        "GRA": _clean_text(row.get("justification_GRA")),
    }
    just = {k: v for k, v in just.items() if v}
    return {
        "task_type": int(row["Task_Type"]) if row.get("Task_Type", "").strip() in ("1", "2") else 0,
        "prompt": _clean_text(row.get("Question")),
        "body": _clean_text(row.get("Essay")),
        "overall_band": _band(row.get("Overall")),
        "ta_band": _band(row.get("Task_Response")),
        "cc_band": _band(row.get("Coherence_Cohesion")),
        "lr_band": _band(row.get("Lexical_Resource")),
        "gra_band": _band(row.get("Range_Accuracy")),
        "examiner_comment": None,
        "source": "ielts_writing_dataset.csv",
        "tier": "silver",
        "justifications": json.dumps(just, ensure_ascii=False) if just else None,
    }


def _from_dataset1(row: dict) -> dict:
    """dataset_1.csv：仅 Overall + 少量真考官评语；四维小分全空。"""
    return {
        "task_type": int(row["Task_Type"]) if row.get("Task_Type", "").strip() in ("1", "2") else 0,
        "prompt": _clean_text(row.get("Question")),
        "body": _clean_text(row.get("Essay")),
        "overall_band": _band(row.get("Overall")),
        "ta_band": _band(row.get("Task_Response")),
        "cc_band": _band(row.get("Coherence_Cohesion")),
        "lr_band": _band(row.get("Lexical_Resource")),
        "gra_band": _band(row.get("Range_Accuracy")),
        "examiner_comment": _clean_text(row.get("Examiner_Commen")) or None,
        "source": "ielts_writing_dataset_1.csv",
        "tier": "silver",
        "justifications": None,
    }


_LOADERS = {
    "ielts_writing_dataset.csv": _from_dataset0,
    "ielts_writing_dataset_1.csv": _from_dataset1,
}


def load_essays(raw_dir: Path) -> list[dict]:
    """读两份 CSV → 归一化 → 清洗 → 跨文件按正文去重。"""
    records: list[dict] = []
    for fname, loader in _LOADERS.items():
        path = raw_dir / fname
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                rec = loader(row)
                if _valid(rec):
                    records.append(_finalize(rec))

    # 去重：同一篇正文（去空白后）只留第一条。
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in records:
        key = " ".join(rec["body"].split()).lower()[:400]
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    return deduped
