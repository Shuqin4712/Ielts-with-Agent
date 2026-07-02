"""长期记忆：学生画像（跨 session，落 SQLite 的 student_profile / grading_history）。

两类记忆（episodic 与 semantic 的区分）：
  - episodic（情景）：原始事件流水，append-only、不改写。→ band_history（走势）+
    grading_history（每次批改一行）。纯确定性，不用 LLM。
  - semantic（语义）：从多次 episodic 蒸馏出的结论（「反复用被动语态」「时态错误高频」）。
    → recurring_errors / weak_criteria / vocab_level。会被**增量**改写。

为什么增量蒸馏（distill）而非全量重算：全量重算要把该生全部历史每次重喂 LLM，成本随
历史线性膨胀、结论抖动、且浪费。增量只喂「已有 semantic + 这一篇的新信号」，让模型做
合并更新——成本恒定、结论单调演进。

⚠️ 护栏：本画像**只用于个性化反馈措辞**，绝不进打分逻辑。打分永远走 build_grade_graph
那条纯管道，学生历史物理上到不了判分。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..db.sqlite import get_conn, init_db
from ..graph.state import CRITERIA
from ..tools._json import call_json

MAX_ERRORS = 6          # semantic 错误模式上限，防画像无限膨胀
_now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── 读 ─────────────────────────────────────────────────────────────
def _loads(v, default):
    try:
        return json.loads(v) if v else default
    except (json.JSONDecodeError, TypeError):
        return default


def load_profile(user_id: str, db_path: str | Path | None = None) -> dict:
    """取该 user 的画像；没有则返回空画像（exists=False）。"""
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        row = conn.execute("SELECT * FROM student_profile WHERE user_id=?",
                           (user_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"user_id": user_id, "recurring_errors": [], "weak_criteria": [],
                "vocab_level": None, "band_history": [], "exists": False}
    return {
        "user_id": user_id,
        "recurring_errors": _loads(row["recurring_errors"], []),
        "weak_criteria": _loads(row["weak_criteria"], []),
        "vocab_level": row["vocab_level"],
        "band_history": _loads(row["band_history"], []),
        "exists": True,
    }


# ── 写（episodic：确定性 append）───────────────────────────────────
def _upsert(user_id, recurring_errors, weak_criteria, vocab_level, band_history,
            db_path=None) -> None:
    conn = get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO student_profile (user_id, recurring_errors, weak_criteria, "
            "vocab_level, band_history, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET recurring_errors=excluded.recurring_errors, "
            "weak_criteria=excluded.weak_criteria, vocab_level=excluded.vocab_level, "
            "band_history=excluded.band_history, updated_at=excluded.updated_at",
            (user_id, json.dumps(recurring_errors, ensure_ascii=False),
             json.dumps(weak_criteria, ensure_ascii=False), vocab_level,
             json.dumps(band_history, ensure_ascii=False), _now()))
        conn.commit()
    finally:
        conn.close()


def _append_history_row(user_id, essay_id, scores, feedback, db_path=None) -> None:
    conn = get_conn(db_path)
    try:
        conn.execute(
            "INSERT INTO grading_history (user_id, essay_id, scores, feedback, created_at) "
            "VALUES (?,?,?,?,?)",
            (user_id, essay_id, json.dumps(scores, ensure_ascii=False), feedback, _now()))
        conn.commit()
    finally:
        conn.close()


def _weak_criteria(band_history: list[dict]) -> list[str]:
    """episodic 派生（确定性）：历次四维均分最低的 1–2 维即薄弱维度。"""
    if not band_history:
        return []
    sums = {c: 0.0 for c in CRITERIA}
    cnts = {c: 0 for c in CRITERIA}
    for ep in band_history:
        for c in CRITERIA:
            v = (ep.get("dims") or {}).get(c)
            if v is not None:
                sums[c] += v
                cnts[c] += 1
    avgs = {c: sums[c] / cnts[c] for c in CRITERIA if cnts[c]}
    if not avgs:
        return []
    order = sorted(avgs, key=lambda c: avgs[c])
    lowest = avgs[order[0]]
    # 取最低维；若次低与最低仅差 ≤0.25，一并算薄弱（并列弱项）。
    return [c for c in order if avgs[c] - lowest <= 0.25][:2]


# ── 写（semantic：LLM 增量蒸馏）────────────────────────────────────
def distill_semantic(existing_errors: list[dict], dimension_scores: dict,
                     overall_band: float, task_type: int) -> dict:
    """喂「已有 recurring_errors + 这一篇的四维依据」→ 增量合并出新的 semantic。

    只看这一篇 + 已有画像（不重扫全部历史）= 增量蒸馏。返回
    {"recurring_errors":[{criterion,pattern,seen}], "vocab_level": str}。
    """
    ev = "\n".join(
        f"- {c}: band {dimension_scores[c]['band']} — {dimension_scores[c]['evidence']}"
        for c in CRITERIA if c in dimension_scores
    )
    system = (
        "You maintain a CONCISE long-term error profile of one IELTS writing student across "
        "essays. You are given their EXISTING recurring-error list and per-criterion examiner "
        "evidence from ONE NEW essay. Produce an UPDATED profile by INCREMENTAL merge: "
        "increment 'seen' for patterns that recur again, keep still-relevant ones, add "
        "genuinely new recurring-style weaknesses, and cap the list at the "
        f"{MAX_ERRORS} most important. This profile is used ONLY to personalize feedback "
        "wording; it must NEVER influence scoring. Respond ONLY JSON: "
        '{"recurring_errors":[{"criterion":"TA|CC|LR|GRA","pattern":"<short phrase>",'
        '"seen":<int>}],"vocab_level":"<short phrase e.g. around band 6 / limited range>"}.'
    )
    human = (
        f"EXISTING recurring_errors (JSON):\n{json.dumps(existing_errors, ensure_ascii=False)}\n\n"
        f"NEW essay — Task {task_type}, overall band {overall_band}. Per-criterion evidence:\n{ev}\n\n"
        "Return the updated JSON profile."
    )
    data = call_json("memory_write", system, human)
    errors = data.get("recurring_errors") or []
    # 归一 + 截断，防脏数据/膨胀。
    clean = []
    for e in errors:
        if not isinstance(e, dict) or not e.get("pattern"):
            continue
        clean.append({
            "criterion": e.get("criterion", ""),
            "pattern": str(e["pattern"]).strip(),
            "seen": int(e.get("seen", 1) or 1),
        })
    return {"recurring_errors": clean[:MAX_ERRORS],
            "vocab_level": (data.get("vocab_level") or "").strip() or None}


# ── 编排：一次批改结束后更新画像 ──────────────────────────────────
def update_after_grading(user_id: str, *, essay_meta: dict, dimension_scores: dict,
                         overall_band: float, feedback: str = "", distill: bool = True,
                         db_path: str | Path | None = None) -> dict:
    """把一次批改的结果写进长期记忆，返回更新后的画像。

    episodic（确定性）：band_history 追加一条 + grading_history 插一行。
    semantic（LLM 增量）：distill=True 时蒸馏更新 recurring_errors / vocab_level；
                          distill=False 时保留旧 semantic（测试/省钱可关）。
    weak_criteria 始终由 band_history 确定性重算。
    """
    prof = load_profile(user_id, db_path)
    episode = {
        "ts": _now(),
        "essay_id": essay_meta.get("essay_id"),
        "task_type": essay_meta.get("task_type"),
        "topic": essay_meta.get("topic"),
        "overall": overall_band,
        "dims": {c: dimension_scores[c]["band"] for c in CRITERIA if c in dimension_scores},
    }
    band_history = prof["band_history"] + [episode]

    if distill:
        sem = distill_semantic(prof["recurring_errors"], dimension_scores,
                               overall_band, essay_meta.get("task_type", 2))
        recurring_errors, vocab_level = sem["recurring_errors"], sem["vocab_level"]
    else:
        recurring_errors, vocab_level = prof["recurring_errors"], prof["vocab_level"]

    weak = _weak_criteria(band_history)
    _upsert(user_id, recurring_errors, weak, vocab_level, band_history, db_path)
    _append_history_row(user_id, essay_meta.get("essay_id"),
                        {c: dimension_scores[c] for c in CRITERIA if c in dimension_scores},
                        feedback, db_path)
    return load_profile(user_id, db_path)


# ── 供反馈注入的可读摘要（批次 C 用；这里先备好）──────────────────
def profile_summary(prof: dict) -> str:
    """把画像压成一小段人读文本，供 feedback 节点注入（不含原始作文，控上下文）。"""
    if not prof.get("exists") and not prof.get("band_history"):
        return ""
    parts = []
    hist = prof.get("band_history") or []
    if hist:
        overalls = [str(ep.get("overall")) for ep in hist if ep.get("overall") is not None]
        parts.append(f"This student has {len(hist)} prior graded essay(s); "
                     f"overall band trend: {', '.join(overalls)}.")
    if prof.get("weak_criteria"):
        parts.append(f"Persistently weak criteria: {', '.join(prof['weak_criteria'])}.")
    errs = prof.get("recurring_errors") or []
    if errs:
        listed = "; ".join(f"{e.get('criterion','')}: {e.get('pattern','')} (seen {e.get('seen',1)}x)"
                           for e in errs)
        parts.append(f"Recurring issues: {listed}.")
    if prof.get("vocab_level"):
        parts.append(f"Vocabulary level: {prof['vocab_level']}.")
    return " ".join(parts)
