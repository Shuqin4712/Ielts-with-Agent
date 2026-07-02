"""阶段 4 演示：同一 user 连批两篇 → 第二篇反馈体现跨篇记忆与个性化。

用法：
  python scripts/demo_memory.py                    # 默认取两篇 gold holdout Task2
  python scripts/demo_memory.py --ids 2269 2273     # 指定两篇 essays.id
  python scripts/demo_memory.py --user bob --keep    # 换 user；--keep 保留画像不清理

跑完默认清理该 demo user 的画像/历史（除非 --keep），避免污染。会真调 DeepSeek。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.sqlite import get_conn
from src.graph.session import build_grading_session_graph
from src.memory import profile as P
from src.graph.state import CRITERIA

DEFAULT_USER = "__demo_memory__"


def _essays(ids):
    conn = get_conn()
    try:
        if ids:
            q = "SELECT id, task_type, prompt, body FROM essays WHERE id IN (%s)" % \
                ",".join("?" * len(ids))
            rows = conn.execute(q, ids).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, task_type, prompt, body FROM essays WHERE tier='gold' "
                "AND split='holdout' AND task_type=2 ORDER BY id LIMIT 2").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _clear(user):
    conn = get_conn()
    conn.execute("DELETE FROM student_profile WHERE user_id=?", (user,))
    conn.execute("DELETE FROM grading_history WHERE user_id=?", (user,))
    conn.commit()
    conn.close()


def _grade(graph, user, e):
    return graph.invoke({
        "user_id": user, "essay": e["body"], "task_type": e["task_type"],
        "prompt": e["prompt"] or "", "essay_id": e["id"], "personalize": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--ids", nargs=2, type=int, help="两篇 essays.id")
    ap.add_argument("--keep", action="store_true", help="跑完保留画像（默认清理）")
    args = ap.parse_args()

    essays = _essays(args.ids)
    if len(essays) < 2:
        sys.exit("需要两篇作文。")
    _clear(args.user)   # 从干净画像开始

    graph = build_grading_session_graph()
    for i, e in enumerate(essays, 1):
        print(f"\n{'='*60}\n第 {i} 篇 (id={e['id']}) — user={args.user}\n{'='*60}")
        out = _grade(graph, args.user, e)
        dims = "  ".join(f"{c}={out['dimension_scores'][c]['band']}" for c in CRITERIA)
        print(f"四维: {dims}  | overall={out['overall_band']}")
        prof = out.get("profile") or {}
        print(f"[画像] 历史{len(prof.get('band_history') or [])}篇 "
              f"薄弱={prof.get('weak_criteria')} 反复问题={len(prof.get('recurring_errors') or [])}项")
        print(f"\n反馈:\n{out['feedback']}")

    print(f"\n{'='*60}\n对比要点：第 2 篇的反馈应能『记得』第 1 篇的问题"
          f"（呼应反复维度 / 走势）。\n{'='*60}")
    if not args.keep:
        _clear(args.user)
        print(f"(已清理 demo user={args.user} 的画像/历史；--keep 可保留)")


if __name__ == "__main__":
    main()
