"""CLI：喂一篇作文 → LangGraph 四维打分 → 终端出分。

用法：
  python scripts/grade_essay.py --gold            # 纯打分（与 eval 同一条管道）
  python scripts/grade_essay.py --gold --task 1   # 指定 task 类型
  python scripts/grade_essay.py --essay-id 123    # 指定 essays.id
  python scripts/grade_essay.py --essay-id 123 --user alice
        # 阶段 4：走个性化全图（load_profile→grade→feedback→memory_write），
        # 出四维 + 个性化反馈，并更新该 user 的长期画像。

无 --user 时走纯打分图（不加载/不写画像，与 eval 对齐）；打分逻辑两者共用，个性化
只改反馈文字，绝不改 band。需 .env 里有 DEEPSEEK_API_KEY。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import sqlite
from src.graph.build import build_grade_graph
from src.graph.session import build_grading_session_graph
from src.memory.checkpoint import get_checkpointer
from src.graph.state import CRITERIA

_LABEL = {"TA": "TA ", "CC": "CC ", "LR": "LR ", "GRA": "GRA"}


def load_essay(essay_id: int | None, gold: bool, task: int) -> dict:
    conn = sqlite.get_conn()
    try:
        if essay_id is not None:
            row = conn.execute(
                "SELECT id, task_type, prompt, body, overall_band, tier "
                "FROM essays WHERE id=?", (essay_id,)).fetchone()
        else:  # 默认取一篇 gold holdout（有官方 band 可肉眼对照）
            row = conn.execute(
                "SELECT id, task_type, prompt, body, overall_band, tier FROM essays "
                "WHERE tier='gold' AND split='holdout' AND task_type=? "
                "ORDER BY id LIMIT 1", (task,)).fetchone()
    finally:
        conn.close()
    if row is None:
        sys.exit("没找到符合条件的作文。")
    return dict(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true", help="取一篇 gold holdout 作文（默认）")
    ap.add_argument("--essay-id", type=int, help="指定 essays.id")
    ap.add_argument("--task", type=int, default=2, choices=(1, 2), help="task 类型（配合 --gold）")
    ap.add_argument("--user", help="指定 user_id → 走个性化全图（加载/更新长期画像 + 出反馈）")
    args = ap.parse_args()

    row = load_essay(args.essay_id, args.gold, args.task)
    print(f"作文来源: id={row['id']} tier={row['tier']} Task{row['task_type']}")
    print(f"题目: {' '.join((row['prompt'] or '').split())[:100]}…\n")

    cfg = {"anchored": True, "reflect": False, "score_tier": "flash"}
    if args.user:
        # 阶段 4：个性化全图。checkpointer 用 user_id 当 thread（同人续跑同一线程）。
        graph = build_grading_session_graph(checkpointer=get_checkpointer())
        final = graph.invoke(
            {"user_id": args.user, "essay": row["body"], "task_type": row["task_type"],
             "prompt": row["prompt"] or "", "essay_id": row["id"], "run_cfg": cfg,
             "personalize": True},
            {"configurable": {"thread_id": f"grade:{args.user}"}})
    else:
        # 纯打分图（与 eval 同一条管道；不加载/不写画像）。
        final = build_grade_graph().invoke({
            "essay": row["body"], "task_type": row["task_type"],
            "prompt": row["prompt"] or "", "run_cfg": cfg,
            "essay_id": row["id"], "anchors": [], "retries": 0,
        })

    print("四维打分：")
    for crit in CRITERIA:
        s = final["dimension_scores"][crit]
        print(f"  {_LABEL[crit]}  band {s['band']:<4}  — {s['evidence']}")
    print(f"\nOverall band: {final['overall_band']}")
    if row["overall_band"] is not None:
        print(f"（该 gold 作文官方 overall = {row['overall_band']}，仅作肉眼对照，未参与打分）")

    if args.user:
        print(f"\n── 个性化反馈（user={args.user}）──\n{final['feedback']}")
        revs = final.get("revision") or []
        if revs:
            print("\n── 改写示范（最弱维度）──")
            for r in revs:
                print(f"  原句: {r.get('original','')}")
                print(f"  改写: {r.get('revised','')}")
                print(f"  为何: {r.get('why','')}\n")
        prof = final.get("profile") or {}
        print(f"\n[画像已更新] 历史 {len(prof.get('band_history') or [])} 篇 | "
              f"薄弱维度={prof.get('weak_criteria')} | 反复问题数={len(prof.get('recurring_errors') or [])}")


if __name__ == "__main__":
    main()
