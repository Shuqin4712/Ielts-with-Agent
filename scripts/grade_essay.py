"""阶段 1 CLI：喂一篇作文 → LangGraph 四维打分 → 终端出分。

用法：
  python scripts/grade_essay.py --gold            # 默认取一篇 gold holdout 作文
  python scripts/grade_essay.py --gold --task 1   # 指定 task 类型
  python scripts/grade_essay.py --essay-id 123    # 指定 essays.id

需 .env 里有 DEEPSEEK_API_KEY（会真调 4 次 flash）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import sqlite
from src.graph.build import build_grade_graph
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
    args = ap.parse_args()

    row = load_essay(args.essay_id, args.gold, args.task)
    print(f"作文来源: id={row['id']} tier={row['tier']} Task{row['task_type']}")
    print(f"题目: {' '.join((row['prompt'] or '').split())[:100]}…\n")

    graph = build_grade_graph()
    # 默认用阶段 2 验证过的最佳配置：锚定开、reflection 关（reflection 本测试集无增益）。
    final = graph.invoke({
        "essay": row["body"],
        "task_type": row["task_type"],
        "prompt": row["prompt"] or "",
        "run_cfg": {"anchored": True, "reflect": False, "score_tier": "flash"},
        "essay_id": row["id"], "anchors": [], "retries": 0,
    })

    print("四维打分：")
    for crit in CRITERIA:
        s = final["dimension_scores"][crit]
        print(f"  {_LABEL[crit]}  band {s['band']:<4}  — {s['evidence']}")
    print(f"\nOverall band: {final['overall_band']}")
    if row["overall_band"] is not None:
        print(f"（该 gold 作文官方 overall = {row['overall_band']}，仅作肉眼对照，未参与打分）")


if __name__ == "__main__":
    main()
