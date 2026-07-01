"""阶段 0 验收 demo：能检索「环境话题的 band 7 Task 2 范文」，并展示 rubric 召回。

用法： python scripts/demo_retrieve.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.rag import store


def show(title, results):
    print(f"\n=== {title} ===")
    for r in results:
        m = r["metadata"]
        print(f"  · dist={r['distance']:.3f} {m} ")
        print(f"    {r['document'][:120].strip()}…")


if __name__ == "__main__":
    # 1) 阶段 0 的标志性产出：按 (task_type=2, band=7, topic=environment) 过滤 + 语义召回。
    ex = store.search(
        config.COLL_EXEMPLAR,
        query="protecting the environment and reducing pollution",
        where={"task_type": 2, "band": 7.0, "topic": "environment"},
        n=3,
    )
    show("环境话题 · Band 7 · Task 2 范文", ex)

    # 2) rubric 按维度 + band 召回（打分阶段会用到）。
    ru = store.search(
        config.COLL_RUBRIC,
        query="range and accuracy of vocabulary",
        where={"task_type": 2, "criterion": "LR", "band": 7},
        n=1,
    )
    show("Rubric · Task2 · LR · Band 7", ru)
