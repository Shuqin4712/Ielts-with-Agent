"""阶段 0 一键构建：SQLite 入库 + ChromaDB 两集合。

用法： python scripts/build_stage0.py
（从仓库根运行；需先 ollama 起着、bge-m3 已 pull、.env 不必填 key——
 阶段 0 不调 DeepSeek，只用本地 embedding。）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.ingest import build_sqlite
from src.data.ingest_gold import ingest_gold
from src.rag.build_index import build_rubric, build_exemplars

if __name__ == "__main__":
    print("[1/3] 归一化 silver 数据入 SQLite …")
    summary = build_sqlite()
    for k, v in summary.items():
        print(f"      {k}: {v}")

    # gold 是 append，必须在 build_sqlite（重建表）之后。已转写的才入库。
    if (config.GOLD_DIR / "manifest.json").exists():
        print("[2/3] 追加剑桥 gold（仅已转写干净的）…")
        for k, v in ingest_gold().items():
            print(f"      {k}: {v}")
    else:
        print("[2/3] 跳过 gold（无 manifest，先跑 scripts/extract_gold.py）")

    print("[3/3] 构建 ChromaDB 索引 …")
    print(f"      rubric_descriptors: {build_rubric()} 块")
    print(f"      exemplar_essays:    {build_exemplars()} 篇")
    print("阶段 0 构建完成 ✅")
