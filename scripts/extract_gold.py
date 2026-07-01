"""剑桥 gold 抽取 → 产出待人工转写的 stub。

机器侧：确定性抽取 band / 考官评语 / 题目 / 范文，写进 data/gold/manifest.json，
并为每条 sample 生成 data/gold/bodies/<id>.txt（预填乱码正文 + 表头，供人工对照 PDF 订正）。
人工侧：对照 PDF 手写图，把每个 stub 分隔线下的正文改对。
之后由 ingest_gold 读 manifest + bodies 入库。

用法：
  python scripts/extract_gold.py 13            # 只抽剑 13（验证用）
  python scripts/extract_gold.py               # 默认抽 9 本有文本层的书

注意：不会覆盖已存在的 body stub（保护你已转写的内容）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.gold_extract import extract_book

# 有文本层、可确定性抽取的 9 本（image-only 的 9/14/15/16/18/19 推后）。
TEXT_LAYER_BOOKS = {
    5: "【5】剑桥雅思真题5.pdf", 6: "【6】剑桥雅思真题6.pdf", 7: "【7】剑桥雅思真题7.pdf",
    8: "【8】剑桥雅思真题8.pdf", 10: "【10】剑桥雅思真题10.pdf", 11: "【11】剑桥雅思真题11.pdf",
    12: "【12】剑桥雅思真题12.pdf", 13: "【13】剑桥雅思真题13.pdf", 17: "剑桥雅思17（A类）.pdf",
}

DELIM = "# ====== ESSAY BODY BELOW (此行以下全部视为正文) ======"


def rec_id(r) -> str:
    return f"{r.source}_t{r.test}_q{r.task_type}_{r.answer_type}"


def write_stub(r, rid: str) -> bool:
    """为一条记录写 body stub；已存在则跳过（保护人工成果）。返回是否新建。"""
    path = config.GOLD_DIR / "bodies" / f"{rid}.txt"
    if path.exists():
        return False
    pages_1idx = ", ".join(str(p + 1) for p in r.pages)
    body_seed = (r.body_raw or "").lstrip(". ").strip()
    # model 印刷体范文天生干净 → 默认 DONE 自动入库；sample 手写需人工转写 → TODO。
    status = "DONE" if r.answer_type == "model" else "TODO"
    note = ("印刷体范文，预填一般已干净；如无误保持 STATUS: DONE 即可。"
            if r.answer_type == "model"
            else "预填为文本层乱码，仅作提示，请对照 PDF 手写图订正；改完把 STATUS 改成 DONE。")
    header = "\n".join([
        f"# GOLD ESSAY — {r.source} Test{r.test} Task{r.task_type} — "
        f"{'Band '+str(r.band) if r.band is not None else 'MODEL(无 band)'}",
        f"# STATUS: {status}",
        f"# PDF 页(1-indexed): {pages_1idx}",
        f"# Prompt: {' '.join(r.prompt.split())[:160]}",
        f"# 说明：{note}",
        DELIM, "",
    ])
    path.write_text(header + body_seed + "\n", encoding="utf-8")
    return True


def main(book_nums: list[int]) -> None:
    (config.GOLD_DIR / "bodies").mkdir(parents=True, exist_ok=True)
    manifest_path = config.GOLD_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    new_stubs = 0
    for bn in book_nums:
        pdf = config.RAW_DIR / TEXT_LAYER_BOOKS[bn]
        recs = extract_book(pdf)
        n_sample = sum(1 for r in recs if r.answer_type == "sample")
        warn = [rec_id(r) for r in recs if not r.prompt_ok]
        print(f"剑{bn}: {len(recs)} 块（sample {n_sample} / model {len(recs)-n_sample}）"
              + (f"  ⚠无题目: {warn}" if warn else ""))
        for r in recs:
            rid = rec_id(r)
            manifest[rid] = {
                "id": rid, "source": r.source, "test": r.test, "task_type": r.task_type,
                "answer_type": r.answer_type, "band": r.band,
                "examiner_comment": r.examiner_comment, "prompt": r.prompt,
                "pages": r.pages, "tt_explicit": r.tt_explicit, "prompt_ok": r.prompt_ok,
            }
            new_stubs += write_stub(r, rid)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nmanifest 记录数: {len(manifest)} | 新建 stub: {new_stubs}")
    print(f"→ 待人工转写: {config.GOLD_DIR / 'bodies'}")


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:]] or list(TEXT_LAYER_BOOKS)
    main(nums)
