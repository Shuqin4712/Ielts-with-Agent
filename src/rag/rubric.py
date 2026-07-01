"""把官方 Writing Band Descriptors PDF 解析成 (criterion, band, task_type) 结构块。

这是「结构感知 chunking」的核心：官方 rubric 是一张
行=band、列=四维(TA/CC/LR/GRA) 的表。纯文本抽取会把四列糊在一起，
所以这里用文字的 x 坐标把列切开、用 band 数字的 y 坐标把行切开，
得到每个 (criterion, band) 单元格独立成块，挂 metadata 后进 ChromaDB。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .. import config

# 四列的 x 坐标锚点 → 边界取相邻锚点中点附近的整数。
# 实测：band≈28 | TA≈55 | CC≈378 | LR≈562 | GRA≈755
_COL_BOUNDS = [
    ("TA", 45, 300),
    ("CC", 300, 470),
    ("LR", 470, 660),
    ("GRA", 660, 10_000),
]
CRITERIA = ["TA", "CC", "LR", "GRA"]


@dataclass
class RubricChunk:
    task_type: int        # 1 or 2
    criterion: str        # TA / CC / LR / GRA（task2 的 TA 实为 Task Response）
    band: int             # 0..9
    text: str


def _page_items(page) -> list[tuple[int, int, str]]:
    """抽出本页所有文字片段的 (x, y, text)。"""
    items: list[tuple[int, int, str]] = []

    def visit(text, cm, tm, fontdict, fontsize):
        if text and text.strip():
            items.append((round(tm[4]), round(tm[5]), text.strip()))

    page.extract_text(visitor_text=visit)
    return items


def _column_of(x: int) -> str | None:
    for name, lo, hi in _COL_BOUNDS:
        if lo <= x < hi:
            return name
    return None


def _parse_table_page(items: list[tuple[int, int, str]], task_type: int) -> list[RubricChunk]:
    """把一页表格切成若干 (criterion, band) 块。"""
    # 1) 找 band 锚点：左列 (x<45) 的单数字。
    anchors = sorted(
        ((y, int(t)) for x, y, t in items if x < 45 and re.fullmatch(r"[0-9]", t)),
        reverse=True,  # y 大在上 → band 9 在最上
    )
    if not anchors:
        return []

    # 2) 行边界：相邻 band 锚点的中点；表头裁掉。
    # band 数字在行内是垂直居中的，用「anchor+常数」会误切最上 band 的首行，
    # 所以表头切线取左列 "Band"/"Score" 标签那一行的下沿，稳定可靠。
    label_ys = [y for x, y, t in items if x < 60 and t in ("Band", "Score")]
    top_cut = (min(label_ys) - 4) if label_ys else (anchors[0][0] + 8)
    bounds: list[tuple[int, int, int]] = []  # (band, y_low, y_high)
    for i, (y, band) in enumerate(anchors):
        y_high = top_cut if i == 0 else (anchors[i - 1][0] + y) // 2
        y_low = -10_000 if i == len(anchors) - 1 else (anchors[i + 1][0] + y) // 2
        bounds.append((band, y_low, y_high))

    # 3) 把每个文字片段按 y 归到 band、按 x 归到 criterion。
    cells: dict[tuple[int, str], list[tuple[int, int, str]]] = {}
    for x, y, t in items:
        if y >= top_cut or x < 45:        # 表头/页眉 或 band 数字本身
            continue
        crit = _column_of(x)
        if crit is None:
            continue
        for band, lo, hi in bounds:
            if lo <= y < hi:
                cells.setdefault((band, crit), []).append((y, x, t))
                break

    # 4) 每个单元格内按 y 降序拼成完整描述。
    #    band 0 在官方表里是一句跨列合并的「未作答」说明，拆列后是碎片，
    #    所以把整行碎片合并成一句，并复制到四维，保持网格统一。
    band0_frags: list[tuple[int, int, str]] = []
    chunks: list[RubricChunk] = []
    for (band, crit), frags in cells.items():
        if band == 0:
            band0_frags.extend(frags)
            continue
        frags.sort(key=lambda f: (-f[0], f[1]))
        text = " ".join(f[2] for f in frags).strip()
        if text:
            chunks.append(RubricChunk(task_type, crit, band, text))

    if band0_frags:
        band0_frags.sort(key=lambda f: (-f[0], f[1]))
        merged = " ".join(f[2] for f in band0_frags).strip()
        for crit in CRITERIA:
            chunks.append(RubricChunk(task_type, crit, 0, merged))
    return chunks


# Task 1 在 PDF 的 2-4 页，Task 2 在 6-8 页（0-indexed）。
_TASK_PAGES = {1: [2, 3, 4], 2: [6, 7, 8]}


def parse_rubric(pdf_path: str | Path | None = None) -> list[RubricChunk]:
    """解析整本 band descriptors，返回 72 个 (criterion, band, task) 块。"""
    path = Path(pdf_path) if pdf_path else config.RAW_DIR / "ielts-writing-band-descriptors.pdf"
    reader = PdfReader(str(path))
    out: list[RubricChunk] = []
    for task_type, pages in _TASK_PAGES.items():
        for p in pages:
            out.extend(_parse_table_page(_page_items(reader.pages[p]), task_type))
    return out


if __name__ == "__main__":
    chunks = parse_rubric()
    print(f"解析得到 {len(chunks)} 块（期望 72 = 2 task × 4 criterion × 9 band）")
    by_task = {1: 0, 2: 0}
    for c in chunks:
        by_task[c.task_type] += 1
    print("task1:", by_task[1], "| task2:", by_task[2])
    sample = next(c for c in chunks if c.task_type == 2 and c.criterion == "LR" and c.band == 7)
    print(f"\n样例 [task2 LR band7]:\n{sample.text[:300]}")
