"""剑桥真题 PDF（文本层）→ gold 记录的确定性抽取。

只处理「有文本层」的书。核心思想：印刷体标签（band / 考官评语 / 题目）
用锚定正则确定性抽取，**不让模型去猜**；手写考生正文是乱码，只做预填，
留给人工对照 PDF 订正（见 scripts/extract_gold.py 产出的待填 stub）。

每条记录两类：
  - sample：真考生作文，有官方 band + 考官评语  → gold 评测集核心
  - model ：剑桥范例作文（印刷体，无 band）       → 高分锚点
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# ── 锚点正则/标记 ──────────────────────────────────────────────────
# 答案块的两个通用信号（跨书稳定）：带逗号的 "TEST n, WRITING TASK m" 头，
# 与每篇 sample 必有的 "answer written by a candidate"。不依赖会变体/被 OCR 弄花的
# 章节页眉（"Sample answers" / "Model and sample answers" / "Sample Writing answers"）。
BAND_RE = re.compile(r"achieved a Band\s*([0-9]+(?:\.[0-9])?)", re.I)
COMMENT_MARK = "examiner's comment:"
# 答案头分隔符在不同书里被 OCR 成逗号/中文逗号/波浪号等；必须有分隔符才算答案头，
# 否则会和题目页的 "Test n WRITING TASK m"（无分隔符）混淆。
TEST_RE = re.compile(r"TEST\s*([0-9])\s*[，,~∼·]\s*WRITING TASK\s*([0-9])", re.I)
# General Training 段（TEST A/B）：体裁不同（Task1 是书信），本项目只要 Academic，跳过。
GT_RE = re.compile(r"TEST\s*[A-Z]\s*[，,~∼·]\s*WRITING TASK", re.I)
ANSWER_WRITTEN = "answer written by a candidate"   # 每篇 sample 的固定起句
PROMPT_HINT = "you should spend about"             # 题目页的固定句

# 乱码信号：花体手写被劣质数字化后的典型特征。
# 注意：'|' '\' '[]' 是考官评语里引用例词的分隔符（干净文本），不能算乱码。
_JUNK_CHARS = re.compile(r"[\^~丨匕生伊咀]")


def book_tag(path: str | Path) -> str:
    """从文件名取一个短 source 标签，如 cam13 / cam17。"""
    m = re.search(r"(\d+)", Path(path).stem)
    return f"cam{m.group(1)}" if m else Path(path).stem


def _is_junk(word: str) -> bool:
    """一个 token 看起来是不是手写乱码。"""
    if _JUNK_CHARS.search(word):
        return True
    if re.search(r"[A-Za-z][0-9]|[0-9][A-Za-z]", word):   # 字母数字混杂
        return True
    if re.search(r"[a-z]-[a-z]", word):                   # 词内连字符（be-nefits / re-spe-ct）
        return True
    if any(ord(c) > 0x2000 for c in word):                # 混入 CJK 等
        return True
    return False


def _split_comment_essay(text: str) -> tuple[str, str]:
    """把 "examiner's comment:" 之后的文本切成 (干净评语, 乱码正文)。

    评语是干净英文、正文是乱码：定位首个乱码 token，再回退到它之前最近的
    句号处切分——评语正好停在 "...is incomplete." 之类的句末，正文从 "W i6 clear…" 起。
    """
    words = text.split()
    k = next((i for i, w in enumerate(words) if _is_junk(w)), None)
    if k is None:
        return text.strip(), ""             # 全干净（无乱码正文）
    prefix = " ".join(words[:k])            # 与 text 同为单空格，下标可直接复用
    cut = prefix.rfind(". ")
    cut = (cut + 1) if cut != -1 else len(prefix)
    return text[:cut].strip(), text[cut:].strip()


@dataclass
class GoldRecord:
    source: str
    test: int
    task_type: int
    answer_type: str           # 'sample' | 'model'
    band: float | None
    examiner_comment: str | None
    prompt: str
    body_raw: str              # 文本层抽出的（可能乱码）正文，供人工订正
    pages: list[int] = field(default_factory=list)
    tt_explicit: bool = True   # (test,task) 来自显式 TEST 标记（True）还是顺延推断（False）
    prompt_ok: bool = True      # 是否匹配到题目


def _page_texts(reader: PdfReader) -> list[str]:
    return [" ".join((p.extract_text() or "").split()) for p in reader.pages]


def _next_tt(last: tuple[int, int] | None) -> tuple[int, int]:
    """无显式标记时顺延：task1→同 test 的 task2；task2→下一 test 的 task1。"""
    if last is None:
        return (1, 1)
    return (last[0], 2) if last[1] == 1 else (last[0] + 1, 1)


_MODEL_PREAMBLE = re.compile(
    r"This model has been prepared by an examiner.*?approaches\.\s*",
    re.I | re.S)


def _clean_model_body(text: str) -> str:
    """model 印刷体范文：去掉页眉/TEST 标记/范例声明样板，留正文。"""
    text = TEST_RE.sub(" ", text)
    for junk in ("Model and sample answers for Writing tasks",
                 "Sample answers for Writing tasks", "Sample Writing answers",
                 "MODEL ANSWER", "SAMPLE ANSWER"):
        text = text.replace(junk, " ")
    text = _MODEL_PREAMBLE.sub("", text)          # 去掉每篇 model 开头的固定声明
    return " ".join(text.split()).strip()


def extract_book(pdf_path: str | Path) -> list[GoldRecord]:
    reader = PdfReader(str(pdf_path))
    pages = _page_texts(reader)
    src = book_tag(pdf_path)

    prompts_in_order = [t for t in pages if PROMPT_HINT in t.lower()]

    # 切块：页内有 TEST 标记 或 "answer written by a candidate" → 起新块；
    # 进入答案区后、既非新块又非题目页 → 续上一块（正文溢出页）。
    blocks: list[dict] = []
    in_answers = False
    last_tt: tuple[int, int] | None = None
    for i, t in enumerate(pages):
        if in_answers and GT_RE.search(t):
            break                       # 进入 GT 段即停（GT 总在 Academic 之后）
        marker = TEST_RE.search(t)
        is_start = bool(marker) or (ANSWER_WRITTEN in t.lower())
        if is_start:
            tt = (int(marker.group(1)), int(marker.group(2))) if marker else _next_tt(last_tt)
            blocks.append({"text": t, "pages": [i], "tt": tt, "explicit": bool(marker)})
            last_tt = tt
            in_answers = True
        elif in_answers and PROMPT_HINT not in t.lower():
            blocks[-1]["text"] += " " + t
            blocks[-1]["pages"].append(i)

    # 题目按出现顺序与答案块一一对应（同为 test 升序、task1→task2）。
    pmap: dict[tuple[int, int], str] = {}
    for i, blk in enumerate(blocks):
        if i < len(prompts_in_order):
            pmap[blk["tt"]] = prompts_in_order[i]

    records: list[GoldRecord] = []
    for blk in blocks:
        text = blk["text"]
        test, task = blk["tt"]
        bandm = BAND_RE.search(text)
        is_model = bandm is None
        band = float(bandm.group(1)) if bandm else None

        ci = text.lower().find(COMMENT_MARK)
        if ci >= 0:
            comment, body_raw = _split_comment_essay(text[ci + len(COMMENT_MARK):])
        else:
            comment, body_raw = None, _clean_model_body(text)

        records.append(GoldRecord(
            source=src, test=test, task_type=task,
            answer_type="model" if is_model else "sample",
            band=band, examiner_comment=comment or None,
            prompt=pmap.get((test, task), ""), body_raw=body_raw,
            pages=blk["pages"], tt_explicit=blk["explicit"],
            prompt_ok=bool(pmap.get((test, task))),
        ))
    return records


if __name__ == "__main__":
    import sys
    recs = extract_book(sys.argv[1] if len(sys.argv) > 1
                        else "data/raw/【13】剑桥雅思真题13.pdf")
    print(f"抽出 {len(recs)} 块")
    for r in recs:
        flag = ("" if r.tt_explicit else " (tt顺延)") + ("" if r.prompt_ok else " ⚠无题目")
        print(f"  {r.source} T{r.test} task{r.task_type} {r.answer_type} "
              f"band={r.band} pages={r.pages} comment={len(r.examiner_comment or '')}c "
              f"body={len(r.body_raw)}c{flag}")
