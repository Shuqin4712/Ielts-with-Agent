"""save_to_library：把工具产出的条目一键存进词库 / 素材库。

vocab_upgrade → target='vocab'；deconstruct_article / exemplar_provide → target='material'。
是纯函数（便于测试），批次 D 再包成 LangChain tool 给 agent。
"""
from __future__ import annotations

from ..db import library


def save_to_library(target: str, item: dict, *, user_id: str = "default", db_path=None) -> dict:
    """target: 'vocab' | 'material'。item 为字段 dict。返回 {saved_id, target}。"""
    if target == "vocab":
        sid = library.save_vocab(
            item["word"], item.get("context_sentence", ""), item.get("alternatives", []),
            item.get("nuance_note", ""), user_id=user_id,
            source_essay_id=item.get("source_essay_id"), db_path=db_path)
    elif target == "material":
        sid = library.save_material(
            item.get("type", "exemplar"), item.get("content", ""), user_id=user_id,
            outline=item.get("outline"), topic=item.get("topic"), band=item.get("band"),
            tags=item.get("tags"), source_excerpt=item.get("source_excerpt"),
            note=item.get("note"), db_path=db_path)
    else:
        raise ValueError(f"未知 target: {target}（应为 'vocab' 或 'material'）")
    return {"saved_id": sid, "target": target}
