"""ChromaDB 封装：持久化 client + 两个集合 + 写入/检索。

设计要点（DESIGN §4.4）：
- 集合不绑内置 embedding_function，向量由我们自己的本地 bge-m3 算好后传入，
  让 embedding 层与向量库解耦（换 embedding 模型不动 Chroma）。
- 检索 = metadata 过滤(where) + 向量召回，是「结构感知 RAG」的落地。
"""
from __future__ import annotations

import chromadb

from .. import config
from . import embeddings


def get_client() -> chromadb.ClientAPI:
    """持久化 client，落在 config.CHROMA_DIR。"""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def reset_collection(name: str) -> chromadb.Collection:
    """删了重建，保证每次构建索引是干净的（幂等）。"""
    client = get_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    # cosine 距离更适合文本语义相似度。
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def add_documents(coll: chromadb.Collection, ids, documents, metadatas) -> None:
    """批量写入：先用本地模型把文档编码成向量，再连同 metadata 入库。"""
    vectors = embeddings.embed_texts(list(documents))
    coll.add(ids=list(ids), embeddings=vectors,
             documents=list(documents), metadatas=list(metadatas))


def _to_where(filters: dict | None) -> dict | None:
    """Chroma 多条件需要显式 $and；单条件直接传。"""
    if not filters:
        return None
    if len(filters) == 1:
        return filters
    return {"$and": [{k: v} for k, v in filters.items()]}


def get_by_meta(name: str, where: dict) -> list[dict]:
    """只按 metadata 过滤取（不做向量召回），返回 [{document, metadata}]。

    用于「取某 criterion 的整条 band 阶梯」这类结构化取数：不需要语义相似度，
    只要精确 metadata 匹配，故用 collection.get 而非 query。
    """
    coll = get_client().get_collection(name)
    res = coll.get(where=_to_where(where))
    return [{"document": d, "metadata": m}
            for d, m in zip(res["documents"], res["metadatas"])]


def search(name: str, query: str, *, where: dict | None = None, n: int = 3) -> list[dict]:
    """先 metadata 过滤再向量召回，返回 [{id, document, metadata, distance}]。"""
    coll = get_client().get_collection(name)
    qvec = embeddings.embed_text(query)
    res = coll.query(query_embeddings=[qvec], n_results=n, where=_to_where(where))
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return out
