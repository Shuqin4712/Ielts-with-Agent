"""本地 embedding 封装（经 Ollama）。

DeepSeek 不提供 embedding，所以这一层完全独立于 LLM：
LLM = DeepSeek API，Embedding = 本地 Ollama（bge-m3）。
业务代码只调 `embed_text` / `embed_texts`，不关心底层是哪个模型。
"""
from __future__ import annotations

import ollama

from .. import config


def _client() -> ollama.Client:
    return ollama.Client(host=config.OLLAMA_HOST)


def embed_text(text: str, *, model: str | None = None) -> list[float]:
    """把一段文本编码成向量。"""
    model = model or config.EMBED_MODEL
    resp = _client().embeddings(model=model, prompt=text)
    return list(resp["embedding"])


def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """批量编码。Ollama embeddings 接口是单条的，这里循环即可（量大时再优化）。"""
    return [embed_text(t, model=model) for t in texts]
