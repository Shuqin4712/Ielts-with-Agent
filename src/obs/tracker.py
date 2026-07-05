"""LLM 用量追踪：LangChain callback + JSONL 落盘 + 操作标签。

一行一条调用记录，写 `data/logs/llm_calls.jsonl`：
  {ts, op, model, tier, input_tokens, output_tokens, total_tokens, latency_ms, error?}

- **tier** 由 model 名反查（含 "pro" → pro，否则 flash），用来算成本路由省了多少。
- **op**（operation）是粗粒度标签（grade / chat / lookup / 工具名…），靠 contextvar 传，
  只在**胶水边界**设置（api 端点、call_json），不侵入打分逻辑。
- 并行 eval 会多线程并发写，故 append 带线程锁；contextvar 每线程独立、天然隔离。
- `OBS_LOG=0` 可整体关闭（默认开）。关闭时 callback 变纯空操作。
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .. import config

# 落盘位置：与主库/向量库分开，放 data/logs/。
LOG_PATH = config.DATA_DIR / "logs" / "llm_calls.jsonl"

_lock = threading.Lock()
# 粗粒度操作标签；默认 None（未分类，如直接跑图/CLI 未包 operation 的调用）。
_current_op: contextvars.ContextVar[str | None] = contextvars.ContextVar("obs_op", default=None)


def _enabled() -> bool:
    return os.getenv("OBS_LOG", "1") != "0"


def _tier_of(model: str | None) -> str:
    """model 名 → 成本档位。config 里 pro 名含 'pro'，flash 含 'flash'。"""
    if model and "pro" in model:
        return "pro"
    return "flash"


@contextlib.contextmanager
def operation(label: str):
    """给这段内的所有 LLM 调用打上操作标签（供日志归类）。

    只在胶水层用（不改打分/记忆逻辑）：
        with operation("grade"):
            graph.invoke(...)

    ⚠️ 不要用它包住会被 StreamingResponse 消费的**生成器体**——Starlette 把同步
    生成器的每次 next() 放进独立的线程上下文跑，token 会跨 context 失效。
    那种场景用 wrap_operation()。
    """
    token = _current_op.set(label)
    try:
        yield
    finally:
        try:
            _current_op.reset(token)
        except ValueError:          # 万一跨 context：观测绝不炸业务
            _current_op.set(None)


def wrap_operation(label: str, gen):
    """给生成器**每次迭代**打操作标签（SSE 流式端点用）。

    set/reset 成对发生在同一次 next() 的线程上下文里；LLM 调用都在 next()
    执行期间发生，故标签能正确命中，且不会跨 yield 持有 token。
    """
    while True:
        token = _current_op.set(label)
        try:
            item = next(gen)
        except StopIteration:
            return
        finally:
            try:
                _current_op.reset(token)
            except ValueError:
                _current_op.set(None)
        yield item


def _extract_tokens(response) -> tuple[int, int, int]:
    """从 LLMResult 抽 (input, output, total) token；流式/缺失时尽力而为，取不到记 0。"""
    out = getattr(response, "llm_output", None) or {}
    usage = out.get("token_usage") or out.get("usage") or {}
    pin = usage.get("prompt_tokens")
    pout = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if pin is None and pout is None:
        # 退回到 message.usage_metadata（流式对话常走这里；可能仍缺）。
        try:
            for gens in response.generations:
                for g in gens:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None)
                    if um:
                        pin = (pin or 0) + (um.get("input_tokens") or 0)
                        pout = (pout or 0) + (um.get("output_tokens") or 0)
        except Exception:
            pass
    pin, pout = int(pin or 0), int(pout or 0)
    return pin, pout, int(total if total is not None else pin + pout)


class LlmUsageCallback(BaseCallbackHandler):
    """被动记录每次 LLM 调用。不抛异常干扰主流程（观测失败绝不影响业务）。"""

    def __init__(self) -> None:
        self._starts: dict[Any, float] = {}
        self._models: dict[Any, str] = {}

    # chat 模型（ChatOpenAI）走 on_chat_model_start；补 on_llm_start 以防万一。
    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, **kw):
        self._begin(run_id, serialized, metadata, kw)

    def on_llm_start(self, serialized, prompts, *, run_id=None, metadata=None, **kw):
        self._begin(run_id, serialized, metadata, kw)

    def _begin(self, run_id, serialized, metadata, kw):
        if not _enabled():
            return
        self._starts[run_id] = time.perf_counter()
        model = None
        if metadata:
            model = metadata.get("ls_model_name")
        if not model:
            params = kw.get("invocation_params") or {}
            model = params.get("model")
        if not model and serialized:
            model = (serialized.get("kwargs") or {}).get("model")
        self._models[run_id] = model or "unknown"

    def on_llm_end(self, response, *, run_id=None, **kw):
        if not _enabled():
            return
        pin, pout, total = _extract_tokens(response)
        model = None
        out = getattr(response, "llm_output", None) or {}
        model = out.get("model_name") or self._models.get(run_id, "unknown")
        self._write(run_id, model, pin, pout, total, error=None)

    def on_llm_error(self, error, *, run_id=None, **kw):
        if not _enabled():
            return
        self._write(run_id, self._models.get(run_id, "unknown"), 0, 0, 0, error=str(error)[:200])

    def _write(self, run_id, model, pin, pout, total, error):
        start = self._starts.pop(run_id, None)
        self._models.pop(run_id, None)
        latency_ms = round((time.perf_counter() - start) * 1000, 1) if start else None
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "op": _current_op.get(),
            "model": model,
            "tier": _tier_of(model),
            "input_tokens": pin,
            "output_tokens": pout,
            "total_tokens": total,
            "latency_ms": latency_ms,
        }
        if error:
            rec["error"] = error
        try:
            with _lock:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 观测绝不影响业务


# 进程级单例：get_llm 统一挂它。
usage_callback = LlmUsageCallback()
