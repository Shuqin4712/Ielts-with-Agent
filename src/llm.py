"""DeepSeek LLM 封装 + 成本路由。

用法模式：业务代码不直接 new ChatOpenAI，而是 `get_llm("flash")` /
`get_llm("pro")`，把「用哪个档位的模型」这个决策收口到一处，方便日后
统一调参、加缓存、加并发限流。
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config
from . import obs


def get_llm(tier: str = "flash", *, thinking: bool = False,
            timeout: float = 60, temperature: float | None = None,
            **kwargs) -> ChatOpenAI:
    """返回一个接好 DeepSeek 的 ChatOpenAI。

    tier:     "flash"（快/省）或 "pro"（推理）。
    thinking: 仅 pro 有意义；开启 DeepSeek 的思考模式。注意该模式下
              temperature 等采样参数无效，因此不要再传。
    timeout:  单次请求超时（秒）。**关键**：卡住的调用要快速失败而非永久阻塞
              （否则批量评测会挂死）。pro+thinking 慢，可调大。

    DeepSeek 是 OpenAI 兼容接口，所以这里复用 langchain-openai 的
    ChatOpenAI，只把 base_url 指到 DeepSeek 即可。max_retries=0：客户端不自动
    重试，重试统一交给业务侧的 with_backoff（带指数退避，专治 429）。
    """
    model = config.MODEL_PRO if tier == "pro" else config.MODEL_FLASH

    extra_body = {}
    if thinking:
        # DeepSeek 思考模式开关；reasoning_effort 控制思考深度。
        extra_body["thinking"] = {"type": "enabled"}
        kwargs.setdefault("reasoning_effort", "medium")
    elif temperature is not None:
        # thinking 模式下 temperature 无效、不能设；仅非 thinking 时钉温以求可复现。
        kwargs["temperature"] = temperature

    # 可观测性：挂一个 passive callback 记录 token/延迟/档位（阶段 6）。
    # 只观察不改行为；OBS_LOG=0 时它自身变空操作。业务侧无感知。
    callbacks = kwargs.pop("callbacks", None) or []
    callbacks = [*callbacks, obs.usage_callback]

    return ChatOpenAI(
        model=model,
        api_key=config.require_api_key(),
        base_url=config.DEEPSEEK_BASE_URL,
        extra_body=extra_body or None,
        timeout=timeout,
        max_retries=0,
        callbacks=callbacks,
        **kwargs,
    )


# DeepSeek 有并发限流（HTTP 429），调用要带指数退避重试。
# 这是一个可复用的装饰器骨架：业务侧把「真正发起调用」的函数包起来即可。
def with_backoff(fn):
    """给任意可调用对象套指数退避重试（最多 5 次，2→4→8…秒）。"""
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )(fn)
