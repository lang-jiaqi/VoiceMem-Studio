"""回复层：核心交出 ``Turn`` 之后的那一步——两条路，一个口子。

``voicemem/stream.py`` 是输入侧（音频 → 记忆），这里是输出侧（记忆 → 回复）。两条路：

    # 路 A：用内置的（OpenAI 兼容 api，流式）
    vm = VoiceMem.from_config({"reply": {"provider": "openai",
                                         "config": {"model": "gpt-4o-mini"}}})

    # 路 B：用自己的模型/函数
    vm = VoiceMem(reply=my_fn)

两条路拿到的调用口完全一样::

    answer = await vm.reply(turn)                      # 收全，返回整串
    async for delta in vm.reply_stream(turn):  ...     # 流式，逐字吐

``my_fn`` 写成下面任意一种都行，``normalize()`` 会把它们统一成异步生成器::

    def       my_fn(text, memory_context) -> str          # 同步：自动丢线程，不阻塞事件循环
    async def my_fn(text, memory_context) -> str          # 协程
    async def my_fn(text, memory_context): yield delta    # 异步生成器（流式）

**TTS 不在这里。** 回复层只产出文本；要出声用 ``voicemem/tts.py``——
``speak_stream(vm.reply_stream(turn))`` 边生成边合成，见 examples/03_simple_agent_with_voicemem_memory.py。
"""
from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from voicemem.llm_config import resolve_api_key, resolve_base_url, resolve_model
from voicemem.prompt_trace import record_request


@dataclass(frozen=True)
class ReplyRequestOptions:
    """Provider-neutral, task-local options for one reply request."""

    reasoning_effort: str = "none"


_REQUEST_OPTIONS: ContextVar[ReplyRequestOptions | None] = ContextVar(
    "voicemem_reply_request_options", default=None)


@contextmanager
def reply_request_options(*, reasoning_effort: str = "none"):
    """Apply per-request options across async provider iteration."""
    if reasoning_effort not in {"none", "low", "high", "max"}:
        raise ValueError(f"unsupported reasoning_effort={reasoning_effort!r}")
    token = _REQUEST_OPTIONS.set(ReplyRequestOptions(reasoning_effort))
    try:
        yield
    finally:
        _REQUEST_OPTIONS.reset(token)


async def apply_reply_request_options(stream, *, reasoning_effort: str = "none"):
    """Iterate a normalized reply stream with task-local provider options."""
    with reply_request_options(reasoning_effort=reasoning_effort):
        async for item in stream:
            yield item

# memory_context 只是「记得关于用户的哪些事」，本身不含人设/风格要求，所以内置
# provider 把它接在人设后面，而不是拿它整个当 system prompt。
#
# 人设在 voicemem/persona.py，**按这个空间的语言选**（建空间时定一次）。以前这里
# 写死一句中文，英文库拿到的也是中文人设；而人设是 system 里唯一稳定的前缀，
# 写死意味着它跟记忆语言可以不一致，模型每轮都要自己调和这个矛盾。
def default_system() -> str:
    from voicemem import persona
    return persona.system_prompt()


def compose_system(memory_context: str, system: str | None = None) -> str:
    """人设 + 记忆 → system prompt。两边都可能为空。"""
    parts = [system or default_system()]
    if memory_context:
        parts.append(memory_context)
    return "\n\n".join(parts)


def openai_reply(model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, system: str | None = None) -> Callable:
    """内置回复 provider：OpenAI 兼容 api，流式吐字。返回一个异步生成器函数。

    模型走 ``reply`` 角色：``model`` 参数 → ``VOICEMEM_REPLY_MODEL`` → 跟随 ``chat``。
    回复是用户直接听得见的一路，所以单独留了一个角色让它能和后台整理记忆的模型
    分开配；不配就跟着 chat 走，不会出现"设了模型但回复还在用默认值"这种一半生效。
    ``import voicemem`` 不会因此要求有 key（client 首次调用时才建）。
    """
    client = None

    async def fn(text: str, memory_context: str = "",
                 history: list | None = None) -> AsyncIterator[str]:
        """``history``：user/assistant 交替的历史消息，排在 system 和本轮之间。

        为什么历史要单独传、而不是拼进 memory_context：服务端的 prompt 缓存复用的是
        **最长公共前缀**。记忆每轮都变，把它和历史一起塞进 system，前缀从人设之后
        就断了，历史再长也复用不了。拆开之后顺序是

            [system 人设]  [历史各轮]  [这轮记忆 + 这轮的话]
             └── 稳定，可复用 ──┘      └ 变的全在最后

        不传 history 就是老行为（两条消息），既有调用方不受影响。
        """
        nonlocal client
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=resolve_api_key(api_key),
                base_url=resolve_base_url(base_url),
            )
        msgs = [{"role": "system", "content": system or default_system()}]
        msgs += list(history or [])
        # 记忆跟本轮的话放同一条消息：它是"回答这句话时该知道的事"，本来就属于
        # 这一轮；单独一条 system 会把它变成前缀的一部分，缓存又断了。
        msgs.append({"role": "user",
                     "content": f"{memory_context}\n\n{text}" if memory_context else text})
        request = {"model": resolve_model(model, "reply"), "stream": True, "messages": msgs}
        if str(request["model"]).startswith("qwen"):
            options = _REQUEST_OPTIONS.get() or ReplyRequestOptions()
            request["extra_body"] = {"enable_thinking": options.reasoning_effort != "none"}
        record_request("llm", "openai", request)
        stream = await client.chat.completions.create(**request)
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        finally:
            await stream.close()

    return fn


def deepseek_reply(model: str | None = None, api_key: str | None = None,
                   base_url: str | None = None, system: str | None = None,
                   protocol: str = "deepseek") -> Callable:
    """DeepSeek 流式语音回复：独立凭据，不改变后台记忆整理的厂商配置。"""
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DeepSeek 回复需要 DEEPSEEK_API_KEY；不要把密钥写进仓库")
    import json

    import httpx
    model = model or os.environ.get("VOICEMEM_DEEPSEEK_MODEL", "deepseek-v4-flash")
    url = (base_url or "https://api.deepseek.com").rstrip("/") + "/chat/completions"
    first_token_timeout = max(
        0.2, float(os.environ.get("VOICEMEM_DEEPSEEK_FIRST_TOKEN_TIMEOUT", "2.0")))
    client = None

    async def reset_client(expected=None):
        """Drop a stalled pooled connection without closing a newer one."""
        nonlocal client
        if client is None or (expected is not None and client is not expected):
            return
        old, client = client, None
        await old.aclose()

    async def stream_once(active_client, request):
        """Yield one HTTP attempt. The caller owns first-token timing/retry."""
        async with active_client.stream(
                "POST", url, headers={"Authorization": f"Bearer {key}"},
                json=request) as response:
            response.raise_for_status()
            data = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    payload = "\n".join(data)
                    data.clear()
                    if payload == "[DONE]":
                        return
                    chunk = json.loads(payload)
                    if "error" in chunk:
                        raise RuntimeError("DeepSeek stream returned an error")
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        reasoning = delta.get("reasoning_content")
                        if reasoning:
                            yield "reasoning", reasoning
                        content = delta.get("content")
                        if content:
                            yield "content", content
            raise RuntimeError("DeepSeek stream ended before [DONE]")

    async def fn(text: str, memory_context: str = "", history: list | None = None):
        nonlocal client
        messages = [{"role": "system", "content": system or default_system()}]
        messages += list(history or [])
        messages.append({"role": "user", "content":
                         f"{memory_context}\n\n{text}" if memory_context else text})
        # Consume SSE directly so early generator closure releases httpcore cleanly.
        options = _REQUEST_OPTIONS.get() or ReplyRequestOptions()
        effort = options.reasoning_effort
        default_budget = {"none": 512, "low": 1024, "high": 4096, "max": 8192}[effort]
        max_tokens = int(os.environ.get(
            f"VOICEMEM_DEEPSEEK_MAX_TOKENS_{effort.upper()}", default_budget))
        request = {"model": model, "messages": messages, "stream": True,
                   "thinking": {"type": "disabled" if effort == "none" else "enabled"},
                   "max_tokens": max_tokens}
        if protocol == "qwen":
            request.pop("thinking")
            request["enable_thinking"] = effort != "none"
        elif effort != "none":
            request["reasoning_effort"] = effort
        record_request("llm", protocol, request)
        last_error = None
        for attempt in range(2):
            if client is None:
                client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
            active_client = client
            stream = stream_once(active_client, request)
            try:
                # SSE keepalives renew httpx's read timeout. Enforce wall-clock
                # time so a live connection cannot wait forever without content.
                first = await asyncio.wait_for(
                    anext(stream), timeout=first_token_timeout)
            except asyncio.CancelledError:
                await stream.aclose()
                raise
            except httpx.HTTPStatusError:
                await stream.aclose()
                raise                         # Explicit HTTP failures are not transient stalls.
            except (asyncio.TimeoutError, httpx.TransportError,
                    RuntimeError, StopAsyncIteration) as exc:
                last_error = exc
                await stream.aclose()
                if attempt == 0:
                    print(f"[llm] DeepSeek 首字超时/断流，立即换连接重试一次："
                          f"{type(exc).__name__}", flush=True)
                    await reset_client(active_client)
                    continue
                await reset_client(active_client)
                if isinstance(exc, asyncio.TimeoutError):
                    raise TimeoutError(
                        f"DeepSeek 连续两次在 {first_token_timeout:g}s 内没有返回首字") from exc
                if isinstance(exc, StopAsyncIteration):
                    raise RuntimeError("DeepSeek 连续两次返回了空回复") from exc
                raise
            else:
                try:
                    kind, value = first
                    if kind == "content":
                        yield value
                    async for kind, value in stream:
                        if kind == "content":
                            yield value
                    return
                finally:
                    await stream.aclose()
        raise RuntimeError("DeepSeek reply failed") from last_error

    async def close():
        await reset_client()

    fn.aclose = close
    return fn


def normalize(fn: Callable) -> Callable:
    """把任意形状的回复函数规格化成「异步生成器函数」这一种。

    同步函数走 ``asyncio.to_thread``——回复生成是秒级的，直接在事件循环里跑会卡住
    读麦克风那条线。返回值若本身是异步可迭代对象（例如一个包装别人生成器的
    lambda），照样按流式展开。
    """
    # 可调用**对象**（``VoiceMem(reply=LocalLLM())`` 这种）要看它的 __call__：
    # inspect 的那几个判断只认函数，对实例一律返回 False，于是一个流式的
    # provider 会被当成普通同步函数走到最后那条分支——history 被丢掉、还白跑
    # 一趟 to_thread。实测后果：本地模型永远收不到对话历史，预热的 KV 前缀
    # 也就永远对不上，首字从 1.3s 退回 2.7s。
    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        call = getattr(type(fn), "__call__", None)
        if call is not None and (inspect.isasyncgenfunction(call)
                                 or inspect.iscoroutinefunction(call)):
            fn = fn.__call__

    if inspect.isasyncgenfunction(fn):
        try:
            if "history" in inspect.signature(fn).parameters:
                return fn
        except (TypeError, ValueError):
            pass

        async def wrap(text: str, memory_context: str = "",
                       history: list | None = None) -> AsyncIterator[str]:
            async for d in fn(text, memory_context):   # 老签名：把 history 吞掉
                yield d
        return wrap

    # history 只传给**接得住它的**函数：用户自己写的 reply 多半只有两个参数
    # （文档里就是这么写的），硬塞第三个会直接 TypeError。
    def _accepts_history(f) -> bool:
        try:
            return "history" in inspect.signature(f).parameters
        except (TypeError, ValueError):
            return False

    if inspect.iscoroutinefunction(fn):
        async def gen(text: str, memory_context: str = "",
                      history: list | None = None) -> AsyncIterator[str]:
            out = await (fn(text, memory_context, history) if _accepts_history(fn)
                         else fn(text, memory_context))
            if hasattr(out, "__aiter__"):
                async for delta in out:
                    yield delta
            else:
                yield out
        return gen

    async def gen(text: str, memory_context: str = "",
                  history: list | None = None) -> AsyncIterator[str]:
        out = await asyncio.to_thread(fn, text, memory_context)
        if hasattr(out, "__aiter__"):
            async for delta in out:
                yield delta
        else:
            yield out
    return gen


async def capture(deltas: AsyncIterator[str], on_done: Callable[[str], None]) -> AsyncIterator[str]:
    """原样透传每个 delta，说完时把整句交给 ``on_done``。

    agent 说的那半也该进记忆，但不该让调用方多写一行、也不能等收全再吐。
    被打断时 ``finally`` 交出已吐出去的那部分——用户听到多少就记多少。
    """
    parts: list[str] = []
    try:
        async for delta in deltas:
            parts.append(delta)
            yield delta
    finally:
        on_done("".join(parts))


def unpack(turn_or_text, memory_context: str = "") -> tuple[str, str]:
    """``vm.reply(turn)`` 的便利：Turn / StreamState 直接拆成 (text, memory_context)。"""
    text = getattr(turn_or_text, "text", None)
    if text is not None and hasattr(turn_or_text, "memory_context"):
        return text, (memory_context or turn_or_text.memory_context)
    return turn_or_text, memory_context
