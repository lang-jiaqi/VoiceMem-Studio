"""Normalize injected reply callables and stream API-generated text.

Providers receive user text, memory context and optional dialogue history.
Request-local options control provider behavior without changing memory state.
This layer produces text; Studio speech adapters synthesize and play audio."""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from voicemem.llm_config import resolve_api_key, resolve_base_url, resolve_model
from voicemem.utils.common.prompt_trace import record_request


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
        try:
            async for item in stream:
                yield item
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

def default_system() -> str:
    """Load the existing Studio default lazily when no reply system prompt is supplied."""
    from studio.core.utils.prompts import legacy_persona as persona
    return persona.system_prompt()


def compose_system(memory_context: str, system: str | None = None) -> str:
    """Combine the existing persona and memory blocks, allowing either to be empty."""
    parts = [system or default_system()]
    if memory_context:
        parts.append(memory_context)
    return "\n\n".join(parts)


def compose_reply_messages(text: str, memory_context: str = "", history=None,
                           *, system: str | None = None,
                           context_as_system: bool = False) -> list[dict]:
    """Build provider messages, optionally separating backend context from user speech.

    The default retains the library's existing context-in-user format. Studio
    opts into the leading system message so its private dialogue directives do
    not become attributed user input or replace the persona on providers that
    treat later system messages as complete prompt updates.
    """
    prompt = (compose_system(memory_context, system) if context_as_system
              else (system or default_system()))
    messages = [{"role": "system", "content": prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": (
        f"{memory_context}\n\n{text}" if memory_context and not context_as_system else text)})
    return messages


def openai_reply(model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, system: str | None = None,
                 *, context_as_system: bool = False) -> Callable:
    """Return a lazy OpenAI-compatible streaming reply provider.

    Model and credentials resolve through the reply role. The SDK client is
    created on first use. ``context_as_system`` separates backend context from
    user input without changing the returned callable's argument contract.
    """
    client = None

    async def fn(text: str, memory_context: str = "",
                 history: list | None = None) -> AsyncIterator[str]:
        """Yield reply deltas with dialogue history between system and current input.

        Library defaults append context to user input for history prefix reuse;
        Studio opts into system context to keep private directives out of speech.
        """
        nonlocal client
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=resolve_api_key(api_key),
                base_url=resolve_base_url(base_url),
            )
        msgs = compose_reply_messages(text, memory_context, history, system=system,
                                      context_as_system=context_as_system)
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
                   protocol: str = "deepseek", *,
                   context_as_system: bool = False) -> Callable:
    """Stream DeepSeek or Qwen replies with optional system-scoped backend context."""
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
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason:
                            yield "finish", finish_reason
            raise RuntimeError("DeepSeek stream ended before [DONE]")

    async def fn(text: str, memory_context: str = "", history: list | None = None):
        nonlocal client
        messages = compose_reply_messages(text, memory_context, history, system=system,
                                          context_as_system=context_as_system)
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
        last_error = None
        for attempt in range(2):
            if client is None:
                client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
            active_client = client
            record_request("llm", protocol, request)
            stream = stream_once(active_client, request)
            started = time.monotonic()
            content_chars = reasoning_chars = 0
            has_content = False
            finish_reason = "unknown"
            status = "failed"
            try:
                # Keepalives renew the read timeout; bound the first content or
                # reasoning event separately from the remaining token stream.
                first = await asyncio.wait_for(
                    anext(stream), timeout=first_token_timeout)
                kind, value = first
                while True:
                    if kind == "content":
                        content_chars += len(value)
                        has_content = has_content or bool(value.strip())
                        yield value
                    elif kind == "reasoning":
                        if not reasoning_chars:
                            print(f"[llm] {protocol} thinking started effort={effort} "
                                  f"attempt={attempt + 1}", flush=True)
                        reasoning_chars += len(value)
                    elif kind == "finish":
                        finish_reason = value
                    try:
                        kind, value = await anext(stream)
                    except StopAsyncIteration:
                        break
                if not has_content:
                    raise RuntimeError(
                        f"{protocol} returned no answer content "
                        f"(finish_reason={finish_reason}, reasoning_chars={reasoning_chars})")
                status = "complete"
                return
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except GeneratorExit:
                status = "closed"
                raise
            except httpx.HTTPStatusError:
                raise                         # Explicit HTTP failures are not transient stalls.
            except (asyncio.TimeoutError, httpx.TransportError,
                    RuntimeError, StopAsyncIteration) as exc:
                last_error = exc
                await stream.aclose()
                # Never replay an answer whose content has already been emitted.
                if has_content:
                    raise
                if attempt == 0:
                    print(f"[llm] {protocol} 首字超时/断流/空正文，换连接重试一次："
                          f"{type(exc).__name__} finish_reason={finish_reason}", flush=True)
                    await reset_client(active_client)
                    continue
                await reset_client(active_client)
                if isinstance(exc, asyncio.TimeoutError):
                    raise TimeoutError(
                        f"DeepSeek 连续两次在 {first_token_timeout:g}s 内没有返回首字") from exc
                if isinstance(exc, StopAsyncIteration):
                    raise RuntimeError("DeepSeek 连续两次返回了空回复") from exc
                raise
            finally:
                await stream.aclose()
                print(f"[llm] {protocol} stream status={status} effort={effort} "
                      f"attempt={attempt + 1} elapsed_ms={(time.monotonic() - started) * 1000:.0f} "
                      f"finish_reason={finish_reason} content_chars={content_chars} "
                      f"reasoning_chars={reasoning_chars}", flush=True)
        raise RuntimeError("DeepSeek reply failed") from last_error

    async def close():
        await reset_client()

    fn.aclose = close
    return fn


def normalize(fn: Callable) -> Callable:
    """Normalize synchronous, asynchronous or streaming reply callables to async delta iteration."""
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
    """Forward deltas and notify with emitted text on completion or closure."""
    parts: list[str] = []
    try:
        async for delta in deltas:
            parts.append(delta)
            yield delta
    finally:
        try:
            close = getattr(deltas, "aclose", None)
            if close is not None:
                await close()
        finally:
            on_done("".join(parts))


def unpack(turn_or_text, memory_context: str = "") -> tuple[str, str]:
    """Resolve text and memory context from raw input, Turn or StreamState."""
    text = getattr(turn_or_text, "text", None)
    if text is not None and hasattr(turn_or_text, "memory_context"):
        return text, (memory_context or turn_or_text.memory_context)
    return turn_or_text, memory_context
