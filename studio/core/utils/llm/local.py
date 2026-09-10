"""Studio local implementation."""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from voicemem.utils.gpu_loop import gpu_loop

MODEL = 'mlx-community/Qwen3.5-4B-4bit'

DEBUG = False

_MARK = "\x00__VM__\x00"

class LocalLLM:
    """Stream local MLX replies through the process GPU scheduler."""

    def __init__(self, model_name: str = MODEL, system: str | None = None):
        self.model_name = model_name

        from voicemem.reply import default_system
        self.system = system or default_system()
        self._m = self._tok = None

        self._base = None
        self._base_ids: list[int] = []
        self._cache = None
        self._cached_prefix: list[int] = []

    def set_system(self, system: str) -> None:
        """Replace the system prompt and invalidate its cached prefix."""
        if system and system != self.system:
            self.system = system
            self._base, self._base_ids = None, []
            self._cache, self._cached_prefix = None, []

    def load(self):
        """Load the local model on the shared GPU scheduler."""
        if self._m is None:
            from mlx_lm import load
            from studio.core.utils.llm.detokenizer_cache import cache_bpe_vocabulary

            def load_cached():
                model, tokenizer = load(self.model_name)
                if cache_bpe_vocabulary(tokenizer) and DEBUG:
                    print("[llm] BPE 只读词表已缓存，每轮解码状态独立", flush=True)
                return model, tokenizer

            loop = gpu_loop()
            if loop.on_thread():
                self._m, self._tok = load_cached()
            else:
                self._m, self._tok = loop.call(load_cached)
        return self._m, self._tok

    def _render(self, msgs) -> str:
        _, tok = self.load()
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)

    def _msgs(self, text: str, memory_context: str, history):
        out = [{"role": "system", "content": self.system}] + list(history or [])
        out.append({"role": "user",
                    "content": f"{memory_context}\n\n{text}" if memory_context else text})
        return out

    def _fork(self, src_cache):
        from mlx_lm.models.cache import make_prompt_cache
        model, _ = self.load()
        out = make_prompt_cache(model)
        for dst, src in zip(out, src_cache):
            st = src.state
            dst.state = (list(st) if isinstance(st, list)
                         else tuple(st) if isinstance(st, tuple) else st)
        return out

    def _prefix_ids(self, memory_context, history) -> list[int]:
        _, tok = self.load()
        full = self._render(self._msgs(_MARK, memory_context, history))
        ids = tok.encode(full[:full.index(_MARK)])

        return ids[:-1]

    def prewarm(self, history=None, memory_context: str = "", *, cancelled=None) -> int:
        """Prepare reusable prefix state, skipping work when cancellation is signaled."""

        def run():
            if cancelled is not None and cancelled.is_set():
                return 0
            return self._prewarm(history, memory_context)

        if cancelled is not None and cancelled.is_set():
            return 0
        return gpu_loop().call(run)

    def _prewarm(self, history, memory_context) -> int:
        import mlx.core as mx
        model, _ = self.load()
        ids = self._prefix_ids(memory_context, history)
        if not ids:
            return 0
        if ids == self._cached_prefix and self._cache is not None:
            return len(ids)
        cache, done, kind = self._pick_base(ids)
        if done < len(ids):
            model(mx.array(ids[done:])[None], cache=cache)
            mx.eval([c.state for c in cache])
        self._cache, self._cached_prefix = cache, ids

        if self._base is None:
            self._base, self._base_ids = cache, list(ids)
            kind = "人设底座 "
        if DEBUG:
            print(f"[llm] 母本{kind}{len(ids) - done} token → 共 {len(ids)}", flush=True)
        return len(ids)

    def _pick_base(self, ids):
        from mlx_lm.models.cache import make_prompt_cache
        for cache, pre, kind in ((self._cache, self._cached_prefix, "续 "),
                                 (self._base, self._base_ids, "接人设底座 ")):
            if cache is not None and len(pre) <= len(ids) and ids[:len(pre)] == pre:
                return self._fork(cache), len(pre), f"{kind}{len(pre)}+"
        return make_prompt_cache(self.load()[0]), 0, "建 "

    def drop_cache(self) -> None:
        """Release prefix cache state before the next generation."""
        self._cache, self._cached_prefix = self._base, list(self._base_ids)

    def _gen_iter(self, text, memory_context, history, submitted=0.0):
        import time as _t
        from mlx_lm import stream_generate
        _t1 = _t.perf_counter()
        _queued = (_t1 - submitted) * 1000 if submitted else 0.0
        model, tok = self.load()
        ids = tok.encode(self._render(self._msgs(text, memory_context, history)))

        cache = prefix = None
        _tf = _t.perf_counter()
        for c, pre in ((self._cache, self._cached_prefix),
                       (self._base, self._base_ids)):
            if c is not None and len(pre) < len(ids) and ids[:len(pre)] == pre:
                cache, prefix = self._fork(c), pre
                break
        _fork_ms = (_t.perf_counter() - _tf) * 1000
        hit = cache is not None
        if not hit:
            prefix = []

        tail = ids[len(prefix):] if hit else ids
        if DEBUG:
            print(f"[llm] prompt {len(ids)} token｜"
                  f"{'命中缓存 ' + str(len(prefix)) if hit else '没命中'}｜"
                  f"要算 {len(ids) - (len(prefix) if hit else 0)} token", flush=True)
            if not hit and prefix:
                i = next((k for k, (a, b) in enumerate(zip(ids, prefix)) if a != b),
                         min(len(ids), len(prefix)))
                print(f"[llm]   前缀在第 {i}/{len(prefix)} 个 token 分叉｜"
                      f"生成 {tok.decode(ids[i:i+24])!r}｜"
                      f"预热 {tok.decode(prefix[i:i+24])!r}", flush=True)

        progress = {}

        def prompt_progress(done, total):
            now = _t.perf_counter()
            if done == 0:
                progress["start"] = progress["prefill"] = now
            elif done < total:
                progress["prefill"] = now
            else:
                progress["ready"] = now

        _g0 = _t.perf_counter()
        first_token = True
        first_text = True
        empty_tokens = 0
        try:
            for r in stream_generate(model, tok, tail, max_tokens=512,
                                     prompt_cache=cache if hit else None,
                                     **({"prompt_progress_callback": prompt_progress} if DEBUG else {})):
                now = _t.perf_counter()
                if DEBUG and first_token:
                    print(f"[llm] 首token {(now - _g0) * 1000:.0f}ms"
                          f"（排队等 GPU 线程 {_queued:.0f} / 渲染+分词 "
                          f"{(_g0 - _t1) * 1000 - _fork_ms:.0f} / fork {_fork_ms:.0f}）",
                          flush=True)
                    if "ready" in progress and "start" in progress:
                        print(f"[llm-detail] 生成器准备 {(progress['start']-_g0)*1000:.0f}ms"
                              f" · 尾部prefill {(progress['prefill']-progress['start'])*1000:.0f}ms"
                              f" · 首token求值/流水准备 {(progress['ready']-progress['prefill'])*1000:.0f}ms"
                              f" · 解码返回 {(now-progress['ready'])*1000:.0f}ms"
                              f" · 未缓存 {len(tail)} token", flush=True)
                first_token = False
                if first_text:
                    if r.text:
                        if DEBUG:
                            print(f"[llm-text] 首个可显示文字 {(now-_g0)*1000:.0f}ms"
                                  f" · 前置空文本token {empty_tokens}", flush=True)
                        first_text = False
                    else:
                        empty_tokens += 1
                yield r.text
        finally:

            del cache
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

    async def __call__(self, text: str, memory_context: str = "",
                       history: list | None = None) -> AsyncIterator[str]:
        """Yield reply text while keeping reasoning private and honoring cancellation."""
        loop = asyncio.get_running_loop()
        import time as _t
        _sub = _t.perf_counter()

        def stamped():
            gen = self._gen_iter(text, memory_context, history, _sub)
            try:
                for value in gen:
                    yield value, _t.perf_counter()
            finally:
                gen.close()

        job = gpu_loop().iter(stamped)
        first_received = True
        try:
            while True:
                item = await loop.run_in_executor(None, job.out.get)
                if item is None:
                    return
                kind, val = item
                if kind == "err":
                    print(f"[llm] 本地生成失败：{type(val).__name__}: {val}", flush=True)
                    return
                val, produced_at = val
                if val:
                    if first_received and DEBUG:
                        now = _t.perf_counter()
                        print(f"[llm-delivery] 首文字回主循环 {(now-produced_at)*1000:.0f}ms"
                              f" · 提交到收到 {(now-_sub)*1000:.0f}ms", flush=True)
                    first_received = False
                    yield val
        finally:

            job.cancel()
