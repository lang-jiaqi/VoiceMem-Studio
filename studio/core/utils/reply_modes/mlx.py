"""Quantized reasoning-depth inference on the process MLX scheduler."""
from __future__ import annotations

import copy
import asyncio
import time
from contextlib import aclosing

from .component import QwenThinkingRouter
from studio.harness.reply_modes.policy import SYSTEM, EXAMPLES


class MLXThinkingRouter(QwenThinkingRouter):
    """Reuse the depth policy with local int8 weights and a static prompt cache.

    Model and KV state belong to the shared GPU thread. Only the system prompt
    and examples enter the prefix cache; each request gets its own mutable copy.
    """

    _PREFILL_STEP = 128

    def __init__(self, model: str | None = None) -> None:
        super().__init__(model=model)
        self._prefix_key = None
        self._prefix_tokens = []
        self._prefix_cache = None

    def _load_mlx(self):
        if self._model is None:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.utils import quantize_model

            model, tokenizer, config = load(
                self.model_name, lazy=True, return_config=True)
            if not config.get('quantization'):
                model, _ = quantize_model(model, config, group_size=64, bits=8)
            mx.eval(model.parameters())
            model.eval()
            self._model, self._tokenizer = model, tokenizer
            print('[thinking] router ready: MLX with static prefix cache', flush=True)
        return self._tokenizer, self._model

    def _predict(self, system: str, examples, prompt: str) -> str:
        from voicemem.engine import gpu_loop

        self._ensure_model_source()
        submitted = time.perf_counter()
        return gpu_loop().run_steps(
            lambda: self._timed_prediction(system, examples, prompt, submitted),
            label="reply_router")

    def _short_text(self, system, prompt, max_tokens, cancelled, deadline):
        from .short_text import mlx_short_text
        return mlx_short_text(self, system, prompt, max_tokens, cancelled, deadline)

    async def _classify_async(self, text, memory_prefetch_hint, history):
        from voicemem.engine import gpu_loop

        prompt = self._context_prompt(text, history, memory_prefetch_hint)
        from studio.harness.prompts import resolve
        system = resolve("router", SYSTEM)
        cached = self._cache.get(self._classification_key(text, prompt, system))
        if cached is not None:
            return cached
        if self._download_default:
            await asyncio.to_thread(self._ensure_model_source)
        output = ''
        submitted = time.perf_counter()
        async with aclosing(gpu_loop().aiter(
                lambda: self._timed_prediction(system, EXAMPLES, prompt, submitted),
                label="reply_router")) as steps:
            async for value in steps:
                if value is not None:
                    output = value
        return self._remember_output(text, prompt, output, system=system)

    def _timed_prediction(self, system, examples, prompt, submitted):
        started = time.perf_counter()
        work_s = max_step_s = 0.0
        count = 0
        status = 'cancelled'
        steps = self._prediction_steps(system, examples, prompt)
        try:
            while True:
                step_started = time.perf_counter()
                try:
                    value = next(steps)
                except StopIteration:
                    status = 'complete'
                    break
                finally:
                    step_s = time.perf_counter() - step_started
                    work_s += step_s
                    max_step_s = max(max_step_s, step_s)
                    count += 1
                yield value
        except Exception:
            status = 'error'
            raise
        finally:
            close = getattr(steps, 'close', None)
            if close is not None:
                close()
            print(f'[route-gpu] queue_ms={(started-submitted)*1000:.0f}'
                  f' compute_ms={work_s*1000:.0f}'
                  f' max_step_ms={max_step_s*1000:.0f} steps={count}'
                  f' status={status}', flush=True)

    def _predict_on_gpu(self, system: str, examples, prompt: str) -> str:
        output = ''
        for value in self._prediction_steps(system, examples, prompt):
            if value is not None:
                output = value
        return output

    def _prediction_steps(self, system: str, examples, prompt: str):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from voicemem.engine import prefill_steps

        tokenizer, model = self._load_mlx()
        examples = tuple(examples)
        messages = [{'role': 'system', 'content': system}]
        for text, label in examples:
            messages.extend(({'role': 'user', 'content': text},
                             {'role': 'assistant', 'content': label}))

        def tokenize(text, generation):
            return tokenizer.apply_chat_template(
                messages + [{'role': 'user', 'content': text}],
                tokenize=True, add_generation_prompt=generation, enable_thinking=False)

        key = (system, examples)
        if key != self._prefix_key:
            # Include a final user message when rendering: Qwen templates can
            # otherwise format earlier assistant turns differently. The common
            # prefix of two distinct stubs stops before request-specific text.
            left, right = tokenize('', False), tokenize('x', False)
            count = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                         min(len(left), len(right)))
            prefix = left[:count]
            cache = make_prompt_cache(model)
            for _ in prefill_steps(model, prefix, cache, step_size=self._PREFILL_STEP):
                yield None
            # Publish only a fully materialized prefix. Cancellation leaves
            # private partial state on the scheduler's owning thread.
            self._prefix_tokens, self._prefix_cache = prefix, cache
            self._prefix_key = key

        tokens = tokenize(prompt, True)
        count = len(self._prefix_tokens)
        prefix_hit = count < len(tokens) and tokens[:count] == self._prefix_tokens
        if prefix_hit:
            tokens = tokens[count:]
            cache = copy.deepcopy(self._prefix_cache)
        else:
            cache = make_prompt_cache(model)
        print(f'[route-input] tokens={len(tokens)} prefix_tokens={count}'
              f' prefix_hit={int(prefix_hit)} prefill_step_tokens={self._PREFILL_STEP}', flush=True)
        # A four-token greedy classifier does not need the text-generation
        # wrapper, which changes the process-wide wired-memory limit and clears
        # allocator caches also used by TTS. Keep inference on the owned stream.
        # Decode from the last bounded batch directly; short inputs need no
        # separate prefill pass just to compute their first greedy token.
        prefill_count = max(0, (len(tokens) - 1) // self._PREFILL_STEP * self._PREFILL_STEP)
        for _ in prefill_steps(model, tokens[:prefill_count], cache,
                               step_size=self._PREFILL_STEP):
            yield None
        pending = mx.array(tokens[prefill_count:])
        generated = []
        for _ in range(4):
            logits = model(pending[None], cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1).item()
            if token in tokenizer.eos_token_ids:
                break
            generated.append(token)
            pending = mx.array([token])
            yield None
        yield tokenizer.decode(generated).strip()
