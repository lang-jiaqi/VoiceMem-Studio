"""Local discriminative depth routing using a shared encoder and trained head."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import math
from pathlib import Path
import threading

import numpy as np

from .component import FAST, SLOW, ThinkingDecision
from .decision_features import VERSION, audio_vector, combine, texts


class LocalDecisionRouter:
    """Return depth probabilities without loading a generative routing model.

    The checkpoint pins its encoder and preprocessing. Acoustic inputs are
    optional observations, never permission to end a turn or skip memory.
    """

    accepts_audio_state = True
    filler_label = 'independent'
    prefetch_filler_text = True

    @property
    def filler_text_timeout_s(self):
        from .independent_filler import generation_timeout
        return generation_timeout()

    def __init__(self, model, *, allow_prototype=False):
        path = Path(model).expanduser()
        self.metadata = json.loads((path / 'decision.json').read_text(encoding='utf-8'))
        if self.metadata.get('version') != VERSION:
            raise ValueError('Unsupported decision checkpoint version')
        if self.metadata.get('prototype', True) and not allow_prototype:
            raise ValueError('Prototype checkpoint requires allow_prototype=true')
        from voicemem.leftbrain.local_embedder import resolve
        self.spec = resolve(self.metadata.get('encoder'), language='zh')
        if self.metadata.get('encoder_repo') != self.spec.repo:
            raise ValueError('Decision encoder identity does not match checkpoint')
        if self.metadata.get('dimensions') != self.spec.dims:
            raise ValueError('Decision encoder dimensions do not match checkpoint')
        # The memory encoder is shared only if its identity already matches.
        if resolve(language='zh').key != self.spec.key:
            raise ValueError('Decision checkpoint must match the active memory encoder')
        with np.load(path / 'head.npz', allow_pickle=False) as data:
            self.weights = np.asarray(data['weights'], dtype=np.float64)
            self.bias = float(data['bias'].item())
        if (self.weights.shape != (3 * self.spec.dims + 6,)
                or not np.isfinite(self.weights).all() or not math.isfinite(self.bias)):
            raise ValueError('Invalid decision head weights')
        self.temperature = float(self.metadata['temperature'])
        self.threshold = float(self.metadata['threshold'])
        if (not math.isfinite(self.temperature) or self.temperature <= 0
                or not math.isfinite(self.threshold) or not 0 < self.threshold < 1):
            raise ValueError('Invalid decision calibration')
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._jobs = {}

    def _encode(self, values):
        from voicemem.leftbrain.local_embedder import resolve_path, shared_model
        model = shared_model(resolve_path(self.spec), self.spec.tokenizer_kwargs)
        return np.asarray(model.encode(
            [self.spec.query_prefix + value for value in values],
            normalize_embeddings=True, show_progress_bar=False))

    def _input_key(self, text, history, audio_state):
        return (*texts(text, history), audio_vector(audio_state).tobytes())

    def classify(self, text, memory_prefetch_hint=False, history=None, *, audio_state=None):
        """Run off-loop; inference caches include every observed input signal."""
        from voicemem.engine.scheduling.cancellation import check_cancelled
        current, context = texts(text, history)
        if not current:
            return ThinkingDecision(FAST, 'decision:empty')
        signals = audio_vector(audio_state)
        key = (current, context, signals.tobytes())
        check_cancelled()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        vectors = self._encode([current, context] if context else [current])
        check_cancelled()
        context_vector = vectors[1] if context else np.zeros_like(vectors[0])
        feature = combine(vectors[0], context_vector, audio_state)
        logit = float(feature @ self.weights + self.bias) / self.temperature
        probability = float(1 / (1 + np.exp(-np.clip(logit, -60, 60))))
        result = ThinkingDecision(SLOW if probability >= self.threshold else FAST,
                                  f'decision:p_deep={probability:.6f}')
        check_cancelled()
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
        return result

    async def classify_async(self, text, memory_prefetch_hint=False, history=None, *, audio_state=None):
        """Share identical jobs; cancelling the final waiter cancels native admission."""
        from voicemem.engine.scheduling.cancellation import run_cancellable
        from voicemem.engine.scheduling.probes import input_executor
        history = [dict(row) for row in history or []]
        audio_state = dict(audio_state) if audio_state is not None else None
        key = (asyncio.get_running_loop(), *self._input_key(text, history, audio_state))
        job = self._jobs.get(key)
        if job is None:
            task = asyncio.create_task(run_cancellable(
                lambda: self.classify(text, memory_prefetch_hint, history, audio_state=audio_state),
                executor=input_executor))
            job = {'task': task, 'waiters': 0}
            self._jobs[key] = job

            def finished(done):
                if self._jobs.get(key) is job:
                    self._jobs.pop(key)
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished)
        job['waiters'] += 1
        try:
            return await asyncio.shield(job['task'])
        finally:
            job['waiters'] -= 1
            if not job['waiters'] and not job['task'].done():
                if self._jobs.get(key) is job:
                    self._jobs.pop(key)
                job['task'].cancel()
                await asyncio.gather(job['task'], return_exceptions=True)

    async def generate_short_text_async(self, system, prompt, *, max_tokens=40, timeout_s=1.2):
        """Keep the legacy adapter interface while delegating optional wording."""
        from .independent_filler import generate
        return await generate(system, prompt, max_tokens=max_tokens, timeout_s=timeout_s)

    def warmup(self):
        """Validate the checkpoint and shared encoder without any LLM calls."""
        from .independent_filler import configuration
        configuration()
        _ = self.filler_text_timeout_s
        return self.classify('这是一条路由预热消息。')


def create(*, model, base_url='', api_key='', options=None):
    """Construct the installed router adapter from a local checkpoint directory."""
    options = dict(options or {})
    if base_url or api_key or set(options) - {'allow_prototype'}:
        raise ValueError('Local decision router accepts only a checkpoint and allow_prototype')
    if 'allow_prototype' in options and not isinstance(options['allow_prototype'], bool):
        raise ValueError('allow_prototype must be a boolean')
    return LocalDecisionRouter(model, **options)
