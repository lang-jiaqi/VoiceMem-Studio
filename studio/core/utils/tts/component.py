"""Local Breeze TTS 2 with the supplied cached depth decoder (MLX 0.32.2)."""
from __future__ import annotations

import asyncio
import math
import os
import time
from contextlib import aclosing
from pathlib import Path
from uuid import uuid4

import numpy as np

from voicemem.utils.audio.stream_io import resample
from studio.prompt_config import tts_prompts
from .safety import SpeechGuard, speech_budget_seconds

SAMPLE_RATE = 24000

class BreezeMLXTTS:
    SERIAL = True
    DEFAULT_MODEL = "mlx-community/Breeze-TTS-2-mlx-4bit"

    def __init__(self, model=None, ref_audio=None, ref_text=None, instruction=None,
                 cfg_scale=1.0, seed=42, chunk_frames=2, max_tokens=750,
                 depth_mode="cached", first_frames=None, repetition_penalty=1.1):
        self.model_name = model or self.DEFAULT_MODEL
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        if self.ref_audio:
            ref = Path(self.ref_audio).expanduser().resolve()
            if not ref.is_file():
                raise ValueError(f"Breeze reference audio not found: {ref}")
            self.ref_audio = str(ref)
            if not self.ref_text and ref.with_suffix(".txt").is_file():
                self.ref_text = ref.with_suffix(".txt").read_text(encoding="utf-8").strip()
            if not self.ref_text:
                raise ValueError("Breeze reference audio requires its transcript (.txt or ref_text).")
        self.instruction = (instruction or tts_prompts()['breeze_default_instruction'])
        self.cfg_scale = float(cfg_scale)
        self.seed = int(seed)
        self.repetition_penalty = float(repetition_penalty)
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError('repetition_penalty must be finite and positive')
        self.chunk_frames = int(chunk_frames)
        # Match the first codec batch to the browser's admission buffer. A
        # one-frame server chunk cannot start 160 ms playback by itself, so it
        # only adds an extra codec call before the browser waits for frame two.
        # Other players can supply first_frames explicitly.
        configured_first = (
            first_frames if first_frames is not None
            else self.chunk_frames
        )
        self.first_frames = int(configured_first)
        self.max_tokens = int(max_tokens)
        if self.chunk_frames < 1 or self.first_frames < 1 or self.max_tokens < 1:
            raise ValueError(
                "chunk_frames, first_frames and max_tokens must be positive")
        if depth_mode not in {"cached", "native", "compiled"}:
            raise ValueError("depth_mode must be cached, native, or compiled")
        self.depth_mode = depth_mode
        self.voice = "breeze:" + (Path(self.ref_audio).stem if self.ref_audio else "female-design")
        import hashlib
        digest = hashlib.sha256()
        digest.update(f"{self.model_name}|{self.ref_text}|{self.instruction}|{self.cfg_scale}|{self.seed}|{self.repetition_penalty}".encode())
        if self.ref_audio:
            digest.update(Path(self.ref_audio).read_bytes())
        self.cache_voice_id = self.voice + ":" + digest.hexdigest()[:16]
        self._m = self._depth = self._reference_cache = None
        self._lock = asyncio.Lock()

    def _load(self):
        if self._m is None:
            from importlib.metadata import version
            if version("mlx-audio") != "0.5.1" or version("mlx") != "0.32.2":
                raise RuntimeError("Breeze fast requires mlx-audio==0.5.1 and mlx==0.32.2")
            from mlx_audio.tts.utils import load_model
            from mlx_audio.tts.models.breeze_tts.breeze_tts import Model
            from studio.core.utils.tts.cache import FastDepth, ReferencePromptCache
            m = load_model(self.model_name)
            if not isinstance(m, Model):
                raise ValueError("Expected Breeze TTS 2 MLX weights")
            codec = m.audio_tokenizer
            rate = getattr(codec, "decode_upsample_rate", None)
            if rate is None:
                rate = getattr(codec.decoder, "decode_upsample_rate", None)
            if rate is None or rate <= 0:
                raise RuntimeError("Breeze codec is missing decode_upsample_rate")
            step = min(self.first_frames, self.chunk_frames)
            self.interval = (step + 1e-6) * rate / m.sample_rate
            self._coalesce = max(1, self.chunk_frames // step)
            depth = FastDepth(m, self.depth_mode)
            depth.install()
            try:
                depth.warmup(2, self.cfg_scale, self.instruction)
            except BaseException:
                depth.close()
                raise
            self._m, self._depth = m, depth

            self._reference_cache = ReferencePromptCache(
                m, prefix_kv=True)
            checks = [fn.snapshot() for fn in depth.functions.values()
                      if hasattr(fn, 'snapshot')]
            depth_note = (f", verified_depth={sum(item['enabled'] for item in checks)}/{len(checks)}"
                          if checks else "")
            note = ""
            if self.ref_audio and self._reference_cache.prefix_kv:
                import time as _t
                t0 = _t.perf_counter()
                ok, err = self._reference_cache.verify(
                    "你好，测试。", voice=None, instruct=self.instruction,
                    ref_audio=self.ref_audio, ref_text=self.ref_text)
                note = (f", prefix_kv={self._reference_cache.prefix_len} tok"
                        f" (err {err:.1e}, {(_t.perf_counter()-t0)*1000:.0f}ms)"
                        if ok else f", prefix_kv OFF (mismatch err {err:.1e})")
            print(f"[tts] Breeze MLX ready: {self.voice}, depth={self.depth_mode}{depth_note}, chunk={self.chunk_frames} frames{note}", flush=True)
        return self._m

    def _segments(self, text, instruction):
        # Called only on the shared GPU thread. Codec state must not overlap.
        import mlx.core as mx
        m = self._load()
        if self._reference_cache is not None:
            self._reference_cache.install()
        results = m.generate(
            text=text, instruct=instruction or self.instruction,
            ref_audio=self.ref_audio, ref_text=self.ref_text,
            cfg_scale=self.cfg_scale, seed=self.seed, max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            stream=True, streaming_interval=self.interval,
        )
        guard = SpeechGuard(text, m.sample_rate)
        def to_bytes(pcm):
            if m.sample_rate != SAMPLE_RATE:
                pcm = resample(pcm, src=m.sample_rate, dst=SAMPLE_RATE)
            return (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()

        try:
            first, buf = True, []
            for result in results:
                if not result.is_streaming_chunk or result.token_count <= 0:
                    raise RuntimeError("Breeze did not produce streaming acoustic frames")
                if int(result.sample_rate) != int(m.sample_rate):
                    raise RuntimeError("Breeze stream sample rate changed")
                pcm = np.array(result.audio.astype(mx.float32), copy=True)
                if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
                    raise RuntimeError("Breeze returned invalid PCM")
                guard.accept(pcm.size)
                if first:
                    first = False
                    yield to_bytes(pcm)
                    continue
                buf.append(pcm)
                if len(buf) >= self._coalesce:
                    yield to_bytes(np.concatenate(buf)); buf = []
            if buf:
                yield to_bytes(np.concatenate(buf))
        finally:
            try:
                results.close()
            finally:
                try:
                    m.audio_tokenizer.decoder.reset_streaming_state()
                finally:
                    if self._reference_cache is not None:
                        self._reference_cache.close()

    def _profiled_segments(self, text, instruction, submitted, request_id):
        from voicemem.engine import speech_scope
        with speech_scope():
            started = time.perf_counter()
            work_s = max_step_s = audio_s = 0.0
            first_s = None
            count = 0
            status = 'cancelled'
            segments = self._segments(text, instruction)
            try:
                while True:
                    step_started = time.perf_counter()
                    try:
                        pcm = next(segments)
                    except StopIteration:
                        status = 'complete'
                        break
                    finally:
                        step_s = time.perf_counter() - step_started
                        work_s += step_s
                        max_step_s = max(max_step_s, step_s)
                    produced = time.perf_counter()
                    if first_s is None:
                        first_s = produced - started
                    count += 1
                    audio_s += len(pcm) / (SAMPLE_RATE * 2)
                    from voicemem.engine import observe_speech
                    observe_speech(len(pcm) / (SAMPLE_RATE * 2), step_s, first=count == 1)
                    yield pcm, produced
            except Exception:
                status = 'error'
                raise
            finally:
                try:
                    close = getattr(segments, 'close', None)
                    if close is not None:
                        close()
                finally:
                    print(f'[tts-gpu] id={request_id} queue_ms={(started-submitted)*1000:.0f}'
                          f' first_pcm_ms={(first_s or 0)*1000:.0f}'
                          f' compute_ms={work_s*1000:.0f} max_step_ms={max_step_s*1000:.0f}'
                          f' audio_ms={audio_s*1000:.0f} chunks={count}'
                          f' status={status}', flush=True)

    async def _raw(self, text, instruction=None):
        from voicemem.engine import Service, gpu_loop, speech_scope
        if not text.strip():
            return
        async with self._lock:
            from voicemem.utils.common.prompt_trace import record_request
            record_request("tts", "breeze_mlx", {
                "model": self.model_name, "text": text,
                "instruct": instruction or self.instruction,
                "ref_audio": self.ref_audio, "ref_text": self.ref_text,
                "cfg_scale": self.cfg_scale, "seed": self.seed,
                "max_tokens": self.max_tokens,
                "repetition_penalty": self.repetition_penalty,
                "audio_limit_s": speech_budget_seconds(text),
            })
            submitted = time.perf_counter()
            request_id = uuid4().hex[:8]
            delivered = 0
            audio_s = max_delivery_s = max_gap_s = 0.0
            previous = None
            status = 'cancelled'
            try:
                with speech_scope():
                    async with aclosing(gpu_loop().aiter(
                        lambda: self._profiled_segments(text, instruction, submitted, request_id),
                        weight=4, solo_first=True, max_pending=2,
                        service=Service.AUDIO, label="tts")) as chunks:
                        async for pcm, produced in chunks:
                            now = time.perf_counter()
                            max_delivery_s = max(max_delivery_s, now - produced)
                            if previous is not None:
                                max_gap_s = max(max_gap_s, now - previous)
                            previous = now
                            delivered += 1
                            audio_s += len(pcm) / (SAMPLE_RATE * 2)
                            yield pcm
                status = 'complete'
            except Exception:
                status = 'error'
                raise
            finally:
                elapsed = time.perf_counter() - submitted
                print(f'[tts-delivery] id={request_id} chunks={delivered:.0f}'
                      f' audio_ms={audio_s*1000:.0f} elapsed_ms={elapsed*1000:.0f}'
                      f' max_gap_ms={max_gap_s*1000:.0f}'
                      f' max_handoff_ms={max_delivery_s*1000:.0f}'
                      f' rtf={elapsed/audio_s if audio_s else 0:.3f}'
                      f' status={status}', flush=True)

    async def stream(self, text, instruction=None):
        # Already sample-aligned PCM16; explicitly close the inner generator on
        # early exit so cancellation releases the per-model serialization lock.
        async with aclosing(self._raw(text, instruction)) as chunks:
            async for chunk in chunks:
                yield chunk
