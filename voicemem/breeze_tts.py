"""Local Breeze TTS 2 with the supplied cached depth decoder (MLX 0.32.2)."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np

from voicemem.utils.audio.stream_io import resample
from voicemem.prompt_config import tts_prompts

SAMPLE_RATE = 24000


class BreezeMLXTTS:
    SERIAL = True
    DEFAULT_MODEL = "mlx-community/Breeze-TTS-2-mlx-4bit"

    def __init__(self, model=None, ref_audio=None, ref_text=None, instruction=None,
                 cfg_scale=1.0, seed=42, chunk_frames=2, max_tokens=750,
                 depth_mode="cached", first_frames=None):
        self.model_name = model or os.environ.get("VOICEMEM_BREEZE_MLX_MODEL") or self.DEFAULT_MODEL
        self.ref_audio = ref_audio or os.environ.get("VOICEMEM_BREEZE_REF_AUDIO") or None
        self.ref_text = ref_text or os.environ.get("VOICEMEM_BREEZE_REF_TEXT") or None
        if self.ref_audio:
            ref = Path(self.ref_audio).expanduser().resolve()
            if not ref.is_file():
                raise ValueError(f"Breeze reference audio not found: {ref}")
            self.ref_audio = str(ref)
            if not self.ref_text and ref.with_suffix(".txt").is_file():
                self.ref_text = ref.with_suffix(".txt").read_text(encoding="utf-8").strip()
            if not self.ref_text:
                raise ValueError("Breeze reference audio requires its transcript (.txt or ref_text).")
        self.instruction = (instruction or os.environ.get("VOICEMEM_BREEZE_INSTRUCTION")
                            or tts_prompts()["breeze_default_instruction"])
        self.cfg_scale = float(os.environ.get("VOICEMEM_BREEZE_CFG_SCALE", cfg_scale))
        self.seed = int(os.environ.get("VOICEMEM_BREEZE_SEED", seed))
        self.chunk_frames = int(chunk_frames)
        #: 首块几帧。默认 1：首块是"多久能出声"，主干+depth+codec 少算一帧就少
        #: 40~60ms；播放端有 160ms 预缓冲，1 帧（约 80ms 音频）足够起播。后面的块
        #: 仍按 chunk_frames 攒——codec 每块调一次，块太碎开销多。
        self.first_frames = int(first_frames or os.environ.get("VOICEMEM_BREEZE_FIRST_FRAMES", "1"))
        self.max_tokens = int(max_tokens)
        if self.chunk_frames < 1 or self.max_tokens < 1:
            raise ValueError("chunk_frames and max_tokens must be positive")
        if depth_mode not in {"cached", "native", "compiled"}:
            raise ValueError("depth_mode must be cached, native, or compiled")
        self.depth_mode = depth_mode
        self.voice = "breeze:" + (Path(self.ref_audio).stem if self.ref_audio else "female-design")
        import hashlib
        digest = hashlib.sha256()
        digest.update(f"{self.model_name}|{self.ref_text}|{self.instruction}|{self.cfg_scale}|{self.seed}".encode())
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
            from voicemem.breeze_fast import FastDepth, ReferencePromptCache
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
            self._coalesce = max(1, self.chunk_frames // step)   # 首块之后几个流块攒成一块
            depth = FastDepth(m, self.depth_mode)
            depth.install()
            try:
                depth.warmup(2, self.cfg_scale, self.instruction)
            except BaseException:
                depth.close()
                raise
            self._m, self._depth = m, depth
            # 参考前缀的 backbone KV 只算一次（VOICEMEM_BREEZE_PREFIX_KV=0 关掉）。
            # 原来每段都把 8 秒参考音频的 codes 重新过一遍 3B 主干，首帧里最大的一块。
            self._reference_cache = ReferencePromptCache(
                m, prefix_kv=os.environ.get("VOICEMEM_BREEZE_PREFIX_KV", "1") != "0")
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
            print(f"[tts] Breeze MLX ready: {self.voice}, depth={self.depth_mode}, chunk={self.chunk_frames} frames{note}", flush=True)
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
            stream=True, streaming_interval=self.interval,
        )
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
                if first:                      # 首块立刻放出去，越早出声越好
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

    async def _raw(self, text, instruction=None):
        from voicemem.utils.gpu_loop import gpu_loop
        if not text.strip():
            return
        async with self._lock:
            from voicemem.prompt_trace import record_request
            record_request("tts", "breeze_mlx", {
                "model": self.model_name, "text": text,
                "instruct": instruction or self.instruction,
                "ref_audio": self.ref_audio, "ref_text": self.ref_text,
                "cfg_scale": self.cfg_scale, "seed": self.seed,
                "max_tokens": self.max_tokens,
            })
            loop = asyncio.get_running_loop()
            # 首块出来之前独占 GPU 线程（见 gpu_loop.iter 的 solo_first）。
            job = gpu_loop().iter(lambda: self._segments(text, instruction), weight=4,
                                  solo_first=True)
            try:
                while True:
                    item = await loop.run_in_executor(None, job.out.get)
                    if item is None:
                        return
                    kind, chunk = item
                    if kind == "err":
                        raise chunk
                    yield chunk
            finally:
                job.cancel()

    async def stream(self, text, instruction=None):
        # Already sample-aligned PCM16; explicitly close the inner generator on
        # early exit so cancellation releases the per-model serialization lock.
        from contextlib import aclosing
        async with aclosing(self._raw(text, instruction)) as chunks:
            async for chunk in chunks:
                yield chunk
