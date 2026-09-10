"""Studio providers implementation."""
from __future__ import annotations

import asyncio
import os

import numpy as np

from voicemem.utils.audio.stream_io import resample
from studio.core.utils.tts.audio_timing import TimedAudioChunk, TextTimestamp
from voicemem.llm_config import resolve_model

TTS_MODEL = resolve_model(role="tts")
TTS_BACKEND = 'breeze_mlx'   # openai(api) | local | voxcpm

TTS_VOICE = 'alloy'

TTS_INSTRUCTIONS = ''

BREEZE_SEED = 42
SAMPLE_RATE = 24000

_SENT_END  = "。！？!?…\n"
_SOFT_END  = "，,、；;：: "
#

_FIRST_MIN = 2

_FIRST_MAX = 24
_SENT_MIN  = 2

_SENT_MAX  = int('200' if TTS_BACKEND == 'qwen' else '60')

_FIRST_SOFT = True

_CUT_AT_COMMA = True
_COMMA = "，,、；;：:"

def cut_point(buf: str, first: bool) -> bool:
    """Return the next safe speech boundary or zero while more text is needed."""
    s = buf.strip()
    if not s:
        return False
    if first:
        ends = _SENT_END + _SOFT_END if _FIRST_SOFT else _SENT_END
        if len(s) >= _FIRST_MIN and s[-1] in ends:
            return True

        return len(s) >= _FIRST_MAX and (buf[-1:].isspace() or not s[-1].isalnum())
    return ((len(s) >= _SENT_MIN and s[-1] in _SENT_END)
            or (_CUT_AT_COMMA and len(s) >= 12 and s[-1] in _COMMA)
            or len(s) >= _SENT_MAX)

class BaseTTS:
    """Stream mono PCM16 audio with optional text timing metadata."""

    SERIAL = False

    async def stream(self, text: str, instruction: str | None = None):
        """Yield speech chunks for text and an optional synthesis instruction."""
        tail = b""
        async for chunk in self._raw(text, instruction):
            timed = chunk if isinstance(chunk, TimedAudioChunk) else None
            raw = timed.pcm if timed is not None else chunk
            buf = tail + raw
            cut = len(buf) & -2
            tail = buf[cut:]
            if cut:
                if timed is None:
                    yield buf[:cut]
                else:
                    yield TimedAudioChunk(
                        pcm=buf[:cut], timestamps=timed.timestamps,
                        sample_rate=timed.sample_rate)
        if tail:
            yield tail + b"\x00"

    def _raw(self, text: str, instruction: str | None = None):
        raise NotImplementedError

class OpenAITTS(BaseTTS):
    """Stream speech from the configured OpenAI-compatible TTS endpoint."""

    def __init__(self, model=None, voice=None, instructions=None,
                 api_key=None, base_url=None):
        self.model = resolve_model(model, "tts")
        self.voice = voice or TTS_VOICE
        self.instructions = TTS_INSTRUCTIONS if instructions is None else instructions
        self._key = api_key
        self._base = base_url
        self._client = None

    def _cli(self):
        if self._client is None:
            from openai import AsyncOpenAI
            kw = {}
            if self._key:
                kw["api_key"] = self._key
            if self._base:
                kw["base_url"] = self._base
            self._client = AsyncOpenAI(**kw)
        return self._client

    async def _raw(self, text, instruction=None):
        kw = {"model": self.model, "voice": self.voice,
              "input": text, "response_format": "pcm"}
        ins = instruction or self.instructions
        if ins:
            kw["instructions"] = ins
        async with self._cli().audio.speech.with_streaming_response.create(**kw) as resp:
            async for chunk in resp.iter_bytes():
                yield chunk

class PiperTTS(BaseTTS):
    """Stream speech from a local Piper model."""

    def __init__(self, model=None):
        self.model = resolve_model(model, "tts", default=None)
        self._voice = None

    def _load(self):
        if self._voice is None:
            if not self.model:
                raise ValueError(
                    "piper 后端要指定 voice 文件：设 VOICEMEM_TTS_MODEL 指向 .onnx，"
                    '或 config 里给 {"provider": "piper", "config": {"model": "…/x.onnx"}}')
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.model)
        return self._voice

    async def _raw(self, text, instruction=None):
        v = self._load()
        old = getattr(v, "synthesize_stream_raw", None)
        if old is not None:
            sr = getattr(getattr(v, "config", None), "sample_rate", 22050)
            for raw in old(text):
                f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                out = resample(f, src=sr, dst=SAMPLE_RATE)
                yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            return
        for chunk in v.synthesize(text):

            sr = int(getattr(chunk, "sample_rate", 22050))
            raw = getattr(chunk, "audio_int16_bytes", None)
            f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            out = resample(f, src=sr, dst=SAMPLE_RATE)
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

class QwenTTS(BaseTTS):
    """Stream speech from the configured local Qwen provider."""

    DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

    SERIAL = True

    BASE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

    def __init__(self, model=None, voice=None, instruction=None,
                 streaming_interval=None, ref_audio=None, ref_text=None):
        self.model_name = (model or self.DEFAULT_MODEL)
        self.voice = voice or 'Serena'
        self.instruction = instruction
        self.interval = float(streaming_interval
                              or '0.2')

        self.temperature = 0.6

        self.seed = 0

        self.ref_audio = ref_audio
        self.ref_text = ref_text
        if self.ref_audio and not self.ref_text:
            txt = os.path.splitext(self.ref_audio)[0] + ".txt"
            if os.path.isfile(txt):
                self.ref_text = open(txt, encoding="utf-8").read().strip()
        if self.ref_audio and not self.ref_text:
            raise ValueError(f"参考音频 {self.ref_audio} 缺转写：设 VOICEMEM_QWEN_TTS_REF_TEXT "
                             "或放一个同名 .txt")
        if self.ref_audio and not (model):
            self.model_name = self.BASE_MODEL
        if self.ref_audio:

            self.voice = "ref:" + os.path.splitext(os.path.basename(self.ref_audio))[0]
        self._m = None

    def _load(self):
        if self._m is None:
            from mlx_audio.tts.utils import load_model
            self._m = load_model(self.model_name)
        return self._m

    def _segments(self, kw):
        m = self._load()
        sr = getattr(m, "sample_rate", SAMPLE_RATE)
        if self.seed >= 0:
            import mlx.core as mx
            mx.random.seed(self.seed)
        for seg in m.generate(**kw):
            a = np.asarray(seg.audio, np.float32).reshape(-1)
            o = a if sr == SAMPLE_RATE else resample(a, src=sr, dst=SAMPLE_RATE)
            yield (np.clip(o, -1, 1) * 32767).astype(np.int16).tobytes()

    async def _raw(self, text, instruction=None):
        import asyncio as _a
        from voicemem.utils.gpu_loop import gpu_loop
        kw = {"text": text, "stream": True,
              "streaming_interval": self.interval, "temperature": self.temperature}
        if self.ref_audio:

            kw["ref_audio"], kw["ref_text"] = self.ref_audio, self.ref_text
        else:
            kw["voice"] = self.voice
            ins = instruction or self.instruction
            if ins:
                kw["instruct"] = ins
        loop = _a.get_running_loop()

        job = gpu_loop().iter(lambda: self._segments(kw), weight=4)
        try:
            while True:
                item = await loop.run_in_executor(None, job.out.get)
                if item is None:
                    return
                kind, chunk = item
                if kind == "err":
                    print(f"[tts] Qwen3-TTS 合成失败：{type(chunk).__name__}: {chunk}",
                          flush=True)
                    return
                yield chunk
        finally:
            job.cancel()

class KokoroTTS(BaseTTS):
    """Stream speech using the language-specific Kokoro pipeline."""

    DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"

    SERIAL = True

    def __init__(self, model=None, voice=None, voice_zh=None, lang=None,
                 speed=None, split_pattern=None):
        self.model_name = (model or self.DEFAULT_MODEL)
        self.voice_en = voice or 'af_heart'
        self.voice_zh = voice_zh or 'zf_xiaoyi'

        self.lang = lang
        self.speed = float(speed or '1.0')

        self.split_pattern = split_pattern or r"\n+"

        self.voice = self.voice_en
        self._m = None

    def _load(self):
        if self._m is None:
            from mlx_audio.tts.utils import load_model
            self._m = load_model(self.model_name)
        return self._m

    def _pick(self, text):
        if self.lang:
            v = self.voice_zh if self.lang == "z" else self.voice_en
            return self.lang, v
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return "z", self.voice_zh
        return "a", self.voice_en

    def _segments(self, text):
        m = self._load()
        sr = getattr(m, "sample_rate", SAMPLE_RATE)
        lang, voice = self._pick(text)
        for seg in m.generate(text, voice=voice, lang_code=lang, speed=self.speed,
                              split_pattern=self.split_pattern):
            a = np.asarray(seg.audio, np.float32).reshape(-1)
            o = a if sr == SAMPLE_RATE else resample(a, src=sr, dst=SAMPLE_RATE)
            yield (np.clip(o, -1, 1) * 32767).astype(np.int16).tobytes()

    async def _raw(self, text, instruction=None):
        from voicemem.utils.gpu_loop import gpu_loop
        loop = asyncio.get_running_loop()
        job = gpu_loop().iter(lambda: self._segments(text), weight=4)
        try:
            while True:
                item = await loop.run_in_executor(None, job.out.get)
                if item is None:
                    return
                kind, chunk = item
                if kind == "err":
                    print(f"[tts] Kokoro 合成失败：{type(chunk).__name__}: {chunk}",
                          flush=True)
                    return
                yield chunk
        finally:
            job.cancel()

class VoxCPMTTS(BaseTTS):
    """Stream local VoxCPM speech with serialized synthesis."""

    SERIAL = True

    def __init__(self, model=None, ref_audio=None, ref_text=None, voice=None,
                 instruction=None, hifi=None, device=None, cfg_value=None,
                 inference_timesteps=None):
        self.model = resolve_model(model, "tts", default=None) or "openbmb/VoxCPM2"
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        if self.ref_audio and not self.ref_text:
            txt = os.path.splitext(self.ref_audio)[0] + ".txt"
            if os.path.isfile(txt):
                self.ref_text = open(txt, encoding="utf-8").read().strip()
        self.hifi = (False) if hifi is None else bool(hifi)
        if self.hifi and not (self.ref_audio and self.ref_text):
            raise ValueError("hifi 克隆要 ref_audio + ref_text（同名 .txt 或 VOICEMEM_VOXCPM_REF_TEXT）")
        self.voice = voice
        self.instruction = instruction
        self.device = device
        self.cfg_value = float(cfg_value or '2.0')
        self.steps = int(inference_timesteps or '10')
        self._m = None

    def _load(self):
        if self._m is None:
            from voxcpm import VoxCPM
            self._m = VoxCPM.from_pretrained(self.model, load_denoiser=False, device=self.device)
        return self._m

    def _prefix(self, instruction):
        parts = [] if self.ref_audio else ([self.voice] if self.voice else [])
        ins = instruction or self.instruction
        if ins:
            parts.append(ins)
        return f"({', '.join(parts)})" if parts else ""

    def _kw(self):
        kw = dict(cfg_value=self.cfg_value, inference_timesteps=self.steps)
        if self.ref_audio:
            kw["reference_wav_path"] = self.ref_audio
            if self.hifi:
                kw["prompt_wav_path"] = self.ref_audio
                kw["prompt_text"] = self.ref_text
        return kw

    async def _raw(self, text, instruction=None):
        m = self._load()
        sr = m.tts_model.sample_rate
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        text = self._prefix(instruction) + text

        def work():
            try:
                for f in m.generate_streaming(text=text, **self._kw()):
                    loop.call_soon_threadsafe(q.put_nowait, np.asarray(f, np.float32).reshape(-1))
                loop.call_soon_threadsafe(q.put_nowait, None)
            except BaseException as e:
                loop.call_soon_threadsafe(q.put_nowait, e)

        loop.run_in_executor(None, work)
        while True:
            f = await q.get()
            if f is None:
                return
            if isinstance(f, BaseException):
                raise f
            out = resample(f, src=sr, dst=SAMPLE_RATE)
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

class BreezeTTS(BaseTTS):
    """Stream speech through the configured Breeze service adapter."""

    def __init__(self, base_url=None, instruction=None, cfg_scale=None,
                 ref_audio=None, ref_text=None, seed=None, timeout=60.0, model=None):
        self.base_url = (base_url or 'http://127.0.0.1:7860').rstrip("/")
        self.instruction = instruction or ''
        if cfg_scale is None:
            env_cfg = None

            cfg_scale = float(env_cfg) if env_cfg else (4 if self.instruction else None)
        self.cfg_scale = cfg_scale

        self.ref_audio = ref_audio
        self.ref_text = ref_text

        self.seed = BREEZE_SEED if seed is None else seed
        self.timeout = timeout
        self._client = None
        self._ref_bytes = None

        self._lock = asyncio.Lock()

        self.model = model

    def _cli(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0))
        return self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _raw(self, text, instruction=None):
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError("Breeze 后端要 httpx：pip install httpx") from e

        ins = instruction or self.instruction
        cfg = self.cfg_scale if self.cfg_scale is not None else (4 if ins else None)
        fields = {"text": text}
        if ins:
            fields["instruction"] = ins
        if cfg is not None:
            fields["cfg_scale"] = str(cfg)
        if self.ref_text:
            fields["ref_text"] = self.ref_text
        if self.seed is not None:
            fields["seed"] = str(self.seed)

        files = {k: (None, v) for k, v in fields.items()}
        if self.ref_audio:
            if self._ref_bytes is None:
                from pathlib import Path
                ref = Path(self.ref_audio)
                self._ref_bytes = (ref.name, ref.read_bytes())
            files["ref_audio"] = self._ref_bytes

        url = f"{self.base_url}/v1/audio/speech"
        async with self._lock:
            async with self._cli().stream("POST", url, files=files) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

from studio.core.utils.tts.component import BreezeMLXTTS
from studio.core.utils.tts.cuda import BreezeCUDATTS

TTS_PROVIDERS = {
    "openai": OpenAITTS,
    "local":  PiperTTS,
    "piper":  PiperTTS,
    "voxcpm": VoxCPMTTS,
    "breeze": BreezeTTS,
    "breeze_mlx": BreezeMLXTTS,
    "breeze_cuda": BreezeCUDATTS,
    "qwen":   QwenTTS,
    "kokoro": KokoroTTS,
}

_INSTANCES: dict = {}

def make_tts(provider: str | None = None, **cfg):
    """Resolve and cache a TTS provider; reject unknown provider names."""
    name = (provider or TTS_BACKEND).lower()
    cls = TTS_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"未知的 tts.provider={provider!r}；"
                         f"可选：{' / '.join(sorted(set(TTS_PROVIDERS)))}")
    key = (name, str(sorted(cfg.items())))
    if key not in _INSTANCES:
        _INSTANCES[key] = cls(**cfg)
    return _INSTANCES[key]

def _tts_cfg(reply):
    seg = (reply or {}).get("tts") or {}
    return seg.get("provider"), (seg.get("config") or {})

async def tts_stream(text, reply=None, instruction=None):
    """Yield PCM from the selected provider, propagating cancellation."""
    provider, cfg = _tts_cfg(reply)
    async for pcm in make_tts(provider, **cfg).stream(text, instruction):
        yield pcm

async def speak_stream(deltas, reply=None, on_delta=None, tts=None,
                       instruction=None):
    """Segment streamed reply text and yield ordered speech output."""
    queue: asyncio.Queue = asyncio.Queue()
    out: asyncio.Queue = asyncio.Queue()

    async def synth():
        while (seg := await queue.get()) is not None:
            gen = (tts.stream(seg, instruction) if tts is not None
                   else tts_stream(seg, reply, instruction))
            async for pcm in gen:
                await out.put(pcm)
        await out.put(None)

    worker = asyncio.create_task(synth())

    async def feed():
        buf, sent = "", 0
        try:
            async for d in deltas:
                if on_delta:
                    on_delta(d)
                buf += d
                if cut_point(buf, first=sent == 0):
                    await queue.put(buf.strip())
                    buf, sent = "", sent + 1
            if buf.strip():
                await queue.put(buf.strip())
        finally:
            await queue.put(None)

    feeder = asyncio.create_task(feed())
    try:
        while (pcm := await out.get()) is not None:
            yield pcm
    finally:
        for t in (feeder, worker):
            if not t.done():
                t.cancel()
        await asyncio.gather(feeder, worker, return_exceptions=True)
