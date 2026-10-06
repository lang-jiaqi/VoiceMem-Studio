"""Public VoiceMem facade for memory ingestion, retrieval and optional replies.

The facade owns reply normalization and delegates memory operations to one
Orchestrator. Legacy runtime methods remain accessible through __getattr__.
Provider injection, Space selection and memory modes retain their existing
contracts; Studio-specific interaction policy lives outside this facade."""

from __future__ import annotations

from pathlib import Path

from voicemem.orchestrator import Orchestrator, SearchResult, Utils
from voicemem.llm_config import MODELS

# Re-export public result and capability types for existing imports.
__all__ = ["VoiceMem", "SearchResult", "Utils"]


class VoiceMem:
    """Public memory facade with optional reply generation.

    Constructor options select the Space, capability profile and injected providers.
    Memory operations use the existing runtime; replies use a separately normalized
    callable. Model overrides are process-wide, while language defaults belong to
    the selected instance and operation."""

    # Public mode aliases map to the existing internal capability profiles.
    MODE_ALIASES = {
        "normal":         "multi_modal",
        "leftbrain_only": "left_brain_single",
        "text":           "text_mode",
    }

    def __init__(self, api_key=None, mode="text_mode", memory_root=None,
                 user_id="voice_user", base_url=None, reply=None,
                 openai_key=None, top_k=5, space=None, models=None,
                 memory_language=None, follow_input_language=False, **kw):
        # Model overrides are process-wide. Per-instance language defaults and
        # per-operation language scopes are managed separately. openai_key remains an api_key alias.
        if models:
            MODELS.update(models)
        # The runtime resolves space and memory_root before language defaults are read.
        self._o = Orchestrator(api_key=api_key or openai_key,
                               mode=self.MODE_ALIASES.get(mode, mode),
                               memory_root=memory_root, space=space,
                               user_id=user_id, base_url=base_url, **kw)
        # Reply providers are normalized separately from capability factories:
        # a provider can itself be a function and must not be called as a zero-argument factory.
        self._reply_src = reply
        self._reply_norm = None
        self._top_k = top_k  # Default retrieval limit.
        # Resolve the instance language from the selected Space, then enable input following if requested.
        from voicemem.lang import resolve_for_space
        self._o.memory_language = resolve_for_space(self._o._memory_root, memory_language)
        self._o.follow_input_language = bool(follow_input_language)
        self.mode = self._o.mode
        self.utils = self._o.utils
        self.left_brain = self._o._left  # Existing component instance.
        self.right_brain = self._o._right

    @classmethod
    def from_config(cls, config: dict) -> "VoiceMem":
        """Construct from the declarative provider mapping accepted by build_kwargs()."""
        from voicemem.config import build_kwargs
        return cls(**build_kwargs(config))

    # Public convenience methods delegate to the existing runtime.

    def ingest(self, text=None, audio=None, **kw):
        """Store text, an audio file, or both.

        Audio-only input is transcribed first. Text-only input uses the account-owner
        speaker identity; audio input retains voiceprint identification. Remaining
        options and completion callbacks are forwarded to the existing runtime."""
        if audio is not None:
            from voicemem import sample_audio  # Import locally to avoid a package initialization cycle.
            audio = sample_audio(audio)
        if text is None:
            if audio is None:
                raise ValueError("ingest() 要么给 text，要么给 audio")
            text = self.transcribe(audio)
        # Text-only input belongs to the account owner. Audio input retains voiceprint-based speaker identification.
        if audio is None:
            kw.setdefault("speaker", "user")
        return self._o.Ingest(text, audio_path=audio, **kw)

    def transcribe(self, audio) -> str:
        """Transcribe a file with the configured ASR capability, reusing its loaded model."""
        from voicemem.utils.audio.stream_io import transcribe_file
        return transcribe_file(self.utils.get("asr"), audio)

    def search(self, query, **kw):
        kw.setdefault("top_k", self._top_k)
        return self._o.Search(query, **kw)

    def classify(self, query):                return self._o.Classify(query)
    def preprocess(self, text, audio=None):   return self._o.preprocess(text, audio_path=audio)
    def flush(self):                          return self._o.Flush()

    def warmup(self, *, audio: bool = True, verbose: bool = True) -> None:
        """Load configured capabilities and run representative warmup probes.

        Set audio=False for text-only warmup and verbose=False to suppress progress.
        Probes reuse cached components; repeated calls may still perform inference."""
        import sys
        import time

        total = 4 if audio else 1
        # Use terminal progress updates only for interactive output; redirected output receives complete lines.
        bar_ok = verbose and sys.stdout.isatty()
        done = [0]

        def draw(label, finished=False):
            if not verbose:
                return
            if not bar_ok:
                if finished:
                    print(f"[warmup] {label}", flush=True)
                return
            width = 24
            filled = int(width * done[0] / total)
            bar = "█" * filled + "░" * (width - filled)
            pct = int(100 * done[0] / total)
            end = "\n" if done[0] >= total else ""
            sys.stdout.write(f"\r\033[31m{bar}\033[0m {pct:3d}%  {label:<28}{end}")
            sys.stdout.flush()

        def step(name, fn):
            draw(f"正在加载 {name} …")
            t0 = time.time()
            try:
                fn()
                label = f"{name} {time.time() - t0:.1f}s"
            except Exception as e:  # Optional dependency or model unavailable.
                label = f"{name} 跳过（{type(e).__name__}）"
            done[0] += 1
            draw(label if done[0] < total else "模型就绪", finished=True)

        step("embedding / slot 分类", lambda: self.classify("你好"))
        if not audio:
            return

        import numpy as np

        def warm_asr():
            asr = self.utils.get("asr")
            asr.feed(np.zeros(9600, dtype=np.float32))  # Load the model and perform a representative decode.
            asr.reset()

        step("ASR", warm_asr)
        step("VAD", lambda: self.utils.get("vad").is_speech(np.zeros(512, dtype=np.float32)))

        # Audio perception requires a file-backed probe; keep it separate from text-only warmup.
        def warm_perceive():
            import tempfile
            import soundfile as sf
            from pathlib import Path
            p = Path(tempfile.gettempdir()) / "voicemem_warmup.wav"
            sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
            try:
                self.preprocess("预热", audio=str(p))
            finally:
                p.unlink(missing_ok=True)

        step("感知（场景 / 声纹 / 情绪）", warm_perceive)

    def stream(self, **kw):
        """Create an input session that yields confirmed text and memory results."""
        from voicemem.stream import VoiceStream
        return VoiceStream(self, **kw)

    # Reply generation after memory retrieval.

    def _reply_fn(self):
        """Normalize the configured reply callable once, loading the default provider lazily."""
        if self._reply_norm is None:
            from voicemem.reply import normalize, openai_reply
            self._reply_norm = normalize(self._reply_src or openai_reply())
        return self._reply_norm

    def reply_stream(self, turn_or_text, memory_context="", history=None):
        """Yield reply deltas and register the completed assistant exchange."""
        from voicemem.reply import capture, unpack
        text, ctx = unpack(turn_or_text, memory_context)
        return capture(self._reply_fn()(text, ctx, history),
                       lambda answer: self._o.remember_reply(text, answer))

    async def reply(self, turn_or_text, memory_context="", history=None):
        """Collect the reply_stream() deltas into one string."""
        return "".join([d async for d in self.reply_stream(turn_or_text, memory_context,
                                                           history)])

    def test(self):
        """Probe the current mode's capabilities and return the existing startup report."""
        from voicemem.startup_check import run_util_report
        return run_util_report(self.utils)

    def __getattr__(self, name):
        # Preserve legacy runtime methods and attributes through delegation.
        return getattr(self.__dict__["_o"], name)
