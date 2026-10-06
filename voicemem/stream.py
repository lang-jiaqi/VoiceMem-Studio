"""Streaming memory input with ASR, turn confirmation and speculative retrieval.

Each VoiceStream owns its input state. Streaming ASR runs on its serial worker;
final ASR uses the process-wide executor in this module. Confirmed turns carry
text and memory results to the caller, which owns reply generation and playback.
Typed text and external ASR use the same confirmation and retrieval interfaces."""
from __future__ import annotations

import asyncio
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from voicemem import gate as _gate_mod
from dataclasses import dataclass
from voicemem.memory_api import build_memory_context
from voicemem.utils.audio.stream_io import resample


@dataclass
class Turn:
    """Confirmed input text, its memory result, the original streaming text and final route."""
    text: str
    result: object
    # Streaming text before final-ASR refinement.
    raw_text: str = ""
    # Final memory route. Non-deep routes carry a gated empty result.
    route: str = _gate_mod.DEEP

    @property
    def memory_context(self) -> str:
        return build_memory_context(self.result)

@dataclass
class StreamState:
    """Input snapshot containing transcript, silence, EOT and retrieval state.

    Acoustic properties compute only when accessed and reuse the existing provider
    owner. The snapshot keeps streaming text separate from final-ASR wording."""
    state: str  # <speak>, <silence>, or turn_over.
    text: str  # Current cumulative transcript.
    memory: object | None  # Latest speculative retrieval, or None.
    turn: Turn | None  # Confirmed turn, or None.
    _vm: object = None  # Provider owner for lazy perception.
    _pcm: object = None  # Mono 16 kHz audio for this snapshot.
    # Current silence duration in seconds; also used by the application's pause and backchannel policies.
    silence: float = 0.0
    # Whether voiced input has occurred in this turn.
    spoke: bool = False
    # Keep the streaming transcript for speculation comparisons; final-ASR wording is a separate field.
    raw_text: str = ""
    # Latest EOT score in [0, 1]. Confirmation also requires silence; the application may use earlier scores for speculation.
    eot_score: float = 0.0
    speech_end: float = 0.0  # Server monotonic time of the last voiced input frame.

    @property
    def memory_context(self) -> str:
        m = self.turn.result if self.turn else self.memory
        return build_memory_context(m) if m is not None else ""

    @property
    def route(self) -> str:
        """Return the confirmed route; before confirmation, retain the existing deep default."""
        return self.turn.route if self.turn else _gate_mod.DEEP

    # Structured memory access.

    @property
    def _result(self):
        return self.turn.result if self.turn else self.memory

    @property
    def transcript(self) -> str:
        return self.text

    @property
    def result_leftbrain(self) -> list[str]:
        r = self._result
        return list(r.result_leftbrain) if r is not None else []

    @property
    def result_rightbrain(self) -> list[str]:
        r = self._result
        return list(r.result_rightbrain) if r is not None else []

    @property
    def entity(self) -> list[str]:
        """Read named entities from the existing classification result without reclassification."""
        r = self._result
        return list(getattr(r.classification, "entities", []) or []) if r is not None else []

    @property
    def schema(self) -> list[str]:
        """Read slots from the existing classification result."""
        r = self._result
        return list(getattr(r.classification, "slots", []) or []) if r is not None else []

    # Lazy acoustic and text perception.

    @property
    def _perception(self):
        if getattr(self, "_p_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._p_cache = _perceive(self._vm, self._pcm, self.text)
        return self._p_cache

    @property
    def emotion(self) -> str:
        p = self._perception
        return getattr(p, "emotion", "") if p else ""

    @property
    def speaker_id(self) -> str:
        p = self._perception
        return (getattr(p, "person_id", None) or "") if p else ""

    @property
    def speaker_voiceprint(self):
        """Compute and cache the speaker vector lazily; return None when unavailable."""
        if getattr(self, "_vp_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._vp_cache = _voiceprint(self._vm, self._pcm)
        return self._vp_cache

    @property
    def text_embedding(self):
        """Compute and cache the transcript vector lazily; return None for empty text."""
        if getattr(self, "_emb_cache", None) is None:
            if self._vm is None or not self.text.strip():
                return None
            self._emb_cache = _embed(self._vm, self.text)
        return self._emb_cache

def _tmp_wav(pcm) -> str:
    """Write mono 16 kHz audio to a temporary WAV; the caller removes the file."""
    import tempfile, wave
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return path

def _perceive(vm, pcm, text):
    """Reuse the memory facade's audio preprocessing and remove its temporary recording."""
    path = _tmp_wav(pcm)
    try:
        return vm.preprocess(text or "", audio=path)
    except Exception as e:
        print(f"[stream] 感知失败: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass

def _voiceprint(vm, pcm):
    path = _tmp_wav(pcm)
    try:
        return vm.utils.get("voiceprint").embed(Path(path))
    except Exception as e:
        print(f"[stream] 声纹提取失败: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass

def _embed(vm, text):
    try:
        emb = vm.utils.get("embedding")
        fn = getattr(emb, "embed_texts", None)
        return fn([text])[0] if fn else emb.embed(text)
    except Exception as e:
        print(f"[stream] 文本向量失败: {e}", flush=True)
        return None


def empty_result():
    """Return a gated empty SearchResult, preserving the normal result schema."""
    from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
    from voicemem.orchestrator import SearchResult
    return SearchResult(hits=[], classification=QueryClassification(slots=[], entities=[]),
                        related_summaries={}, slot_mem_ids=set(),
                        final_candidate_ids=set(), search_mode="gated")


# Maximum retained audio per turn: 30 seconds of mono 16 kHz samples.
_MAX_TURN_SAMPLES = 30 * 16000

# Retain 300 ms of audio before speech onset to preserve initial phonemes.
_PREROLL_SAMPLES = int(0.3 * 16000)

# Minimum duration for a sound-only turn when ASR has no transcript.
MIN_SOUND_ONLY_S = float(os.environ.get("VOICEMEM_SOUND_ONLY_S", "5"))

# Sound-only turns use a separate silence window so musical pauses do not split recordings.
SOUND_ONLY_SILENCE_S = float(os.environ.get("VOICEMEM_SOUND_ONLY_SILENCE_S", "3.0"))

# Use RMS while no transcript exists: speech VAD alone cannot distinguish continuing music from silence.
SOUND_LEVEL = float(os.environ.get("VOICEMEM_SOUND_LEVEL", "0.01"))
# Warn when queued streaming audio exceeds this duration.
ASR_BACKLOG_WARN_S = float(os.environ.get("VOICEMEM_ASR_BACKLOG_WARN", "0.4"))
ASR_DEBUG = os.environ.get("VOICEMEM_ASR_DEBUG", "0") == "1"
# Final decoding gets a short quality preference, not an exclusive wait path.
ASR_FINAL_GRACE_S = 0.08
ASR_FINISH_TIMEOUT_S = 1.0
# Throttle speculative retrieval unless enough time or new text has accumulated.
SPEC_MIN_INTERVAL_S = float(os.environ.get("VOICEMEM_SPEC_MIN_INTERVAL", "0.8"))
SPEC_MIN_GROWTH = int(os.environ.get("VOICEMEM_SPEC_MIN_GROWTH", "6"))
# Gate inference and query encoding share the process-level Torch lock.
from voicemem.utils.torch_lock import TORCH_LOCK as _EMBED_LOCK  # Process-wide Torch serialization.
# One shared recognizer may serve multiple sessions; never decode it concurrently.
# Separate from the default executor so ingestion cannot queue ahead of final ASR.
_FINAL_ASR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-final")


class _AsrWorker:
    """Serial streaming-ASR worker owned by one input session.

    push() queues audio and returns the latest cumulative transcript. flush() places
    a barrier after queued chunks and returns a Future for the final decode. Epoch
    checks reject pre-reset results; close() invalidates queued work while allowing
    an in-flight native call to finish. Queue and inference metrics use 16 kHz audio."""

    def __init__(self, asr):
        self.asr = asr
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.text = ""
        self.epoch = 0  # Reject results from pre-reset inference.
        self.backlog = 0  # Queued samples at 16 kHz.
        self.stats = {"chunks": 0, "busy_s": 0.0, "max_backlog": 0, "last_ms": 0.0}
        self._last_warn = 0.0
        self.closed = False
        self.t = threading.Thread(target=self._run, name="asr-worker", daemon=True)
        self.t.start()

    def push(self, frame) -> str:
        with self.lock:
            if self.closed:
                raise RuntimeError("ASR worker is closed")
            self.backlog += len(frame)
            self.stats["max_backlog"] = max(self.stats["max_backlog"], self.backlog)
            backlog_s, text = self.backlog / 16000.0, self.text
            self.q.put(("feed", frame, None, self.epoch))
        now = time.monotonic()
        if backlog_s >= ASR_BACKLOG_WARN_S and now - self._last_warn > 2.0:
            self._last_warn = now
            print(f"[asr] 积压 {backlog_s*1000:.0f}ms 音频没识别"
                  f"（上一块推理 {self.stats['last_ms']:.0f}ms）→ ASR 跟不上实时",
                  flush=True)
        return text

    def flush(self):
        fut = asyncio.get_running_loop().create_future()
        loop = asyncio.get_running_loop()
        with self.lock:
            if self.closed:
                fut.set_result("")
                return fut
            self.q.put(("flush", loop, fut, self.epoch))
        return fut

    def reset(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.epoch += 1
            self.text = ""
            self.backlog = 0
            self.stats = {"chunks": 0, "busy_s": 0.0, "max_backlog": 0, "last_ms": 0.0}
            # The full snapshot is assigned to final ASR. Discard queued chunks; in-flight inference finishes under its epoch guard.
            while True:
                try:
                    cmd, arg, fut, _ = self.q.get_nowait()
                except queue.Empty:
                    break
                if cmd == "flush":
                    self._resolve(arg, fut, "")
            self.q.put(("reset", None, None, self.epoch))

    def close(self) -> None:
        """Invalidate queued work and stop after the current native call returns."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.epoch += 1
            self.text, self.backlog = "", 0
            while True:
                try:
                    cmd, arg, fut, _ = self.q.get_nowait()
                except queue.Empty:
                    break
                if cmd == "flush":
                    self._resolve(arg, fut, "")
            self.q.put(("stop", None, None, self.epoch))

    @staticmethod
    def _resolve(loop, future, value):
        try:
            loop.call_soon_threadsafe(
                lambda: (not future.done()) and future.set_result(value))
        except RuntimeError:  # The caller loop may already be closed.
            pass

    def report(self) -> str:
        s = self.stats
        n = max(1, s["chunks"])
        return (f"{s['chunks']} 块 · 均 {s['busy_s']/n*1000:.0f}ms/块 · "
                f"积压峰值 {s['max_backlog']/16000*1000:.0f}ms")

    def _run(self):
        while True:
            cmd, arg, fut, epoch = self.q.get()
            if cmd == "stop":
                self.asr = None
                return
            try:
                with self.lock:
                    obsolete = epoch != self.epoch
                if obsolete:
                    if cmd == "flush":
                        self._resolve(arg, fut, "")
                    continue
                if cmd == "feed":
                    t0 = time.perf_counter()
                    txt = self.asr.feed(arg)
                    dt = time.perf_counter() - t0
                    with self.lock:
                        if epoch != self.epoch:
                            continue
                        self.text = txt
                        self.backlog = max(0, self.backlog - len(arg))
                        if dt > 0.005:  # Count only chunks that performed decoding.
                            self.stats["chunks"] += 1
                            self.stats["busy_s"] += dt
                            self.stats["last_ms"] = dt * 1000
                elif cmd == "flush":
                    fl = getattr(self.asr, "flush", None)
                    txt = fl() if fl is not None else self.text
                    with self.lock:
                        if epoch == self.epoch:
                            self.text = txt or self.text
                        out = self.text if epoch == self.epoch else ""
                    self._resolve(arg, fut, out)
                elif cmd == "reset":
                    self.asr.reset()
            except Exception as e:  # noqa: BLE001 - keep the worker alive after decode errors.
                print(f"[asr] 线程里出错（{type(e).__name__}: {e}）", flush=True)
                if cmd == "flush":
                    with self.lock:
                        out = self.text if epoch == self.epoch else ""
                    self._resolve(arg, fut, out)

# Canonical placeholder for recorded sound without speech.
# Ingestion recognizes this value to preserve sound-only tagging and replay behavior.
SOUND_ONLY_TEXT = "用户放了一段声音给我听。"

# Optional logging for semantic and timeout-based turn confirmation.
EOT_DEBUG = os.environ.get("VOICEMEM_EOT_DEBUG", "0") != "0"
# Optional logging for streaming and final transcript comparison.
REFINE_DEBUG = os.environ.get("VOICEMEM_REFINE_DEBUG", "0") != "0"
# Require sustained high EOT scores before acoustic confirmation.
EOT_HOT_LEVEL = float(os.environ.get("VOICEMEM_EOT_HOT_LEVEL", "0.85"))
# Text endings supplement acoustic EOT for punctuation and Chinese final particles.
_TEXT_DONE = re.compile(r"[。！？!?…]\s*$|[吗呢吧嘛呗]\s*[。！？!?]?\s*$")
# Text endings cannot override a score below this acoustic minimum.
EOT_TEXT_MIN = float(os.environ.get("VOICEMEM_EOT_TEXT_MIN", "0.3"))
EOT_HOT_S = float(os.environ.get("VOICEMEM_EOT_HOT_S", "0.15"))
_UNSET = object()


class VoiceStream:
    """Input session for typed text, external-ASR partials or PCM16 audio.

    The session owns its mutable turn state, ASR adapter and speculative tasks.
    gamble_s controls early retrieval; confirm_s is the fallback silence timeout.
    EOT and optional application pause policy participate in confirmation. Replies
    and audio playback remain owned by the caller."""

    def __init__(self, vm, *, on_partial=None, spec_min_chars=6,
                 gamble_s=0.2, confirm_s=0.3, src_rate=24000, vad_threshold=None,
                 emotion=None, gate=_gate_mod.route, eot=None, textless_confirm_s=None,
                 turn_end_guard=None,
                 eot_min_s=float(os.environ.get("VOICEMEM_EOT_MIN_SILENCE", "0")),
                 eot_ends_turn=os.environ.get("VOICEMEM_EOT_ENDS_TURN", "1") != "0"):
        self.vm = vm
        # An omitted EOT provider loads the built-in detector lazily; None explicitly disables it.
        self.eot = eot
        # Minimum silence before EOT may confirm a turn. The silence timeout remains a fallback.
        self.eot_min_s = eot_min_s
        # When enabled, EOT may confirm a complete utterance before the fallback timeout.
        # Confirmation still requires the silence and pause-policy conditions in feed().
        self.eot_ends_turn = eot_ends_turn
        # An omitted gate uses the built-in router. None explicitly enables memory retrieval for every turn.
        self.gate = gate
        # Caller-provided emotion hint for affective retrieval.
        # Reuse known context rather than synchronously running lazy acoustic perception during speculation.
        self.emotion = emotion
        self.on_partial = on_partial
        self.spec_min_chars = spec_min_chars
        self.gamble_s = gamble_s
        self.confirm_s = confirm_s
        # Dialogue can probe short voiced input before streaming ASR emits a word.
        # Opt-in: library/music-only callers retain the original sound-only path.
        self.textless_confirm_s = textless_confirm_s
        # Optional application turn-taking policy. Applies to both EOT and timeout.
        self.turn_end_guard = turn_end_guard
        self._held_final_text = ""
        self._vad_silence = self._voiced_s = 0.0
        self._textless_probed = False
        self.src_rate = src_rate
        self.vad_threshold = vad_threshold
        # Text-only and external-ASR feeds do not initialize audio models.
        self._asr = None
        self._vad = None
        self._closed = False
        self._snapshot_tasks = set()
        # Mutable state belongs to this input session and is reset at turn boundaries.
        self._text = ""
        self._silence = 0.0
        self._spoke = False
        self._speech_end = 0.0
        self._spec = None
        self._spec_text = ""
        self._spec_started = 0.0  # Speculation throttle timestamp.
        self._gate_pre = None  # Gate task started at silence onset.
        self._gate_pre_text = ""
        self._eot_wait = 0.0  # Elapsed time since the last EOT probe.
        self._eot_score = 0.0  # Latest EOT score.
        self._raw_text = ""  # Streaming transcript before final decoding.
        self._eot_hot = 0.0  # Duration of sustained high EOT scores.
        self._final_asr = _UNSET  # Lazy final-ASR provider.
        self._route = _gate_mod.DEEP  # Current route.
        self._last_memory = None  # Latest completed speculative result.
        self._pcm = []  # Mono 16 kHz turn audio.
        self._pcm_len = 0  # Retained sample count.
        self._preroll = []  # Audio before speech onset.

    @property
    def asr(self):
        self._check_open()
        if self._asr is None:
            self._asr = self._new_audio_adapter("asr")
        return self._asr

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Voice stream is closed")

    def _new_audio_adapter(self, name):
        if name == "vad" and self.vad_threshold is not None:
            from voicemem.utils.audio.stream_io import make_vad
            return make_vad(threshold=self.vad_threshold)
        adapter = self.vm.utils.get(name)
        fork = getattr(adapter, "new_stream", None)
        if fork is not None:
            return fork()
        if name == "asr":
            adapter.reset()
        return adapter

    async def prepare_audio(self):
        """Load weights off the event loop and retain connection-owned state."""
        self._check_open()
        asr, vad = self._asr, self._vad

        def load():
            return (asr if asr is not None else self._new_audio_adapter("asr"),
                    vad if vad is not None else self._new_audio_adapter("vad"))

        loaded_asr, loaded_vad = await asyncio.to_thread(load)
        if self._closed:
            raise RuntimeError("Voice stream is closed")
        self._asr, self._vad = loaded_asr, loaded_vad

    async def aclose(self):
        """Release decoder work without cancelling another connection's models."""
        if self._closed:
            return
        self._closed = True
        tasks = {task for task in (self._spec, self._gate_pre, *self._snapshot_tasks)
                 if task is not None}
        for task in tasks:
            task.cancel()
        worker = getattr(self, "_asr_w", None)
        if worker is not None:
            worker.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        if worker is not None:
            await asyncio.to_thread(worker.t.join, ASR_FINISH_TIMEOUT_S)
        self._asr = self._vad = self._asr_w = None
        self._pcm, self._preroll = [], []

    @property
    def asr_worker(self) -> "_AsrWorker":
        """Load the connection's serial streaming-ASR worker lazily."""
        self._check_open()
        w = getattr(self, "_asr_w", None)
        if w is None:
            w = self._asr_w = _AsrWorker(self.asr)
        return w

    @property
    def vad(self):
        self._check_open()
        if self._vad is None:
            self._vad = self._new_audio_adapter("vad")
        return self._vad

    # Routing is re-evaluated as text grows. The final confirmed turn determines memory eligibility.
    def _gate_locked(self, text) -> str:
        with _EMBED_LOCK:
            return self._gate(text)

    def _gate(self, text) -> str:
        if self.gate is None:
            return _gate_mod.DEEP
        try:
            return self.gate(text)
        except Exception as e:
            # On gate failure, retain memory eligibility rather than suppressing retrieval.
            print(f"[gate] 判定失败（{type(e).__name__}: {e}）→ 按 deep 走", flush=True)
            return _gate_mod.DEEP

    # Speculative retrieval runs outside the event loop.
    async def _speculate(self, text) -> Turn:
        t0 = time.perf_counter()

        def work():
            from voicemem.leftbrain.query_embedding import query_embedding_scope
            started = time.perf_counter()
            # Hold TORCH_LOCK only for model inference, not while waiting for Search().
            # Search waits on other workers whose encoders need the same cross-thread lock.
            route = self._gate_locked(text)
            if route != _gate_mod.DEEP:
                return route, None, started, started, time.perf_counter()
            with query_embedding_scope():
                c = self.vm.classify(text)
                classified = time.perf_counter()
                result = self.vm.search(text, slots=c.slots, entities=c.entities,
                                        emotion=self.emotion or None)
            searched = time.perf_counter()
            return route, result, started, classified, searched

        route, result, started, classified, searched = await asyncio.to_thread(work)
        self._route = route
        if result is None:
            return Turn(text, empty_result(), raw_text=text, route=route)
        print(f"[speculate] {text[:24]!r} -> {len(result.hits)} hits  "
              f"{(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
        if os.environ.get("VOICEMEM_SEARCH_DEBUG", "0") != "0":
            print(f"[search-detail] 线程排队 {(started-t0)*1000:.0f}ms"
                  f" · 分类 {(classified-started)*1000:.0f}ms"
                  f" · 检索 {(searched-classified)*1000:.0f}ms"
                  f" · 回主循环 {(time.perf_counter()-searched)*1000:.0f}ms"
                  f" · 内部 {getattr(result, 'timing', {})}", flush=True)
        return Turn(text, result, raw_text=text, route=_gate_mod.DEEP)

    def _kick(self, text):
        """Restart speculation only when text growth and throttle conditions permit."""
        if not (text and text != self._spec_text and len(text) >= self.spec_min_chars):
            return
        now = time.monotonic()
        if (self._spec_started and now - self._spec_started < SPEC_MIN_INTERVAL_S
                and len(text) - len(self._spec_text) < SPEC_MIN_GROWTH):
            return  # Throttle when time and text growth are both insufficient.
        # Evaluate the gate in the speculation worker, outside the event loop.
        if self._spec:
            self._spec.cancel()
        self._spec_text, self._spec_started = text, now
        self._spec = asyncio.create_task(self._speculate(text))

    def _ready_memory(self):
        """Return completed speculation, otherwise retain the previous result or None."""
        if self._spec is not None and self._spec.done() and not self._spec.cancelled():
            try:
                self._last_memory = self._spec.result().result
            except Exception:
                pass
        return self._last_memory

    @property
    def final_asr(self):
        """Load final ASR lazily; None retains the existing streaming-text fallback."""
        if self._final_asr is _UNSET:
            try:
                self._final_asr = self.vm.utils.get("asr_final")
            except Exception as e:
                print(f"[asr] 离线复核不可用（{type(e).__name__}: {e}）", flush=True)
                self._final_asr = None
        return self._final_asr

    def _transcribe_final(self, pcm):
        """Worker returns text only: cancellation must not overwrite a newer turn."""
        if pcm is None or not len(pcm):
            return None
        started = time.monotonic()
        model = self.final_asr
        loaded = time.monotonic()
        if model is None:
            return None
        try:
            return model.transcribe(pcm)
        except Exception as e:
            print(f"[asr] 复核失败（{type(e).__name__}: {e}）→ 沿用流式转写", flush=True)
            return None
        finally:
            if ASR_DEBUG:
                print(f"[asr-final] 获取/加载 {(loaded-started)*1000:.0f}ms"
                      f" · 推理 {(time.monotonic()-loaded)*1000:.0f}ms", flush=True)

    def _apply_refined(self, text):
        if text and text != self._text:
            if REFINE_DEBUG:
                print(f"[asr] 复核：{self._text!r} → {text!r}", flush=True)
            self._raw_text = self._text
            self._text = text

    def _refine(self, pcm) -> None:
        """Synchronous compatibility helper; the audio loop uses _refine_async."""
        self._apply_refined(self._transcribe_final(pcm))

    async def _refine_async(self, pcm) -> None:
        text = await self._final_text_async(pcm)
        self._apply_refined(text)

    async def _final_text_async(self, pcm):
        """Only return text: competing tasks never mutate turn state."""
        if pcm is None or not len(pcm):
            return
        submitted = time.monotonic()

        def work():
            if ASR_DEBUG:
                print(f"[asr-final] 排队 {(time.monotonic()-submitted)*1000:.0f}ms", flush=True)
            return self._transcribe_final(pcm)

        future = asyncio.get_running_loop().run_in_executor(_FINAL_ASR_EXECUTOR, work)
        try:
            return await asyncio.wait_for(future, timeout=ASR_FINISH_TIMEOUT_S)
        except asyncio.TimeoutError:
            # A running native decode cannot be stopped safely. Its result stays
            # private to this cancelled future and cannot mutate a later turn.
            print(f"[asr-final] 离线复核等待超时（{ASR_FINISH_TIMEOUT_S*1000:.0f}ms，含排队），使用转写回退",
                  flush=True)
            return None

    def refine_current_snapshot(self) -> "asyncio.Task[str]":
        """Freeze current audio and return a background final-ASR transcript task.

        The snapshot is immutable: audio and streaming text received after this
        call cannot change the returned transcript.  If final ASR is unavailable
        or produces no text, the frozen streaming transcript is returned.
        """
        fallback = self._text.strip()
        pcm = np.concatenate(self._pcm).copy() if self._pcm else None

        async def resolve() -> str:
            refined = await self._final_text_async(pcm)
            return (refined or fallback).strip()

        task = asyncio.create_task(resolve())
        self._snapshot_tasks.add(task)
        task.add_done_callback(self._snapshot_tasks.discard)
        return task

    async def _finish_asr(self, pcm):
        """Race both decoders after a final-ASR grace period, with one deadline."""
        started = time.monotonic()
        deadline = started + ASR_FINISH_TIMEOUT_S
        final = asyncio.create_task(self._final_text_async(pcm))
        flush = asyncio.ensure_future(self.asr_worker.flush())
        pending = {final, flush}
        text = stream_text = None
        timed_out = False
        try:
            # Unlike wait_for, wait does not cancel a useful late final result.
            await asyncio.wait({final}, timeout=min(ASR_FINAL_GRACE_S, ASR_FINISH_TIMEOUT_S))
            while True:
                for task in tuple(pending):
                    if not task.done():
                        continue
                    pending.remove(task)
                    if task.cancelled():
                        continue
                    try:
                        result = task.result()
                    except Exception as exc:
                        kind = "离线复核" if task is final else "流式收尾"
                        print(f"[asr-final] {kind}失败（{type(exc).__name__}），尝试另一路转写",
                              flush=True)
                        continue
                    if result and result.strip():
                        if task is final:
                            text = result
                        else:
                            stream_text = result
                if text or stream_text or not pending:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                await asyncio.wait(pending, timeout=remaining,
                                   return_when=asyncio.FIRST_COMPLETED)
            if text:
                if stream_text:
                    self._text = stream_text
                    self._apply_refined(text)
                    source = "完整复核（流式已就绪）"
                else:
                    # Partial streaming text cannot prove full snapshot coverage.
                    self._text = self._raw_text = text
                    self.asr_worker.reset()
                    source = "完整复核（绕过流式积压）"
            elif stream_text:
                self._text = stream_text
                source = "流式收尾"
            else:
                self.asr_worker.reset()
                source = "最近可用转写回退"
            if timed_out:
                print(f"[asr-final] 收尾等待超时（{ASR_FINISH_TIMEOUT_S*1000:.0f}ms），使用最近可用转写",
                      flush=True)
            if ASR_DEBUG:
                print(f"[asr-final] {source} · ASR收尾 {(time.monotonic()-started)*1000:.0f}ms",
                      flush=True)
        finally:
            for task in (final, flush):
                if not task.done():
                    task.cancel()
            await asyncio.gather(final, flush, return_exceptions=True)

    async def _confirm(self) -> Turn:
        """Apply the gate to final text and return a confirmed turn with its memory result."""
        route, pre = None, self._gate_pre
        self._gate_pre = None
        if pre is not None and len(self._text) - len(self._gate_pre_text) < SPEC_MIN_GROWTH:
            try:
                route = await pre  # Reuse completed speculation when available.
            except Exception:
                route = None
        if route is None:
            route = await asyncio.to_thread(self._gate_locked, self._text)
        self._route = route
        if self._route != _gate_mod.DEEP:
            if self._spec:
                self._spec.cancel()
                self._spec = None
            print(f"[gate] {self._route}：{self._text[:16]!r} → 不检索", flush=True)
            return Turn(self._text, empty_result(), raw_text=self._raw_text or self._text,
                        route=self._route)
        try:
            # A final deep route needs retrieval even if earlier partial text was shallow.
            turn = await (self._spec or self._speculate(self._text))
        except asyncio.CancelledError:
            turn = await self._speculate(self._text)
        # Final ASR supplies reply text; completed speculative memory can be reused without another search.
        if turn.text != self._text:
            turn = Turn(self._text, turn.result, raw_text=self._raw_text or self._text,
                        route=self._route)
        return turn

    def _reset_turn(self):
        if getattr(self, "_asr_w", None) is not None:
            self._asr_w.reset()
        elif self._asr is not None:
            self._asr.reset()
        self._text, self._silence, self._spoke = "", 0.0, False
        self._vad_silence = self._voiced_s = 0.0
        self._textless_probed = False
        self._speech_end = 0.0
        self._held_final_text = ""
        self._spec, self._spec_text, self._last_memory = None, "", None
        self._spec_started, self._gate_pre, self._gate_pre_text = 0.0, None, ""
        self._route = _gate_mod.DEEP
        self._eot_wait = 0.0
        self._eot_score = 0.0
        self._raw_text = ""
        self._eot_hot = 0.0
        self._pcm, self._pcm_len = [], 0
        self._preroll = []

    async def feed_text(self, text) -> Turn:
        """Confirm typed input and retrieve memory only when its final gate permits."""
        self._check_open()
        self._route = self._gate(text)
        if self._route != _gate_mod.DEEP:
            return Turn(text, empty_result(), route=self._route)
        return await self._speculate(text)

    async def feed_partial(self, text, ended: bool = False) -> StreamState:
        """Accept cumulative external-ASR text; ended=True confirms the turn."""
        self._check_open()
        text = (text or "").strip()
        new = bool(text) and text != self._text
        if text:
            self._text = text
        if new and self.on_partial:
            self.on_partial(self._text)
        self._kick(self._text)
        if ended and self._text:
            turn = await self._confirm()
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm)
        return StreamState("<speak>" if new else "<silence>", self._text,
                           self._ready_memory(), None, self.vm,
                           silence=self._silence, spoke=self._spoke)

    async def feed(self, pcm_bytes) -> StreamState:
        """Process a PCM16 frame at src_rate, returning the existing input-state snapshot."""
        received_at = time.monotonic()
        if getattr(self, "_asr_w", None) is None or self._vad is None:
            await self.prepare_audio()
        frame = resample(np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0,
                         src=self.src_rate)
        self._text = self.asr_worker.push(frame)  # Queue audio; decoding runs on the ASR worker.
        speaking = self.vad.is_speech(frame)
        if speaking:
            self._held_final_text = ""
        elif self._held_final_text:
            self._text = self._held_final_text
        if speaking:
            self._speech_end = received_at
            self._voiced_s += len(frame) / 16000.0
            self._vad_silence = 0.0
            self._textless_probed = False
        else:
            self._vad_silence += len(frame) / 16000.0

        # Retain voiced-turn audio and a short preroll for perception and replay.
        # Inter-turn silence stays outside the bounded turn buffer.
        if speaking or self._spoke:
            if not self._spoke and self._preroll:  # Prepend the onset preroll.
                self._pcm.extend(self._preroll)
                self._pcm_len += sum(len(f) for f in self._preroll)
                self._preroll = []
            self._pcm.append(frame)
            self._pcm_len += len(frame)
            while self._pcm_len > _MAX_TURN_SAMPLES and len(self._pcm) > 1:
                self._pcm_len -= len(self._pcm.pop(0))
        else:
            self._preroll.append(frame)  # Keep only recent pre-speech audio.
            while sum(len(f) for f in self._preroll) > _PREROLL_SAMPLES and len(self._preroll) > 1:
                self._preroll.pop(0)
        # Without text, continuing sound is measured by energy. With text, VAD silence controls the speech boundary.
        audible = speaking or (
            not self._text.strip() and self._spoke
            and float(np.sqrt(np.mean(frame * frame))) >= SOUND_LEVEL)
        if audible:
            if speaking and self._silence > 0 and self._spec:  # New speech rejects an unfinished speculative turn.
                self._spec.cancel(); self._spec, self._spec_text = None, ""
            if speaking:
                self._spoke = True
                self._gate_pre = None  # New speech invalidates the silence-time gate probe.
            self._silence = 0.0
        else:
            if self._silence == 0.0 and self._spoke and self._text.strip():
                # Start gate inference when silence begins, concurrently with turn confirmation.
                self._gate_pre_text = self._text
                self._gate_pre = asyncio.ensure_future(asyncio.to_thread(self._gate_locked, self._text))
            self._silence += len(frame) / 16000.0
        if self._text.strip() and self.on_partial:
            self.on_partial(self._text)
        # Prefetch while text grows, including the early-confirmation window.
        if self._spoke and self._text.strip() and \
                (self._silence == 0.0 and len(self._text) >= self.spec_min_chars
                 or self._silence >= self.gamble_s):
            self._kick(self._text)
        # Sound-only turns require sustained audio; short background noises do not become dialogue turns.
        sound_only = (not self._text.strip()
                      and self._pcm_len >= MIN_SOUND_ONLY_S * 16000)
        # Sound-only recordings use their own silence threshold.
        need_silence = self.confirm_s if self._text.strip() else SOUND_ONLY_SILENCE_S
        # EOT may confirm before the timeout once minimum silence and pause-policy checks pass.
        semantic_done = False
        # Scores during speech inform speculation; only silence scores participate in turn confirmation.
        if self.eot is not None and self._spoke and self._text.strip() and self._pcm:
            self._eot_wait += len(frame) / 16000.0
            if self._eot_wait >= 0.04:  # Limit EOT probe frequency.
                self._eot_wait = 0.0
                try:
                    _p = self.eot.score(np.concatenate(self._pcm))
                    # Sustained high scores reject isolated mid-utterance spikes.
                    self._eot_hot = (self._eot_hot + 0.04) if _p >= EOT_HOT_LEVEL else 0.0
                    # Accept sustained acoustic completion or a text ending with sufficient acoustic support.
                    _text_done = bool(_TEXT_DONE.search(self._text or ""))
                    self._eot_score = _p if (
                        self._eot_hot >= EOT_HOT_S
                        or (_text_done and _p >= EOT_TEXT_MIN)) else 0.0
                    # Confirmation requires observed silence. A high score during speech is insufficient.
                    semantic_done = (self.eot_ends_turn and _p >= self.eot.threshold
                                     and self._silence > 0
                                     and self._silence >= self.eot_min_s)
                    if EOT_DEBUG and self._silence > 0:
                        print(f"[eot] 静音 {self._silence*1000:3.0f}ms  分数 {_p:.2f}"
                              f"  {'→ 结束' if semantic_done else ''}", flush=True)
                except Exception as e:
                    print(f"[eot] 判定失败（{type(e).__name__}: {e}）→ 退回掐表",
                          flush=True)
                    self.eot = None
        # "ok" often has no streaming hypothesis at all. Do not classify that
        # as music and wait for 5s of audio + 3s silence. Use VAD silence (not
        # residual speaker energy) for one offline probe per voiced burst.
        fast_final = False
        if (self.textless_confirm_s is not None and self._spoke
                and not self._text.strip() and not self._textless_probed
                and self._voiced_s >= 0.12
                and self._vad_silence >= max(self.confirm_s, self.textless_confirm_s)):
            self._textless_probed = True
            _asr_started = time.monotonic()
            pcm = np.concatenate(self._pcm) if self._pcm else None
            refined = await self._final_text_async(pcm)
            if refined and refined.strip():
                self._text = self._raw_text = refined.strip()
                self.asr_worker.reset()
                fast_final = True
                if ASR_DEBUG:
                    print(f"[asr-final] 无流式字短句快速复核 · VAD静音 {self._vad_silence*1000:.0f}ms", flush=True)
            # Empty/failed recognition is not a user turn; keep the sound-only
            # buffer intact and do not repeatedly probe the same silence.
        allow_end = (self.turn_end_guard is None or self.turn_end_guard(
            self._text, self._silence, speaking, len(frame) / 16000.0,
            float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0))
        if fast_final and not allow_end:
            self._held_final_text = self._text
        if allow_end and (fast_final or (self._spoke and (semantic_done or self._silence >= need_silence)
                          and (self._text.strip() or sound_only))):
            if EOT_DEBUG and self.eot is not None:
                print(f"[eot] 回合结束：{'短句快速复核' if fast_final else '语义判定' if semantic_done else f'兜底掐表 {need_silence*1000:.0f}ms'}"
                      f"（静音 {self._silence*1000:.0f}ms）", flush=True)
            if not fast_final:
                _asr_started = time.monotonic()
            if ASR_DEBUG:
                print(f"[asr] 本轮 {self.asr_worker.report()}", flush=True)
            pcm = np.concatenate(self._pcm) if self._pcm else None
            # Final ASR refines the confirmed snapshot; its text is used for the reply and later memory ingestion.
            if not fast_final and not self._held_final_text:
                await self._finish_asr(pcm)
                # Final ASR may reveal a dangling clause absent from the partial.
                # Retain that decode while the app leaves room for continuation.
                if self.turn_end_guard is not None and not self.turn_end_guard(
                        self._text, self._silence, speaking, 0.0,
                        float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0):
                    self._held_final_text = self._text
                    return StreamState("<silence>", self._text, self._ready_memory(),
                                       None, self.vm, silence=self._silence,
                                       spoke=self._spoke, eot_score=0.0,
                                       speech_end=self._speech_end)
            _refined = time.monotonic()
            turn = await self._confirm()  # Yield the confirmed turn and reset its input state.
            if ASR_DEBUG:
                print(f"[asr-final] 判停/VAD后等待 {(_asr_started-self._speech_end)*1000:.0f}ms"
                      f" · ASR并行收尾 {(_refined-_asr_started)*1000:.0f}ms"
                      f" · 回合确认 {(time.monotonic()-_refined)*1000:.0f}ms", flush=True)
            speech_end = self._speech_end
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm, pcm,
                               speech_end=speech_end)
        return StreamState("<speak>" if speaking else "<silence>", self._text,
                           self._ready_memory(), None, self.vm,
                           silence=self._silence, spoke=self._spoke,
                           eot_score=self._eot_score, speech_end=self._speech_end)
