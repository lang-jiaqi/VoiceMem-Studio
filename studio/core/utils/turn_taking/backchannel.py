"""Studio backchannel implementation."""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field



ENABLED = True

def emitting() -> bool:
    return ENABLED

DEBUG = False

from studio.harness.turn_taking.policy import (BackchannelPolicy, SessionFrequencyCurve, _ASK, _INVITE, _DISCLOSE)

ZH_AFFIRMATIVE_TOKENS = ("哦", "哦哦", "嗯嗯", "嗯", "对", "明白")
ZH_QUESTION_TOKENS = ("哦？", "嗯？", "是吗？")
ZH_SYNTHESIS_TEXT = {
    token: token if token.endswith("？") else f"{token}。"
    for token in (*ZH_AFFIRMATIVE_TOKENS, *ZH_QUESTION_TOKENS)
}
ZH_VARIANTS = {
    **{token: 2 for token in ZH_AFFIRMATIVE_TOKENS},
    **{token: 5 for token in ("哦", "嗯嗯", "嗯")},
    **{token: 1 for token in ZH_QUESTION_TOKENS},
}

# Chinese acknowledgements deliberately stay within two restrained intents:
# attentive affirmation and mild curiosity. Strong praise, surprise, sympathy,
# or dismay makes a cached clip sound emotionally wrong when context is noisy.
_TOKENS = {
    "zh": {
        "continuer":  ["嗯", "嗯嗯"],
        "support":    ["嗯", "嗯嗯", "明白"],
        "agree":      ["嗯", "对", "明白"],
        "surprise":   list(ZH_QUESTION_TOKENS),
        "assess_good":["嗯嗯", "对"],
        "assess_bad": ["嗯", "明白"],
        "neutral":    ["嗯", "嗯嗯", "哦", "哦哦", "对", "明白"],
    },
    "en": {
        "continuer":  ["mm-hmm", "mmm", "uh-huh"],
        "support":    ["mm-hmm", "mmm", "oh", "i see", "okay"],
        "agree":      ["yeah", "right", "uh-huh", "exactly", "for sure"],
        "surprise":   ["really", "oh wow", "no way", "oh really", "huh"],
        "assess_good":["that's great", "nice", "amazing", "that's awesome"],
        "assess_bad": ["that's rough", "oh no", "sorry to hear that", "that's tough"],
        "neutral":    ["mm-hmm", "uh-huh", "yeah", "okay", "right", "mmm"],
    },
}

_NEWS = re.compile(
    r"(升职|加薪|辞职|离职|跳槽|失业|分手|结婚|订婚|离婚|怀孕|搬家|搬去|考上|"
    r"录取|中了|拿到|买了|卖了|住院|生病|出事|退休|毕业|第一次|终于)"
    r"|\b(got promoted|quit|resigned|laid off|broke up|got married|engaged|"
    r"pregnant|moving|moved|got in|got accepted|won|bought|sold|"
    r"in hospital|retired|graduated|finally|for the first time)\b",
    re.IGNORECASE)

_GOOD = re.compile(
    r"(升职|加薪|考上|录取|中了|拿到|通过|成功|终于|搞定|赢了|结婚|订婚|怀孕|毕业)"
    r"|\b(got promoted|got a raise|got in|got accepted|won|passed|made it|"
    r"finally|nailed it|got married|engaged|graduated)\b", re.IGNORECASE)
_BAD = re.compile(
    r"(失业|裁员|辞退|分手|离婚|住院|生病|去世|没通过|挂了|失败|取消|丢了|出事|被拒)"
    r"|\b(laid off|fired|broke up|divorced|in hospital|passed away|failed|"
    r"rejected|cancell?ed|lost)\b", re.IGNORECASE)
_NEGATIVE = {"难过", "悲伤", "委屈", "焦虑", "孤独", "疲惫", "纠结",
             "sad", "wronged", "anxious", "lonely", "tired", "conflicted"}

def pick_group(text: str, emotion: str) -> str:
    """Choose the acknowledgement group for the current text and emotion."""
    t = (text or "").strip()
    good, bad = bool(_GOOD.search(t)), bool(_BAD.search(t))
    if good != bad:
        return "assess_good" if good else "assess_bad"
    if _NEWS.search(t):
        return "surprise"
    if (emotion or "").strip() in _NEGATIVE or _DISCLOSE.search(t):
        return "support"
    if _INVITE.search(t):
        return "agree"
    return "neutral"

def pick_token(text: str, emotion: str, lang: str, rng: random.Random,
               recent=(), available=None) -> str:
    """Choose an available language-specific token while avoiding repeats."""
    bank = _TOKENS.get(lang) or _TOKENS["en"]
    group = pick_group(text, emotion)
    if group != "neutral" and rng.random() < 0.1:
        group = "neutral"
    pool = bank[group]
    if available is not None:
        pool = [t for t in pool if t in available]
        if not pool:
            pool = [t for t in bank["neutral"] if t in available]
        if not pool:
            return ""
    choices = [t for t in pool if t not in recent] or pool
    return rng.choice(choices)

@dataclass
class Backchannel:
    """Track per-session acknowledgement decisions and emission cooldown."""
    policy: BackchannelPolicy = field(default_factory=BackchannelPolicy)
    rng: random.Random = field(default_factory=random.Random)
    _last_at: float | None = None
    _last_token: str = ""

    _recent: list = field(default_factory=list)
    _armed: bool = True
    _speech_started: float = 0.0
    _completed_turns: int = 0
    _phase_targets: list = field(default_factory=lambda: [None] * 4)
    _phase_emitted: list = field(default_factory=lambda: [0] * 4)

    @property
    def completed_turns(self) -> int:
        return self._completed_turns

    def reset_turn(self) -> None:
        self._speech_started = 0.0
        self._armed = True
        self._phase_targets = [None] * 4
        self._phase_emitted = [0] * 4

    def complete_turn(self) -> None:
        """Finish one accepted user turn and reset turn-local state."""
        self._completed_turns += 1
        self.reset_turn()

    def can_emit(self, now: float | None = None) -> bool:
        """Return whether the shared session cooldown permits another clip."""
        now = now if now is not None else time.monotonic()
        return (emitting() and
                (self._last_at is None or
                 now - self._last_at >= max(0.0, self.policy.refractory_s)))

    def choose(self, *, text: str, emotion: str = "", lang: str = "zh",
               available=None, now: float | None = None) -> str | None:
        """Choose an acknowledgement without committing it as emitted."""
        if not self.can_emit(now):
            return None
        token = pick_token(text, emotion, lang, self.rng, self._recent, available)
        return token

    def mark_emitted(self, token: str, now: float | None = None) -> None:
        """Commit a clip only after the transport has accepted it."""
        if not token:
            return
        self._last_at = now if now is not None else time.monotonic()
        self._last_token = token
        self._recent = (self._recent + [token])[-3:]

    def offer(self, *, text: str, silence: float, spoke: bool, speech_s: float,
              emotion: str = "", tail_rms: float = 0.0, prev_rms: float = 0.0,
              lang: str = "zh", now: float | None = None, available=None,
              unfinished: bool = False) -> str | None:
        """Return an eligible acknowledgement or None without synthesizing audio."""
        now = now if now is not None else time.monotonic()
        if silence < 0.01:
            self._armed = True
            if spoke and not self._speech_started:
                self._speech_started = now
            return None
        if not (emitting() and spoke and self._armed):
            return None
        if silence < self.policy.gap_s or (not unfinished and silence >= self.policy.max_gap_s):
            return None
        clean = "".join(ch for ch in (text or "") if ch.isalnum())
        # Very short turns are usually complete greetings or acknowledgements;
        # do not fill their first silence. Unfinished clauses still qualify once
        # they contain enough signal to be intentional speech.
        if not unfinished and _ASK.search(text or "") and not _INVITE.search(text or ""):
            return None
        if not clean or len(clean) < (2 if unfinished else self.policy.min_chars):
            return None
        self._armed = False
        if self._last_at is not None and now - self._last_at < max(0.0, self.policy.refractory_s):
            return None
        speech = speech_s or (now - self._speech_started if self._speech_started else 0.0)
        phase = self.policy.session_curve.phase(speech)
        target = self._phase_targets[phase]
        if target is None:
            target = self.policy.session_curve.draw_target(speech, self.rng.random())
            self._phase_targets[phase] = target
        if DEBUG:
            print(f"[backchannel] turn={self._completed_turns + 1} "
                  f"阶段={phase + 1} 目标={target} 已说={self._phase_emitted[phase]} "
                  f"说了{speech:.1f}s "
                  f"{text[-12:]!r}", flush=True)
        if self._phase_emitted[phase] >= target:
            return None
        if unfinished:
            pool = (_TOKENS.get(lang) or _TOKENS["en"])["continuer"]
            pool = [t for t in pool if available is None or t in available]
            token = self.rng.choice([t for t in pool if t not in self._recent] or pool) if pool else ""
        else:
            token = pick_token(text, emotion, lang, self.rng, self._recent, available)
        if not token:
            return None
        self.mark_emitted(token, now)
        self._phase_emitted[phase] += 1
        return token

#

#

#

from voicemem.prompt_config import tts_prompts
_STYLES = tts_prompts()["backchannel_styles"]

def _cache_root():
    from studio.paths import ROOT
    return ROOT / "cache/backchannel"

class BackchannelVoice:
    """Cache acknowledgement PCM using the main reply voice."""

    def __init__(self, tts, *, lang: str = "zh", voice_id: str = ""):
        self.tts = tts
        self.lang = lang if lang in _TOKENS else "en"

        self.voice_id = getattr(tts, "cache_voice_id", "") or voice_id or getattr(tts, "voice", "") or type(tts).__name__
        self._clips: dict[str, list[bytes]] = {}
        self._ready = False
        self.primed = False

    @property
    def available(self):
        return set(self._clips)

    @property
    def ready(self) -> bool:
        return self._ready

    def _path(self, token: str, style_idx: int):
        import hashlib
        spoken = self._spoken_text(token)
        key = (f"{self.voice_id}|{self.lang}|{token}|{spoken}|{style_idx}|"
               f"{_STYLES[self.lang][style_idx]}")
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return _cache_root() / f"{h}.pcm"

    def _spoken_text(self, token: str) -> str:
        """Return synthesis text with punctuation that supplies natural prosody."""
        if self.lang == "zh":
            return ZH_SYNTHESIS_TEXT.get(token, token)
        return token

    def _style_indices(self, token: str, variants=None) -> range:
        """Return the configured variants for one token."""
        count = len(_STYLES[self.lang])
        if variants is not None:
            count = min(count, max(0, int(variants)))
        if self.lang == "zh":
            count = min(count, ZH_VARIANTS.get(token, 1))
        return range(count)

    @staticmethod
    def _bundled_filename(token: str, style_idx: int) -> str:
        suffix = "" if style_idx == 0 else f"_{style_idx + 1}"
        return f"OK_{token}{suffix}.wav"

    def _uses_bundled_voice(self) -> bool:
        if self.lang != "zh":
            return False
        from pathlib import Path
        from studio.paths import ROOT as root
        bundled_ref = root / "voice" / "noctelle_ref_short.wav"
        ref = getattr(self.tts, "ref_audio", None)
        try:
            return bool(ref and Path(ref).read_bytes() == bundled_ref.read_bytes())
        except OSError:
            return False

    def _bundled_pcm(self, token: str, style_idx: int) -> bytes:
        """Load a reviewed clip when this TTS uses the bundled Noctelle voice.

        The WAV bank is deliberately voice-specific.  Comparing the reference
        bytes prevents a custom voice with the same filename from silently
        playing Noctelle's acknowledgements.
        """
        if not self._uses_bundled_voice() or "/" in token or "\\" in token:
            return b""
        from pathlib import Path
        from studio.paths import ROOT as root
        try:
            clip = (root / "voice" / "backchannel" /
                    self._bundled_filename(token, style_idx))
            if not clip.is_file():
                return b""
            import wave
            with wave.open(str(clip), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                    raise ValueError(f"bundled backchannel has wrong format: {clip}")
                return wav.readframes(wav.getnframes())
        except OSError:
            return b""

    def _tokens(self) -> list[str]:
        seen, out = set(), []
        for group in _TOKENS[self.lang].values():
            for t in group:
                if t not in seen:
                    seen.add(t); out.append(t)
        return out

    # Reject pathological generations instead of cutting through a syllable.
    MAX_MS = 3000
    # RMS threshold used only to locate leading and trailing silence.
    SILENCE = 0.006
    LEADING_MS = 20
    TRAILING_MS = 120
    MIN_NATURAL_TAIL_MS = 60

    TARGET_DB = -24.0

    @staticmethod
    def _trim(pcm: bytes, max_ms: int = MAX_MS) -> bytes:
        """Trim silence while preserving the complete final syllable."""
        import numpy as np
        a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        if a.size == 0:
            return pcm
        window = 240
        power = np.convolve(
            a * a, np.full(window, 1.0 / window, dtype=np.float32), mode="same")
        loud = np.sqrt(power) > BackchannelVoice.SILENCE
        if not loud.any():
            return b""
        i, j = int(np.argmax(loud)), int(a.size - np.argmax(loud[::-1]))
        leading = round(BackchannelVoice.LEADING_MS * 24)
        trailing = round(BackchannelVoice.TRAILING_MS * 24)
        natural_tail = max(0, a.size - j)
        minimum_tail = round(BackchannelVoice.MIN_NATURAL_TAIL_MS * 24)
        if natural_tail < minimum_tail:
            edge_n = min(a.size, 480)
            edge_rms = float(np.sqrt(np.mean(a[-edge_n:] * a[-edge_n:])))
            voiced_rms = float(np.sqrt(np.mean(a[i:j] * a[i:j])))
            if edge_rms > max(BackchannelVoice.SILENCE, voiced_rms * 0.2):
                return b""
        start, end = max(0, i - leading), min(a.size, j + trailing)
        a = a[start:end]
        missing_tail = trailing - max(0, end - j)
        if missing_tail > 0:
            a = np.pad(a, (0, missing_tail))
        cap = int(max_ms * 24000 / 1000)
        if a.size > cap:
            return b""
        fade = min(240, a.size // 4)
        if fade > 0:
            a[:fade] *= np.linspace(0, 1, fade)
            a[-fade:] *= np.linspace(1, 0, fade)
        return BackchannelVoice._normalize((a * 32767).astype(np.int16).tobytes())

    @staticmethod
    def _normalize(pcm: bytes) -> bytes:
        import numpy as np
        a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        if a.size == 0:
            return pcm
        rms, peak = float(np.sqrt(np.mean(a * a))), float(np.abs(a).max())
        if rms < 1e-5 or peak < 1e-5:
            return pcm
        g = min(10 ** (BackchannelVoice.TARGET_DB / 20) / rms, 0.95 / peak)
        if abs(g - 1.0) < 0.02:
            return pcm
        return (np.clip(a * g, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    async def _synth_one(self, text: str, instruction: str) -> bytes:
        buf = bytearray()
        spoken = self._spoken_text(text)
        try:
            stream = self.tts.stream(spoken, instruction)
        except TypeError:
            stream = self.tts.stream(spoken)
        async for chunk in stream:
            buf.extend(chunk)
        return self._trim(bytes(buf))

    async def prime(self, tokens=None, variants=None, cache_only=False) -> int:
        """Load or synthesize missing clips; cache_only never generates new audio."""
        import time as _t
        t0 = _t.time()
        root = _cache_root()
        root.mkdir(parents=True, exist_ok=True)
        tokens = self._tokens() if tokens is None else list(tokens)
        jobs = [(token, i) for token in tokens
                for i in self._style_indices(token, variants)]
        if cache_only and self._uses_bundled_voice():
            # The checked-in WAV directory is the user's reviewed selection.
            # Missing files intentionally disable variants even if an older
            # generated cache entry still exists.
            jobs = [job for job in jobs if self._bundled_pcm(*job)]
        active_jobs = set(jobs)
        bundled = 0
        for token, i in jobs:
            path = self._path(token, i)
            pcm = self._bundled_pcm(token, i)
            if pcm:
                # The reviewed bank is authoritative for the bundled voice.
                # A colleague may already have clips generated by an older
                # checkout; replace those too, otherwise pulling the WAVs
                # would appear to work while playback still used old audio.
                pcm = self._normalize(pcm)
                if not path.is_file() or path.read_bytes() != pcm:
                    path.write_bytes(pcm)
                bundled += 1
        cached = sum(1 for token, i in jobs if self._path(token, i).is_file())
        total = len(jobs)
        print(f"[backchannel] 开始预合成：{self.lang} / {self.voice_id} · "
              f"{total} 条，其中 {cached} 条已有缓存"
              + (f"（仓库定稿 {bundled} 条）" if bundled else "")
              + ("，仅加载缓存" if cache_only else
                 "" if cached == total else f"，要现合成 {total - cached} 条（首次启动）"),
              flush=True)
        n = 0
        for token in tokens:
            clips = []
            for i in self._style_indices(token, variants):
                if (token, i) not in active_jobs:
                    continue
                style = _STYLES[self.lang][i]
                path = self._path(token, i)
                try:
                    if path.is_file() and path.stat().st_size > 0:
                        clips.append(self._normalize(path.read_bytes()))
                    elif not cache_only:
                        pcm = await self._synth_one(token, style)
                        if pcm:
                            path.write_bytes(pcm)
                            clips.append(pcm)
                except Exception as e:
                    print(f"[backchannel] 合成 {token!r} 失败（跳过）："
                          f"{type(e).__name__}: {e}", flush=True)
            if clips:
                self._clips[token] = clips
                n += len(clips)

                self._ready = True
        self._ready = bool(self._clips)
        self.primed = True
        print(f"[backchannel] 预合成完成：{len(self._clips)} 个词 / {n} 条音频 "
              f"（{_t.time() - t0:.1f}s）", flush=True)
        return n

    def get(self, token: str, rng: random.Random | None = None) -> bytes | None:
        """Return a cached PCM variant or None when unavailable."""
        clips = self._clips.get(token)
        if not clips:
            return None
        return (rng or random).choice(clips)
