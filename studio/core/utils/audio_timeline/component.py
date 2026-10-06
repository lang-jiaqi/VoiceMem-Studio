"""Shared output-audio timeline for chained TTS and speech-to-speech modes."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from voicemem.audio_timing import OUTPUT_SAMPLE_RATE, TextTimestamp

DEFAULT_SPEECH_UNITS_PER_SECOND = 5.2

class SpeechRateEstimator:
    def __init__(self, initial: float = DEFAULT_SPEECH_UNITS_PER_SECOND):
        self.value = float(initial)

    def update(self, observed: float) -> None:
        if 1.0 <= observed <= 20.0:
            self.value = self.value * 0.7 + observed * 0.3

def _text_cost(ch: str) -> float:
    if "\u3400" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff":
        return 1.0
    if ch in "。！？!?":
        return 1.15
    if ch in "，、；;：:,.":
        return 0.55
    if ch.isspace():
        return 0.16
    return 0.34

def _prefix_end(text: str, start: int, end: int, units: float) -> int:
    if units <= 0:
        return start
    used = 0.0
    for index in range(start, end):
        cost = _text_cost(text[index])
        if used + cost > units + 1e-9:
            return index
        used += cost
    return end

def _range_cost(text: str, start: int, end: int) -> float:
    return sum(_text_cost(ch) for ch in text[start:end])

@dataclass(frozen=True)
class AudioChunkTimestamp:
    output_id: str
    sequence: int
    pts_samples: int
    frame_count: int
    sample_rate: int

@dataclass(frozen=True)
class TextAlignment:
    text_start: int
    text_end: int
    audio_start_samples: int
    audio_end_samples: int
    source: str = "provider"
    confidence: float = 1.0

@dataclass
class _Segment:
    segment_id: int
    text_start: int
    text_end: int
    audio_start_samples: int
    audio_end_samples: int | None = None
    complete: bool = False
    alignments: list[TextAlignment] = field(default_factory=list)

class AudioTimeline:
    """Track generated PCM, delivery and playback on one media clock.

    Studio enables ``track_delivery`` and records successful sends separately.
    Legacy direct callers may treat ``append_audio`` as emission; only that mode
    retains elapsed-time playback estimation before a checkpoint arrives.
    """

    def __init__(self, sample_rate: int = OUTPUT_SAMPLE_RATE,
                 prebuffer_seconds: float = 0.0,
                 speech_units_per_second: float = DEFAULT_SPEECH_UNITS_PER_SECOND,
                 rate_estimator: SpeechRateEstimator | None = None,
                 track_delivery: bool = False):
        self.output_id = uuid.uuid4().hex
        self.sample_rate = int(sample_rate)
        self.prebuffer_seconds = max(0.0, float(prebuffer_seconds))
        self.rate_estimator = rate_estimator or SpeechRateEstimator(
            speech_units_per_second)
        self.generated_text = ""
        self.source_text: str | None = None
        self._source_ends: list[int] = []
        self.generated_samples = 0
        self.sent_samples = 0
        self.rendered_samples = 0
        self.sequence = 0
        self.generation_complete = False
        self.context_saved = False
        self.reply_tone = ""
        self.context_managed = False
        self.track_delivery = track_delivery
        self.interrupted = False
        self._frozen_samples: int | None = None
        self._frozen_text: str | None = None
        self.checkpoint_seen = False
        self.playback_state = "idle"
        self._first_audio_at: float | None = None
        self._segments: list[_Segment] = []
        self._segment_by_id: dict[int, _Segment] = {}
        self._alignments: list[TextAlignment] = []
        self._playback_done = asyncio.Event()

    def append_text(self, delta: str) -> None:
        self.generated_text += delta or ""

    def append_source_text(self, delta: str) -> None:
        """Keep display Markdown separate from the text synthesized as speech."""
        if self.source_text is None:
            self.source_text = ""
        self.source_text += delta

    def append_speech_text(self, delta: str, source_ends: tuple[int, ...], consumed: int,
                           *, previous_end: int = 0) -> None:
        """Map synthesized code points to monotonic source boundaries."""
        if self.source_text is None or len(delta) != len(source_ends):
            raise ValueError("Speech text requires one source boundary per code point")
        previous = self._source_ends[-1] if self._source_ends else 0
        previous = max(previous, min(previous_end, len(self.source_text)))
        for end in source_ends:
            if not previous <= end <= len(self.source_text):
                raise ValueError("Speech source boundaries must be monotonic and in range")
            previous = end
        if self._source_ends:
            self._source_ends[-1] = max(self._source_ends[-1], min(previous_end, len(self.source_text)))
        self.generated_text += delta
        self._source_ends.extend(source_ends)
        if self._source_ends:
            self._source_ends[-1] = max(self._source_ends[-1], min(consumed, len(self.source_text)))

    def _heard_prefix(self, text_end: int) -> str:
        if self.source_text is None:
            return self.generated_text[:text_end].rstrip()
        if text_end <= 0 or not self._source_ends:
            return ""
        source_end = self._source_ends[min(text_end, len(self._source_ends)) - 1]
        return self.source_text[:source_end].rstrip()

    def begin_segment(self, text_start: int, text_end: int) -> int:
        segment_id = len(self._segments)
        segment = _Segment(
            segment_id=segment_id,
            text_start=max(0, int(text_start)),
            text_end=max(0, int(text_end)),
            audio_start_samples=self.generated_samples,
        )
        self._segments.append(segment)
        self._segment_by_id[segment_id] = segment
        return segment_id

    def append_audio(self, pcm: bytes) -> AudioChunkTimestamp:
        if len(pcm) % 2:
            raise ValueError("PCM16 audio chunks must contain complete samples")
        frame_count = len(pcm) // 2
        stamp = AudioChunkTimestamp(
            output_id=self.output_id,
            sequence=self.sequence,
            pts_samples=self.generated_samples,
            frame_count=frame_count,
            sample_rate=self.sample_rate,
        )
        self.sequence += 1
        self.generated_samples += frame_count
        if not self.track_delivery:
            self.mark_sent(frame_count)
        return stamp

    def mark_sent(self, frame_count: int) -> None:
        """Record successful transport delivery, independently of generated PCM."""
        self.sent_samples = min(self.generated_samples,
                                self.sent_samples + max(0, int(frame_count)))
        if self.sent_samples and self._first_audio_at is None:
            self._first_audio_at = time.monotonic()

    def add_segment_timestamps(self, segment_id: int,
                               timestamps: tuple[TextTimestamp, ...]) -> None:
        segment = self._segment_by_id.get(segment_id)
        if segment is None:
            return
        existing = {
            (a.text_start, a.text_end, a.audio_start_samples, a.audio_end_samples)
            for a in segment.alignments
        }
        for timestamp in timestamps:
            rate = max(1, int(timestamp.sample_rate))
            scale = self.sample_rate / rate
            alignment = TextAlignment(
                text_start=min(segment.text_end,
                               segment.text_start + max(0, int(timestamp.text_start))),
                text_end=min(segment.text_end,
                             segment.text_start + max(0, int(timestamp.text_end))),
                audio_start_samples=(segment.audio_start_samples
                                     + round(timestamp.audio_start_samples * scale)),
                audio_end_samples=(segment.audio_start_samples
                                   + round(timestamp.audio_end_samples * scale)),
                confidence=float(timestamp.confidence),
            )
            key = (alignment.text_start, alignment.text_end,
                   alignment.audio_start_samples, alignment.audio_end_samples)
            if alignment.text_end > alignment.text_start and key not in existing:
                segment.alignments.append(alignment)
                existing.add(key)
        segment.alignments.sort(key=lambda item: item.audio_start_samples)

    def add_timestamps(self, timestamps: tuple[TextTimestamp, ...]) -> None:
        existing = {
            (a.text_start, a.text_end, a.audio_start_samples, a.audio_end_samples)
            for a in self._alignments
        }
        for timestamp in timestamps:
            rate = max(1, int(timestamp.sample_rate))
            scale = self.sample_rate / rate
            alignment = TextAlignment(
                text_start=max(0, int(timestamp.text_start)),
                text_end=max(0, int(timestamp.text_end)),
                audio_start_samples=round(timestamp.audio_start_samples * scale),
                audio_end_samples=round(timestamp.audio_end_samples * scale),
                confidence=float(timestamp.confidence),
            )
            key = (alignment.text_start, alignment.text_end,
                   alignment.audio_start_samples, alignment.audio_end_samples)
            if alignment.text_end > alignment.text_start and key not in existing:
                self._alignments.append(alignment)
                existing.add(key)
        self._alignments.sort(key=lambda item: item.audio_start_samples)

    def finish_segment(self, segment_id: int, complete: bool = True) -> None:
        segment = self._segment_by_id.get(segment_id)
        if segment is None or segment.audio_end_samples is not None:
            return
        segment.audio_end_samples = self.generated_samples
        segment.complete = bool(complete)
        duration = segment.audio_end_samples - segment.audio_start_samples
        units = _range_cost(self.generated_text, segment.text_start, segment.text_end)
        if complete and duration > 0 and units > 0:
            observed = units * self.sample_rate / duration
            self.rate_estimator.update(observed)

    def mark_generation_complete(self) -> None:
        self.generation_complete = True
        if not self._segments and self.sent_samples > 0 and self.generated_text:
            units = _range_cost(self.generated_text, 0, len(self.generated_text))
            self.rate_estimator.update(units * self.sample_rate / self.sent_samples)

    def update_checkpoint(self, rendered_samples: int, sample_rate: int,
                          state: str = "playing") -> bool:
        if self._frozen_samples is not None:
            return False
        try:
            rate = max(1, int(sample_rate or self.sample_rate))
        except (TypeError, ValueError, OverflowError):
            rate = self.sample_rate
        try:
            rendered = max(0, int(rendered_samples))
        except (TypeError, ValueError, OverflowError):
            rendered = 0
        converted = round(rendered * self.sample_rate / rate)
        self.rendered_samples = min(self.generated_samples if self.track_delivery else self.sent_samples,
                                    max(self.rendered_samples, converted))
        self.checkpoint_seen = True
        self.playback_state = state or "playing"
        if self.playback_state == "drained":
            if not self.track_delivery:
                self.rendered_samples = self.sent_samples
            self._playback_done.set()
        elif self.playback_state == "interrupted":
            self._playback_done.set()
        return True

    def assume_drained(self) -> None:
        if self.track_delivery:
            self.freeze_playback()
            self._playback_done.set()
            return
        self.rendered_samples = self.sent_samples
        self.playback_state = "drained"
        self._playback_done.set()

    def mark_interrupted(self) -> None:
        self.freeze_playback()
        self.interrupted = True
        self.playback_state = "interrupted"
        self._playback_done.set()

    def freeze_playback(self) -> None:
        """Freeze both sample and text cutoffs before cancellation can change alignment."""
        if self._frozen_samples is None:
            self._frozen_samples = self._effective_rendered_samples()
            self._frozen_text = self.heard_text()

    async def wait_playback_done(self) -> None:
        await self._playback_done.wait()

    @property
    def playback_done(self) -> bool:
        return self._playback_done.is_set()

    def _effective_rendered_samples(self) -> int:
        if self._frozen_samples is not None:
            return self._frozen_samples
        if self.track_delivery or self.checkpoint_seen or self._first_audio_at is None:
            return min(self.sent_samples, self.rendered_samples)
        elapsed = max(0.0, time.monotonic() - self._first_audio_at
                      - self.prebuffer_seconds)
        return min(self.sent_samples, round(elapsed * self.sample_rate))

    def rendered_ms(self) -> int:
        return round(self._effective_rendered_samples() * 1000 / self.sample_rate)

    def rendered_cutoff_samples(self) -> int:
        return self._effective_rendered_samples()

    def _segment_text_end(self, segment: _Segment, rendered: int) -> int:
        if rendered <= segment.audio_start_samples:
            return segment.text_start
        audio_end = segment.audio_end_samples
        if audio_end is not None and audio_end <= segment.audio_start_samples:
            return segment.text_start
        if segment.complete and audio_end is not None and rendered >= audio_end:
            return segment.text_end
        if segment.alignments:
            text_end = segment.text_start
            for alignment in segment.alignments:
                if alignment.audio_end_samples <= rendered:
                    text_end = max(text_end, alignment.text_end)
                elif alignment.audio_start_samples < rendered:
                    text_end = max(text_end, alignment.text_start)
                    break
            return text_end
        if segment.complete and audio_end is not None and audio_end > segment.audio_start_samples:
            fraction = ((rendered - segment.audio_start_samples)
                        / (audio_end - segment.audio_start_samples))
            total = _range_cost(self.generated_text, segment.text_start, segment.text_end)
            return _prefix_end(self.generated_text, segment.text_start,
                               segment.text_end, total * max(0.0, min(1.0, fraction)))
        seconds = ((min(rendered, audio_end) if audio_end is not None else rendered)
                   - segment.audio_start_samples) / self.sample_rate
        return _prefix_end(
            self.generated_text, segment.text_start, segment.text_end,
            seconds * self.rate_estimator.value)

    def heard_text(self) -> str:
        if self._frozen_text is not None:
            return self._frozen_text
        if not self.generated_text:
            return ""
        rendered = self._effective_rendered_samples()
        if rendered <= 0:
            return ""
        if self._alignments:
            text_end = 0
            for alignment in self._alignments:
                if alignment.audio_end_samples <= rendered:
                    text_end = max(text_end, alignment.text_end)
                elif alignment.audio_start_samples < rendered:
                    text_end = max(text_end, alignment.text_start)
                    break
            return self._heard_prefix(min(len(self.generated_text), text_end))
        if self._segments:
            text_end = 0
            for segment in self._segments:
                if rendered <= segment.audio_start_samples:
                    break
                text_end = max(text_end, self._segment_text_end(segment, rendered))
                if (not segment.complete or segment.audio_end_samples is None
                        or rendered < segment.audio_end_samples):
                    break
            return self._heard_prefix(text_end)
        if self.generation_complete and self.sent_samples > 0:
            fraction = max(0.0, min(1.0, rendered / self.sent_samples))
            units = _range_cost(self.generated_text, 0, len(self.generated_text)) * fraction
        else:
            units = rendered / self.sample_rate * self.rate_estimator.value
        text_end = _prefix_end(self.generated_text, 0, len(self.generated_text), units)
        return self._heard_prefix(text_end)
