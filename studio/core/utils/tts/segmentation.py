"""Sentence-first text buffering, independent of synthesis and PCM transport."""
from __future__ import annotations

from dataclasses import dataclass


FIRST_WAIT_S = 0.45
NEXT_WAIT_S = 0.60
BUFFERED_WAIT_S = 1.20
FIRST_SOFT_CHARS = 28
NEXT_SOFT_CHARS = 60
FIRST_MAX_CHARS = 48
NEXT_MAX_CHARS = 100

_SENTENCE_END = "。！？!?…\n"
_SOFT_END = "，,、；;：:"
_CLOSERS = "\"'”’」』】）》)]}"


def _sentence_ends(text: str):
    for index, char in enumerate(text):
        end = index + 1
        if char == '.':
            following = text[end:end + 1]
            # Do not treat a partial decimal or a dot inside a word as a stop.
            if following and not (following.isspace() or following in _CLOSERS):
                continue
            if not following and index and text[index - 1].isdigit():
                continue
        elif char not in _SENTENCE_END:
            continue
        while end < len(text) and text[end] in _SENTENCE_END + '.' + _CLOSERS:
            end += 1
        yield end


def _cut_index(text: str, first: bool, *, force: bool = False) -> int:
    """Return a prefix boundary, preserving original punctuation and offsets."""
    if not text.strip():
        return len(text) if force else 0
    leading = len(text) - len(text.lstrip())
    limit = leading + (FIRST_MAX_CHARS if first else NEXT_MAX_CHARS)
    for end in _sentence_ends(text):
        if end <= limit and any(char.isalnum() for char in text[:end]):
            return end
    soft_min = FIRST_SOFT_CHARS if first else NEXT_SOFT_CHARS
    for index, char in enumerate(text[:limit]):
        if char in _SOFT_END and index + 1 - leading >= soft_min:
            return index + 1
    if not force and len(text) < limit:
        return 0

    # At a deadline or length limit, keep a substantial phrase where possible.
    window = text[:limit]
    for index in range(len(window) - 1, leading + 10, -1):
        if window[index] in _SOFT_END:
            return index + 1
    for index in range(len(window) - 1, leading, -1):
        if window[index].isspace():
            return index + 1
    return min(len(text), limit)


def cut_point(buf: str, first: bool) -> bool:
    """Compatibility predicate for callers that submit whole buffered phrases."""
    return bool(_cut_index(buf, first))


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    start: int
    end: int


class SpeechBuffer:
    """Own unsent text and its bounded wait within one assistant reply."""

    def __init__(self):
        self.text = ""
        self.offset = 0
        self.first = True
        self.started: float | None = None

    def append(self, delta: str, now: float) -> None:
        self.text += delta
        if self.started is None and self.text.strip():
            self.started = now

    def remaining_wait(self, now: float, buffered_audio_s: float = 0.0) -> float | None:
        if self.started is None:
            return None
        budget = FIRST_WAIT_S if self.first else max(
            NEXT_WAIT_S, min(BUFFERED_WAIT_S, buffered_audio_s - 0.4))
        return max(0.0, self.started + budget - now)

    def ready(self, now: float, buffered_audio_s: float = 0.0,
              *, final: bool = False) -> list[SpeechSegment]:
        remaining = self.remaining_wait(now, buffered_audio_s)
        expired = remaining is not None and remaining <= 0
        output = []
        while self.text:
            end = _cut_index(self.text, self.first, force=final or expired)
            if not end:
                break
            raw = self.text[:end]
            text = raw.strip()
            start = self.offset + len(raw) - len(raw.lstrip())
            if any(char.isalnum() for char in text):
                output.append(SpeechSegment(text, start, start + len(text)))
                self.first = False
            self.offset += end
            self.text = self.text[end:]
            self.started = now if self.text.strip() else None
            # A timeout releases one useful prefix, not every incomplete tail.
            expired = False
        return output
