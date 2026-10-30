"""Bound pathological local speech without using device speed as a deadline."""
import math
import re

_SYLLABIC = re.compile(r'[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]')
_WORDS = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_PAUSES = re.compile(r'[,，。.!！？?;；:：]')


def speech_budget_seconds(text: str) -> float:
    """Estimate a generous audio ceiling, allowing slow speech and expanded numbers.

    This is an anomaly limit, not a target speaking duration. It operates on
    emitted audio so scheduling delays cannot consume a valid speech budget.
    """
    syllables = len(_SYLLABIC.findall(text))
    words = _WORDS.findall(_SYLLABIC.sub(' ', text))
    word_units = sum(max(1, len(word) / (2 if word.isupper() else 8)) for word in words)
    digits = sum(char.isdecimal() for char in text)
    return max(6.0, 3.0 + syllables + word_units * 1.4
               + digits + len(_PAUSES.findall(text)) * .3)


class SpeechGuard:
    """Own one segment's sample budget and raise before delivering excess audio."""

    def __init__(self, text: str, sample_rate: int):
        self.limit_seconds = speech_budget_seconds(text)
        self.limit_samples = math.ceil(self.limit_seconds * sample_rate)
        self.samples = 0

    def accept(self, samples: int) -> None:
        """Count generated samples; raise RuntimeError on an abnormal overrun."""
        self.samples += samples
        if self.samples > self.limit_samples:
            raise RuntimeError(f'Breeze output exceeded speech budget ({self.limit_seconds:.1f}s)')
