"""Conservative text guard for residual speaker echo after browser AEC."""
import unicodedata
from functools import lru_cache


class UtteranceGuard:
    """Keep capture-time context until even a late final ASR result is handled.

    A playback checkpoint can mark the assistant idle while ASR is still working.
    Neither that checkpoint nor an interrupt should erase this turn's reference.
    """

    def __init__(self):
        self.active = False
        self.started_busy = False
        self.reference = ""
        self._partial = ""
        self._updates = 0

    def observe(self, *, active, busy, reference):
        if active and not self.active:
            self.active = True
            self.started_busy = busy
        if self.active and reference:
            if reference.startswith(self.reference):
                self.reference = reference
            elif not self.reference.endswith(reference):
                self.reference += "\n" + reference
            self.reference = self.reference[-4096:]

    def allow_partial(self, text, *, echo, final=False, backchannel=False,
                      explicit=False, confirmed=False):
        if echo:
            self._partial, self._updates = "", 0
            return False
        if not (self.started_busy or self.reference) or final or explicit or confirmed or backchannel:
            return True
        # A single uncertain fragment during playback is often garbled echo.
        # Require growth, not simply another audio frame with the same hypothesis.
        clean = normalize(text)
        if clean and clean != self._partial:
            self._updates = self._updates + 1 if clean.startswith(self._partial) else 1
            self._partial = clean
        return self._updates >= 2


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text or "").casefold()
                   if c.isalnum())


@lru_cache(maxsize=256)
def is_echo(probe, reference, window=300, ratio=.6, fuzzy_min=4):
    s, hay = normalize(probe), normalize((reference or "")[-window:])
    if not s or not hay:
        return False
    if s in hay:
        return True
    if len(s) < fuzzy_min:
        return False
    # Keep the old continuous-match criterion, but ignore ASR punctuation.
    prev, best = [0] * (len(hay) + 1), 0
    for c in s:
        cur = [0] * (len(hay) + 1)
        for j, h in enumerate(hay, 1):
            if c == h:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    if best / len(s) >= ratio:
        return True
    # One insertion/substitution in a >=6-character continuous phrase, e.g.
    # 在呢刚刚在整理 vs 在呢刚在整理. Do not match scattered topic words.
    if len(s) < 6:
        return False
    prev = [0] * (len(hay) + 1)
    for i, c in enumerate(s, 1):
        cur = [i]
        for j, h in enumerate(hay, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (c != h)))
        prev = cur
    return min(prev) <= 1
