"""Studio interruption implementation."""
from voicemem import gate

class Interruption:
    def _is_backchannel(self, new_chars: str) -> bool:
        return bool(self.BACKCHANNEL_ON) and gate.is_backchannel(new_chars)

    def _is_echo(self, new_chars: str, said: str) -> bool:
        from studio.core.utils.echo_guard.component import is_echo
        return is_echo(new_chars, said, max(self.ECHO_WINDOW, min(len(said), 4096)),
                       self.ECHO_RATIO, self.ECHO_FUZZY_MIN)

    def _barge_text(self, text: str) -> str:
        return self._FILLER_PREFIX.sub("", self._bc_norm(text))

    def _is_explicit_interrupt(self, text: str) -> bool:
        clean = self._barge_text(text)
        return bool(clean) and any(clean.startswith(prefix) for prefix in self._INTERRUPT_PREFIXES)

    def _has_barge_content(self, text: str) -> bool:
        clean = self._barge_text(text)
        if not clean or len(set(clean)) == 1 or self._is_backchannel(text):
            return False
        cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in clean)
        latin = sum(ch.isascii() and ch.isalnum() for ch in clean)
        return cjk >= self.BARGE_MIN_CHARS or latin >= max(3, self.BARGE_MIN_CHARS)

    def _has_strong_final_barge(self, text: str) -> bool:
        clean = self._barge_text(text)
        cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in clean)
        latin = sum(ch.isascii() and ch.isalnum() for ch in clean)
        # Two-character Chinese turns such as "你好" are complete user turns.
        # Keep the threshold aligned with _has_barge_content so short greetings
        # are not discarded while the assistant is still speaking.
        return cjk >= self.BARGE_MIN_CHARS or latin >= max(3, self.BARGE_MIN_CHARS)
