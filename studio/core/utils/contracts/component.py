import asyncio
import time
from dataclasses import dataclass
from studio.core.utils.reply_modes.initialize import DIRECT
from voicemem import gate

@dataclass
class Pending:
    """Confirmed turn with speculative memory that the final reply router validates."""
    text: str
    memory_context: str
    result: object
    spoken: bool = True
    audio_path: str = ""

    stranger: bool = False
    replay: str = ""
    emotion: str = ""
    route: str = gate.DEEP       # Provisional gate route; the final reply router may replace it.
    reply_mode: str = DIRECT     # direct / memory / memory_cot; the final router owns this.

    early_ok: bool = False
    continuation_prompt: bool = False
    # Monotonic time of the last voiced VAD frame. Latency must include the
    # configured turn confirmation and final-ASR work before reply handoff.
    speech_end: float = 0.0

class ReplySink:
    """Buffer ordered speculative events and PCM until final turn confirmation."""

    def __init__(self, send, send_audio):
        self._send, self._send_audio = send, send_audio
        self._buf: list[tuple[str, object]] = []
        self.live = False
        self._lock = asyncio.Lock()
        self._audio_ready = asyncio.Event()
        self._output_resolved = asyncio.Event()
        self.first_audio_at = 0.0

    async def send(self, msg):
        if isinstance(msg, dict) and msg.get("type") in {"answer_done", "error"}:
            self._output_resolved.set()
        async with self._lock:
            if self.live:
                await self._send(msg)
            else:
                self._buf.append(("json", msg))

    async def send_audio(self, pcm: bytes):
        async with self._lock:
            if pcm:
                if not self.first_audio_at:
                    self.first_audio_at = time.monotonic()
                self._audio_ready.set()
            if self.live:
                await self._send_audio(pcm)
            else:
                self._buf.append(("pcm", pcm))

    @property
    def buffered_ms(self) -> float:
        n = sum(len(p) for k, p in self._buf if k == "pcm")
        return n / 2 / 24000 * 1000

    async def wait_for_audio(self) -> None:
        await self._audio_ready.wait()

    async def wait_for_output(self) -> None:
        """Wait until main audio exists or generation resolves without audio."""
        if self._audio_ready.is_set() or self._output_resolved.is_set():
            return
        audio = asyncio.create_task(self._audio_ready.wait())
        resolved = asyncio.create_task(self._output_resolved.wait())
        try:
            await asyncio.wait((audio, resolved), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (audio, resolved):
                if not task.done():
                    task.cancel()
            await asyncio.gather(audio, resolved, return_exceptions=True)

    async def commit(self, *, final_user_text: str | None = None) -> None:
        """Release buffered output, using the confirmed transcript in the UI."""
        async with self._lock:
            for kind, payload in self._buf:
                if kind == "json":
                    if (final_user_text is not None and isinstance(payload, dict)
                            and payload.get("type") == "user_transcript"):
                        payload = {**payload, "text": final_user_text}
                    await self._send(payload)
                else:
                    await self._send_audio(payload)
            self._buf.clear()
            self.live = True
