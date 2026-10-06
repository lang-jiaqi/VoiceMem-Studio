"""Duplex audio, echo cancellation and barge-in for examples 04 and 05.

One sounddevice callback supplies microphone input and the matching playback
reference to WebRTC APM. This reduces feedback from assistant speech into ASR.

    audio = AudioIO(loop, on_barge_in=stop.set)
    audio.start()
    pcm = await audio.mic.get()      # Echo-cancelled 16 kHz PCM16.
    audio.play(pcm24k)               # Resampled internally to 16 kHz.

Example 03 keeps its own audio helper so it can be read as a single-file agent.
"""
import asyncio
import threading

import numpy as np
import sounddevice as sd
from pywebrtc_audio import AudioProcessor

from voicemem.utils.audio.stream_io import resample

SR = 16000
BLOCK = 160       # 10 ms frames for WebRTC APM.
PLAY_SR = 24000


class AudioIO:
    """Capture microphone PCM, buffer playback and detect sustained user speech.

    ``on_barge_in`` runs on the event loop after the playback buffer is cleared.
    ``barge_s`` requires sustained speech; ``grace_s`` delays interruption detection
    after assistant speech starts so the echo canceller can settle.
    """

    def __init__(self, loop, on_barge_in=None, *, threshold=0.3,
                 barge_s=0.5, grace_s=0.6, mic_backlog=100):
        self.loop = loop
        self.mic: asyncio.Queue = asyncio.Queue(maxsize=mic_backlog)
        self.on_barge_in = on_barge_in
        self.threshold, self.barge_s, self.grace_s = threshold, barge_s, grace_s

        self._buf = bytearray()                 # Playback PCM16 at 16 kHz.
        self._lock = threading.Lock()
        self._active = threading.Event()
        self._spoke_s = 0.0                     # Consecutive detected speech duration.
        self._said_s = 0.0                      # Time since assistant speech started.

        self.aec = AudioProcessor(sample_rate=SR, echo_cancellation=True,
                                  noise_suppression=True, auto_gain_control=False,
                                  stream_delay_ms=0)
        self.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=1,
                                dtype="float32", callback=self._cb)

    # The audio callback must not perform inference or network I/O.
    def _pull(self, n) -> np.ndarray:
        with self._lock:
            take = bytes(self._buf[:n * 2])
            del self._buf[:n * 2]
        out = np.zeros(n, dtype=np.float32)
        f = np.frombuffer(take, np.int16).astype(np.float32) / 32768.0
        out[:len(f)] = f
        return out

    def _cb(self, indata, outdata, frames, _time, _status):
        far = self._pull(frames)
        outdata[:, 0] = far
        clean = self.aec.process(indata[:, 0].copy(), far)

        if self._active.is_set():
            self._said_s += frames / SR
            self._spoke_s = (self._spoke_s + frames / SR
                             if self.aec.speech_probability >= self.threshold else 0.0)
            if self._spoke_s >= self.barge_s and self._said_s >= self.grace_s:
                self._active.clear()
                self._spoke_s = 0.0
                self.stop_playing()
                if self.on_barge_in:
                    self.loop.call_soon_threadsafe(self.on_barge_in)
        else:
            self._spoke_s = 0.0

        pcm = (np.clip(clean, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        # Drop overflow rather than growing the microphone backlog without bound.
        self.loop.call_soon_threadsafe(
            lambda: None if self.mic.full() else self.mic.put_nowait(pcm))

    def play(self, pcm24: bytes):
        """Buffer 24 kHz PCM16 as 16 kHz playback and echo-reference samples."""
        f = np.frombuffer(pcm24, np.int16).astype(np.float32) / 32768.0
        f16 = resample(f, src=PLAY_SR, dst=SR)
        with self._lock:
            self._buf += (np.clip(f16, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def stop_playing(self):
        with self._lock:
            self._buf.clear()

    def busy(self) -> bool:
        with self._lock:
            return bool(self._buf)

    def assistant_started(self):
        self._spoke_s = self._said_s = 0.0
        self._active.set()

    def assistant_done(self):
        self._active.clear()

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()
