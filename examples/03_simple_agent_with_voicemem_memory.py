"""Run a standalone microphone agent with VoiceMem, OpenAI replies and OpenAI TTS.

Install the project and its audio dependencies before running this script.
This example owns its own audio and dialogue loop; it is not the Studio pipeline.
"""


import asyncio
import contextlib
import io
import os
import queue
import re
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from openai import AsyncOpenAI
from pywebrtc_audio import AudioProcessor
from voicemem import VoiceMem


# ── config ────────────────────────────────────────────────────────────
SR = 16000
BLOCK = 160                    # 10 ms: WebRTC AEC native frame size
BARGE_IN_S = 0.5
BARGE_THRESHOLD = 0.30

OPENAI_MODEL = "gpt-4o-mini"

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_NATIVE_SR = 24000
TTS_MIN_CHARS = 6              # lower = faster first audio, but more fragmented

PROMPT = """你是一个语音助手。结合左脑中本轮有价值的信息，以及右脑中过去沉淀的情景和性格记忆，
给用户一个温暖、有价值、适合直接说出口的回答。不要直接描述或暴露你看到的记忆。

[左脑·事实记忆]
{left}

[右脑·情景记忆]
{right}

[用户刚说]
{text}
"""


# Playback buffer shared with the audio callback.
class AudioBuffer:
    def __init__(self):
        self.q = queue.Queue()
        self.cur = np.empty(0, dtype=np.float32)

    def put(self, audio):
        if len(audio):
            self.q.put(np.asarray(audio, dtype=np.float32))

    def clear(self):
        self.cur = np.empty(0, dtype=np.float32)
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def pull(self, n):
        out = np.zeros(n, dtype=np.float32)
        pos = 0

        while pos < n:
            if not len(self.cur):
                try:
                    self.cur = self.q.get_nowait()
                except queue.Empty:
                    break

            k = min(n - pos, len(self.cur))
            out[pos:pos + k] = self.cur[:k]
            self.cur = self.cur[k:]
            pos += k

        return out


# ── one duplex device: speaker reference -> AEC -> VoiceMem ──────────
class AudioIO:
    def __init__(self, loop):
        self.loop = loop
        self.mic = asyncio.Queue(maxsize=100)
        self.playback = AudioBuffer()

        self.assistant_active = threading.Event()
        self.stop_reply = threading.Event()

        self.spoke_s = 0.0

        self.aec = AudioProcessor(
            sample_rate=SR,
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=False,
            stream_delay_ms=0,
        )

        self.stream = sd.Stream(
            samplerate=SR,
            blocksize=BLOCK,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def _push_mic(self, pcm):
        if not self.mic.full():
            self.mic.put_nowait(pcm)

    def _callback(self, indata, outdata, frames, _time, status):
        far = self.playback.pull(frames)
        outdata[:, 0] = far

        near = indata[:, 0].copy()
        clean = self.aec.process(near, far)

        # Only detect barge-in while assistant is speaking.
        if self.assistant_active.is_set():
            if self.aec.speech_probability >= BARGE_THRESHOLD:
                self.spoke_s += frames / SR
            else:
                self.spoke_s = 0.0

            if self.spoke_s >= BARGE_IN_S:
                self.stop_reply.set()
                self.playback.clear()
        else:
            self.spoke_s = 0.0

        pcm = (
            np.clip(clean, -1.0, 1.0) * 32767
        ).astype(np.int16).tobytes()

        self.loop.call_soon_threadsafe(self._push_mic, pcm)

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()


# One OpenAI client and worker thread process queued speech segments.
class TTS:
    def __init__(self, audio: AudioIO):
        from openai import OpenAI

        self.audio = audio
        self.text_q = queue.Queue()
        self.closed = False
        self.client = OpenAI()

        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def feed(self, text):
        if text:
            self.text_q.put(text)

    def reset(self):
        while True:
            try:
                self.text_q.get_nowait()
            except queue.Empty:
                break
        self.audio.playback.clear()

    def _worker(self):
        while not self.closed:
            text = self.text_q.get()

            if text is None:
                return

            if self.audio.stop_reply.is_set():
                continue

            try:
                # Queue received PCM without waiting for the full synthesis response.
                with self.client.audio.speech.with_streaming_response.create(
                    model=TTS_MODEL, voice=TTS_VOICE, input=text,
                    response_format="pcm",          # 24 kHz mono PCM16.
                ) as resp:
                    tail = b""
                    for chunk in resp.iter_bytes(4096):
                        if self.audio.stop_reply.is_set():
                            break
                        buf = tail + chunk
                        cut = len(buf) & ~1          # Preserve complete PCM16 samples.
                        tail = buf[cut:]
                        if not cut:
                            continue
                        wav = np.frombuffer(buf[:cut], np.int16).astype(np.float32) / 32768.0
                        # Playback and the AEC reference use the same 16 kHz signal.
                        self.audio.playback.put(resample_poly(wav, SR, TTS_NATIVE_SR))
            except Exception as e:
                print(f"\n[tts] 合成失败：{type(e).__name__}: {e}", flush=True)


# ── LLM token stream -> terminal + short TTS chunks ─────────────────
def tts_chunks():
    buf = ""

    def push(delta):
        nonlocal buf
        buf += delta

        # Release at punctuation or the configured character limit.
        if re.search(r"[。！？!?；;，,\n]$", buf) or len(buf) >= TTS_MIN_CHARS:
            x, buf = buf, ""
            return x
        return None

    def flush():
        nonlocal buf
        x, buf = buf, ""
        return x

    return push, flush


async def answer(client, tts, audio, text, left, right):
    prompt = PROMPT.format(
        left="\n".join(left) or "（无）",
        right="\n".join(right) or "（无）",
        text=text,
    )

    push, flush = tts_chunks()
    full = []

    audio.stop_reply.clear()
    audio.assistant_active.set()

    try:
        stream = await client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            stream=True,
        )

        async for event in stream:
            if audio.stop_reply.is_set():
                break

            if event.type != "response.output_text.delta":
                continue

            delta = event.delta
            full.append(delta)

            print(delta, end="", flush=True)

            chunk = push(delta)
            if chunk:
                tts.feed(chunk)

        tail = flush()
        if tail and not audio.stop_reply.is_set():
            tts.feed(tail)

    finally:
        if audio.stop_reply.is_set():
            tts.reset()
            print("\n[barge-in]", flush=True)

        audio.assistant_active.clear()

    return "".join(full).strip()


# ── terminal formatting ──────────────────────────────────────────────
def show_turn(st):
    print(f"\n\n你：{st.transcript}")

    print("\n┌─ 左脑 ─────────────────────")
    for x in st.result_leftbrain:
        print(f"│ {x}")
    print("└────────────────────────────")

    print("┌─ 右脑 ─────────────────────")
    for x in st.result_rightbrain:
        print(f"│ {x}")
    print("└────────────────────────────")

    print("\n助手：", end="", flush=True)


# Initialize providers before starting microphone capture.
async def main():
    key = os.environ["OPENAI_API_KEY"]

    loop = asyncio.get_running_loop()
    client = AsyncOpenAI(api_key=key)

    vm = VoiceMem(openai_key=key)
    # Load lazy audio capabilities before accepting the first utterance.
    print("[warmup] loading local models…", flush=True)
    await asyncio.to_thread(vm.warmup)
    stream = vm.stream(src_rate=SR, vad_threshold=BARGE_THRESHOLD)

    audio = AudioIO(loop)
    tts = await asyncio.to_thread(TTS, audio)   # Starts the API worker, not a local model.

    audio.start()
    print("[ready] 一切准备完成，说话吧。", flush=True)

    try:
        while True:
            pcm = await audio.mic.get()
            st = await stream.feed(pcm)

            # Terminal does NOT print intermediate ASR states.
            if st.state != "turn_over":
                continue

            show_turn(st)

            reply = await answer(
                client, tts, audio,
                st.transcript,
                st.result_leftbrain,
                st.result_rightbrain,
            )

            print()

            if reply:
                # Pass assistant text by keyword; the second positional argument is audio.
                vm.ingest(st.transcript, agent_reply=reply)

    finally:
        audio.close()


if __name__ == "__main__":
    asyncio.run(main())
