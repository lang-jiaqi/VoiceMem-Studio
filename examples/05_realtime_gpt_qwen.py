"""Connect a standalone VoiceMem microphone agent to a realtime speech API.

    pip install sounddevice pywebrtc-audio "websockets>=14"

    OPENAI_API_KEY=sk-...    python examples/05_realtime_gpt_qwen.py gpt
    DASHSCOPE_API_KEY=sk-... python examples/05_realtime_gpt_qwen.py qwen

Echo-cancelled microphone audio feeds both the remote speech API and VoiceMem.
Separate consumers keep audio upload independent of local turn processing.
Confirmed turns attach their memory context to response.create. Provider-specific
session payloads and sample rates are configured below, independently of Studio.
"""
import argparse
import asyncio
import base64
import json
import os

import numpy as np
import websockets
from _audio import AudioIO, SR

from voicemem import VoiceMem
from voicemem.utils.audio.stream_io import resample

PERSONA = ("你是语音助手。用你记得的事，但别念出来，也别用「我记得你说过」开头。"
           "句子短，一次说一两句就停。")

GPT = {
    "url": "wss://api.openai.com/v1/realtime?model=gpt-realtime",
    "key_env": "OPENAI_API_KEY",
    "in_rate": 24000,
    # Disable automatic replies so confirmed local turns can attach memory.
    # Server VAD still interrupts active responses when it detects speech.
    "session": {"type": "realtime", "audio": {
        "input": {"turn_detection": {"type": "server_vad", "create_response": False,
                                     "interrupt_response": True}},
        "output": {"voice": "marin"}}},
    # Fact extraction uses a separate chat request.
    "llm": {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": None},
}

QWEN = {
    "url": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime",
    "key_env": "DASHSCOPE_API_KEY",
    "in_rate": 16000,
    "session": {"modalities": ["text", "audio"], "voice": "Chelsie",
                "input_audio_format": "pcm16", "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad", "create_response": False,
                                   "interrupt_response": True}},
    # Fact extraction uses DashScope's OpenAI-compatible endpoint.
    "llm": {"model": "qwen-plus", "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
}

# Server VAD may already have committed the buffer when the local turn commits it.
# Local and server interruption can also race to cancel the same response.
_EXPECTED_ERRORS = ("input_audio_buffer_commit_empty", "response_cancel_not_active")


def to_wire(pcm: bytes, rate: int) -> str:
    """Resample microphone PCM16 to the provider's input rate and encode as base64."""
    if rate != SR:
        f = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        out = resample(f, src=SR, dst=rate)
        pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode()


async def main(p, name):
    vm = VoiceMem.from_config({
        "mode": "normal",
        "embedding": {"provider": "local"},
        "slots":     {"provider": "local"},
        "llm": {"provider": "openai", "config": p["llm"]},
    })

    loop = asyncio.get_running_loop()
    to_ws: asyncio.Queue = asyncio.Queue()
    to_vm: asyncio.Queue = asyncio.Queue()
    stream = vm.stream(src_rate=SR)
    turn = {"text": "", "reply": "", "live": False}

    print("[warmup] 正在加载模型…", flush=True)
    await asyncio.to_thread(vm.warmup)
    await asyncio.to_thread(vm.search, "预热")
    await stream.feed(b"\x00" * 320)

    key = os.environ[p["key_env"]]

    async with websockets.connect(
            p["url"], additional_headers={"Authorization": f"Bearer {key}"}) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": p["session"]}))

        def on_barge_in():
            """Cancel remote generation after AudioIO has stopped local playback."""
            if not turn["live"]:
                return
            turn["live"] = False
            print("\n[打断]", flush=True)
            asyncio.create_task(ws.send(json.dumps({"type": "response.cancel"})))

        audio = AudioIO(loop, on_barge_in=on_barge_in)

        async def split():
            """Send each microphone block to both consumers."""
            while True:
                pcm = await audio.mic.get()
                to_ws.put_nowait(pcm)
                to_vm.put_nowait(pcm)

        async def uplink():
            """Upload audio independently of local turn processing."""
            while True:
                pcm = await to_ws.get()
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": to_wire(pcm, p["in_rate"])}))

        async def pump():
            """Consume remote events without blocking on playback or ingestion."""
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type", "")

                if t.endswith("audio.delta"):                  # gpt: response.output_audio.delta
                    if turn["live"]:
                        audio.play(base64.b64decode(ev["delta"]))
                elif t.endswith("audio_transcript.delta"):
                    if turn["live"]:
                        turn["reply"] += ev["delta"]
                        print(ev["delta"], end="", flush=True)
                elif t.endswith("input_audio_buffer.speech_started"):
                    # Remote cancellation cannot remove PCM already buffered locally.
                    if turn["live"]:
                        turn["live"] = False
                        audio.stop_playing()
                elif t.endswith("response.done") or t.endswith("response.cancelled"):
                    turn["live"] = False
                    audio.assistant_done()
                    # Keep synchronous ingestion outside the WebSocket event consumer.
                    asyncio.create_task(asyncio.to_thread(
                        vm.ingest, turn["text"], agent_reply=turn["reply"], async_facts=True))
                    turn["reply"] = ""
                elif t == "error":
                    if (ev.get("error") or {}).get("code") not in _EXPECTED_ERRORS:
                        print(f"\n[{name}] {ev.get('error')}", flush=True)

        tasks = [asyncio.create_task(f()) for f in (pump, uplink, split)]

        audio.start()
        print(f"[ready] {name}，说话吧", flush=True)

        try:
            while True:
                st = await stream.feed(await to_vm.get())
                if st.state != "turn_over":
                    continue

                turn.update(text=st.transcript, reply="", live=True)
                print(f"\n你：{st.transcript}\n助手：", end="", flush=True)

                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                # Attach the current memory snapshot to this response only.
                await ws.send(json.dumps({"type": "response.create", "response": {
                    "instructions": f"{PERSONA}\n\n{st.memory_context}"}}))
                audio.assistant_started()
        finally:
            for t in tasks:
                t.cancel()
            audio.close()


ap = argparse.ArgumentParser()
ap.add_argument("provider", choices=["gpt", "qwen"], nargs="?", default="gpt")
args = ap.parse_args()

asyncio.run(main({"gpt": GPT, "qwen": QWEN}[args.provider], args.provider))
