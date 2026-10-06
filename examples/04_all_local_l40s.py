"""Run a standalone local voice agent with vLLM and VoxCPM.

    # Start vLLM in a separate terminal.
    vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.5

    pip install sounddevice voxcpm pywebrtc-audio
    bash scripts/download_models.sh
    python examples/04_all_local_l40s.py

Intended for a CUDA server such as an L40S. Inference uses local audio models,
embeddings and slot classification; fact extraction and replies use the local
OpenAI-compatible vLLM endpoint. AudioIO supplies echo cancellation and barge-in.
Model downloads and hardware requirements are separate from Studio startup.
"""
import asyncio
from contextlib import aclosing

from _audio import AudioIO, SR

from voicemem import VoiceMem
from voicemem.tts import speak_stream, tts_stream

# Placeholder key for a local vLLM server started without API-key authentication.
VLLM = {"model": "Qwen/Qwen3-8B", "api_key": "EMPTY",
        "base_url": "http://127.0.0.1:8000/v1"}

# VoxCPM can share GPU resources with vLLM. Piper is an alternative provider;
# it requires a voice model configured through VOICEMEM_TTS_MODEL.
TTS = {"tts": {"provider": "voxcpm"}}

# Use an embedding model suited to the input languages. Changing that model
# requires re-embedding stored vectors; equal dimensions do not imply compatibility.
vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},
    "slots":     {"provider": "local"},
    "llm":   {"provider": "openai", "config": VLLM},
    "reply": {"provider": "openai", "config": VLLM},
})


async def warmup(stream):
    """Initialize local capabilities, retrieval and TTS before microphone capture."""
    await asyncio.to_thread(vm.warmup)
    await asyncio.to_thread(vm.search, "预热")
    await stream.feed(b"\x00" * 320)
    async for _ in tts_stream("你好。", TTS):
        break


async def main():
    loop = asyncio.get_running_loop()
    stream = vm.stream(src_rate=SR)

    stop = asyncio.Event()
    audio = AudioIO(loop, on_barge_in=stop.set)

    print("[warmup] 正在加载模型…", flush=True)
    await warmup(stream)

    audio.start()
    print("[ready] 说话吧", flush=True)

    try:
        while True:
            st = await stream.feed(await audio.mic.get())
            if st.state != "turn_over":
                continue

            print(f"\n你：{st.transcript}\n助手：", end="", flush=True)

            # Use the confirmed turn's memory snapshot and stream speech as text arrives.
            stop.clear()
            audio.assistant_started()
            try:
                # Closing on barge-in also cancels pending text and synthesis work.
                async with aclosing(speak_stream(
                        vm.reply_stream(st), TTS,
                        on_delta=lambda d: print(d, end="", flush=True))) as speech:
                    async for pcm in speech:
                        if stop.is_set():
                            print("\n[打断]", flush=True)
                            break
                        audio.play(pcm)
            finally:
                audio.assistant_done()

            # Keep synchronous ingestion outside the microphone loop. reply_stream
            # captures generated text; this example does not track what was heard.
            asyncio.create_task(asyncio.to_thread(vm.ingest, st.transcript, async_facts=True))
    finally:
        audio.close()


asyncio.run(main())
