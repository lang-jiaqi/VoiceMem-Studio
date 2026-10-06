"""Inspect microphone transcription, memory retrieval and background ingestion.

    export OPENAI_API_KEY=sk-...
    python examples/06_mic_memory.py

This example generates no assistant reply or speech. Fact extraction during
ingestion can still call the configured LLM. The ingestion task runs in the
background; its scheduling message does not confirm that persistence has finished.
Press Ctrl-C to exit.
"""
import asyncio
import os
import queue
import sys
from pathlib import Path

import sounddevice as sd

from voicemem import VoiceMem

SR = 16000
BLOCK = 512         # 32 ms of microphone audio at 16 kHz.

vm = VoiceMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},
    "slots": {"provider": "local"},
    "api_key": os.environ["OPENAI_API_KEY"],
    # Share the local-embedding example store with examples 01 and 02.
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
})


def show_partial(text):
    print(f"\r[听] {text}", end="", flush=True)


async def main():
    vm.warmup()

    # Transfer audio from the sounddevice callback thread to the async consumer.
    blocks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        blocks.put(bytes(indata))

    stream = vm.stream(src_rate=SR, on_partial=show_partial)

    with sd.RawInputStream(samplerate=SR, blocksize=BLOCK, dtype="int16",
                           channels=1, callback=on_audio):
        print("说话吧（Ctrl-C 退出）\n", flush=True)
        while True:
            pcm = await asyncio.to_thread(blocks.get)
            st = await stream.feed(pcm)
            if st.state != "turn_over":
                continue

            print(f"\r[听] {st.transcript}")

            left = st.result_leftbrain or []
            right = st.result_rightbrain or []
            if left or right:
                print("[查] 说完这一刻已经检索好的记忆：")
                for m in left:
                    print(f"       左脑  {m}")
                for m in right:
                    print(f"       右脑  {m}")
            else:
                print("[查] 还没有相关记忆（库是空的，多说几句就有了）")

            # Schedule synchronous ingestion outside the microphone loop.
            asyncio.create_task(asyncio.to_thread(vm.ingest, st.transcript))
            print("[存] 已写入，下次可被检索\n", flush=True)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\n再见。")
    sys.exit(0)
