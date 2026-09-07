"""Synthesize a fixed-voice Breeze sample without starting the memory demo.

Run from the repo: python3 tools/test_breeze_tts.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def synthesize(args):
    # Fail on this thread instead of hanging on an unavailable GPU worker.
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError("Breeze requires an Apple Silicon terminal with Metal access")
    from voicemem.breeze_tts import BreezeMLXTTS
    model = ROOT / "models/tts/Breeze-TTS-2-mlx-4bit"
    tts = BreezeMLXTTS(
        model=os.environ.get("VOICEMEM_BREEZE_MLX_MODEL") or
              (str(model) if (model / "config.json").is_file() else None),
        ref_audio=args.ref_audio,
    )
    # Finish one short synthesis; subsequent latency excludes model loading.
    async for _ in tts.stream("你好。"): pass
    started, first, chunks = time.perf_counter(), None, []
    async for chunk in tts.stream(args.text):
        if first is None:
            first = time.perf_counter() - started
        chunks.append(chunk)
    elapsed = time.perf_counter() - started
    pcm = b"".join(chunks)
    if not pcm:
        raise RuntimeError("Breeze produced no audio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    seconds = len(pcm) / 48000
    print(f"{args.output}\nvoice={tts.voice}; first PCM={first:.3f}s; "
          f"audio={seconds:.2f}s; RTF={elapsed / seconds:.2f}; chunks={len(chunks)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default="你好呀，我是诺可。今天过得怎么样？可以慢慢跟我说，我在听。")
    p.add_argument("--ref-audio", default=str(ROOT / "voice/noctelle_ref.wav"))
    p.add_argument("--output", type=Path, default=ROOT / "results/breeze_noctelle.wav")
    asyncio.run(synthesize(p.parse_args()))
