"""Generate the reviewed Chinese Breeze backchannel bank with one model load."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import time
import wave


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def generate(output: Path, ref_audio: Path) -> None:
    import mlx.core as mx
    if not mx.metal.is_available():
        raise RuntimeError("Breeze backchannel generation requires Apple Silicon Metal")

    from harness.turn_taking.backchannel import (
        BackchannelVoice,
        ZH_AFFIRMATIVE_TOKENS,
        ZH_QUESTION_TOKENS,
        _STYLES,
    )
    from voicemem.breeze_tts import BreezeMLXTTS

    model = ROOT / "models/tts/Breeze-TTS-2-mlx-4bit"
    tts = BreezeMLXTTS(
        model=str(model) if (model / "config.json").is_file() else None,
        ref_audio=str(ref_audio),
    )
    voice = BackchannelVoice(tts, lang="zh")
    output.mkdir(parents=True, exist_ok=True)
    tokens = (*ZH_AFFIRMATIVE_TOKENS, *ZH_QUESTION_TOKENS)
    jobs = [(token, i) for token in tokens
            for i in voice._style_indices(token, variants=2)]
    started = time.perf_counter()
    base_seed = tts.seed
    with tempfile.TemporaryDirectory(prefix=".backchannel-", dir=output) as tmp:
        staging = Path(tmp)
        for index, (token, style_idx) in enumerate(jobs, 1):
            pcm = b""
            for attempt in range(4):
                tts.seed = base_seed + index * 101 + attempt
                pcm = await voice._synth_one(token, _STYLES["zh"][style_idx])
                if pcm:
                    break
            if not pcm:
                raise RuntimeError(
                    f"Breeze did not produce a complete natural ending for {token!r} "
                    f"variant {style_idx + 1}; existing clips were kept")
            name = voice._bundled_filename(token, style_idx)
            path = staging / name
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(pcm)
            print(f"[{index:02d}/{len(jobs)}] {token} v{style_idx + 1} "
                  f"{len(pcm) / 48:.0f}ms", flush=True)
        keep = {voice._bundled_filename(token, i) for token, i in jobs}
        for path in staging.glob("OK_*.wav"):
            os.replace(path, output / path.name)
        removed = 0
        for path in output.glob("OK_*.wav"):
            if path.name not in keep:
                path.unlink()
                removed += 1
    print(f"Generated {len(jobs)} clips in {time.perf_counter() - started:.1f}s", flush=True)
    if removed:
        print(f"Removed {removed} clips outside the reviewed bank", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "voice/backchannel")
    parser.add_argument("--ref-audio", type=Path,
                        default=ROOT / "voice/noctelle_ref_short.wav")
    args = parser.parse_args()
    asyncio.run(generate(args.output, args.ref_audio))
