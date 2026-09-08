"""Live API timing with synthetic examples, no memory DB or GPU model loading.

DEEPSEEK_API_KEY must be set, or pass --key-stdin for a non-echoed pipe.
Reports text TTFT / first speakable segment, NOT microphone-to-playback latency.
"""
import argparse
import asyncio
from contextlib import aclosing
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def main(args):
    if args.key_stdin:
        os.environ["DEEPSEEK_API_KEY"] = sys.stdin.readline().strip()
    # Match the demo's first-clause settings before importing the splitter.
    os.environ.setdefault("VOICEMEM_TTS_FIRST_MIN", "2")
    os.environ.setdefault("VOICEMEM_TTS_FIRST_SOFT", "1")
    from voicemem import persona
    from voicemem import tts_control
    from voicemem.reply import deepseek_reply
    from voicemem.tts import cut_point
    provider = deepseek_reply(
        system=persona.system_prompt("zh") + "\n\n" + tts_control.prompt_rule("zh"))
    examples = [
        ("今天工作有点累。", "", []),
        ("你还记得我喜欢什么饮料吗？", "可信记忆：用户喜欢无糖拿铁。", []),
        ("那明天我试试。", "", [
            {"role": "user", "content": "最近总是很晚睡。"},
            {"role": "assistant", "content": "可以先试着比今晚早十分钟放下手机。"}]),
    ]
    rows = []
    try:
        for i in range(args.rounds):
            text, memory, history = examples[i % len(examples)]
            started = time.perf_counter()
            first = None
            buf = ""
            tone_done = False
            async with aclosing(provider(text, memory, history)) as deltas:
                async for delta in deltas:
                    if first is None:
                        first = (time.perf_counter() - started) * 1000
                    buf += delta
                    if not tone_done:
                        tone, rest = tts_control.split(buf)
                        if tone:
                            buf, tone_done = rest, True
                        elif len(buf) >= 26 or any(c in buf for c in "]】"):
                            tone_done = True
                        else:
                            continue
                    if cut_point(buf, first=True):
                        break
            segment = (time.perf_counter() - started) * 1000
            if first is None:
                raise RuntimeError("API returned no speakable text")
            rows.append((first, segment))
            print(f"round={i+1} first_text_ms={first:.0f} first_segment_ms={segment:.0f}", flush=True)
    finally:
        await provider.aclose()
    print(f"median_first_text_ms={statistics.median(x[0] for x in rows):.0f} "
          f"median_first_segment_ms={statistics.median(x[1] for x in rows):.0f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 20:
        parser.error("rounds must be 1..20")
    try:
        asyncio.run(main(args))
    except Exception as exc:
        # Never print headers, credentials or request/response bodies.
        print(f"benchmark_failed={type(exc).__name__} status={getattr(exc, 'status_code', 'n/a')}",
              file=sys.stderr)
        raise SystemExit(1)
