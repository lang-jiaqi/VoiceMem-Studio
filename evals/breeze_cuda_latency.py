"""Native GPU benchmark: warmed PCM delivery, optionally with same-GPU ASR.

Run from the repository root with .venv-cuda/bin/python -m evals.breeze_cuda_latency.
Uses synthetic text/audio, never opens a Memory Space or calls an LLM API.
"""
import argparse
import asyncio
from contextlib import aclosing
import json
import threading
import time

from studio.core.utils.tts.cuda import BreezeCUDATTS
from studio.paths import MODELS, ROOT


async def benchmark(args):
    tts = BreezeCUDATTS(ref_audio=str(ROOT / 'voice/noctelle_ref_short.wav'),
                        device=args.device, depth_mode=args.depth_mode)
    stopped = threading.Event()
    asr_thread = None
    failures = []
    results = []
    try:
        async with aclosing(tts.stream('你好，预热测试。')) as chunks:
            await anext(chunks)
        await asyncio.sleep(.3)
        if args.with_asr:
            from voicemem.utils.audio.asr import FunASRStreamingASR
            import numpy as np
            asr = FunASRStreamingASR(
                model=str(MODELS / 'asr/funasr-paraformer-zh-streaming'), device=args.device)

            def listen():
                try:
                    frame = np.zeros(320, dtype=np.float32)
                    while not stopped.wait(.02):
                        asr.feed(frame)
                except Exception as exc:
                    failures.append(exc)

            asr_thread = threading.Thread(target=listen, name='benchmark-asr')
            asr_thread.start()
        for index in range(args.repeats):
            texts = [args.text]
            if args.segmented:
                from studio.core.utils.tts.segmentation import SpeechBuffer
                buffer = SpeechBuffer()
                texts = []
                for character in args.text:
                    buffer.append(character, 0.0)
                    texts.extend(segment.text for segment in buffer.ready(0.0))
                texts.extend(segment.text for segment in buffer.ready(0.0, final=True))

            async def response():
                for text in texts:
                    async with aclosing(tts.stream(text)) as chunks:
                        async for pcm in chunks:
                            yield pcm

            started = time.monotonic()
            first = None
            size = count = 0
            deficit = 0.0
            async with aclosing(response()) as chunks:
                async for pcm in chunks:
                    now = time.monotonic()
                    if first is None:
                        first = now
                    deficit = max(deficit, now - first - size / 48000)
                    size += len(pcm)
                    count += 1
            elapsed = time.monotonic() - started
            if not size:
                raise RuntimeError('No PCM generated')
            audio = size / 48000
            row = dict(run=index + 1, depth=args.depth_mode, same_gpu_asr=args.with_asr,
                       segments=len(texts),
                       audio_s=round(audio, 3), elapsed_s=round(elapsed, 3),
                       rtf=round(elapsed / audio, 3), first_ms=round((first - started) * 1000),
                       delivery_deficit_ms=round(deficit * 1000), chunks=count)
            results.append(row)
            print(json.dumps(row), flush=True)
        if failures:
            raise failures[0]
        if args.assert_realtime and any(r['rtf'] >= 1 or r['delivery_deficit_ms'] > 40 for r in results):
            raise RuntimeError('PCM delivery does not sustain realtime playback')
    finally:
        stopped.set()
        if asr_thread is not None:
            await asyncio.to_thread(asr_thread.join)
        await tts.aclose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--depth-mode', choices=('cached', 'compiled'), default='compiled')
    parser.add_argument('--with-asr', action='store_true')
    parser.add_argument('--segmented', action='store_true', help='Use Studio text segmentation')
    parser.add_argument('--assert-realtime', action='store_true')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--text', default='你好呀，我会用自然的语气回答你的问题，接下来我们检查语音是否能够连续播放。')
    args = parser.parse_args()
    if args.repeats < 1 or not args.text.strip():
        parser.error('repeats must be positive and text must not be empty')
    asyncio.run(benchmark(args))
