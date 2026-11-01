"""Compare default and post-run-idle CPU sessions on nonpersonal audio fixtures.

Run with existing Studio model weights, without other performance workloads:
    python -m evals.asr_runtime sample-zh.wav sample-en.wav --component both

A uses plain CPU; B uses Studio's packaged session configuration. Model weights,
threads, audio and decoding settings remain identical. Measurements cover native
recognition calls and CPU time in a short post-run idle window, not capture,
EOT, TTS, playback, power or temperature. Nothing is recorded, played, downloaded
or written; transcript digests are printed without their contents.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from studio.core.utils.asr.component import XASRStreamingASR, cpu_provider
from studio.core.utils.asr.profile import XASR
from studio.paths import MODELS
from voicemem.utils.audio.asr import OfflineASR, StreamingASR
from voicemem.utils.audio.stream_io import read_wav, resample


class StreamingRuntime(XASRStreamingASR):
    """Keep the Studio decoder and explicit parameters with a selectable provider."""

    def __init__(self, provider):
        StreamingASR.__init__(self, str(MODELS / "asr" / XASR), prefer_int8=True,
                              num_threads=1, model_type="zipformer2", provider=provider)

    def transcribe(self, audio):
        self.reset()
        for start in range(0, len(audio), 640):
            self.feed(audio[start:start + 640])
        return self.flush()


def run(args):
    samples = []
    for path in args.samples:
        audio, rate = read_wav(path)
        audio = resample(audio, src=rate)
        if not 0 < len(audio) <= 30 * 16000 or not np.isfinite(audio).all():
            raise ValueError("Samples must contain finite audio longer than 0 and at most 30 seconds")
        samples.append(audio)
    print(json.dumps({"environment": {
        "system": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "sherpa_onnx": version("sherpa-onnx"),
        "order": args.order, "repeats": args.repeats, "tail_ms": args.tail_ms}}, sort_keys=True))
    components = ("streaming", "final") if args.component == "both" else (args.component,)
    for component in components:
        results = []
        for mode in args.order:
            provider = "cpu" if mode == "A" else cpu_provider()
            model = (StreamingRuntime(provider) if component == "streaming" else
                     OfflineASR(str(MODELS / "asr/sensevoice-int8"), provider=provider,
                                allowed_languages=("zh", "en", "yue")))
            model.transcribe(samples[0])
            time.sleep(args.tail_ms / 1000)
            for repeat in range(args.repeats):
                for index, audio in enumerate(samples, 1):
                    cpu, wall = time.process_time(), time.perf_counter()
                    text = model.transcribe(audio)
                    compute_s, compute_cpu_s = time.perf_counter() - wall, time.process_time() - cpu
                    time.sleep(args.tail_ms / 1000)
                    tail_cpu_s = time.process_time() - cpu - compute_cpu_s
                    row = {"component": component, "mode": mode, "repeat": repeat + 1,
                           "sample": index, "audio_ms": round(len(audio) / 16, 2),
                           "asr_ms": round(compute_s * 1000, 2),
                           "asr_cpu_ms": round(compute_cpu_s * 1000, 2),
                           "post_run_cpu_ms": round(tail_cpu_s * 1000, 2),
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                           "text_chars": len(text)}
                    results.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
            del model
            gc.collect()
        summary = {"component": component, "same_text": all(
            len({r["text_sha256"] for r in results if r["sample"] == sample}) == 1
            for sample in range(1, len(samples) + 1))}
        for mode in ("A", "B"):
            group = [row for row in results if row["mode"] == mode]
            summary[mode] = {key: round(statistics.mean(row[key] for row in group), 2)
                             for key in ("asr_ms", "asr_cpu_ms", "post_run_cpu_ms")}
        print(json.dumps({"comparison": summary}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path)
    parser.add_argument("--component", choices=("streaming", "final", "both"), default="both")
    parser.add_argument("--order", default="ABBA")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tail-ms", type=float, default=80)
    args = parser.parse_args()
    if set(args.order) != {"A", "B"}:
        parser.error("--order must contain both A and B, with no other characters")
    if not 1 <= args.repeats <= 100:
        parser.error("--repeats must be between 1 and 100")
    if not np.isfinite(args.tail_ms) or not 0 <= args.tail_ms <= 1000:
        parser.error("--tail-ms must be between 0 and 1000")
    run(args)


if __name__ == "__main__":
    main()
