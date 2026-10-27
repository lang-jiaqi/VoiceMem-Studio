"""Prepare paired waveform/ASR examples using local, reference-free speech."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .train import load_rows


def perturb(pcm, seed):
    """Add reproducible channel noise; retain actual samples for observation."""
    rng = np.random.default_rng(seed)
    gain = float(rng.uniform(.55, .9))
    rms = max(float(np.sqrt(np.mean(pcm * pcm))), .001)
    noise = rng.normal(0, rms / 10 ** (18 / 20), len(pcm)).astype(np.float32)
    return np.clip(gain * pcm + noise, -1, 1).astype(np.float32)


def augmentation_seed(identity):
    """Keep perturbations identical across worker partitions and resumed runs."""
    return int(hashlib.sha256((identity+':augmentation-v1').encode()).hexdigest()[:8], 16)


def observe(pcm, vad, eot):
    """Measure voiced span, quiet duration and actual native EOT predictions."""
    first = last = None
    quiet_s = rms_slow = 0.0
    peak = 0.0
    # Use native 16 kHz VAD frames; EOT samples the actual growing waveform.
    for offset in range(0, len(pcm), 512):
        frame = pcm[offset:offset + 512]
        padded = np.pad(frame, (0, 512-len(frame)))
        speaking = bool(vad.is_speech(padded))
        rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
        if speaking:
            first = offset if first is None else first
            last = offset + len(frame)
            rms_slow = .9 * rms_slow + .1 * rms if rms_slow else rms
        quiet = rms < max(.008, .25*rms_slow)
        quiet_s = quiet_s + len(frame) / 16000 if quiet else 0.0
        if offset >= 16000 and offset % 8192 == 0:
            peak = max(peak, float(eot.score(pcm[:offset + len(frame)])))
    peak = max(peak, float(eot.score(pcm)))
    result = {'eot_probability': peak, 'silence_s': quiet_s}
    if first is not None and last is not None:
        result['span_s'] = (last-first)/16000
    return result


def recognize(pcm, final_asr, streaming_factory):
    """Retain both observed recognizer outputs, without substituting oracle text."""
    stream = streaming_factory()
    partial = ''
    try:
        for offset in range(0, len(pcm), 9600):
            partial = stream.feed(pcm[offset:offset+9600]) or partial
        flushed = stream.flush() or partial
    finally:
        close = getattr(stream, 'close', None)
        if close is not None:
            close()
    final = final_asr.transcribe(pcm)
    return {'streaming': flushed.strip(), 'final': final.strip()}


async def prepare(args):
    from scipy.signal import resample_poly
    import soundfile as sf
    from studio.core.utils.tts.cuda import BreezeCUDATTS
    from studio.core.utils.asr.initialize import final, streaming
    from studio.core.utils.component_config.component import vad as create_vad
    from studio.core.utils.eot.initialize import create as create_eot
    rows = load_rows(args.data)
    selection = getattr(args, 'select_file', None)
    if selection:
        selected_ids = set(json.loads(Path(selection).read_text()))
        selected = [row for row in rows if row['id'] in selected_ids]
        if len(selected) != len(selected_ids):
            raise ValueError('Selected speech identities are missing from source data')
    else:
        selected = [row for row in rows if row['id'].endswith('-0')]
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        raise ValueError('Invalid data preparation worker partition')
    selected = [row for index, row in enumerate(selected) if index % args.workers == args.worker]
    if args.limit:
        selected = selected[:args.limit]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    wavs = output / 'wav'
    if args.reuse_wav and not wavs.exists():
        wavs.symlink_to(os.path.relpath(args.reuse_wav.resolve(), output.resolve()), target_is_directory=True)
    else:
        wavs.mkdir(exist_ok=True)
    manifest = output / ('paired.jsonl' if args.workers == 1 else f'paired-worker-{args.worker}.jsonl')
    finished = set()
    for existing in output.glob('paired*.jsonl'):
        finished.update(json.loads(line)['id'] for line in existing.read_text().splitlines() if line.strip())
    recognizer, eot = final(), create_eot()
    # No reviewed reference recording or individual voice is used for synthesis.
    tts = BreezeCUDATTS(ref_audio=None, instruction='自然清晰的普通话，语速适中。')
    try:
        with manifest.open('a', encoding='utf-8') as dest:
            for index, row in enumerate(selected):
                utterance = row['id']
                if all(f'{utterance}-{variant}-{asr}' in finished
                       for variant in ('clean', 'noise') for asr in ('streaming', 'final')):
                    continue
                stem = ('case-'+hashlib.sha256(utterance.encode()).hexdigest()[:24]
                        if '/' in utterance else utterance)
                wav = wavs / (stem + '.wav')
                if wav.exists():
                    pcm, rate = sf.read(wav, dtype='float32')
                    if rate != 16000 or pcm.ndim != 1:
                        raise ValueError(f'Invalid prepared wave: {wav.name}')
                else:
                    if args.reuse_wav:
                        raise FileNotFoundError(f'Prepared source wave is missing: {wav.name}')
                    seed = int(hashlib.sha256(utterance.encode()).hexdigest()[:8], 16)
                    tts.seed = seed
                    parts = []
                    async with aclosing(tts.stream(row['text'])) as chunks:
                        async for data in chunks:
                            parts.append(np.frombuffer(data, '<i2').astype(np.float32)/32768)
                    if not parts:
                        raise ValueError(f'No synthesized audio for {utterance}')
                    pcm = resample_poly(np.concatenate(parts), 2, 3).astype(np.float32)
                    sf.write(wav, pcm, 16000, subtype='PCM_16')
                noise_seed = augmentation_seed(utterance)
                variants = {'clean': pcm, 'noise': perturb(pcm, noise_seed)}
                for variant, samples in variants.items():
                    samples = np.pad(samples, (0, 4000)).astype(np.float32)
                    detector = create_vad()
                    fresh = getattr(detector, 'new_stream', None)
                    detector = fresh() if fresh else detector
                    observed = await asyncio.to_thread(observe, samples, detector, eot)
                    recognized = await asyncio.to_thread(recognize, samples, recognizer, streaming)
                    for engine, text in recognized.items():
                        identity = f'{utterance}-{variant}-{engine}'
                        if identity in finished or not text:
                            continue
                        item = {**row, 'id': identity, 'text': text,
                                'audio_state': observed,
                                'source': 'breeze-designed-speech/native-asr-v1',
                                'reference_text': row['text'], 'asr': engine,
                                'augmentation_seed': noise_seed if variant == 'noise' else None,
                                'audio_path': str(Path('wav') / wav.name), 'audio_variant': variant,
                                'audio_sha256': hashlib.sha256(samples.tobytes()).hexdigest()}
                        dest.write(json.dumps(item, ensure_ascii=False)+'\n')
                        dest.flush()
                        finished.add(identity)
                if (index+1) % 10 == 0 or index == 0:
                    print(json.dumps({'completed_utterances': index+1, 'total': len(selected),
                                      'observed_rows': len(finished)}), flush=True)
    finally:
        await tts.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--reuse-wav', type=Path)
    parser.add_argument('--select-file', type=Path)
    args = parser.parse_args()
    asyncio.run(prepare(args))


if __name__ == '__main__':
    main()
