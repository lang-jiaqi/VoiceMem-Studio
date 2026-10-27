"""Prepare public speech for training with explicitly weak consensus labels."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

from .audio_data import observe, recognize
from .public_audio import DATASET, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--laya-source', type=Path, required=True)
    parser.add_argument('--laya-model', type=Path, required=True)
    parser.add_argument('--qwen-model', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.laya_source.resolve()))
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    from studio.core.utils.asr.initialize import final, streaming
    from studio.core.utils.eot.initialize import create as create_eot
    from studio.core.utils.component_config.component import vad
    from studio.core.utils.reply_modes.component import QwenThinkingRouter, SLOW
    from studio.core.utils.reply_modes.laya import LayaDecisionRouter

    table = pq.read_table(args.parquet, columns=['split_name', 'index', 'round',
                                               'question', 'question_audio', 'answer'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'wav').mkdir(exist_ok=True)
    manifest = args.out / 'paired.jsonl'
    done = {json.loads(line)['source_row'] for line in manifest.read_text().splitlines()} if manifest.exists() else set()
    history, counts = {}, {'source_rows': 0, 'accepted': 0, 'disagreement': 0, 'reserved': 0}
    first = QwenThinkingRouter(model=str(args.qwen_model), device='cuda:0')
    second = LayaDecisionRouter(args.laya_model, device='cuda:0', allow_prototype=True)
    recognizer, eot = final(), create_eot()
    with manifest.open('a', encoding='utf-8') as dest:
        for index, row in enumerate(table.to_pylist()):
            if args.limit and index >= args.limit:
                break
            group = f'public-{row["split_name"]}-{row["index"]}'
            prior = list(history.setdefault(group, [])[-4:])
            history[group].extend(({'role': 'user', 'content': row['question']},
                                   {'role': 'assistant', 'content': row['answer']}))
            identity = f'{args.parquet.stem}-{index}'
            counts['source_rows'] += 1
            if identity in done:
                continue
            # Teacher-generated labels are training-only. They never become an
            # evaluation target used to claim either teacher's accuracy.
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
            if bucket >= 6:
                counts['reserved'] += 1
                continue
            a = first.classify(row['question'], history=prior)
            b = second.classify(row['question'], history=prior)
            if a.level != b.level:
                counts['disagreement'] += 1
                continue
            raw = row['question_audio']['bytes']
            pcm, rate = sf.read(io.BytesIO(raw), dtype='float32')
            if pcm.ndim == 2:
                pcm = pcm.mean(axis=1)
            factor = math.gcd(rate, 16000)
            pcm = resample_poly(pcm, 16000//factor, rate//factor).astype(np.float32)
            wave = args.out / 'wav' / (identity+'.wav')
            sf.write(wave, pcm, 16000, subtype='PCM_16')
            samples = np.pad(pcm, (0, 4000))
            detector = vad().new_stream()
            measured = observe(samples, detector, eot)
            observed_text = recognize(samples, recognizer, streaming)
            for engine, text in observed_text.items():
                if not text:
                    continue
                item = {'id': f'{identity}-{engine}', 'group': group, 'split': 'train',
                        'text': text, 'history': prior, 'needs_deep': a.level == SLOW,
                        'audio_state': measured, 'reference_text': row['question'],
                        'source': 'weak-consensus:'+DATASET, 'source_revision': REVISION,
                        'source_row': identity, 'label_provenance': 'local-qwen-and-laya-agreement',
                        'asr': engine, 'audio_path': str(Path('wav')/wave.name),
                        'audio_sha256': hashlib.sha256(raw).hexdigest()}
                dest.write(json.dumps(item, ensure_ascii=False)+'\n')
                dest.flush()
            counts['accepted'] += 1
            if counts['accepted'] % 20 == 0:
                print(json.dumps(counts), flush=True)
    (args.out / 'report.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
