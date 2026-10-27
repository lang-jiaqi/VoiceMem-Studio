"""Qualify a local router on held-out wording families and optional private canaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from finetune.decision_router.dialogue_eval import summarize
from finetune.decision_router.train import load_rows


def evaluate(router, rows, *, audio_mode='full'):
    """Clear result caching and preserve authored categories and acoustic missingness."""
    predicted, timings, evidence = [], [], []
    for row in rows:
        state = row.get('audio_state')
        if audio_mode == 'none':
            state = None
        elif audio_mode == 'early' and state:
            state = {name: value for name, value in state.items() if name != 'span_s'}
        router._cache.clear()
        start = time.perf_counter()
        decision = router.classify(row['text'], history=row.get('history'), audio_state=state)
        elapsed = (time.perf_counter()-start)*1000
        guess = decision.level == 'slow'
        predicted.append(guess)
        timings.append(elapsed)
        evidence.append({'id': row['id'], 'expected': row['needs_deep'], 'predicted': guess,
                         'raw': decision.raw, 'ms': elapsed})
    result = summarize(rows, predicted)
    ordered = sorted(timings)
    result.update(median_ms=statistics.median(timings), p95_ms=ordered[int(.95*(len(ordered)-1))],
                  audio_mode=audio_mode, cases=evidence)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--laya-source', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'calibration', 'test'], default='test')
    parser.add_argument('--canary', type=Path)
    parser.add_argument('--backend', choices=['eager', 'onnx'], default='eager')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audio-mode', choices=['full', 'early', 'none'], default='full')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.laya_source:
        sys.path.insert(0, str(args.laya_source.resolve()))
    if (args.checkpoint/'depth_encoder.json').is_file():
        from studio.core.utils.reply_modes.encoder_depth import EncoderDepthRouter as Router
    else:
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter as Router
    from evals.router_quality import HOLDOUT_CASES, evaluate as frozen_evaluate
    rows = [row for row in load_rows(args.data) if row['split'] == args.split]
    router = Router(args.checkpoint, device=args.device, backend=args.backend, allow_prototype=True)
    start = time.perf_counter()
    router.classify('普通的本地预热检查。')
    result = {'checkpoint': str(args.checkpoint), 'backend': args.backend, 'device': args.device,
              'cold_load_s': time.perf_counter()-start, 'split': args.split,
              'heldout': evaluate(router, rows, audio_mode=args.audio_mode),
              'frozen': frozen_evaluate(router, HOLDOUT_CASES)}
    if args.canary:
        canaries = [json.loads(line) for line in args.canary.read_text().splitlines() if line.strip()]
        result['canary'] = evaluate(router, canaries, audio_mode='none')
        result['canary_note'] = 'Private regression only; no original acoustic state; not independent natural-dialogue accuracy.'
    with args.out.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps({name: {key: value for key, value in item.items() if key != 'cases'}
                      if isinstance(item, dict) else item for name, item in result.items()},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
