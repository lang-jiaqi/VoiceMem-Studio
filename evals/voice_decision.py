"""Evaluate routing on observed ASR inputs with paired acoustic state."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
import platform


def memory():
    """Return process memory evidence with explicit platform units."""
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(peak if sys.platform == 'darwin' else peak*1024)
    current = None
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                current = int(line.split()[1])*1024
    return {'rss_bytes': current, 'peak_rss_bytes': peak}

from finetune.decision_router.train import load_rows


def evaluate(router, rows, audio):
    cases, groups = [], defaultdict(list)
    for row in rows:
        router._cache.clear()
        started = time.perf_counter()
        kwargs = {'history': row.get('history')}
        if audio and router.accepts_audio_state:
            kwargs['audio_state'] = row.get('audio_state')
        decision = router.classify(row['text'], **kwargs)
        elapsed = (time.perf_counter()-started)*1000
        predicted = decision.level == 'slow'
        correct = predicted == row['needs_deep']
        groups[row['group']].append(correct)
        cases.append({'id': row['id'], 'group': row['group'], 'expected': row['needs_deep'],
                      'predicted': predicted, 'correct': correct, 'ms': elapsed,
                      'raw': decision.raw, 'asr': row.get('asr')})
    ordinary = [row for row in cases if not row['expected']]
    deep = [row for row in cases if row['expected']]
    times = sorted(row['ms'] for row in cases)
    result = {'n': len(cases), 'groups': len(groups),
              'accuracy': statistics.mean(row['correct'] for row in cases),
              'macro_group_accuracy': statistics.mean(statistics.mean(values) for values in groups.values()),
              'false_deep_rate': statistics.mean(row['predicted'] for row in ordinary) if ordinary else None,
              'deep_recall': statistics.mean(row['predicted'] for row in deep) if deep else None,
              'median_ms': statistics.median(times), 'p95_ms': times[int(.95*(len(times)-1))],
              'by_asr': {}, 'cases': cases}
    for name in sorted({row.get('asr') or 'text' for row in rows}):
        selected = [row for row in cases if (row['asr'] or 'text') == name]
        result['by_asr'][name] = {'n': len(selected), 'accuracy': statistics.mean(row['correct'] for row in selected)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--laya-source', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'calibration', 'test'], default='test')
    parser.add_argument('--compare-qwen', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--backend', choices=['eager', 'onnx'], default='eager')
    parser.add_argument('--onnx-path', type=Path)
    parser.add_argument('--audio-mode', choices=['full', 'early', 'none'], default='full')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.laya_source:
        sys.path.insert(0, str(args.laya_source.resolve()))
    if (args.checkpoint/'depth_encoder.json').is_file():
        from studio.core.utils.reply_modes.encoder_depth import EncoderDepthRouter as Router
    else:
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter as Router
    from studio.core.utils.reply_modes.component import QwenThinkingRouter
    rows = [row for row in load_rows(args.data, require_splits=False)
            if row['split'] == args.split and row.get('audio_state')]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error('no observed-audio evaluation rows')
    if args.audio_mode == 'early':
        rows = [{**row, 'audio_state': {key: value for key, value in row['audio_state'].items()
                                      if key != 'span_s'}} for row in rows]
    elif args.audio_mode == 'none':
        rows = [{**row, 'audio_state': None} for row in rows]
    before = memory()
    options = {'device': args.device, 'allow_prototype': True, 'backend': args.backend}
    if args.onnx_path:
        if (args.checkpoint/'depth_encoder.json').is_file():
            parser.error('Task-specific encoder bundles use their own model.onnx')
        options['onnx_path'] = args.onnx_path
    candidate = Router(args.checkpoint, **options)
    started = time.perf_counter()
    candidate.classify('路由预热测试')
    report = {'split': args.split, 'device': args.device, 'backend': args.backend,
              'audio_mode': args.audio_mode, 'observed_audio': True,
              'platform': platform.platform(), 'cold_load_s': time.perf_counter()-started,
              'memory_before': before, 'memory_loaded': memory(),
              'candidate': evaluate(candidate, rows, True)}
    report['memory_after_candidate'] = memory()
    if args.compare_qwen:
        reference = QwenThinkingRouter(str(args.compare_qwen), args.device)
        reference.warmup()
        report['qwen'] = evaluate(reference, rows, False)
    with args.out.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({key: {name: value for name, value in item.items() if name != 'cases'}
                      if isinstance(item, dict) else item for key, item in report.items()}, indent=2))


if __name__ == '__main__':
    main()
