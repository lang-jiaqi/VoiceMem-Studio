"""Compare an offline-trained depth head with Qwen on frozen synthetic holdouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

from evals.router_quality import HOLDOUT_CASES, evaluate
from studio.core.utils.reply_modes.component import QwenThinkingRouter
from studio.core.utils.reply_modes.decision import LocalDecisionRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backend', choices=['linear', 'laya'], default='linear')
    parser.add_argument('--laya-source', type=Path)
    parser.add_argument('--runtime', choices=['eager', 'onnx'], default='eager')
    parser.add_argument('--onnx-path', type=Path)
    parser.add_argument('--compare-qwen', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.backend == 'laya':
        if args.laya_source:
            sys.path.insert(0, str(args.laya_source.resolve()))
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter
        candidate = LayaDecisionRouter(args.checkpoint, device=args.device, allow_prototype=True,
                                       backend=args.runtime, onnx_path=args.onnx_path)
    else:
        candidate = LocalDecisionRouter(args.checkpoint, allow_prototype=True)
    start = time.perf_counter()
    candidate.classify('路由预热测试')
    report = {'platform': platform.platform(), 'device': args.device, 'backend': args.backend,
              'runtime': args.runtime,
              'synthetic_only': True, 'audio_evaluated': False,
              'candidate_cold_load_s': time.perf_counter()-start,
              'candidate': evaluate(candidate, HOLDOUT_CASES)}
    if args.compare_qwen:
        baseline = QwenThinkingRouter(model=str(args.compare_qwen), device=args.device)
        baseline.warmup()
        report['qwen'] = evaluate(baseline, HOLDOUT_CASES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({key: {name: value for name, value in item.items() if name != 'cases'}
                      if isinstance(item, dict) else item for key, item in report.items()},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
