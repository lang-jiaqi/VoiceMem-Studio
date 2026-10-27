"""Assemble text and observed-speech rows with disjoint validation/calibration."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .train import load_rows
from studio.core.utils.reply_modes.decision_features import texts, audio_vector


def split_for(group, weak=False):
    if weak:
        return 'train'
    bucket = int(hashlib.sha256(('voice-depth-v2:'+group).encode()).hexdigest()[:8], 16) % 10
    return 'test' if bucket < 2 else 'calibration' if bucket == 2 else 'validation' if bucket == 3 else 'train'


def assemble(paths, output):
    rows, ids, cases, ambiguous = [], {}, {}, set()
    counts = Counter()
    for path in paths:
        for raw in Path(path).read_text(encoding='utf-8').splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row['id'] in ids:
                # Resumed preparation may repeat metadata, but never silently
                # accept two observations/labels under one case identity.
                previous = ids[row['id']]
                keys = ('text', 'history', 'needs_deep', 'audio_state', 'audio_sha256')
                if any(previous.get(key) != row.get(key) for key in keys):
                    raise ValueError(f'Conflicting repeated case: {row["id"]}')
                counts['repeated_ids'] += 1
                continue
            ids[row['id']] = row
            state = audio_vector(row.get('audio_state'))
            key = (*texts(row['text'], row.get('history')), state.tobytes())
            if key in cases:
                previous = cases[key]
                if previous['group'] != row['group'] or previous['needs_deep'] != row['needs_deep']:
                    counts['ambiguous_duplicate_inputs'] += 1
                    ambiguous.add(key)
                    continue
                counts['duplicate_inputs'] += 1
                continue
            weak = row.get('label_provenance') == 'local-qwen-and-laya-agreement'
            row['split'] = split_for(row['group'], weak)
            cases[key] = row
            rows.append(row)
    rows = [row for row in rows if (*texts(row['text'], row.get('history')),
                                    audio_vector(row.get('audio_state')).tobytes()) not in ambiguous]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False)+'\n')
    load_rows(output)
    report = {'rows': len(rows), 'observed_audio_rows': sum(bool(row.get('audio_state')) for row in rows),
              'weak_label_rows': sum(row.get('label_provenance') == 'local-qwen-and-laya-agreement' for row in rows),
              'splits': {split: {'rows': sum(row['split'] == split for row in rows),
                                 'groups': len({row['group'] for row in rows if row['split'] == split}),
                                 'deep': sum(row['split'] == split and row['needs_deep'] for row in rows)}
                         for split in ('train', 'validation', 'calibration', 'test')},
              'deduplication': dict(counts)}
    output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.data, args.out), indent=2))


if __name__ == '__main__':
    main()
