"""Preserve authored wording splits and reuse only supervised legacy training rows."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .dialogue_data import audit
from .train import load_rows
from studio.core.utils.reply_modes.decision_features import texts, audio_vector


def assemble(authored, legacy, speech, output, *, excluded_ids=()):
    """Keep test/calibration frozen and explicitly discard consensus weak labels."""
    rows = [json.loads(line) for line in Path(authored).read_text().splitlines() if line.strip()]
    counts = Counter()
    for line in Path(legacy).read_text().splitlines():
        row = json.loads(line)
        if row.get('label_provenance') == 'local-qwen-and-laya-agreement':
            counts['excluded_weak'] += 1
            continue
        if row['split'] != 'train':
            counts['legacy_frozen_not_reused'] += 1
            continue
        if row['id'] in excluded_ids:
            counts['quarantined_legacy_views'] += 1
            continue
        rows.append({**row, 'category': 'legacy_deep' if row['needs_deep'] else 'legacy_ordinary'})
        counts['legacy_training'] += 1
    source_families = {row['group'] for row in rows}
    for path in speech:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row['id'] in excluded_ids:
                counts['quarantined_audio_views'] += 1
                continue
            if row['group'] not in source_families:
                raise ValueError('Observed speech has no authored source family')
            rows.append(row)
            counts['observed_new'] += 1
    unique, ids, cases = [], set(), {}
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate input identity')
        ids.add(row['id'])
        key = (*texts(row['text'], row.get('history')), audio_vector(row.get('audio_state')).tobytes())
        if key in cases:
            previous = cases[key]
            if previous['split'] != row['split'] or previous['needs_deep'] != row['needs_deep']:
                raise ValueError('Conflicting input or cross-partition duplicate')
            counts['deduplicated'] += 1
            continue
        cases[key] = row
        unique.append(row)
    report = audit(unique)
    report.update(counts=counts, observed_audio_rows=sum(bool(row.get('audio_state')) for row in unique),
                  sources={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for path in [authored, legacy, *speech]})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as dest:
        for row in unique:
            dest.write(json.dumps(row, ensure_ascii=False)+'\n')
    load_rows(output)
    report['data_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authored', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--speech', type=Path, nargs='*', default=[])
    parser.add_argument('--exclude', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    excluded = json.loads(args.exclude.read_text()) if args.exclude else []
    print(json.dumps(assemble(args.authored, args.legacy, args.speech, args.out,
                              excluded_ids=excluded), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
