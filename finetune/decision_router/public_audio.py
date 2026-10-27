"""Fetch bounded public voice examples with source/conversation provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import time

DATASET = 'gpt-omni/VoiceAssistant-400K'
REVISION = '65dab707fcdd3d43dcfb834398aa9fed4116be3a'


def fetch(args):
    """Use the public row API; no user tokens or application credentials are sent."""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'wav').mkdir(exist_ok=True)
    selected, histories = [], {}
    for offset in range(args.offset, args.offset+args.count, 100):
        query = urllib.parse.urlencode({'dataset': DATASET, 'config': 'default',
            'split': 'train', 'offset': offset, 'length': min(100, args.offset+args.count-offset)})
        with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+query, timeout=60) as response:
            page = json.load(response)
        for entry in page['rows']:
            row = entry['row']
            group = f'public-{row["split_name"]}-{row["index"]}'
            history = histories.setdefault(group, [])
            question = str(row['question']).strip()
            sources = row.get('question_audio') or []
            if question and sources:
                selected.append({'id': f'public-row-{entry["row_idx"]}', 'group': group,
                    'text': question, 'history': list(history[-4:]),
                    'source': DATASET, 'source_revision': REVISION,
                    'audio_url': sources[0]['src']})
            history.extend(({'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': str(row.get('answer') or '')}))
    def download(row):
        url = row.pop('audio_url')
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != 'https' or not (parts.hostname == 'huggingface.co'
                                          or parts.hostname.endswith('.huggingface.co')):
            raise ValueError('Unexpected public dataset audio host')
        path = args.out / 'wav' / (row['id']+'.audio')
        if not path.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=20) as response:
                        audio = response.read(8*1024*1024+1)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(.5*(attempt+1))
            if not audio or len(audio) > 8*1024*1024:
                raise ValueError('Unexpected public audio size')
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(audio)
            temporary.replace(path)
        row['audio_path'] = str(path.relative_to(args.out))
        row['audio_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Split entire conversations, not adjacent audio turns.
        bucket = int(hashlib.sha256(row['group'].encode()).hexdigest()[:8], 16) % 10
        row['split'] = 'test' if bucket < 2 else 'calibration' if bucket < 4 else 'train'
        return row
    rows, failures = [], []
    with (args.out / 'unlabelled.jsonl').open('w', encoding='utf-8') as file, ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download, dict(row)): row['id'] for row in selected}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                failures.append({'id': futures[future], 'error': type(error).__name__})
                continue
            rows.append(row)
            file.write(json.dumps(row, ensure_ascii=False)+'\n')
            file.flush()
    (args.out / 'download-failures.json').write_text(json.dumps(failures, indent=2))
    print(json.dumps({'public_audio_rows': len(rows), 'conversations': len(histories),
                      'failures': len(failures), 'dataset': DATASET, 'revision': REVISION}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--offset', type=int, default=350000)
    parser.add_argument('--count', type=int, default=160)
    fetch(parser.parse_args())


if __name__ == '__main__':
    main()
