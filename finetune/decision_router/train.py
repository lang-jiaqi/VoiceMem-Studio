"""Train and calibrate a decision head without modifying the shared encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from studio.core.utils.reply_modes.decision_features import VERSION, combine, texts, audio_vector


def load_rows(path, *, require_splits=True):
    """Reject ambiguous labels, duplicate cases and group leakage across splits."""
    rows, ids, groups, cases = [], set(), {}, {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {'id', 'group', 'split', 'text', 'needs_deep', 'source'}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f'Row {number}: missing decision fields')
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ('id', 'group', 'text', 'source')):
            raise ValueError(f'Row {number}: expected nonempty text fields')
        if row['split'] not in {'train', 'validation', 'calibration', 'test'} or type(row['needs_deep']) is not bool:
            raise ValueError(f'Row {number}: invalid split or label')
        history = row.get('history', [])
        if not isinstance(history, list) or any(
                not isinstance(item, dict) or item.get('role') not in {'user', 'assistant'}
                or not isinstance(item.get('content'), str) for item in history):
            raise ValueError(f'Row {number}: invalid history')
        acoustic = audio_vector(row.get('audio_state'))
        if row['id'] in ids:
            raise ValueError(f'Row {number}: duplicate id')
        ids.add(row['id'])
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError(f'Row {number}: group crosses data splits')
        groups[row['group']] = row['split']
        case = (*texts(row['text'], history), acoustic.tobytes())
        if case in cases and cases[case] != (row['group'], row['needs_deep']):
            raise ValueError(f'Row {number}: duplicate case has inconsistent group or label')
        cases[case] = (row['group'], row['needs_deep'])
        rows.append(row)
    for split in ('train', 'calibration', 'test'):
        if not require_splits and not any(row['split'] == split for row in rows):
            continue
        if {row['needs_deep'] for row in rows if row['split'] == split} != {False, True}:
            raise ValueError(f'{split} must contain both labels')
    validation = [row for row in rows if row['split'] == 'validation']
    if validation and {row['needs_deep'] for row in validation} != {False, True}:
        raise ValueError('validation must contain both labels')
    return rows


def encode_rows(rows, spec, *, batch_size=32):
    """Batch distinct semantic inputs using the same frozen runtime encoder."""
    from voicemem.leftbrain.local_embedder import resolve_path, shared_model
    encoder = shared_model(resolve_path(spec), spec.tokenizer_kwargs)
    pairs = [texts(row['text'], row.get('history')) for row in rows]
    unique = list(dict.fromkeys(value for pair in pairs for value in pair if value))
    vectors = {}
    for offset in range(0, len(unique), batch_size):
        batch = unique[offset:offset + batch_size]
        encoded = encoder.encode([spec.query_prefix + item for item in batch],
                                 normalize_embeddings=True, show_progress_bar=False,
                                 batch_size=batch_size)
        vectors.update(zip(batch, encoded))
    return np.stack([combine(vectors[current], vectors.get(context, np.zeros(spec.dims)),
                             row.get('audio_state'))
                     for row, (current, context) in zip(rows, pairs)]).astype(np.float64)


def probabilities(logits, temperature=1.0):
    return 1 / (1 + np.exp(-np.clip(logits / temperature, -60, 60)))


def fit(features, labels, *, regularization=.001):
    """Fit a balanced binary head; no generative model or teacher API is used."""
    from scipy.optimize import minimize
    labels = np.asarray(labels, dtype=np.float64)
    counts = np.bincount(labels.astype(int), minlength=2)
    if np.any(counts == 0):
        raise ValueError('Training requires both labels')
    sample_weight = len(labels) / (2 * counts[labels.astype(int)])

    def objective(theta):
        logits = features @ theta[:-1] + theta[-1]
        loss = np.mean(sample_weight * (np.logaddexp(0, logits) - labels * logits))
        loss += .5 * regularization * np.dot(theta[:-1], theta[:-1])
        residual = sample_weight * (probabilities(logits) - labels) / len(labels)
        gradient = np.r_[features.T @ residual + regularization * theta[:-1], residual.sum()]
        return loss, gradient

    result = minimize(objective, np.zeros(features.shape[1] + 1), jac=True,
                      method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-7})
    if not result.success:
        raise RuntimeError(f'Decision training did not converge: {result.message}')
    return result.x[:-1], float(result.x[-1])


def calibrate(logits, labels):
    """Fit temperature on a disjoint calibration split, never training/test rows."""
    from scipy.optimize import minimize_scalar

    def loss(log_temperature):
        values = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0, values) - labels * values))

    result = minimize_scalar(loss, bounds=(-4, 4), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature calibration failed')
    return float(np.exp(result.x))


def metrics(labels, scores, threshold=.5):
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    return {'n': len(labels), 'accuracy': float(np.mean(predicted == positive)),
            'deep_recall': float(np.mean(predicted[positive])),
            'false_deep_rate': float(np.mean(predicted[negative])),
            'brier': float(np.mean((scores-labels)**2))}


def train(path, output, *, encoder='e5', batch_size=32):
    """Export NumPy weights and reproducible metadata; checkpoints stay prototypes."""
    from voicemem.leftbrain.local_embedder import resolve
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Checkpoint output directory must be empty')
    rows = load_rows(path)
    spec = resolve(encoder, language='zh')
    started = time.perf_counter()
    features = encode_rows(rows, spec, batch_size=batch_size)
    labels = np.asarray([row['needs_deep'] for row in rows], dtype=np.float64)
    masks = {split: np.asarray([row['split'] == split for row in rows])
             for split in ('train', 'calibration', 'test')}
    weights, bias = fit(features[masks['train']], labels[masks['train']])
    logits = features @ weights + bias
    temperature = calibrate(logits[masks['calibration']], labels[masks['calibration']])
    scores = probabilities(logits, temperature)
    report = {split: metrics(labels[mask], scores[mask]) for split, mask in masks.items()}
    report.update(elapsed_s=time.perf_counter()-started,
                  observed_audio_rows=sum(bool(row.get('audio_state')) for row in rows),
                  sources=sorted({row['source'] for row in rows}))
    metadata = {'version': VERSION, 'prototype': True, 'encoder': spec.key,
                'encoder_repo': spec.repo, 'dimensions': spec.dims,
                'temperature': temperature, 'threshold': .5,
                'data_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'report': report}
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'head.npz', weights=weights, bias=np.asarray(bias))
    (output / 'decision.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--encoder', default='e5')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    print(json.dumps(train(args.data, args.out, encoder=args.encoder,
                           batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
