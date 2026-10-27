"""Train with upstream Laya objectives and explicitly disjoint dialogue groups."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

from finetune.decision_router.train import load_rows
from studio.core.utils.reply_modes.decision_features import decision_state
from finetune.decision_router.dialogue_eval import summarize, record_predictions, validation_score


QUESTION = {
    'type': 'choice',
    'instructions': (
        'Judge only whether the current user request needs deep reasoning. Use history only '
        'to resolve references and unfinished work. Recall, casual talk, simple explanations '
        'and comparisons, quoting reasoning instructions, and requests not to think deeply '
        'are ordinary. Explicit requests to think deeply, rigorous proofs, multi-constraint '
        'tradeoffs and multi-cause diagnosis are deep. Do not decide memory retrieval or '
        'whether the speaker has finished.'),
    'criteria': {
        'ordinary': 'Ordinary reasoning, factual recall, simple explanation or casual conversation.',
        'deep': 'Explicit deep reasoning request or a genuinely complex reasoning task.',
    },
}


def typed_row(row, smoothing=0.0):
    return {'state': decision_state(row['text'], row.get('history'), row.get('audio_state')),
            'questions': {'depth': QUESTION},
            'gold': {'depth': {'probabilities': {
                'ordinary': (1-smoothing)*float(not row['needs_deep']) + smoothing/2,
                'deep': (1-smoothing)*float(row['needs_deep']) + smoothing/2}}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--laya-source', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--loss', choices=['soft-ce', 'rlcd'], default='soft-ce')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--ignore-audio', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--encoder-lr', type=float, default=2.5e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.label_smoothing < 1:
        parser.error('label-smoothing must be in [0, 1)')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('output directory must be empty')
    if not (args.laya_source / 'laya/train.py').is_file():
        parser.error('laya-source must point to the inspected upstream checkout')
    rows = load_rows(args.data)
    observed_audio_rows = sum(bool(row.get('audio_state')) for row in rows)
    observed_training_rows = sum(bool(row.get('audio_state')) and row['split'] == 'train' for row in rows)
    if args.ignore_audio:
        rows = [{key: value for key, value in row.items() if key != 'audio_state'} for row in rows]
    sys.path.insert(0, str(args.laya_source.resolve()))
    import torch
    from laya.train import (TrainConfig, load_checkpoint, items_from_rows, train_model,
                            calibration_records, evaluate_records, save_checkpoint, optimizer_updates)
    from laya.calibrate import fit_temperature_map
    from laya.common import uses_parallel_layout

    started = time.perf_counter()
    model, tokenizer, cfg = load_checkpoint(str(args.model))
    config = TrainConfig(loss=args.loss, epochs=args.epochs, micro_batch=args.micro_batch,
                         grad_accum=args.grad_accum, seed=42, shuffle_options=('choice',),
                         freeze_encoder=args.freeze_encoder, max_len=512, head_max_len=256,
                         encoder_lr=args.encoder_lr, head_lr=args.head_lr,
                         label_smoothing=args.label_smoothing, log_every=50)
    config.validate()
    device = torch.device(args.device)
    model.to(device)
    parallel = uses_parallel_layout(cfg)
    items = {}
    split_rows = {}
    splits = ['train', 'calibration', 'test']
    if any(row['split'] == 'validation' for row in rows):
        splits.append('validation')
    for split in splits:
        split_rows[split] = [row for row in rows if row['split'] == split]
        selected = [typed_row(row, args.label_smoothing if split == 'train' else 0.0)
                    for row in split_rows[split]]
        items[split], skipped = items_from_rows(tokenizer, selected, 512, 256)
        if skipped or len(items[split]) != len(selected):
            raise ValueError(f'{split}: upstream preprocessing skipped training examples: {skipped}')
    best, validation_history = {'score': None, 'state': None, 'epoch': None}, []

    def validate_epoch(epoch, loss):
        if 'validation' not in items:
            return
        values = calibration_records(model, tokenizer, items['validation'], device,
                                     512, 256, batch_size=args.micro_batch, parallel=parallel)
        measured = evaluate_records(values)
        semantic = summarize(split_rows['validation'], record_predictions(values))
        measured['semantic'] = semantic
        validation_history.append({'epoch': epoch+1, **measured})
        score = (*validation_score(semantic), -measured['loss'])
        if best['score'] is None or score > best['score']:
            best.update(score=score, epoch=epoch+1,
                        state={name: parameter.detach().cpu().clone()
                               for name, parameter in model.state_dict().items()})
        print('VOICE_VALIDATION=' + json.dumps(validation_history[-1]), flush=True)
        model.train()
        if config.freeze_encoder:
            model.encoder.eval()

    losses = train_model(model, tokenizer, items['train'], config, device, 512, 256,
                         parallel=parallel, on_epoch_end=validate_epoch)
    if best['state'] is not None:
        model.load_state_dict(best['state'], strict=True)
    # The upstream item-level random split is intentionally bypassed: all
    # paraphrases and snapshots of one dialogue stay in a single split.
    records = {split: calibration_records(model, tokenizer, items[split], device,
                                          512, 256, batch_size=args.micro_batch,
                                          parallel=parallel)
               for split in ('calibration', 'test')}
    calibration = fit_temperature_map(records['calibration'])
    temperature = calibration['temperature']
    buckets = calibration.get('temperature_by_options', {})
    report = {split: evaluate_records(values, temperature, buckets)
              for split, values in records.items()}
    for split, values in records.items():
        report[split]['semantic'] = summarize(split_rows[split], record_predictions(values))
    report.update(loss=args.loss, losses=losses, elapsed_s=time.perf_counter()-started,
                  observed_audio_rows=observed_audio_rows,
                  observed_audio_training_rows=observed_training_rows,
                  trained_audio_state=bool(observed_training_rows and not args.ignore_audio),
                  train_items=len(items['train']), source_revision=args.source_revision,
                  model_revision=args.model_revision, best_epoch=best['epoch'],
                  validation_history=validation_history)
    cfg.update(temperature=temperature, temperature_by_options=buckets,
               max_len=512, head_max_len=256)
    cfg['training'] = {'fine_tuned_from_checkpoint': True, 'epochs_completed': args.epochs,
                       'selected_epoch': best['epoch'] or args.epochs,
                       'updates': optimizer_updates(len(items['train']), args.micro_batch,
                                                    args.grad_accum, args.epochs),
                       'loss': args.loss, 'base_revision': args.model_revision}
    args.out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, tokenizer, cfg, str(args.out))
    metadata = {'version': 1, 'prototype': True, 'task': 'voice-dialogue-depth',
                'question': QUESTION, 'training': asdict(config), 'report': report,
                'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (args.out / 'voice_decision.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('VOICE_DECISION_RESULT=' + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
