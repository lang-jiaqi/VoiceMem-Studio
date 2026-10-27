"""Fine-tune a task-specific encoder while isolating current text from history."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from .train import load_rows, calibrate, probabilities, metrics
from .dialogue_eval import summarize, validation_score
from studio.core.utils.reply_modes.decision_features import VERSION
from studio.core.utils.reply_modes.depth_network import DepthEncoder, features


def prepare(rows, tokenizer, max_length):
    cache, result = {}, []
    for row in rows:
        current, recent, vector = features(row['text'], row.get('history'), row.get('audio_state'))
        tokens = []
        for text in (current, recent):
            if text not in cache:
                cache[text] = tokenizer(text, truncation=True, max_length=max_length)['input_ids']
            tokens.append(cache[text])
        result.append((tokens, vector, int(row['needs_deep'])))
    return result


def batch(items, tokenizer, device):
    encoded = tokenizer.pad({'input_ids': [tokens for item in items for tokens in item[0]]},
                            padding=True, return_tensors='pt')
    return {'input_ids': encoded['input_ids'].to(device),
            'attention_mask': encoded['attention_mask'].to(device),
            'observed': torch.from_numpy(np.stack([item[1] for item in items])).to(device)}, \
           torch.tensor([item[2] for item in items], dtype=torch.long, device=device)


@torch.inference_mode()
def evaluate(model, items, rows, tokenizer, device):
    model.eval()
    values = []
    for start in range(0, len(items), 24):
        inputs, _ = batch(items[start:start+24], tokenizer, device)
        values.append(model(**inputs).float().cpu().numpy())
    logits = np.concatenate(values)
    return logits, summarize(rows, logits.argmax(axis=1).astype(bool).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--micro-batch', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=.04)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--init-checkpoint', type=Path)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('output directory must be empty')
    if min(args.epochs, args.micro_batch, args.grad_accum) < 1 or not 0 <= args.label_smoothing < 1:
        parser.error('invalid training budget or smoothing')
    torch.manual_seed(42)
    started = time.perf_counter()
    all_rows = load_rows(args.data)
    rows = {split: [row for row in all_rows if row['split'] == split]
            for split in ('train', 'validation', 'calibration', 'test')}
    if not rows['validation']:
        parser.error('independent validation is required')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = DepthEncoder(config, pretrained=args.model)
    if args.init_checkpoint:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(args.init_checkpoint/'model.safetensors')), strict=True)
    device = torch.device(args.device)
    model.to(device)
    if device.type == 'cuda':
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    items = {split: prepare(selected, tokenizer, args.max_length) for split, selected in rows.items()}
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': args.encoder_lr},
                                   {'params': model.classifier.parameters(), 'lr': args.head_lr}], weight_decay=.01)
    micro_steps = math.ceil(len(items['train'])/args.micro_batch)
    total_steps = math.ceil(micro_steps/args.grad_accum)*args.epochs
    warmup = max(1, int(.1*total_steps))

    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        return .05 + .95*.5*(1+math.cos(math.pi*(step-warmup)/max(1, total_steps-warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    loss_function = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best, history = {'score': None, 'state': None, 'epoch': None}, []
    for epoch in range(args.epochs):
        model.train()
        order = list(items['train'])
        random.Random(42+epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for index, start in enumerate(range(0, len(order), args.micro_batch)):
            inputs, targets = batch(order[start:start+args.micro_batch], tokenizer, device)
            window = min(args.grad_accum, micro_steps-(index//args.grad_accum)*args.grad_accum)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                logits = model(**inputs)
                loss = loss_function(logits.float(), targets)
            (loss/window).backward()
            total_loss += float(loss.detach())
            if (index+1) % args.grad_accum == 0 or index+1 == micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (index+1) % 100 == 0:
                print(f'epoch {epoch+1}/{args.epochs} step {index+1} loss {loss.item():.4f}', flush=True)
        _, measured = evaluate(model, items['validation'], rows['validation'], tokenizer, device)
        score = validation_score(measured)
        history.append({'epoch': epoch+1, 'mean_loss': total_loss/micro_steps, 'semantic': measured})
        if best['score'] is None or score > best['score']:
            best.update(score=score, epoch=epoch+1,
                        state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
        print('ENCODER_VALIDATION='+json.dumps(history[-1]), flush=True)
    model.load_state_dict(best['state'], strict=True)
    calibration_logits, calibration_metrics = evaluate(model, items['calibration'], rows['calibration'], tokenizer, device)
    labels = np.asarray([row['needs_deep'] for row in rows['calibration']], np.float64)
    temperature = calibrate(calibration_logits[:, 1]-calibration_logits[:, 0], labels)
    _, test = evaluate(model, items['test'], rows['test'], tokenizer, device)
    report = {'best_epoch': best['epoch'], 'validation_history': history, 'calibration': calibration_metrics,
              'calibrated': metrics(labels, probabilities(calibration_logits[:, 1]-calibration_logits[:, 0], temperature)),
              'test': test, 'elapsed_s': time.perf_counter()-started,
              'observed_audio_training_rows': sum(bool(row.get('audio_state')) for row in rows['train']),
              'trained_audio_state': any(row.get('audio_state') for row in rows['train']),
              'parameter_count': sum(parameter.numel() for parameter in model.parameters())}
    args.out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
              str(args.out/'model.safetensors'))
    tokenizer.save_pretrained(args.out/'tokenizer')
    config.save_pretrained(args.out/'encoder')
    metadata = {'version': 1, 'prototype': True, 'task': 'voice-depth-encoder', 'features_version': VERSION,
                'labels': ['ordinary', 'deep'], 'max_length': args.max_length, 'temperature': temperature,
                'model_revision': args.model_revision, 'base': str(args.model),
                'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
                'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'training': {'epochs': args.epochs, 'micro_batch': args.micro_batch, 'grad_accum': args.grad_accum,
                             'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr,
                             'label_smoothing': args.label_smoothing, 'seed': 42, 'freeze_encoder': False},
                'report': report}
    (args.out/'depth_encoder.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('ENCODER_RESULT='+json.dumps({key: value for key, value in report.items() if key != 'validation_history'}), flush=True)


if __name__ == '__main__':
    main()
