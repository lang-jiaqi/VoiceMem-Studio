"""Export a calibrated depth encoder to portable CPU ONNX with dynamic checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from studio.core.utils.reply_modes.depth_network import DepthEncoder, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('export output directory must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.checkpoint/'depth_encoder.json').read_text())
    config = AutoConfig.from_pretrained(args.checkpoint/'encoder', local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint/'tokenizer', local_files_only=True,
                                             trust_remote_code=False)
    model = DepthEncoder(config).eval()
    model.load_state_dict(load_file(str(args.checkpoint/'model.safetensors')), strict=True)
    torch.set_num_threads(2)

    def inputs(count, length):
        sequences, observations = [], []
        for index in range(count):
            current, recent, vector = features('用于图验证的合成输入。'*length,
                [{'role': 'user', 'content': '先前的合成上下文。'*length}],
                {'eot_probability': .8, 'silence_s': .25, 'span_s': 3})
            sequences.extend([current, recent])
            observations.append(vector)
        encoded = tokenizer(sequences, truncation=True, max_length=metadata['max_length'],
                            padding=True, return_tensors='pt')
        return (encoded['input_ids'], encoded['attention_mask'], torch.from_numpy(np.stack(observations)))

    graph = args.out/'model.onnx'
    torch.onnx.export(model, inputs(1, 1), str(graph), opset_version=17, dynamo=False,
                      input_names=['input_ids', 'attention_mask', 'observed'], output_names=['logits'],
                      dynamic_axes={'input_ids': {0: 'sequences', 1: 'tokens'},
                                    'attention_mask': {0: 'sequences', 1: 'tokens'},
                                    'observed': {0: 'cases'}, 'logits': {0: 'cases'}})
    import onnx
    import onnxruntime as ort
    from voicemem.engine import cpu_session_options
    onnx.checker.check_model(str(graph))
    session = ort.InferenceSession(str(graph), sess_options=cpu_session_options(ort),
                                   providers=['CPUExecutionProvider'])
    checks = []
    for count, length in ((1, 1), (2, 5), (3, 32)):
        values = inputs(count, length)
        with torch.inference_mode():
            expected = model(*values).numpy()
        actual = session.run(['logits'], {name: value.numpy() for name, value in
                                          zip(('input_ids', 'attention_mask', 'observed'), values)})[0]
        error = float(np.max(np.abs(expected-actual)))
        if error > .0002 or not np.array_equal(expected.argmax(1), actual.argmax(1)):
            raise ValueError(f'ONNX parity failed: {count} cases, error={error}')
        checks.append({'cases': count, 'tokens': int(values[0].shape[1]), 'max_abs_error': error})
    for name in ('encoder', 'tokenizer'):
        shutil.copytree(args.checkpoint/name, args.out/name)
    metadata['deployment'] = {'backend': 'onnx', 'precision': 'fp32'}
    (args.out/'depth_encoder.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    report = {'graph_bytes': graph.stat().st_size, 'dynamic_parity': checks,
              'calibration': 'Preserves source temperature; behavior qualification still required.'}
    (args.out/'export-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
