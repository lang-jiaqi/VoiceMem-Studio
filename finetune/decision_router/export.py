"""Export a voice decision checkpoint into bounded CPU deployment bundles."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def half_gather_weights(source, destination):
    """Store large Gather-only tables in fp16, casting gathered rows back to fp32.

    Laya checkpoints already store these parameters as fp16. This preserves
    their values while avoiding an expanded fp32 vocabulary table in memory.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper, helper, TensorProto
    model = onnx.load(str(source))
    changed, extra = [], []
    for tensor in model.graph.initializer:
        uses = [(node, index) for node in model.graph.node
                for index, name in enumerate(node.input) if name == tensor.name]
        if (tensor.data_type != TensorProto.FLOAT or len(tensor.dims) != 2
                or tensor.dims[0] < 1000 or not uses
                or any(node.op_type != 'Gather' or index != 0 for node, index in uses)):
            continue
        values = numpy_helper.to_array(tensor)
        half = values.astype(np.float16)
        if not np.array_equal(values, half.astype(np.float32)):
            continue
        name = tensor.name
        tensor.CopyFrom(numpy_helper.from_array(half, name=name))
        for node, _ in uses:
            original = node.output[0]
            node.output[0] = original + '_stored_fp16'
            extra.append((node, helper.make_node('Cast', [node.output[0]], [original],
                                                name=node.name+'_to_fp32', to=TensorProto.FLOAT)))
        changed.append(name)
    nodes = []
    casts = {node.output[0]: cast for node, cast in extra}
    for node in model.graph.node:
        nodes.append(copy.deepcopy(node))
        if node.output[0] in casts:
            nodes.append(casts[node.output[0]])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return changed


def graph_records(graph, tokenizer, rows, question, max_len, head_max_len):
    """Collect raw graph logits on calibration rows, before applying any scaling."""
    import onnxruntime as ort
    from laya.train import items_from_rows, encode_item
    from laya.common import collate_items
    from voicemem.engine import cpu_session_options
    from studio.core.utils.reply_modes.decision_features import decision_state
    selected = [{'state': decision_state(row['text'], row.get('history'), row.get('audio_state')),
                 'questions': {'depth': question},
                 'gold': {'depth': {'probabilities': {'ordinary': float(not row['needs_deep']),
                                                      'deep': float(row['needs_deep'])}}}}
                for row in rows]
    items, skipped = items_from_rows(tokenizer, selected, max_len, head_max_len)
    if skipped:
        raise ValueError(f'Graph calibration skipped rows: {skipped}')
    session = ort.InferenceSession(str(graph), sess_options=cpu_session_options(ort),
                                   providers=['CPUExecutionProvider'])
    records = []
    for offset in range(0, len(items), 8):
        chunk = items[offset:offset+8]
        encoded = [encode_item(tokenizer, item, max_len, head_max_len) for item in chunk]
        batch = collate_items([encoded], tokenizer.pad_token_id)
        feeds = {item.name: batch[item.name].numpy() for item in session.get_inputs()}
        logits = session.run(['logits'], feeds)[0]
        for values, item in zip(logits, chunk):
            records.append((item['qtype'], values[:item['k']], item['target'], item['k']))
    return records


def quantize_encoder(source, destination):
    """Compress only encoder linears; preserve the typed decision head in fp32."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model = onnx.load(str(source))
    nodes = [node.name for node in model.graph.node
             if node.op_type == 'MatMul' and node.name.startswith('/encoder/')]
    if not nodes:
        raise ValueError('Export contains no named encoder MatMul nodes')
    del model.graph.value_info[:]
    quantize_dynamic(model, str(destination), nodes_to_quantize=nodes,
                     op_types_to_quantize=['MatMul'], weight_type=QuantType.QInt8,
                     per_channel=False, extra_options={'MatMulConstBOnly': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--laya-source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--precisions', nargs='+', choices=['mixed', 'int8', 'encoder-int8'],
                        default=['mixed', 'int8', 'encoder-int8'])
    args = parser.parse_args()
    if args.precisions[0] != 'mixed' or len(set(args.precisions)) != len(args.precisions):
        parser.error('precisions must start with mixed and contain no duplicates')
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error('export output directory must be empty')
    sys.path.insert(0, str(args.laya_source.resolve()))
    from transformers import AutoTokenizer
    from laya.calibrate import fit_temperature_map
    from laya.train import evaluate_records
    from finetune.decision_router.train import load_rows
    spec = importlib.util.spec_from_file_location('upstream_laya_export', args.laya_source/'scripts/export_onnx.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.checkpoint/'voice_decision.json').read_text())
    if args.resume and (args.out/'mixed/voice_decision.json').exists():
        existing = json.loads((args.out/'mixed/voice_decision.json').read_text())
        existing.pop('deployment', None)
        if existing != metadata:
            raise ValueError('Resumed export belongs to a different checkpoint')
    original = args.out/'original-fp32.onnx'
    if not args.resume or not original.exists():
        exporter.export_to_onnx(str(args.checkpoint), str(original), verify=True)
    cfg = json.loads((args.checkpoint/'rl_agent_config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint/'tokenizer', local_files_only=True)
    all_rows = load_rows(args.data)
    calibration_rows = [row for row in all_rows if row['split'] == 'calibration']
    validation_rows = [row for row in all_rows if row['split'] == 'validation']
    if not metadata['report'].get('trained_audio_state'):
        calibration_rows = [{key: value for key, value in row.items() if key != 'audio_state'}
                            for row in calibration_rows]
        validation_rows = [{key: value for key, value in row.items() if key != 'audio_state'}
                           for row in validation_rows]
    report = {}
    changed = []
    for precision in args.precisions:
        output = args.out/precision
        output.mkdir(exist_ok=args.resume)
        if args.resume and (output/'evaluation.json').exists():
            report[precision] = json.loads((output/'evaluation.json').read_text())
            continue
        for name in ('tokenizer', 'encoder'):
            if not (output/name).exists():
                shutil.copytree(args.checkpoint/name, output/name)
        graph = output/'laya.onnx'
        if precision == 'mixed':
            if not args.resume or not graph.exists():
                changed = half_gather_weights(original, graph)
        elif precision == 'int8':
            exporter.quantize_model(str(args.out/'mixed/laya.onnx'), str(graph), per_channel=False)
        else:
            quantize_encoder(args.out/'mixed/laya.onnx', graph)
        records = graph_records(graph, tokenizer, calibration_rows, metadata['question'],
                                cfg['max_len'], cfg['head_max_len'])
        fitted = fit_temperature_map(records)
        deployed_cfg = {**cfg, 'temperature': fitted['temperature'],
                        'temperature_by_options': fitted.get('temperature_by_options', {})}
        (output/'rl_agent_config.json').write_text(json.dumps(deployed_cfg, indent=2))
        deployed_metadata = {**metadata, 'deployment': {'backend': 'onnx', 'precision': precision}}
        (output/'voice_decision.json').write_text(json.dumps(deployed_metadata, ensure_ascii=False, indent=2))
        report[precision] = {'graph_bytes': graph.stat().st_size,
                             'calibration': evaluate_records(records, deployed_cfg['temperature'],
                                                              deployed_cfg['temperature_by_options'])}
        if validation_rows:
            valid = graph_records(graph, tokenizer, validation_rows, metadata['question'],
                                  cfg['max_len'], cfg['head_max_len'])
            report[precision]['validation'] = evaluate_records(
                valid, deployed_cfg['temperature'], deployed_cfg['temperature_by_options'])
        print('VOICE_EXPORT=' + json.dumps({precision: report[precision]}), flush=True)
        (output/'evaluation.json').write_text(json.dumps(report[precision], indent=2))
    if not changed:
        import onnx
        restored = onnx.load(str(args.out/'mixed/laya.onnx'))
        changed = [tensor.name for tensor in restored.graph.initializer
                   if tensor.data_type == onnx.TensorProto.FLOAT16
                   and len(tensor.dims) == 2 and tensor.dims[0] >= 1000]
    report['fp16_gather_tables'] = changed
    recommended = 'mixed'
    if (validation_rows and 'encoder-int8' in report and report['encoder-int8']['validation']['accuracy']
            >= report['mixed']['validation']['accuracy'] - .005):
        recommended = 'encoder-int8'
    report['validation_preferred'] = recommended
    (args.out/'export-report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
