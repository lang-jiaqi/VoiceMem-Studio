"""Prepare pinned X-ASR weights in an isolated startup-only subprocess."""
import sys
from pathlib import Path


def main():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    source, target = map(Path, sys.argv[1:])
    for part in ('encoder', 'decoder', 'joiner'):
        quantize_dynamic(
            str(source / f'{part}-480ms.onnx'),
            str(target / f'{part}-480ms.int8.onnx'),
            weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'],
            extra_options={'MatMulConstBOnly': True})


if __name__ == '__main__':
    main()
