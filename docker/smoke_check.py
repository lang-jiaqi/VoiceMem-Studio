"""Validate image imports and packaged assets without GPU, credentials or models."""
from importlib import import_module, metadata
import os
from pathlib import Path
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from studio.core.utils.tts.cuda import source_directory
    from studio.core.utils.cli.component import parse_args
    from studio.core.utils.tts.segmentation import SpeechBuffer
    from voicemem.prompt_config import tts_prompts

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('The Studio image requires Python 3.12')
    if os.geteuid() == 0:
        raise RuntimeError('The image smoke check must run as the service user')
    for path in (root / 'studio/models', root / 'voicemem_memoryspace', root / 'results',
                 root / 'prompt/logs', root / 'cache', Path.home() / '.cache'):
        with tempfile.TemporaryFile(dir=path):
            pass
    args = parse_args([])
    if (args.backend, args.device, args.tts_device) != ('cuda', 'cuda:0', 'cuda:0'):
        raise RuntimeError('The CUDA image must default to GPU 0')
    for name in ('torch', 'torchaudio', 'torchvision', 'funasr', 'modelscope',
                 'sherpa_onnx', 'onnxruntime', 'soundfile', 'sounddevice',
                 'sentence_transformers', 'qwen_tts', 'fastapi', 'uvicorn'):
        import_module(name)
    import torch
    expected = {'torch': '2.8.0', 'torchaudio': '2.8.0', 'torchvision': '0.23.0'}
    for name, required in expected.items():
        installed = metadata.version(name)
        if installed.split('+', 1)[0] != required:
            raise RuntimeError(f'Expected {name} {required}, got {installed}')
    if torch.version.cuda != '12.8':
        raise RuntimeError(f'Expected CUDA 12.8 PyTorch wheels, got {torch.version.cuda}')
    if metadata.version('transformers') != '4.57.3':
        raise RuntimeError('Breeze CUDA requires Transformers 4.57.3')
    source = source_directory()
    sys.path.insert(0, str(source))
    from breeze_infer.runtime import load_runtime
    from models.fast_streaming import FastBreezeStreamingRuntime
    from models.warmup_profile import load_warmup_profile
    from studio.web.transport import build_app
    from studio.core.voiceagent import VoiceAgent
    assert all(callable(item) for item in (load_runtime, FastBreezeStreamingRuntime, build_app, VoiceAgent))
    load_warmup_profile(source / 'configs/fast.json')
    tts_prompts()
    for relative in ('voice/noctelle_ref_short.wav', 'voice/noctelle_ref_short.txt',
                     'studio/web/voicemem.html', 'studio/web/index.html',
                     'studio/web/images/background.webp', 'studio/web/pcm-player-worklet.js',
                     'studio/web/mic-capture-worklet.js'):
        path = root / relative
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'Missing runtime asset: {relative}')
    if not list((root / 'voice/backchannel').glob('*.wav')):
        raise RuntimeError('Missing reviewed backchannel clips')
    buffer = SpeechBuffer()
    buffer.append('你好，我们完整地说完这句话。', 0.0)
    if len(buffer.ready(0.0)) != 1:
        raise RuntimeError('Sentence-first segmentation smoke test failed')
    print('[image] CUDA imports, assets and sentence segmentation verified; no models loaded')


if __name__ == '__main__':
    main()
