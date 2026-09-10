"""Required model artifacts for the unchanged Studio inference profile."""
from .component import Model

BUNDLE = 'zhifeixie/VoiceMem_Default_Models_Env'
HF = ('config.json', '*.safetensors')


def models(args):
    """List every model needed by the selected voice and memory path."""
    shared = [
        Model('E5 记忆向量', 'embedding', 'intfloat/multilingual-e5-small',
              HF + ('tokenizer.json',), ('*.json', '*.safetensors', '*.model', '1_Pooling/*')),
        Model('Silero VAD', 'vad', BUNDLE, ('silero_vad.onnx',), ('vad/*',)),
        Model('声纹', 'speaker', BUNDLE,
              ('3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',), ('speaker/*',)),
        Model('AST 环境感知', 'scene', 'MIT/ast-finetuned-audioset-10-10-0.4593',
              HF + ('preprocessor_config.json',), ('*.json', '*.safetensors')),
        Model('SenseVoice 感知', 'emotion', 'FunAudioLLM/SenseVoiceSmall',
              ('configuration.json', 'config.yaml', 'model.pt', 'am.mvn',
               'chn_jpn_yue_eng_ko_spectok.bpe.model'),
              ('*.json', '*.yaml', '*.pt', '*.mvn', '*.model', '*.txt')),
        Model('SenseVoice 最终转写', 'asr/sensevoice-int8',
              'csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17',
              ('model.int8.onnx', 'tokens.txt'), ('model.int8.onnx', 'tokens.txt'),
              ('asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17',
               'asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17')),
        Model('emotion2vec', 'emotion2vec', 'emotion2vec/emotion2vec_plus_base',
              ('config.yaml', 'configuration.json', 'model.pt', 'tokens.txt'),
              ('*.json', '*.yaml', '*.pt', '*.txt', '*.npy')),
    ]
    if args.lang == 'zh':
        shared.append(Model('流式 ASR', 'asr/funasr-paraformer-zh-streaming',
                            'funasr/paraformer-zh-streaming',
                            ('config.yaml', 'model.pt', 'tokens.json', 'am.mvn', 'seg_dict')))
    else:
        shared.append(Model('流式 ASR', 'asr/sherpa-onnx-streaming-zipformer-en-2023-06-26',
                            'csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26',
                            ('encoder*.onnx', 'decoder*.onnx', 'joiner*.onnx', 'tokens.txt'),
                            ('*.onnx', 'tokens.txt', 'bpe.model')))
    if args.eot:
        shared.append(Model('Smart Turn EOT', 'eot', 'pipecat-ai/smart-turn-v3',
                            ('smart-turn-v3.2-cpu.onnx',), ('smart-turn-v3.2-cpu.onnx',)))
    if args.mode == 'llm_tts':
        shared.extend([
            Model('三级回复路由', 'reply-router/Qwen3-0.6B', 'Qwen/Qwen3-0.6B',
                  HF + ('tokenizer.json',), ('*.json', '*.safetensors', '*.txt', '*.jinja')),
        ])
        if args.backend == 'cuda':
            from studio.core.utils.tts.cuda import model_directory
            from studio.paths import ROOT
            shared.append(Model('Breeze CUDA', str(model_directory()), 'BreezeBlue/breeze-tts-2',
                                ('config.json', 'model.safetensors.index.json', '*.safetensors',
                                 'tokenizer.json', 'audio_tokenizer/config.json',
                                 'audio_tokenizer/*.safetensors'),
                                legacy=(str(ROOT.parent / 'breeze-tts-2'), 'tts/breeze-tts-2')))
        else:
            shared.append(Model('Breeze MLX', 'tts/Breeze-TTS-2-mlx-4bit',
                                'mlx-community/Breeze-TTS-2-mlx-4bit', HF + ('tokenizer.json',)))
        if args.llm == 'local':
            shared.append(Model('本地回复', 'llm/Qwen3.5-4B-4bit', 'mlx-community/Qwen3.5-4B-4bit',
                                HF + ('tokenizer.json',)))
    return shared


def acquire_all(args):
    from studio.paths import MODELS, ROOT
    for model in models(args):
        # Bundled ONNX artifacts retain their category prefix in the upstream repo.
        if model.repository == BUNDLE:
            if model.ready(MODELS):
                continue
            legacy = ROOT / 'models' / model.directory
            target = MODELS / model.directory
            if model.ready(ROOT / 'models') and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(legacy.resolve(), target_is_directory=True)
                print(f'[model] 复用已有权重：{model.name}', flush=True)
                continue
            print(f'[model] 缺少 {model.name}；开始下载 → {target}', flush=True)
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=BUNDLE, local_dir=str(MODELS), allow_patterns=list(model.patterns))
            except Exception as exc:
                raise RuntimeError(f'{model.name} 下载失败（{type(exc).__name__}）；检查网络后重试') from None
            if not model.ready(MODELS):
                raise RuntimeError(f'{model.name} 下载后仍不完整：{target}；' + '；'.join(model.problems(MODELS)))
        else:
            model.acquire(MODELS, ROOT / 'models')
