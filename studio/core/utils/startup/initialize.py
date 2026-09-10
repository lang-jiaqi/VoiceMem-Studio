"""Inspect all startup requirements before importing models or opening memory."""
from importlib import metadata
import os
import platform
import sys
from studio.paths import ROOT, MODELS
from studio.core.utils.models.initialize import models


def inspect(args):
    """Report missing prerequisites together, without printing any secret value."""
    errors = []
    print(f'[startup] Python {platform.python_version()} · {platform.machine()} · {sys.executable}', flush=True)
    if sys.version_info[:2] != (3, 12):
        errors.append('Studio 需要 Python 3.12，请切换原有 Studio 环境')
    required_keys = {'OPENAI_API_KEY': 'VoiceMem 记忆抽取、归因与后台整理'}
    if args.mode == 'llm_tts' and args.llm == 'deepseek':
        required_keys['DEEPSEEK_API_KEY'] = 'DeepSeek 流式回复'
    if args.mode == 'llm_tts' and args.llm == 'qwen':
        required_keys = {'DASHSCOPE_API_KEY': 'Qwen 流式回复与记忆处理'}
    for name, purpose in required_keys.items():
        present = bool(os.environ.get(name, '').strip())
        print(f'[startup] {name}: {"已找到" if present else "缺失"}（{purpose}）', flush=True)
        if not present:
            errors.append(f'缺少环境变量 {name}（{purpose}）；请让启动终端继承该凭据')
    packages = {name: None for name in (
        'numpy', 'torch', 'torchaudio', 'torchvision', 'fastapi', 'uvicorn',
        'websockets', 'openai', 'httpx', 'mem0ai', 'sentence-transformers',
        'funasr', 'modelscope', 'sherpa-onnx', 'soundfile', 'scipy',
        'huggingface-hub', 'onnxruntime', 'accelerate')}
    packages['transformers'] = '5.16.1'
    if args.mode == 'llm_tts':
        packages.update({'mlx': '0.32.2', 'mlx-audio': '0.5.1', 'mlx-lm': None})
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            errors.append('Breeze MLX 需要原生 Apple Silicon macOS 环境')
    for name, expected in packages.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f'缺少依赖：{name}' + (f'=={expected}' if expected else ''))
            continue
        if expected and installed != expected:
            errors.append(f'依赖版本不匹配：{name}={installed}，需要 {expected}')
        elif args.verbose:
            print(f'[startup] {name} {installed}', flush=True)
    assets = ['studio/web/voicemem.html', 'studio/web/index.html',
              'studio/web/mic-capture-worklet.js', 'studio/web/pcm-player-worklet.js',
              'studio/web/images/background.webp', 'prompt/tts.json']
    if args.mode == 'llm_tts':
        assets += ['voice/noctelle_ref_short.wav', 'voice/noctelle_ref_short.txt']
    for relative in assets:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'缺少资源：{relative}（不会生成替代参考录音）')
    try:
        from voicemem.prompt_config import tts_prompts
        tts_prompts()
        from studio.harness.persona.policy import SYSTEM_PROMPT
        from studio.harness.speaking_style.policy import PROMPT
        from studio.harness.reply_modes.policy import SYSTEM
        from studio.harness.turn_taking.policy import CONTROLS, FILLER_PROMPT
        if not all(isinstance(text, str) and text.strip() for text in (SYSTEM_PROMPT, PROMPT, SYSTEM, FILLER_PROMPT)):
            raise ValueError('empty prompt')
        if CONTROLS['unfinished_followup_s'] <= 0:
            raise ValueError('unfinished_followup_s must be positive')
    except Exception as exc:
        errors.append(f'Prompt/控制配置无效：{type(exc).__name__}: {exc}')
    for model in models(args):
        ready = model.ready(MODELS)
        reusable = any(type(model)(model.name, old, model.repository, model.required).ready(ROOT / 'models')
                       for old in model.legacy or (model.directory,))
        status = '就绪' if ready else '复用原有权重' if reusable else '缺失，启动时自动下载'
        print(f'[startup] {model.name}: {status}', flush=True)
        if not ready and not reusable:
            print('[startup]   ' + '；'.join(model.problems(MODELS)), flush=True)
    if args.backchannel:
        count = sum(1 for _ in (ROOT / 'voice/backchannel').glob('*.wav'))
        print(f'[startup] 已审核附和录音：{count} 段' + ('；缺失，只保留静音等待' if count == 0 else ''), flush=True)
    for error in errors:
        print(f'[startup] 错误：{error}', flush=True)
    if errors:
        raise RuntimeError(f'启动检查失败，共 {len(errors)} 项。尚未下载模型或打开 Memory Space。')
    print('[startup] 环境与凭据检查通过；凭据有效性由对应服务首次请求确认。', flush=True)
