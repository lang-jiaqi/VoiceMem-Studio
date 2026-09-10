"""Set the fixed Studio profile before importing shared memory providers."""
import os
from pathlib import Path


def load_environment(root=None):
    """Load repository credentials and options without replacing exported values."""
    from dotenv import load_dotenv
    root = Path(root) if root is not None else Path(__file__).resolve().parents[4]
    load_dotenv(root / '.env', override=False)
    load_dotenv(root / '.env.qwen', override=False)


def prepare(args):
    """Apply the selected backend and shared paths before model imports."""
    from studio.paths import ROOT, MODELS
    # VoiceMem retains its public environment API; Studio supplies that adapter
    # configuration internally so worker subprocesses use the same model roots.
    os.environ.update({
        'STUDIO_BACKEND': args.backend,
        'STUDIO_DEVICE': args.device,
        'STUDIO_TTS_DEVICE': args.tts_device,
        'VOICEMEM_MODELS_DIR': str(MODELS),
        'VOICEMEM_MEMORYSPACE_ROOT': str(ROOT / 'voicemem_memoryspace'),
        'VOICEMEM_MEMORY_LANGUAGE': args.lang,
        'VOICEMEM_VERBOSE': '1' if args.verbose else '0',
        'VOICEMEM_ENVIRONMENT_MODEL_DIR': str(MODELS / 'scene'),
        'VOICEMEM_ASR_MODEL': str(MODELS / 'emotion'),
        'VOICEMEM_FINAL_ASR_DIR': str(MODELS / 'asr/sensevoice-int8'),
        'VOICEMEM_DEEPSEEK_FIRST_TOKEN_TIMEOUT': '2.0',
        'VOICEMEM_DEEPSEEK_MAX_TOKENS_NONE': '512',
        'VOICEMEM_DEEPSEEK_MAX_TOKENS_LOW': '1024',
        'VOICEMEM_DEEPSEEK_MAX_TOKENS_HIGH': '4096',
        'VOICEMEM_DEEPSEEK_MAX_TOKENS_MAX': '8192',
    })
    from voicemem.lang import set_memory_language
    set_memory_language(args.lang)
