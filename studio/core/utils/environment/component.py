"""Set the fixed Studio profile before importing shared memory providers."""
import os
from studio.paths import ROOT, MODELS


def prepare(args):
    """Apply internal paths and tuning; users supply only provider credentials."""
    credential_file = ROOT / '.env.qwen'
    if credential_file.exists():
        for line in credential_file.read_text().splitlines():
            name, sep, value = line.partition('=')
            if sep and name.strip() == 'DASHSCOPE_API_KEY':
                os.environ.setdefault('DASHSCOPE_API_KEY', value.strip())
    # VoiceMem retains its public environment API; Studio supplies that adapter
    # configuration internally so worker subprocesses use the same model roots.
    for key in list(os.environ):
        if key.startswith(('VOICEMEM_', 'BARGE_', 'SPEAKER_')) and not key.endswith('_API_KEY'):
            os.environ.pop(key)
    os.environ.update({
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
