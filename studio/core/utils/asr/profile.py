"""Local streaming model selection shared by acquisition and initialization."""
import os

XASR = 'x-asr-zh-en-480ms-int8'
XASR_REPOSITORY = 'GilgameshWind/X-ASR-zh-en'
XASR_REVISION = '689ff18c584d29910da37b6fe904db0c1489c9d1'
XASR_SOURCE = 'deployment/models/chunk-480ms-model'
XASR_PARTS = ('encoder', 'decoder', 'joiner')
XASR_FILES = tuple(f'{part}-480ms.int8.onnx' for part in XASR_PARTS) + ('tokens.txt', 'prepared.json')
XASR_RECIPE = 'matmul-const-qint8-v1'

ZIPFORMER = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'
ZIPFORMER_FILES = ('encoder-epoch-99-avg-1.int8.onnx',
                   'decoder-epoch-99-avg-1.int8.onnx',
                   'joiner-epoch-99-avg-1.int8.onnx', 'tokens.txt')


def local_backend():
    """Select a bilingual local backend, rejecting invalid explicit overrides."""
    value = os.environ.get('VOICEMEM_ASR', 'bilingual').strip().lower()
    if value == 'funasr':
        return value
    if value in {'sherpa', 'zipformer'}:
        return 'sherpa'
    if value in {'', 'auto', 'bilingual', 'xasr', 'en', 'zh'}:
        return 'xasr'
    raise ValueError('VOICEMEM_ASR must be bilingual, xasr, sherpa or funasr for Studio')
