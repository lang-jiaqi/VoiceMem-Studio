"""Pinned X-ASR artifacts shared by acquisition and initialization."""

XASR = 'x-asr-zh-en-480ms-int8'
XASR_REPOSITORY = 'GilgameshWind/X-ASR-zh-en'
XASR_REVISION = '689ff18c584d29910da37b6fe904db0c1489c9d1'
XASR_SOURCE = 'deployment/models/chunk-480ms-model'
XASR_PARTS = ('encoder', 'decoder', 'joiner')
XASR_FILES = tuple(f'{part}-480ms.int8.onnx' for part in XASR_PARTS) + ('tokens.txt', 'prepared.json')
XASR_RECIPE = 'matmul-const-qint8-v1'
