"""Initialize the existing recognizers from verified local weights."""
from .component import FunASRStreamingASR, StreamingASR, OfflineASR
from studio.paths import MODELS
import os

def streaming(language="zh"):
    """Create the language-appropriate streaming recognizer."""
    if language == "zh":
        device = os.environ.get('STUDIO_DEVICE') if os.environ.get('STUDIO_BACKEND') == 'cuda' else None
        return FunASRStreamingASR(model=str(MODELS / "asr/funasr-paraformer-zh-streaming"), device=device)
    return StreamingASR(str(MODELS / "asr/sherpa-onnx-streaming-zipformer-en-2023-06-26"))

def final():
    """Create the shared CPU final-ASR recognizer."""
    return OfflineASR(str(MODELS / "asr/sensevoice-int8"))
