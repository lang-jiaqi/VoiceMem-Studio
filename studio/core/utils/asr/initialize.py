"""Initialize the existing recognizers from verified local weights."""
from .component import FunASRStreamingASR, StreamingASR, OfflineASR
from studio.paths import MODELS

def streaming(language="zh"):
    """Create the language-appropriate streaming recognizer."""
    if language == "zh":
        return FunASRStreamingASR(model=str(MODELS / "asr/funasr-paraformer-zh-streaming"))
    return StreamingASR(str(MODELS / "asr/sherpa-onnx-streaming-zipformer-en-2023-06-26"))

def final():
    """Create the shared CPU final-ASR recognizer."""
    return OfflineASR(str(MODELS / "asr/sensevoice-int8"))
