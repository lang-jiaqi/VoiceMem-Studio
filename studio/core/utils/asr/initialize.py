"""Initialize the existing recognizers from verified local weights."""
from .component import FunASRStreamingASR, StreamingASR, OfflineASR
from studio.paths import MODELS
import os
from functools import lru_cache


@lru_cache(maxsize=4)
def _streaming_model(kind, device):
    if kind == "funasr":
        return FunASRStreamingASR(
            model=str(MODELS / "asr/funasr-paraformer-zh-streaming"), device=device)
    return StreamingASR(str(MODELS / "asr/sherpa-onnx-streaming-zipformer-en-2023-06-26"))

def streaming(language="auto"):
    """Default to bilingual FunASR; retain the explicit English adapter."""
    from voicemem.utils.audio.asr import pick_device
    from voicemem.utils.torch_lock import TORCH_LOCK
    kind = "funasr" if language in {"auto", "zh"} else "sherpa"
    device = ((os.environ.get('STUDIO_DEVICE')
               if os.environ.get('STUDIO_BACKEND') == 'cuda' else None)
              or os.environ.get('VOICEMEM_ASR_DEVICE') or pick_device()) if kind == "funasr" else "cpu"
    with TORCH_LOCK:
        return _streaming_model(kind, device).new_stream()


@lru_cache(maxsize=1)
def _final_model():
    return OfflineASR(str(MODELS / "asr/sensevoice-int8"),
                      allowed_languages=("zh", "en", "yue"))


def final():
    """Reuse stateless refinement weights without fixing one input language."""
    from voicemem.utils.torch_lock import TORCH_LOCK
    with TORCH_LOCK:
        return _final_model()
