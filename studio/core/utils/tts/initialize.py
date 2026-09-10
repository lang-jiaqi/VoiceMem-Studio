"""Initialize the reviewed Breeze voice and preserve the shared GPU scheduler."""
from functools import lru_cache
import os
from studio.paths import MODELS, ROOT

MODEL = MODELS / "tts/Breeze-TTS-2-mlx-4bit"
REFERENCE = ROOT / "voice/noctelle_ref_short.wav"

def create():
    """Keep VoiceMem's plain-function factory contract while sharing the model."""
    return _create()


@lru_cache(maxsize=1)
def _create():
    """Share one local Breeze provider, using the selected inference backend."""
    if os.environ.get('STUDIO_BACKEND') == 'cuda':
        from .cuda import BreezeCUDATTS
        return BreezeCUDATTS(ref_audio=str(REFERENCE),
                             device=os.environ.get('STUDIO_TTS_DEVICE', 'cuda:0'))
    from .component import BreezeMLXTTS
    return BreezeMLXTTS(model=str(MODEL), ref_audio=str(REFERENCE),
                        cfg_scale=1.0, seed=42, chunk_frames=2,
                        first_frames=2, max_tokens=750, depth_mode="cached")
