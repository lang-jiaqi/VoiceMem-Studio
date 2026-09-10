"""Initialize the reviewed Breeze voice and preserve the shared GPU scheduler."""
from .component import BreezeMLXTTS
from studio.paths import MODELS, ROOT

MODEL = MODELS / "tts/Breeze-TTS-2-mlx-4bit"
REFERENCE = ROOT / "voice/noctelle_ref_short.wav"

def create():
    """Create Breeze with fixed first-chunk, reference, and cache settings."""
    return BreezeMLXTTS(model=str(MODEL), ref_audio=str(REFERENCE),
                        cfg_scale=1.0, seed=42, chunk_frames=2,
                        first_frames=2, max_tokens=750, depth_mode="cached")
