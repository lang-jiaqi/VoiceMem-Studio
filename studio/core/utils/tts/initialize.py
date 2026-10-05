"""Initialize Studio's selected speech provider."""
from functools import lru_cache
from contextlib import asynccontextmanager
import os
from studio.paths import MODELS, VOICE

MODEL = MODELS / "tts/Breeze-TTS-2-mlx-4bit"
REFERENCE = VOICE / "noctelle_ref_short.wav"

def create():
    """Keep VoiceMem's plain-function factory contract while sharing the model."""
    return _create()


@lru_cache(maxsize=1)
def _create():
    """Share one speech provider across Studio memory spaces."""
    from .qwen_audio_api import selected
    if selected():
        from .qwen_audio_api import QwenAudioAPI, QwenDemoTTS
        return QwenDemoTTS() if os.environ.get('STUDIO_PUBLIC_DEMO') == '1' else QwenAudioAPI()
    if os.environ.get('STUDIO_BACKEND') == 'cuda':
        from .cuda import BreezeCUDATTS
        return BreezeCUDATTS(ref_audio=str(REFERENCE),
                             device=os.environ.get('STUDIO_TTS_DEVICE', 'cuda:0'))
    from .component import BreezeMLXTTS
    return BreezeMLXTTS(model=str(MODEL), ref_audio=str(REFERENCE),
                        cfg_scale=1.0, seed=42, chunk_frames=2,
                        first_frames=2, max_tokens=750, depth_mode="cached")


@asynccontextmanager
async def conversation_speech(agent):
    """Lease demo API speech; ordinary App and local-model callers keep their factory."""
    if getattr(agent, 'PUBLIC_DEMO', False):
        from .qwen_audio_api import QwenDemoTTS
        provider = agent.vm.utils.get('tts')
        if isinstance(provider, QwenDemoTTS):
            async with provider.session() as speech:
                yield speech
            return
    yield None
