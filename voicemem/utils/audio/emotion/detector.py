"""Small acoustic emotion classifier shared by memory and Studio."""
from pathlib import Path

_TRANSCRIBER = None


def shared_transcriber():
    """Load one CPU recognizer; initialization and inference use the Torch lock."""
    global _TRANSCRIBER
    from voicemem.utils.torch_lock import TORCH_LOCK
    with TORCH_LOCK:
        if _TRANSCRIBER is None:
            from voicemem.utils.audio.asr import Transcriber
            from voicemem.utils.common.paths import models_dir
            directory = models_dir() / 'emotion'
            source = str(directory) if (directory / 'configuration.json').is_file() else 'FunAudioLLM/SenseVoiceSmall'
            _TRANSCRIBER = Transcriber('cpu', model=source)
        return _TRANSCRIBER


class SmallEmotionDetector:
    """Return an acoustic label without generating explanations or loading an LLM."""
    def __init__(self, transcriber=None):
        self._transcriber = transcriber

    def warmup(self):
        """Load the recognizer eagerly; propagate missing-model errors to startup."""
        if self._transcriber is None:
            self._transcriber = shared_transcriber()

    def detect(self, audio_path: Path) -> str:
        """Classify audio and localize its label; inference errors remain visible."""
        from voicemem.lang import display_emotion, is_zh
        self.warmup()
        _, emotion = self._transcriber.run_with_emotion(str(audio_path))
        canonical = {'中性': '平静', '愤怒': '委屈', '恐惧': '焦虑'}.get(emotion, emotion)
        if canonical in {'厌恶', '惊讶'} and not is_zh():
            return {'厌恶': 'disgusted', '惊讶': 'surprised'}[canonical]
        return display_emotion(canonical)
