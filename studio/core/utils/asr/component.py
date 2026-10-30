"""Native VoiceMem recognizers; worker and epoch ownership stay in VoiceStream."""
from voicemem.utils.audio.asr import StreamingASR, FunASRStreamingASR, OfflineASR
import re
import numpy as np


class XASRStreamingASR(StreamingASR):
    """Bilingual CPU decoding with resumable right context and native casing."""

    FINAL_PAD_SAMPLES = 12800

    def __init__(self, directory):
        super().__init__(directory, prefer_int8=True, num_threads=1, model_type='zipformer2')

    def reset(self):
        super().reset()
        self._zero_tail = 0

    def feed(self, samples):
        self._zero_tail = 0 if np.any(samples) else self._zero_tail + len(samples)
        return super().feed(samples)

    def flush(self):
        """Supply missing right context immediately without closing the stream."""
        if not self._flushed:
            missing = max(0, self.FINAL_PAD_SAMPLES - self._zero_tail)
            if missing:
                self.feed(np.zeros(missing, dtype=np.float32))
            self._flushed = True
        return self._format_text(self.rec.get_result(self.stream))

    def _format_text(self, text):
        text = re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', text)
        text = re.sub(r'\s+([，。！？、；：,.!?;:])', r'\1', text)
        return re.sub(r'([，。！？、；：])\s+(?=[\u3400-\u9fff])', r'\1', text).strip()
