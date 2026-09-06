"""流式输入的音频小工具：重采样 + silero VAD。

原样搬自 web/utils.py（脑图 html 发 24k、流式 ASR 要 16k；silero VAD 判说完），
提升为核心能力，供 voicemem/stream.py 的 VoiceStream 与 web demo 共用（消除重复）。

VAD 现在是可注入能力（``VoiceMem(vad=...)`` / config 的 ``vad`` 段），``make_vad`` 只是
内置的那个 silero 实现；换成自己的只要给个有 ``is_speech(frame) -> bool`` 的对象。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from voicemem.utils.common.paths import model_path, require


def resample(f32, src=24000, dst=16000):        # 脑图 html 发 24k，流式 ASR 要 16k
    """降采样**必须先低通**，否则高频镜像折回语音频段，ASR 拿到的是掺了噪声的信号。

    原来直接 ``np.interp`` 线性插值：那只是极弱的低通，24k→16k 时 8kHz 以上照样
    混叠进来。听感上人不太察觉（语音主要能量在低频），但 ASR 是在频谱上做的，
    这种噪声直接压准确率——一直以为是"模型不行"，其实是喂进去的东西就坏了。

    有 scipy 就用 ``resample_poly``（多相滤波，自带抗混叠）；没有就退回线性插值，
    并且**只在降采样时**打一次提醒——升采样没有混叠问题，不必吓人。
    """
    if src == dst:
        return np.asarray(f32, np.float32)
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(src), int(dst))
        return resample_poly(np.asarray(f32, np.float32),
                             int(dst) // g, int(src) // g).astype(np.float32)
    except ImportError:
        if dst < src and not _RESAMPLE_WARNED:
            _warn_resample()
        n = int(len(f32) * dst / src)
        return np.interp(np.arange(n) * src / dst,
                         np.arange(len(f32)), f32).astype(np.float32)


_RESAMPLE_WARNED = False


def _warn_resample() -> None:
    global _RESAMPLE_WARNED
    _RESAMPLE_WARNED = True
    print("[audio] 没装 scipy → 降采样退回线性插值（有混叠，ASR 会变差）。"
          "pip install scipy 可修。", flush=True)


def read_wav(path) -> tuple[np.ndarray, int]:
    """读音频文件 → (float32 单声道, 采样率)。有 soundfile 就用它（格式全），
    没有就退回标准库 wave（只认 PCM wav，但不多一个依赖）。"""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        return (audio[:, 0] if audio.ndim > 1 else audio), sr
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            sr, n_ch = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        return (pcm[::n_ch] if n_ch > 1 else pcm), sr


def transcribe_file(asr, path, chunk_s: float = 0.6) -> str:
    """用流式 ASR 把整个文件转写成一段文本：分块喂完再 flush 收尾。

    只是把「流式接口」包成「整段接口」，不引入第二套 ASR——ingest(audio=...) 这类
    一次性调用没必要为此再拉一个非流式模型。
    """
    audio, sr = read_wav(path)
    if sr != 16000:
        audio = resample(audio, src=sr)
    asr.reset()
    step, text = max(1, int(16000 * chunk_s)), ""
    for i in range(0, len(audio), step):
        text = asr.feed(audio[i:i + step]) or text
    flush = getattr(asr, "flush", None)          # 块式 ASR 把不足一块的尾巴补零吐出来
    if flush is not None:
        text = flush() or text
    return (text or "").strip()


def make_vad(model: str | None = None, threshold: float = 0.5,
             min_silence_s: float | None = None):
    """内置 VAD：silero（sherpa-onnx 包的）。返回一个只有 ``is_speech(frame)`` 的小对象。

    ``model`` 不给就走 ``VOICEMEM_SILERO_VAD`` / ``VOICEMEM_MODELS_DIR/silero_vad.onnx``。
    这个 .onnx 没有自动下载兜底，缺了就明确报出来（而不是让 sherpa 抛个看不懂的错）。

    ``min_silence_s``：静音要持续多久才算"停了"。silero 默认 **0.5 秒**——那是为
    "把一句话完整切出来"设的，词间的小停顿一律被桥接掉。判回合结束用这个默认值
    正合适；但要判**句子中间的微停顿**（附和就靠它）就得单独建一个短的，比如
    0.08——两个用途参数相反，共用一个实例是做不到的。
    """
    import sherpa_onnx
    path = require(
        Path(model) if model else model_path("silero_vad.onnx", "vad", kind="vad"),
        "silero VAD 模型 silero_vad.onnx",
    )
    silero = sherpa_onnx.SileroVadModelConfig(model=str(path), threshold=threshold)
    if min_silence_s is not None:
        silero.min_silence_duration = float(min_silence_s)
    v = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
        silero_vad=silero, sample_rate=16000), buffer_size_in_seconds=30)

    class _V:
        def is_speech(self, frame): v.accept_waveform(frame); return v.is_speech_detected()
    return _V()
