"""语音转文字：流式识别（实时 partial）+ 非流式精转写（最终文本）。

流式两个实现，接口一致（``feed(samples) -> 累积文本`` / ``flush()`` / ``reset()``），
由 ``utils/defaults.py`` 的 ``asr`` 工厂按 ``VOICEMEM_ASR`` 选：

  · ``FunASRStreamingASR``  FunASR paraformer-zh-streaming（**默认**，中文更准）
  · ``StreamingASR``        sherpa-onnx 流式 zipformer（中英双语、纯 onnx 无 torch）
"""
from __future__ import annotations

import re

import numpy as np

SAMPLE_RATE = 16000

SENSEVOICE_EMOTION_MAP = {
    "NEUTRAL": "中性",
    "HAPPY": "开心",
    "ANGRY": "愤怒",
    "SAD": "悲伤",
    "FEARFUL": "恐惧",
    "FEAR": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}


def pick_device() -> str:
    """自动选最佳设备: cuda > mps(Apple M) > cpu。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


#: 独立的 "I"（以及 I'm / I've / I'll / I'd）要保持大写。
_I_WORD = re.compile(r"\bi\b(?='|\s|$)", re.IGNORECASE)


def _normal_case(text: str) -> str:
    """sherpa 的 zipformer 出的是**全大写**（BPE 词表就是大写的），直接显示和入库都难看。

    只在"整段没有一个小写字母"时才动手——那说明是模型的全大写输出，不是用户真的
    在喊。已经有大小写的文本（比如别的 ASR 喂进来的）原样放过。

    专有名词会被一起小写（"john" 而不是 "John"）：没有模型是判不出来的，
    而全大写比这个更难读。中文不受影响（没有大小写）。
    """
    if not text or any(c.islower() for c in text):
        return text
    if not any(c.isupper() for c in text):
        return text
    out = text.lower()
    out = _I_WORD.sub("I", out)
    # 句首字母大写（以及 . ! ? 之后）
    out = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out


class StreamingASR:
    """sherpa-onnx 流式 zipformer，出实时 partial 文本。``VOICEMEM_ASR=sherpa`` 时启用。"""

    @staticmethod
    def _pick(asr_dir: str, part: str) -> str:
        """在模型目录里找 encoder/decoder/joiner。

        **不能写死文件名**：不同发布版后缀不一样（双语那个是
        ``encoder-epoch-99-avg-1.onnx``，英文那个是
        ``encoder-epoch-99-avg-1-chunk-16-left-128.onnx``），写死就只有一个能用，
        而且报的是 "does not exist"，看不出是版本差异。
        int8 量化版跳过：默认用全精度那份。
        """
        from pathlib import Path as _P
        cands = sorted(f for f in _P(asr_dir).glob(f"{part}-*.onnx")
                       if ".int8." not in f.name)
        if not cands:
            raise FileNotFoundError(
                f"{asr_dir} 里找不到 {part}-*.onnx —— 模型没下全？"
                "跑 scripts/download_models.sh")
        return str(cands[0])

    def __init__(self, asr_dir: str) -> None:
        import sherpa_onnx          # 惰性：默认走 FunASR 时不拉 sherpa
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{asr_dir}/tokens.txt",
            encoder=self._pick(asr_dir, "encoder"),
            decoder=self._pick(asr_dir, "decoder"),
            joiner=self._pick(asr_dir, "joiner"),
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search",
        )
        self.stream = self.rec.create_stream()

    def feed(self, samples):
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return _normal_case(self.rec.get_result(self.stream))

    def flush(self) -> str:
        """接口对齐 FunASRStreamingASR；sherpa 逐帧就把结果吐完了，没有尾巴要补。"""
        return _normal_case(self.rec.get_result(self.stream))

    def reset(self) -> None:
        self.stream = self.rec.create_stream()


# ── 默认流式 ASR：FunASR paraformer-zh-streaming ────────────────────────────────

class FunASRStreamingASR:
    """Streaming FunASR ``paraformer-zh-streaming`` with cumulative text.

    Paraformer decodes 600 ms chunks while callers may provide much shorter
    frames. ``feed()`` buffers those frames and returns all text decoded so far,
    matching the sherpa streaming adapter's contract.

    ``flush()`` appends 50 ms of zero-valued audio to the real tail and sends
    the final piece with ``is_final=True``. The right context lets the decoder
    release trailing tokens without inventing a full 600 ms pause. Audio that
    arrives after a flush starts a fresh decoder cache while retaining the
    cumulative text for the current utterance.
    """

    CHUNK_SIZE = [0, 10, 5]                    # paraformer-streaming 标配
    STRIDE     = CHUNK_SIZE[1] * 960           # 9600 samples @16k = 600ms
    FINAL_PAD_SAMPLES = round(SAMPLE_RATE * 0.050)
    LOOK_BACK  = dict(encoder_chunk_look_back=4, decoder_chunk_look_back=1)

    #: 离线包里的位置。跟回退那套并列放在 asr/ 下——两个都是流式 ASR，区别只是
    #: 默认(FunASR，中文更准) / 回退(sherpa，纯 onnx 不依赖 torch)。
    LOCAL_DIR = "funasr-paraformer-zh-streaming"
    MODEL_CACHE_DIR = (
        "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")

    @staticmethod
    def _complete_model_dir(path):
        """Return a local FunASR directory only when its weight is complete."""
        from pathlib import Path

        directory = Path(path).expanduser()
        required = ("config.yaml", "model.pt", "tokens.json", "am.mvn", "seg_dict")
        if not all((directory / name).is_file() for name in required):
            return None
        # Interrupted ModelScope downloads can leave a tiny placeholder beside
        # valid metadata. Never hand that directory to AutoModel as offline.
        if (directory / "model.pt").stat().st_size < 1024 * 1024:
            return None
        return directory

    @classmethod
    def _cached_model(cls):
        """Find a complete repository bundle or standard ModelScope snapshot."""
        import os
        from pathlib import Path
        from voicemem.utils.common.paths import models_dir

        configured = os.environ.get("VOICEMEM_ASR_MODEL", "").strip()
        if configured:
            complete = cls._complete_model_dir(configured)
            if complete is None:
                raise FileNotFoundError(
                    f"VOICEMEM_ASR_MODEL is not a complete FunASR model: {configured}")
            return complete

        bundled = cls._complete_model_dir(models_dir() / "asr" / cls.LOCAL_DIR)
        if bundled is not None:
            return bundled

        cache_root = Path(os.environ.get(
            "MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
        snapshots = cache_root / "models" / cls.MODEL_CACHE_DIR / "snapshots"
        if snapshots.is_dir():
            for candidate in sorted(
                    snapshots.iterdir(), key=lambda item: item.stat().st_mtime,
                    reverse=True):
                complete = cls._complete_model_dir(candidate)
                if complete is not None:
                    return complete
        return None

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        import logging as _logging
        import os as _os

        # `import funasr` 这一行本身就会把 **root logger** 从 WARNING 拉到 INFO 并挂
        # 一个 handler（实测：import 前 WARNING/0 handlers，import 后 INFO/1）。
        # 后果不只是它自己刷屏——之后 openai/httpx 的 INFO 也全冒出来，一次基础用法
        # 能刷几十行 "HTTP Request: POST ... 200 OK"，真正的结果被埋在中间。
        # 记下 import 前的状态，import 完原样恢复。VOICEMEM_VERBOSE=1 保留原样。
        _quiet = _os.environ.get("VOICEMEM_VERBOSE", "0") == "0"
        _root = _logging.getLogger()
        _lvl, _handlers = _root.level, list(_root.handlers)

        from funasr import AutoModel          # 惰性：只有真用流式 ASR 才拉 funasr

        if _quiet:
            _os.environ.setdefault("TQDM_DISABLE", "1")   # 每转写一块刷一条 rtf 进度条
            _root.setLevel(_lvl)
            for _h in list(_root.handlers):
                if _h not in _handlers:
                    _root.removeHandler(_h)
        # funasr 的 AutoModel 里有一句 logging.basicConfig(level=log_level)，默认
        # INFO —— 它设的是 **root**，于是 openai/httpx 那些库的 INFO 也跟着全冒出来
        # （"HTTP Request: POST ... 200 OK" 刷几十行）。在 voicemem/__init__ 里给
        # 各个 logger 设等级挡不住这个，因为它改的是 root。直接把参数传进去。
        if model is None:
            local = self._cached_model()
            model = str(local) if local is not None else "paraformer-zh-streaming"
        # 跑 MPS。试过挪到 CPU：一块 600ms 音频要 683ms，比实时还慢，说完要排空
        # 几秒积压。MPS 上 55~167ms 一块。多线程撞 MPS 的问题靠 TORCH_LOCK 解决
        # （见 _run）。VOICEMEM_ASR_DEVICE 可强制。
        self.model = AutoModel(model=model,
                               device=device or _os.environ.get("VOICEMEM_ASR_DEVICE") or pick_device(),
                               disable_update=True,
                               log_level="ERROR" if _quiet else "INFO")
        self.reset()

    def _run(self, samples, is_final: bool) -> str:
        from voicemem.utils.torch_lock import TORCH_LOCK
        with TORCH_LOCK:                       # 跟 embedding 等 torch 调用串行
            res = self.model.generate(input=samples, cache=self._cache, is_final=is_final,
                                      chunk_size=self.CHUNK_SIZE, **self.LOOK_BACK)
        if res and res[0].get("text"):
            self._text += res[0]["text"]
        return self._text

    def feed(self, samples) -> str:
        """喂任意长度的 16k float32 帧；攒够 600ms 推一块。返回累积文本。"""
        if self._final:                        # 上一轮 flush 过又来音频 → 起新子流续着攒
            self._cache, self._final = {}, False
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        while len(self._buf) >= self.STRIDE:
            self._run(self._buf[:self.STRIDE], False)
            self._buf = self._buf[self.STRIDE:]
        return self._text

    def flush(self) -> str:
        """Append 50 ms of zero audio and finalize the buffered decoder tail."""
        if self._final:
            return self._text
        tail = np.concatenate([
            self._buf,
            np.zeros(self.FINAL_PAD_SAMPLES, dtype=np.float32),
        ])
        self._buf = np.zeros(0, dtype=np.float32)
        while len(tail) > self.STRIDE:
            self._run(tail[:self.STRIDE], False)
            tail = tail[self.STRIDE:]
        self._final = True
        return self._run(tail, True)

    def reset(self) -> None:
        self._cache: dict = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self._text = ""
        self._final = False


class Transcriber:
    """SenseVoiceSmall 出最终文本（中英），比流式 ASR 更准，锁定一轮时用这个。"""

    def __init__(self, device: str, model: str | None = None) -> None:
        from funasr import AutoModel        # 懒 import：只有用非流式精转写才需要 funasr
        from voicemem.utils.common.paths import hf_model
        _name = model or hf_model("emotion", "FunAudioLLM/SenseVoiceSmall", "asr")
        self.model = AutoModel(model=_name, hub="hf",
                               device=device, disable_update=True,
                               trust_remote_code=False)

    def _generate(self, audio) -> str:
        from voicemem.utils.torch_lock import TORCH_LOCK
        with TORCH_LOCK:
            return self._generate_locked(audio)

    def _generate_locked(self, audio) -> str:
        res = self.model.generate(input=audio, cache={}, language="zh",
                                  use_itn=True, ban_emo_unk=True)
        if not res:
            return ""
        return res[0].get("text", "") or ""

    def run(self, audio) -> str:
        return re.sub(r"<\|[^|]*\|>", "", self._generate(audio)).strip()

    def run_with_emotion(self, audio) -> tuple[str, str]:
        """一次 SenseVoice 推理同时取得文本和声学情绪 token。"""
        raw = self._generate(audio)
        tags = re.findall(r"<\|([^|]+)\|>", raw.upper())
        emotion = next((SENSEVOICE_EMOTION_MAP[tag] for tag in tags
                        if tag in SENSEVOICE_EMOTION_MAP), "中性")
        return re.sub(r"<\|[^|]*\|>", "", raw).strip(), emotion


class OfflineASR:
    """说完那一刻把整轮音频**重转一遍**的离线 ASR（默认 SenseVoice）。

    为什么要两个 ASR：流式模型为了低延迟牺牲了准确率，而这条链上两个需求是分开的——

        partial  打断判定、EOT、闸门要它，**必须低延迟**，转得烂无所谓
                 （它只用来判"有没有人在说连贯的话"）
        final    进记忆、给回复模型的那份，**必须准**，晚 50ms 没人察觉

    实测同一批真实录音（Whisper 转写当标准答案）：

        真实                        流式 zipformer      SenseVoice
        Everything is good.         Everything is going  一字不差
        Why we are helping...       I                    Why there no reply
        Wait, why are you on mute?  Why don't            Wait, why are you

    差距不是调参能补的。SenseVoice 中英日韩粤多语、int8 量化后一轮 33~86ms，
    比 whisper-base 快 4 倍还更准（后者在短句上爱瞎编："of breaking this"）。
    """

    def __init__(self, model_dir: str, num_threads: int = 4):
        import sherpa_onnx
        from pathlib import Path as _P
        d = _P(model_dir)
        model = d / "model.int8.onnx"          # int8 够用，实测和 fp32 没差别
        if not model.is_file():
            model = d / "model.onnx"
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model), tokens=str(d / "tokens.txt"),
            num_threads=num_threads, use_itn=True)   # use_itn：出标点和数字

    def transcribe(self, pcm16k) -> str:
        """整轮 float32 @16k → 文本。"""
        import numpy as np
        a = np.asarray(pcm16k, np.float32).reshape(-1)
        if not a.size:
            return ""
        st = self.rec.create_stream()
        st.accept_waveform(SAMPLE_RATE, a)
        self.rec.decode_stream(st)
        return (st.result.text or "").strip()
