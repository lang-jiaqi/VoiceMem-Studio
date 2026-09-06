"""语义判「说完了没」：Smart Turn v3（pipecat-ai），看波形不看转写。

现在这条链上，「你说完了吗」是靠**掐表**判的：静音满 300ms 就算一轮结束。这一个
数字同时决定两件事，所以怎么调都不对：

    调大 → 句子不再被切碎，但每次回答都晚那么多（生成是回合结束才起跑的）
    调小 → 快，但你中间想一下超过 300ms 就被当成说完了

实测代价看得见：用户一分钟里的六段录音有四段是 1~2 秒的碎片（"Wait."、
"Wait, wait, wait, there's no."），一句话被切成好几轮，各自存一条残缺记忆、
各自触发一次回复。

这个模型把「说完没」和「等多久」解耦：它直接看最后 8 秒的波形，输出「这一轮结束
了」的概率。Whisper Tiny 编码器 + 线性头，8M 参数、32MB onnx，CPU 上实测 15ms
——跟 VAD 同一个运行时，不引入 torch。

    from voicemem.utils.audio.eot import EndOfTurn
    eot = EndOfTurn()
    if eot.done(pcm16k):        # 说完了
        ...

**它判的是语气和句法，不是转写**，所以本地 ASR 转得烂也不影响——这点对这条链
很关键（实测本地流式 ASR 在英文短句上几乎转不出东西）。

模型：https://huggingface.co/pipecat-ai/smart-turn-v3 （BSD-2）
"""
from __future__ import annotations

import os

import numpy as np

#: 判到多少算「说完了」。偏高一点更保守：宁可多等 100ms，也别把人的话切断——
#: 切断的代价是残缺记忆 + 一次白跑的回复 + 用户被抢话。
THRESHOLD = float(os.environ.get("VOICEMEM_EOT_THRESHOLD", "0.6"))
#: 模型固定吃 8 秒窗口（80×800 log-mel）。
WINDOW_S = 8
REPO = "pipecat-ai/smart-turn-v3"
FILE = os.environ.get("VOICEMEM_EOT_MODEL_FILE", "smart-turn-v3.2-cpu.onnx")


class EndOfTurn:
    """``done(pcm)`` / ``score(pcm)``：pcm 是 16k 单声道 float32。"""

    def __init__(self, model_path: str | None = None, threshold: float = THRESHOLD):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        path = model_path or self._resolve()
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1          # 跟官方推理代码一致
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._sess = ort.InferenceSession(path, sess_options=so,
                                          providers=["CPUExecutionProvider"])
        # chunk_length=8 是这个模型的窗口；其余参数用 Whisper 默认（它就是 Whisper 编码器）
        self._fe = WhisperFeatureExtractor(chunk_length=WINDOW_S)
        self.threshold = threshold

    @staticmethod
    def _resolve() -> str:
        """离线包优先（models/eot/），没有就从 HF 拉一次（32MB）。"""
        from voicemem.utils.common.paths import models_dir
        local = models_dir() / "eot" / FILE
        if local.is_file():
            return str(local)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(REPO, FILE)

    #: 送进模型前，尾部只保留这么多静音。见 score() 里那段。
    TAIL_S = float(os.environ.get("VOICEMEM_EOT_TAIL", "0.2"))
    #: 判「哪儿算语音结束」的能量阈值。
    LEVEL = 0.01

    @classmethod
    def _frame(cls, a, sample_rate: int):
        """把音频切到「语音结束 + 固定 200ms」。

        **这一步是必须的，不是优化。** 模型吃固定 8 秒窗口，不够就在前面补零——
        于是尾部静音一变，语音在窗口里的位置就跟着移动，而它对这个位置极度敏感。
        实测同一句话：

            尾部 +200ms → 0.95      尾部 +500ms → 0.01
            尾部 +2.0s  → 0.96      尾部 +4.8s  → 0.45

        这显然不是"说完没说完"的函数。而线上的缓冲区是一直在涨的，每问一次位置
        都不一样——之前看到的分数乱跳，根子全在这。切成固定取景之后，同一批录音
        上完整句稳定 0.95~0.97、残句稳定 0.01~0.23，跟内容对得上了。
        """
        idx = np.nonzero(np.abs(a) > cls.LEVEL)[0]
        if not idx.size:
            return a
        end = min(len(a), int(idx[-1]) + int(cls.TAIL_S * sample_rate))
        return a[:end]

    def score(self, pcm, sample_rate: int = 16000) -> float:
        """0~1，越大越像「说完了」。onnx 输出**已经是概率**，别再套 sigmoid。"""
        n = WINDOW_S * sample_rate
        a = self._frame(np.asarray(pcm, np.float32).reshape(-1), sample_rate)
        # 不够 8 秒**在前面补零**（官方 truncate_audio_to_last_n_seconds 就是这样）：
        # 补在后面等于告诉模型「话说完后还有一段静音」，那正是它要判的东西，
        # 补错一边结论就反了。
        a = a[-n:] if len(a) > n else np.pad(a, (n - len(a), 0))
        feat = self._fe(a, sampling_rate=sample_rate, return_tensors="np",
                        padding="max_length", max_length=n, truncation=True,
                        do_normalize=True).input_features
        out = self._sess.run(None, {"input_features": feat.astype(np.float32)})
        return float(np.asarray(out[0]).reshape(-1)[0])

    def done(self, pcm, sample_rate: int = 16000) -> bool:
        return self.score(pcm, sample_rate) >= self.threshold
