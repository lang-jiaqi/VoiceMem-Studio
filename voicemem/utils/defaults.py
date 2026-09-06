"""voicemem 各能力的内置默认实现工厂（util 名 -> 无参工厂）。

core.py 的 Utils 用它建默认；传函数给 VoiceMem(embedding=..., slots=...) 即覆盖对应项。
九个位子：embedding / schema / entity / emotion / voiceprint / asr / vad /
memory_engine / tts。前八个在核心链路上（按 mode 由 _NEED 决定加载哪些），
tts 不在——记忆系统只到文本为止，出声是可选的一层。
放这里而不是 core.py，是让顶层门面只讲「系统骨架」，不被这些具体默认实现的 import 撑大。
"""
from __future__ import annotations

import os


def default_utils(base_url, memory_root):
    def embedding():
        from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
        return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=base_url))
    def slots():
        # 默认本地句向量分类器：0 LLM、0 网络——投机预取那 0–300ms 预算里不能走网络，
        # 而 Classify 就在那条路上（voicemem/stream.py 的 _speculate）。
        # sentence-transformers 不在基础依赖里（随 [demo] extra 装），缺了就回落到
        # LLM 版并打一行说明——静默回落等于悄悄开始花钱。
        # VOICEMEM_SLOTS=openai 可强制用 LLM 版（要实体抽取 / 子 slot 下钻时）。
        if os.environ.get("VOICEMEM_SLOTS", "local").lower() != "openai":
            try:
                from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
                from voicemem.leftbrain.local_embedder import (
                    resolve, resolve_path, shared_model)
                # 语言显式从**这个空间**读，不走进程全局：同一进程开两个不同语言
                # 的空间时，全局那份是后建的那个（issue #9 的同一个根子）。
                from voicemem.lang import resolve_for_space
                spec = resolve(language=resolve_for_space(memory_root))
                # 和本地 embedder 共享同一份权重（省一份内存），也保证槽描述和
                # 查询编码在同一个向量空间里。
                return LocalQueryClassifier(model=shared_model(resolve_path(spec),
                                                                spec.tokenizer_kwargs))
            except ImportError as e:
                print(f"[slots] 本地分类器不可用（{e}）→ 回落 LLM 版 QuerySlotClassifier。"
                      "装 sentence-transformers（或 pip install -e '.[demo]'）可用本地版。",
                      flush=True)
        from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
        return QuerySlotClassifier()
    def entity():
        from voicemem.leftbrain.cognitive_graph.annotator import CognitiveAnnotator, CognitiveAnnotatorConfig
        return CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=base_url))
    def emotion():
        from voicemem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
        return PaperAlignedEmotionDetector()
    def voiceprint():
        from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
        return SpeakerEncoder(device="cpu")
    def asr():
        """**按空间语言选**：zh → FunASR paraformer-zh，en → sherpa 英文专用。

        一个模型只认一种语言，所以不能跟空间语言脱钩——早先一律默认 FunASR，
        英文库里说英文转出来是一串无意义的中文（"hellolo""今天天气如何何" 这类
        脏记忆就是这么来的）。双语那个也不行：它会把英文吐成中文（实测
        "Wait, why are you on mute?" → "喂我也用"）。

        代价：一个库不能中英混说。这跟 voicemem/lang.py 的既有立场一致——语言是
        库的属性。要混说：VOICEMEM_ASR=bilingual。

        **英文这一档已知不够好**：本地流式小模型对英文短句都很弱（实测 "Wait."
        → 空，"Why we are helping your plan." → "I"），而同一批音频 Whisper 全对。
        这是这一档模型的上限，不是选型问题——要准得在说完之后用大模型重转一遍。
        中文那边没这个问题，FunASR 实测够用。
        """
        from voicemem.utils.common.paths import model_path
        from voicemem.utils.audio.asr import StreamingASR
        _SHERPA = {
            "en": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
            "bilingual": "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        }
        pick = os.environ.get("VOICEMEM_ASR", "").lower()
        if not pick:
            from voicemem.lang import resolve_for_space
            pick = "funasr" if resolve_for_space(memory_root) == "zh" else "en"
        if pick in _SHERPA:
            try:
                return StreamingASR(str(model_path(_SHERPA[pick], kind="asr")))
            except Exception as e:
                print(f"[asr] {_SHERPA[pick]} 不可用（{type(e).__name__}: {e}）"
                      f"→ 回落 FunASR（**只认中文**）。"
                      f"scripts/download_models.sh 可下载。", flush=True)
        from voicemem.utils.audio.asr import FunASRStreamingASR
        return FunASRStreamingASR()

    def asr_final():
        """说完那一刻重转一遍的离线 ASR。没下载模型就返回 None（沿用流式那份文本）。

        单独一个能力位而不是换掉 asr：两者的取舍相反（一个要快、一个要准），
        合成一个就得二选一。见 OfflineASR 的文档。
        """
        if os.environ.get("VOICEMEM_FINAL_ASR", "1") == "0":
            return None
        from voicemem.utils.common.paths import models_dir
        d = models_dir() / "asr" / os.environ.get(
            "VOICEMEM_FINAL_ASR_DIR",
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
        if not (d / "tokens.txt").is_file():
            print(f"[asr] 没有离线复核模型（{d.name}）→ 只用流式那份转写。"
                  "scripts/download_models.sh 可下载。", flush=True)
            return None
        from voicemem.utils.audio.asr import OfflineASR
        return OfflineASR(str(d))

    def vad():
        # 判「说完了」的 VAD。默认内置 silero；换自己的传一个有 is_speech(frame)->bool
        # 的对象即可（VoiceMem(vad=lambda: MyVad()) 或 config 的 vad 段）。
        from voicemem.utils.audio.stream_io import make_vad
        return make_vad()
    def tts():
        # 第九个可替换位。核心链路不用它——记忆系统只到文本为止，出声是调用方的事，
        # 所以 tts 不进 _NEED（warmup 不会拉起来），谁要出声谁 utils.get("tts")。
        # 不把 base_url 传下去：那个通常指向自建 LLM/embedding 服务，多半没有
        # /audio/speech，跟过去只会在出声时才炸。要换端点用 OPENAI_TTS_BASE_URL。
        from voicemem.tts import make_tts
        return make_tts()
    def memory_engine():
        from pathlib import Path
        from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore
        # memory_root 由 Orchestrator 传下来（已解析过默认值）；这里的兜底只在
        # 直接构造 default_utils 时用得上，跟上面保持同一个默认。
        return Mem0BackendStore(embedding(),
                                memory_root=Path(memory_root or Path.cwd() / "voicemem_memory"))
    return {"embedding": embedding, "slots": slots, "entity": entity, "emotion": emotion,
            "asr_final": asr_final,
            "voiceprint": voiceprint, "asr": asr, "vad": vad, "memory_engine": memory_engine,
            "tts": tts}
