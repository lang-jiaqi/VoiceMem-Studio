"""Lazy default factories for injectable VoiceMem capabilities.

Utils in orchestrator.py resolves and caches these factories per instance.
Its mode profile chooses capabilities to warm up. Optional TTS is not part of
memory search and loads only when the caller explicitly requests it.
"""
from __future__ import annotations

import os


def default_utils(base_url, memory_root):
    def embedding():
        from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
        return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=base_url))
    def slots():
        # Local classification shares the embedding model and avoids network
        # work during speculative retrieval. An unavailable optional runtime
        # falls back to the LLM classifier with an explicit notice.
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
                      "安装 sentence-transformers 可用本地版。",
                      flush=True)
        from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
        return QuerySlotClassifier()
    def entity():
        from voicemem.leftbrain.cognitive_graph.annotator import CognitiveAnnotator, CognitiveAnnotatorConfig
        return CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=base_url))
    def emotion():
        from voicemem.utils.audio.emotion.detector import SmallEmotionDetector
        return SmallEmotionDetector()
    def voiceprint():
        from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
        return SpeakerEncoder(device="cpu")
    def asr():
        """Select the core ASR default from the Space language or explicit override.

        Chinese Spaces default to FunASR; English Spaces default to Zipformer.
        ``VOICEMEM_ASR=bilingual`` selects the bilingual Zipformer adapter.
        Studio injects its own shared bilingual FunASR provider independently.
        This selection does not restrict the language of stored memory text.
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
                      f"→ 回落 FunASR。"
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
        configured = os.environ.get("VOICEMEM_FINAL_ASR_DIR", "").strip()
        names = ([configured] if configured else [
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        ])
        candidates = [models_dir() / "asr" / name for name in names]
        d = next((path for path in candidates
                  if (path / "tokens.txt").is_file()
                  and ((path / "model.int8.onnx").is_file()
                       or (path / "model.onnx").is_file())), None)
        if d is None:
            expected = ", ".join(path.name for path in candidates)
            print(f"[asr] 没有离线复核模型（尝试了 {expected}）→ 只用流式那份转写。"
                  "scripts/download_models.sh 可下载。", flush=True)
            return None
        from voicemem.utils.audio.asr import OfflineASR
        print(f"[asr] 最终复核模型：{d.name}（CPU int8 优先）", flush=True)
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
