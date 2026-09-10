"""The sole Studio bridge for VoiceMem initialization and native streaming."""
from voicemem import VoiceMem

def open_memory(config):
    """Create memory with Studio reply/TTS/ASR injected through existing contracts."""
    from voicemem.config import build_kwargs
    from studio.core.utils.llm.initialize import create as reply_model
    from studio.core.utils.tts.initialize import create as speech_model
    from studio.core.utils.asr.initialize import streaming, final
    segment = config["reply"]["llm"]
    memory_config = {k: v for k, v in config.items() if k not in {"reply", "tts"}}
    kwargs = build_kwargs(memory_config)
    language = config.get("memory_language", "zh")
    kwargs.update(reply=reply_model(segment["config"]["system"], segment["provider"]),
                  tts=speech_model, asr=lambda: streaming(language), asr_final=final)
    return VoiceMem(**kwargs)

def open_stream(memory, **options):
    """Return the native stream without wrapping workers or cancellation guards."""
    return memory.stream(**options)


def memory_warmups(memory):
    """Expose strict, read-only model warmups at the memory adapter boundary."""
    import numpy as np

    def asr():
        model = memory.utils.get('asr')
        try:
            model.feed(np.zeros(9600, dtype=np.float32))
        finally:
            model.reset()

    def emotion():
        detector = memory._o._get_emotion_detector()
        detector.warmup()

    return (
        ('Embedding / 槽分类', lambda: memory.classify('你好')),
        ('流式 ASR', asr),
        ('VAD', lambda: memory.utils.get('vad').is_speech(np.zeros(512, dtype=np.float32))),
        ('AST 场景', lambda: memory._o._get_env_detector()._load()),
        ('声纹', lambda: memory._o._get_speaker_encoder()._ensure_started()),
        ('SenseVoice 情绪', emotion),
    )
