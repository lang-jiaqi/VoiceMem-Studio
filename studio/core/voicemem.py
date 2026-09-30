"""Studio's bridge for VoiceMem initialization, streaming, and shared models."""
from voicemem import VoiceMem

def open_memory(config):
    """Create memory with Studio reply/TTS/ASR injected through existing contracts."""
    from voicemem.config import build_kwargs
    from studio.core.utils.llm.initialize import create as reply_model, credential
    from studio.core.utils.tts.initialize import create as speech_model
    from studio.core.utils.asr.initialize import streaming, final
    segment = config["reply"]["llm"]
    memory_config = {k: v for k, v in config.items() if k not in {"reply", "tts"}}
    kwargs = build_kwargs(memory_config)
    language = config.get("memory_language", "zh")
    kwargs.update(reply=reply_model(segment["config"]["system"], segment["provider"],
                                   credential(segment["provider"], "reply")),
                  tts=speech_model, asr=lambda: streaming(language), asr_final=final)
    return VoiceMem(**kwargs)

def open_stream(memory, **options):
    """Return the native stream without wrapping workers or cancellation guards."""
    return memory.stream(**options)


def greeting_memories(memory, *, limit: int = 3) -> list[str]:
    """Read a few low-sensitivity facts from this VoiceMem user's space."""
    repo = memory._o._get_repo()
    user_id = memory._o._user_id
    cognitive = repo._cognitive_store
    if cognitive is None:
        return []
    allowed_slots = {"daily_life", "knowledge", "goals", "work"}
    entries = repo._vector_store.list_entries(user_id=user_id, limit=80)
    entries.sort(key=lambda item: item.get("date", ""), reverse=True)
    selected = []
    for entry in entries:
        if entry.get("role") == "assistant":
            continue
        record = cognitive.get_memory_record(entry["id"])
        if (record is None or record.user_id != user_id
                or record.slot not in allowed_slots
                or record.sensitivity > 0.2):
            continue
        fact = " ".join(entry.get("text", "").split())[:160]
        if fact and fact not in selected:
            selected.append(fact)
        if len(selected) >= limit:
            break
    return selected


def shared_embed_model():
    """Return VoiceMem's process-shared embedding model for Studio perception."""
    from voicemem.leftbrain.local_embedder import resolve, resolve_path, shared_model

    return shared_model(resolve_path(resolve()))


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
