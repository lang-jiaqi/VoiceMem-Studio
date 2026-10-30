"""Initialize the existing recognizers from verified local weights."""
from .component import OfflineASR, XASRStreamingASR
from studio.paths import MODELS
from functools import lru_cache
from .profile import XASR


@lru_cache(maxsize=1)
def _streaming_model():
    print('[asr] X-ASR zh-en · 480ms · CPU · int8 · 1 thread', flush=True)
    return XASRStreamingASR(str(MODELS / 'asr' / XASR))

def streaming(language="auto"):
    """Share bilingual weights; UI language never restricts recognized speech."""
    from studio.core.utils.component_config.component import selection
    value = selection('asr')
    if value['provider'] == 'deepgram':
        from .deepgram import DeepgramASR
        return DeepgramASR(model=value['model'], base_url=value['baseUrl'],
                           api_key=value['apiKey'], language=value['language'])
    if value['provider'] == 'adapter':
        return _custom_streaming().new_stream()
    from voicemem.engine import device_lock
    with device_lock('cpu'):
        return _streaming_model().new_stream()


@lru_cache(maxsize=1)
def _final_model():
    return OfflineASR(str(MODELS / "asr/sensevoice-int8"),
                      allowed_languages=("zh", "en", "yue"))


def final():
    """Reuse stateless refinement weights without fixing one input language."""
    from studio.core.utils.component_config.component import selection
    if selection('asr_final')['provider'] == 'adapter':
        return _custom_final()
    from voicemem.engine import TORCH_LOCK
    with TORCH_LOCK:
        return _final_model()


@lru_cache(maxsize=1)
def _custom_streaming():
    from studio.core.utils.component_config.component import custom
    return custom('asr')


@lru_cache(maxsize=1)
def _custom_final():
    from studio.core.utils.component_config.component import custom
    return custom('asr_final')
