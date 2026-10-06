"""Translate declarative provider configuration into VoiceMem constructor arguments.

Factories remain lazy. Explicit provider settings and injected objects override
built-in defaults through the existing capability interfaces. This module only
composes configuration; it does not select a conversation policy."""
from __future__ import annotations

import os
from voicemem.llm_config import MODELS


def _split(component: dict | None) -> tuple[str, dict]:
    """Read provider and optional config values from one capability mapping."""
    component = component or {}
    provider = component.get("provider")
    cfg = component.get("config") or {}
    return provider, cfg


def _bad(component: str, provider, known) -> None:
    raise ValueError(
        f"未知的 {component}.provider={provider!r}；可选：{' / '.join(known)}"
    )


def _embedding_factory(provider, cfg):
    """Build the selected embedding provider lazily, preserving injected objects."""
    if provider == "local":
        def make():
            from voicemem.leftbrain.local_embedder import LocalEmbedder
            # An omitted model keeps the embedding registry and environment defaults.
            return LocalEmbedder(cfg.get("model"), language=cfg.get("language", ""))
        return make
    if provider == "openai":
        def make():
            from voicemem.leftbrain.local_memory_store import (
                OpenAILocalEmbedder, OpenAILocalEmbedderConfig,
            )
            return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(
                model=cfg.get("model"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url"),
                dimensions=cfg.get("dimensions"),
            ))
        return make

    # Delegate other supported embedding providers to the existing mem0 adapter.
    # bedrock / azure_openai / vertexai / together / lmstudio / fastembed /
    from voicemem.leftbrain.mem0_embedder import mem0_providers
    known = mem0_providers()
    if provider in known:
        def make():
            from voicemem.leftbrain.mem0_embedder import Mem0Embedder
            return Mem0Embedder(provider, cfg)
        return make
    _bad("embedding", provider, ["local", "openai", *sorted(known)])


def _slots_factory(provider, cfg):
    """Build the selected local, API or injected slot classifier lazily."""
    if provider == "local":
        def make():
            from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
            # Share embedding weights by default so query and slot vectors use the same model.
            kw = dict(cfg)
            if "model" not in kw:
                from voicemem.leftbrain.local_embedder import (
                    resolve, resolve_path, shared_model)
                # Keep language in the classifier config so its prefixes match the embedder.
                _spec = resolve(language=kw.get("language", ""))
                kw["model"] = shared_model(resolve_path(_spec), _spec.tokenizer_kwargs)
            return LocalQueryClassifier(**kw)
        return make
    if provider == "openai":
        def make():
            from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
            return QuerySlotClassifier()
        return make
    _bad("slots", provider, ["local", "openai"])


def _vad_factory(provider, cfg):
    """Build the selected speech detector lazily, preserving its configured threshold."""
    if provider in (None, "silero"):
        def make():
            from voicemem.utils.audio.stream_io import make_vad
            return make_vad(model=cfg.get("model"), threshold=cfg.get("threshold", 0.5))
        return make
    if provider == "custom":
        obj = cfg.get("obj")
        if obj is None or not hasattr(obj, "is_speech"):
            raise ValueError(
                'vad.provider="custom" 需要 config.obj 给一个有 is_speech(frame)->bool '
                "的对象；直接注入更省事：VoiceMem(vad=lambda: MyVad())"
            )
        return lambda: obj
    _bad("vad", provider, ["silero", "custom"])


def _memory_engine_factory(provider, cfg):
    """Build the selected vector backend with the existing embedding and Space root."""
    if provider == "mem0":
        # None preserves the built-in memory backend.
        return None
    _bad("memory_engine", provider, ["mem0"])


def _tts_factory(provider, cfg):
    """Build the selected optional speech provider lazily."""
    from voicemem.tts import TTS_PROVIDERS
    if provider is not None and str(provider).lower() not in TTS_PROVIDERS:
        _bad("tts", provider, sorted(set(TTS_PROVIDERS)))

    def make():
        from voicemem.tts import make_tts
        return make_tts(provider, **cfg)
    return make


# Legacy nested reply config separates reply generation, TTS and Web realtime settings.
_REPLY_DEMO_KEYS = ("llm", "tts", "realtime")

# Reject unknown top-level settings rather than silently ignoring them.
_KNOWN_TOP = {
    "api_key", "base_url", "mode", "memory_root", "user_id", "space", "models",
    "embedding", "slots", "vad", "memory_engine", "llm", "tts", "reply",
    "top_k", "memory_language", "follow_input_language",
}


def _check_keys(config: dict) -> None:
    unknown = sorted(set(config) - _KNOWN_TOP)
    if unknown:
        raise ValueError(f"config 里有不认识的键：{', '.join(unknown)}。"
                         f"可用的是：{', '.join(sorted(_KNOWN_TOP))}")
    # Accept flat provider config and the existing nested reply shape.
    seg = config.get("reply")
    if isinstance(seg, dict) and any(k in seg for k in _REPLY_DEMO_KEYS):
        bad = sorted(set(seg) - set(_REPLY_DEMO_KEYS))
        if bad:
            raise ValueError(f"reply 段（嵌套写法）里有不认识的键：{', '.join(bad)}。"
                             f"可用的是：{', '.join(_REPLY_DEMO_KEYS)}")


def _reply_factory(provider, cfg):
    """Resolve an API or injected reply callable without invoking it as a factory."""
    if provider == "qwen":
        from voicemem.reply import deepseek_reply
        key = cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("Qwen 回复需要 DASHSCOPE_API_KEY")
        return deepseek_reply(model=cfg.get("model", "qwen3.6-flash"), api_key=key, protocol="qwen",
                            base_url=cfg.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                            system=cfg.get("system"))
    if provider == "deepseek":
        from voicemem.reply import deepseek_reply
        return deepseek_reply(model=cfg.get("model"), api_key=cfg.get("api_key"),
                              base_url=cfg.get("base_url"), system=cfg.get("system"))
    if provider in (None, "openai"):
        from voicemem.reply import openai_reply
        return openai_reply(model=cfg.get("model"), api_key=cfg.get("api_key"),
                            base_url=cfg.get("base_url"), system=cfg.get("system"))
    if provider == "custom":
        fn = cfg.get("fn")
        if not callable(fn):
            raise ValueError(
                'reply.provider="custom" 需要 config.fn 给一个可调用对象；'
                "直接传函数更省事：VoiceMem(reply=fn)"
            )
        return fn
    _bad("reply", provider, ["openai", "deepseek", "qwen", "custom"])


def build_kwargs(config: dict) -> dict:
    """Translate supported configuration keys into the existing VoiceMem injection arguments."""
    config = config or {}
    kwargs: dict = {}
    _check_keys(config)

    if config.get("api_key") is not None:
        kwargs["api_key"] = config["api_key"]
    if config.get("base_url") is not None:
        kwargs["base_url"] = config["base_url"]
    if config.get("mode") is not None:
        kwargs["mode"] = config["mode"]
    if config.get("memory_root") is not None:
        kwargs["memory_root"] = config["memory_root"]
    if config.get("user_id") is not None:
        kwargs["user_id"] = config["user_id"]
    if config.get("space") is not None:
        kwargs["space"] = config["space"]
    if config.get("top_k") is not None:
        kwargs["top_k"] = config["top_k"]
    if config.get("memory_language") is not None:
        kwargs["memory_language"] = config["memory_language"]
    if "follow_input_language" in config:
        if not isinstance(config["follow_input_language"], bool):
            raise ValueError("follow_input_language must be a boolean")
        kwargs["follow_input_language"] = config["follow_input_language"]

    # Component-level model settings take precedence over role defaults.
    if config.get("models"):
        MODELS.update(config["models"])

    if "embedding" in config:
        provider, cfg = _split(config["embedding"])
        kwargs["embedding"] = _embedding_factory(provider, cfg)

    if "slots" in config:
        provider, cfg = _split(config["slots"])
        kwargs["slots"] = _slots_factory(provider, cfg)

    if "vad" in config:
        provider, cfg = _split(config["vad"])
        kwargs["vad"] = _vad_factory(provider, cfg)

    if "memory_engine" in config:
        provider, cfg = _split(config["memory_engine"])
        factory = _memory_engine_factory(provider, cfg)
        if factory is not None:
            kwargs["memory_engine"] = factory

    # The internal LLM config updates the shared chat role and SDK environment settings.
    if "llm" in config:
        provider, cfg = _split(config["llm"])
        if provider not in (None, "openai", "deepseek", "qwen"):
            _bad("llm", provider, ["openai", "deepseek", "qwen"])
        if provider == "qwen":
            cfg = {"model": "qwen3.6-flash", "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", **cfg}
            cfg.setdefault("api_key", os.environ.get("DASHSCOPE_API_KEY", ""))
        if provider == "deepseek":
            # VoiceMem uses an OpenAI-compatible client for internal workers.
            # Give it a DeepSeek model and endpoint instead of the OpenAI default.
            cfg = {"model": "deepseek-v4-flash",
                   "base_url": "https://api.deepseek.com",
                   **cfg}
            if cfg.get("model", "").startswith("gpt-"):
                cfg["model"] = "deepseek-v4-flash"
            cfg.setdefault("api_key", os.environ.get("DEEPSEEK_API_KEY", ""))
        if cfg.get("model"):
            # Model names live in MODELS; SDK credentials and endpoints also use the environment.
            MODELS.update(chat=cfg["model"])
        if cfg.get("api_key"):
            os.environ["OPENAI_API_KEY"] = cfg["api_key"]
            kwargs.setdefault("api_key", cfg["api_key"])
        if cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
            kwargs.setdefault("base_url", cfg["base_url"])

    # Top-level TTS config takes precedence over nested reply.tts.
    tts_seg = config.get("tts")
    if tts_seg is None:
        _r = config.get("reply") or {}
        # A directly injected reply callable is not a configuration mapping.
        if isinstance(_r, dict) and any(k in _r for k in _REPLY_DEMO_KEYS):
            tts_seg = _r.get("tts")
    if tts_seg is not None:
        provider, cfg = _split(tts_seg)
        kwargs["tts"] = _tts_factory(provider, cfg)

    # Resolve either flat reply config or the nested llm segment.
    if "reply" in config:
        seg = config["reply"] or {}
        # Preserve direct callable injection.
        if callable(seg):
            kwargs["reply"] = seg
        else:
            if any(k in seg for k in _REPLY_DEMO_KEYS):
                seg = seg.get("llm") or {}
            provider, cfg = _split(seg)
            kwargs["reply"] = _reply_factory(provider, cfg)

    return kwargs
