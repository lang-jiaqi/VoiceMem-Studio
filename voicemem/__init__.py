"""Lazy public API for factual, affective and audio-aware memory.

Public components are resolved on first access and cached under their existing
names. Importing this package alone does not construct models or capabilities.
Implementation owners are listed in voicemem/README.md."""

from __future__ import annotations

# Reduce third-party progress output unless VOICEMEM_VERBOSE is enabled.
#
def _quiet_third_party_logs() -> None:
    import logging
    import os

    if os.environ.get("VOICEMEM_VERBOSE", "0") != "0":
        return

    os.environ.setdefault("TQDM_DISABLE", "1")

    # Logger levels apply to child loggers; logger filters do not propagate.
    # "HTTP Request: POST https://api.openai.com/v1/embeddings ..."，
    for name in ("openai", "httpx", "httpx2", "httpcore", "httpcore2",
                 "mem0", "funasr", "modelscope",
                 "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Set warning and tokenizer options without importing model libraries.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


_quiet_third_party_logs()



try:
    from importlib.metadata import PackageNotFoundError, version as _v
    __version__ = _v("voicemem")
except Exception:
    __version__ = "0.0.0.dev"

import importlib
from typing import TYPE_CHECKING

# Resolve each public name on first access and cache it in this module.
_LAZY: dict[str, str] = {
    "VoiceMem":                 "voicemem.core:VoiceMem",
    "Utils":                    "voicemem.orchestrator:Utils",
    "LeftBrain":                "voicemem.leftbrain.brain:LeftBrain",
    "RightBrain":               "voicemem.rightbrain.brain:RightBrain",
    "SearchResult":             "voicemem.orchestrator:SearchResult",
    "RightBrainHit":            "voicemem.rightbrain.brain:RightBrainHit",
    "AudioPerception":          "voicemem.utils.audio.perceiver:AudioPerception",
    "VoiceStream":              "voicemem.stream:VoiceStream",
    "openai_reply":             "voicemem.reply:openai_reply",
    "normalize_reply":          "voicemem.reply:normalize",
    "Turn":                     "voicemem.stream:Turn",
    "StreamState":              "voicemem.stream:StreamState",

    "VoiceInput":               "voicemem.utils.common.voice_input:VoiceInput",
    "VoiceContent":             "voicemem.utils.common.voice_input:VoiceContent",
    "VoiceIngestResult":        "voicemem.utils.common.voice_input:VoiceIngestResult",
    "VoiceprintRegistry":       "voicemem.utils.common.voice_input:VoiceprintRegistry",
    "VoiceprintEntry":          "voicemem.utils.common.voice_input:VoiceprintEntry",
    "ingest_voice_input":       "voicemem.utils.common.voice_input:ingest_voice_input",
    "voice_input_to_messages":  "voicemem.utils.common.voice_input:voice_input_to_messages",
    "emotion_to_affect":        "voicemem.utils.common.voice_input:emotion_to_affect",
    "map_voice_slots_to_slotv2":"voicemem.utils.common.voice_input:map_voice_slots_to_slotv2",

    "SpeakerEncoder":           "voicemem.utils.audio.voiceprint.speaker_encoder:SpeakerEncoder",
    "VoiceprintStore":          "voicemem.utils.audio.voiceprint.voiceprint_store:VoiceprintStore",
    "IdentifyResult":           "voicemem.utils.audio.voiceprint.voiceprint_store:IdentifyResult",
    "parse_self_identification":"voicemem.utils.audio.voiceprint.speaker_identity:parse_self_identification",

    "ASTEnvironmentDetector":   "voicemem.utils.audio.environment.environment_detector_ast:ASTEnvironmentDetector",
    "CLAPEnvironmentDetector":  "voicemem.utils.audio.environment.environment_detector_clap:CLAPEnvironmentDetector",
    "SceneTag":                 "voicemem.utils.audio.environment.scene_classifier:SceneTag",
    "SceneResult":              "voicemem.utils.audio.environment.scene_classifier:SceneResult",
    "classify_scene":           "voicemem.utils.audio.environment.scene_classifier:classify_scene",
    "scene_to_response_directive":"voicemem.utils.audio.environment.scene_classifier:scene_to_response_directive",
    "infer_scene_from_text":    "voicemem.utils.audio.environment.scene_classifier:infer_scene_from_text",

    "SceneTrigger":             "voicemem.utils.audio.environment.scene_trigger:SceneTrigger",
    "SceneTriggerStore":        "voicemem.utils.audio.environment.scene_trigger:SceneTriggerStore",
    "TriggerFireResult":        "voicemem.utils.audio.environment.scene_trigger:TriggerFireResult",
    "parse_trigger_intent":     "voicemem.utils.audio.environment.scene_trigger:parse_trigger_intent",
    "check_and_fire":           "voicemem.utils.audio.environment.scene_trigger:check_and_fire",

    "MusicMemoryStore":         "voicemem.utils.audio.environment.music_memory:MusicMemoryStore",
    "TuneIdentifyResult":       "voicemem.utils.audio.environment.music_memory:TuneIdentifyResult",
    "PlaceMemoryStore":         "voicemem.utils.audio.environment.place_memory:PlaceMemoryStore",
    "PlaceIdentifyResult":      "voicemem.utils.audio.environment.place_memory:PlaceIdentifyResult",
    "RoutineStore":             "voicemem.utils.audio.environment.routine_memory:RoutineStore",
    "bucket_label":             "voicemem.utils.audio.environment.routine_memory:bucket_label",

    "AudioArchive":             "voicemem.utils.audio.audio_archive:AudioArchive",
    "SessionTracker":           "voicemem.utils.common.session_tracker:SessionTracker",
    "VoiceStoreConfig":         "voicemem.utils.common.voice_config:VoiceStoreConfig",

    "EmotionLayer":             "voicemem.utils.audio.emotion:EmotionLayer",
    "EmotionLayerConfig":       "voicemem.utils.audio.emotion:EmotionLayerConfig",
    "EmotionLayerResult":       "voicemem.utils.audio.emotion:EmotionLayerResult",

    "run_startup_check":        "voicemem.startup_check:run_startup_check",
    "check_and_gate":           "voicemem.startup_check:check_and_gate",
    "StartupReport":            "voicemem.startup_check:StartupReport",

    "Memory":                   "voicemem.memory_api:Memory",
    "build_memory_context":     "voicemem.memory_api:build_memory_context",
    "inject":                   "voicemem.memory_api:inject",
    "recall":                   "voicemem.memory_api:recall",
    "remember":                 "voicemem.memory_api:remember",
}

# Expose registered subpackages lazily alongside the component exports.
_SUBPACKAGES: tuple[str, ...] = (
    "leftbrain", "rightbrain", "utils",
)

__all__ = sorted([*_LAZY, *_SUBPACKAGES])


def __getattr__(name: str):
    """Resolve a public component lazily and cache it under its existing exported name."""
    target = _LAZY.get(name)
    if target is not None:
        module_path, _, attr = target.partition(":")
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value
        return value
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"voicemem.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


# Type checking must not import these components at runtime.
if TYPE_CHECKING:  # pragma: no cover
    from voicemem.core import VoiceMem
    from voicemem.orchestrator import SearchResult
    from voicemem.rightbrain.brain import RightBrainHit
    from voicemem.utils.audio.perceiver import AudioPerception
    from voicemem.utils.audio.emotion import (
        EmotionLayer, EmotionLayerConfig, EmotionLayerResult,
    )

def sample_audio(path):
    """Resolve an existing audio path, otherwise use the packaged example with that basename if present."""
    import os
    from pathlib import Path

    if path is None:
        return None
    p = Path(str(path))
    if p.exists():
        return str(path)
    packaged = Path(__file__).resolve().parent / "assets" / p.name
    if packaged.is_file():
        return str(packaged)
    return str(path)
