"""Editable Studio prompt catalog and task-local overrides."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from string import Formatter

_CURRENT = ContextVar("studio_harness_prompts", default=None)


def catalog(language="zh"):
    """Return default prompt metadata without loading models or user data."""
    return _catalog("en" if language == "en" else "zh")


@lru_cache(maxsize=2)
def _catalog(language):
    from studio.harness.persona.policy import SYSTEM_PROMPT
    from studio.harness.speaking_style.policy import PROMPT, TONE_RULE
    from studio.harness.reply_modes.policy import SYSTEM
    from studio.harness.self_harness.policy import CONTROL_RULE
    from studio.harness.turn_taking.policy import FILLER_PROMPT, FILLER_INPUT_PROMPT
    from studio.core.utils.self_harness.component import PROMPT_SCHEMA
    from studio.prompt_config import tts_prompts

    rows = (
        ("main_system", "main", "SYSTEM_PROMPT", SYSTEM_PROMPT),
        ("persona", "persona", "PERSONA_PROMPT", ""),
        ("speaking_style", "speaking_style", "SPEAKING_PROMPT", PROMPT),
        ("tone_rule", "speaking_style", "TONE_RULE", TONE_RULE),
        ("router", "reply_modes", "ROUTER_PROMPT", SYSTEM),
        ("filler_system", "turn_taking", "FILLER_PROMPT", FILLER_PROMPT),
        ("filler_input", "turn_taking", "FILLER_INPUT_PROMPT", FILLER_INPUT_PROMPT),
        ("self_harness", "self_harness", "SELF_HARNESS_PROMPT", CONTROL_RULE),
        ("tts_zh", "tts", "TTS_PROMPT_ZH", tts_prompts()["base"]["zh"]),
        ("tts_en", "tts", "TTS_PROMPT_EN", tts_prompts()["base"]["en"]),
    )
    localized = {}
    if language == "en":
        from studio.harness.prompts_en import DEFAULTS
        localized = DEFAULTS
    return {key: {"group": group, "variable": variable, "default": localized.get(key, default),
                  "max_length": PROMPT_SCHEMA[key]["max_length"] if key in PROMPT_SCHEMA else 20000}
            for key, group, variable, default in rows}


def validate_overrides(value, *, language="zh"):
    """Validate an atomic settings replacement; raise ValueError on invalid input."""
    specs = catalog(language)
    if not isinstance(value, dict) or len(value) > len(specs):
        raise ValueError("Prompt settings must be an object with known fields")
    result = {}
    for key, text in value.items():
        spec = specs.get(key)
        if spec is None or not isinstance(text, str):
            raise ValueError("Unknown prompt or non-text prompt value")
        if len(text) > spec["max_length"] or "\x00" in text:
            raise ValueError(f"{spec['variable']}: prompt is too long or contains NUL")
        if key != "persona" and not text.strip():
            raise ValueError(f"{spec['variable']}: prompt cannot be empty")
        if key in {"filler_system", "filler_input"}:
            expected = ({"task_context"} if key == "filler_system" else
                        {"task_context", "language", "history"})
            try:
                parsed = list(Formatter().parse(text))
                fields = {name for _, name, _, _ in parsed if name is not None}
            except (ValueError, KeyError, IndexError, AttributeError) as exc:
                raise ValueError(f"{spec['variable']}: invalid template fields") from exc
            if (fields != expected or any(fmt or conversion for _, _, fmt, conversion in parsed)):
                raise ValueError(f"{spec['variable']}: retain {', '.join(sorted(expected))}")
        if key == "router" and not all(token in text for token in ("是", "否")):
            raise ValueError("ROUTER_PROMPT: retain the 是 / 否 output protocol")
        if key in {"self_harness", "tone_rule"}:
            required = ("<self_harness>", "</self_harness>")
            if not all(token in text for token in required):
                raise ValueError(f"{spec['variable']}: retain the Self Harness control protocol")
        if text != spec["default"]:
            result[key] = text
    return result


def effective_prompts(snapshot):
    """Resolve localized defaults and verbatim edits from one captured turn."""
    snapshot = snapshot or {}
    defaults = {}
    if snapshot.get("prompt_language") == "en":
        from studio.harness.prompts_en import DEFAULTS
        defaults = DEFAULTS
    return {**defaults, **snapshot.get("prompt_overrides", {})}


def resolve(name, fallback, overrides=None):
    """Resolve a prompt from a captured snapshot or the current task scope."""
    selected = _CURRENT.get() if overrides is None else overrides
    return selected.get(name, fallback) if selected else fallback


@contextmanager
def prompt_scope(overrides):
    """Keep concurrent conversations and background model work isolated."""
    token = _CURRENT.set(dict(overrides or {}))
    try:
        yield
    finally:
        _CURRENT.reset(token)
