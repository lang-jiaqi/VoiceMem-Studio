"""Editable prompts in root prompt/. Restart after editing; no audio-loop I/O.

Wheel installs ship the same files under sys.prefix/prompt (data-files).
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def prompt_directory():
    explicit = os.environ.get("VOICEMEM_PROMPT_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    source = Path(__file__).resolve().parents[1] / "prompt"
    return source if source.is_dir() else Path(sys.prefix) / "prompt"


@lru_cache(maxsize=None)
def read_prompt(name):
    path = prompt_directory() / name
    try:
        return path.read_text(encoding="utf-8").rstrip("\r\n")
    except OSError as exc:
        raise RuntimeError(f"无法读取生效的 prompt 文件 {path}：{exc}") from exc


@lru_cache(maxsize=None)
def _json(name):
    try:
        value = json.loads(read_prompt(name))
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompt 配置格式错误：{prompt_directory() / name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"prompt/{name} 顶层必须是 JSON 对象")
    return value


def _strings(value, where, required=None):
    if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
        raise ValueError(f"{where} 必须是 字符串→字符串 的 JSON 对象")
    if required and not set(required).issubset(value):
        raise ValueError(f"{where} 缺少字段：{sorted(set(required) - value.keys())}")


@lru_cache(maxsize=1)
def context_prompts():
    cfg = _json("llm_context.json")
    for key in ("stranger", "no_memory", "language", "replay", "no_replay", "state_label"):
        _strings(cfg.get(key), f"prompt/llm_context.json:{key}", ("zh", "en"))
    return cfg


@lru_cache(maxsize=1)
def tts_prompts():
    cfg = _json("tts.json")
    _strings(cfg.get("base"), "prompt/tts.json:base", ("zh", "en"))
    labels = {"温和", "共情", "轻快", "认真", "鼓励", "俏皮", "抱歉", "平静"}
    _strings(cfg.get("tones"), "prompt/tts.json:tones", labels)
    if set(cfg["tones"]) != labels:
        raise ValueError("prompt/tts.json:tones 请保留原有八个标签，只修改提示词正文")
    fallback = cfg.get("fallback_by_user_emotion")
    if not isinstance(fallback, dict):
        raise ValueError("prompt/tts.json:fallback_by_user_emotion 必须是对象")
    for lang in ("zh", "en"):
        _strings(fallback.get(lang), f"prompt/tts.json:fallback_by_user_emotion.{lang}")
    if not isinstance(cfg.get("breeze_default_instruction"), str):
        raise ValueError("prompt/tts.json:breeze_default_instruction 必须是字符串")
    styles = cfg.get("backchannel_styles")
    if not isinstance(styles, dict):
        raise ValueError("prompt/tts.json:backchannel_styles 必须是对象")
    for lang in ("zh", "en"):
        items = styles.get(lang)
        if not isinstance(items, list) or not items or not all(isinstance(s, str) for s in items):
            raise ValueError(f"prompt/tts.json:backchannel_styles.{lang} 必须是非空字符串列表")
    return cfg
