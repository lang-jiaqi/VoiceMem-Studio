"""Render a single speaking policy without prompt-language branches."""
import re
from studio.harness.speaking_style.policy import (
    CONTEXT, TONE_RULE, PROMPT, EMOTION_NOTE,
    QWEN36_TTS, QWEN36_INTRO_BRIGHT, QWEN36_INTRO_SOFT,
)

def is_qwen36(reply):
    if not isinstance(reply, dict):
        return False
    return str((reply.get("llm", reply).get("config") or {}).get("model", "")).startswith("qwen3.6")

def qwen_segment_instruction(base, user_text, prefix, qwen=True, language="zh"):
    """Apply a segment-local voice arc without adding spoken control tokens."""
    intro = re.search(r"你是谁|你是什么|介绍.{0,8}(?:自己|你)|自我介绍|who are you|introduce yourself", user_text or "", re.I)
    if intro:
        from studio.core.utils.tts.control import instruction
        quiet = any(word in prefix.lower() for word in ("如果说", "没有身体", "做不到", "no body", "don't have a body", "cannot accompany", "can't accompany"))
        hint = (" Let the ending soften naturally, without sounding sad." if quiet else
                " Sound bright and proud, with natural conversational energy.") if language == "en" else (QWEN36_INTRO_SOFT if quiet else QWEN36_INTRO_BRIGHT)
        return instruction("共情" if quiet else "轻快", language=language) + hint
    hint = " Speak naturally and smoothly; avoid exaggerated pitch changes or a presenter voice." if language == "en" else QWEN36_TTS
    return base + hint if qwen else base

def prompt_rule(lang=None):
    return PROMPT

def content_emotion_note(emotion, lang=None):
    emotion = (emotion or "").strip()
    if emotion and lang == "en":
        return f"Internal acoustic emotion hint: {emotion}. Use only when consistent with the content; never mention this hint."
    return EMOTION_NOTE.format(emotion=emotion) if emotion else ""
