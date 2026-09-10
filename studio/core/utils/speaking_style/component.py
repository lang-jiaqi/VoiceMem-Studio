"""Render a single speaking policy without prompt-language branches."""
from studio.harness.speaking_style.policy import CONTEXT, TONE_RULE, PROMPT, EMOTION_NOTE
import re
from studio.harness.speaking_style.policy import QWEN36_TTS, QWEN36_INTRO_BRIGHT, QWEN36_INTRO_SOFT

def is_qwen36(reply):
    if not isinstance(reply, dict):
        return False
    return str((reply.get("llm", reply).get("config") or {}).get("model", "")).startswith("qwen3.6")

def qwen_segment_instruction(base, user_text, prefix, qwen=True):
    """Apply a segment-local voice arc without adding spoken control tokens."""
    intro = re.search(r"你是谁|你是什么|介绍.{0,8}(?:自己|你)|自我介绍", user_text or "")
    if intro:
        from studio.core.utils.tts.control import instruction
        quiet = any(word in prefix for word in ("如果说", "没有身体", "做不到"))
        return instruction("共情" if quiet else "轻快") + (QWEN36_INTRO_SOFT if quiet else QWEN36_INTRO_BRIGHT)
    return base + QWEN36_TTS if qwen else base

def prompt_rule(lang=None):
    return PROMPT

def content_emotion_note(emotion, lang=None):
    emotion = (emotion or "").strip()
    return EMOTION_NOTE.format(emotion=emotion) if emotion else ""
