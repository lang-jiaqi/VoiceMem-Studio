"""LLM-facing reply-shape and content-emotion guidance."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=2)
def prompt_rule(lang: str = "zh") -> str:
    """Return the stable speaking-style system prompt for a language."""
    code = "en" if str(lang).lower().startswith("en") else "zh"
    path = Path(__file__).with_name(f"prompt_{code}.md")
    return path.read_text(encoding="utf-8").strip()


def content_emotion_note(emotion: str, lang: str = "zh") -> str:
    """Render a private, weak emotional signal for the reply text model."""
    emotion = (emotion or "").strip()
    if not emotion:
        return ""
    if str(lang).lower().startswith("en"):
        return (
            f"Private emotional context (weak signal): the recent detected state is {emotion}. "
            "Let it affect what you acknowledge, wording, and the emotional contour of the "
            "reply. Prefer the user's current words and conversation context if they disagree. "
            "Never mention this label or the detection process."
        )
    return (
        f"内容情绪参考（弱信号，不要念出）：最近检测到的状态是“{emotion}”。"
        "让它影响先接住什么、如何措辞，以及回复内容自身的情绪弧线；如果与用户本轮"
        "文字或上下文冲突，以文字和上下文为准。绝不说出这个标签或检测过程。"
    )


__all__ = ["content_emotion_note", "prompt_rule"]
