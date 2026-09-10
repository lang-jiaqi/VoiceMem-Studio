"""Studio control implementation."""
from __future__ import annotations

import re
from voicemem.prompt_config import read_prompt, tts_prompts

TONES: dict[str, str] = tts_prompts()["tones"]
DEFAULT = "平静"

#:

_T = "温和|共情|轻快|认真|鼓励|俏皮|抱歉|平静"
_TAG = re.compile(
    rf"^\s*(?:"
    rf"([^\n|｜]{{1,24}})[|｜]\s*"
    rf"|[\[【]\s*({_T})\s*[\]】]\s*"
    rf"|({_T})[\s:：]+"
    rf")")

_ALIAS: dict[str, str] = {
    "gentle": "温和", "warm": "温和", "soft": "温和", "caring": "温和",
    "empath": "共情", "compassion": "共情", "understanding": "共情",
    "light": "轻快", "breezy": "轻快", "cheerful": "轻快", "upbeat": "轻快",
    "serious": "认真", "steady": "认真", "matter-of-fact": "认真",
    "encourag": "鼓励", "supportive": "鼓励",
    "playful": "俏皮", "teasing": "俏皮",
    "sorry": "抱歉", "apologetic": "抱歉", "regret": "抱歉",
    "calm": "平静", "neutral": "平静", "plain": "平静",
}

def _from_head(head: str) -> str:
    tag = next((t for t in TONES if t in head), "")
    if tag:
        return tag
    low = (head or "").lower()
    return next((t for k, t in _ALIAS.items() if k in low), DEFAULT)

MAX_STEP = 0.35

_XY: dict[str, tuple[float, float]] = {
    "平静": (0.35, 0.50), "认真": (0.40, 0.45), "温和": (0.35, 0.65),
    "共情": (0.25, 0.55), "抱歉": (0.25, 0.30), "鼓励": (0.70, 0.75),
    "轻快": (0.70, 0.80), "俏皮": (0.80, 0.85),
}

def smooth(prev: str, tag: str, max_step: float = MAX_STEP) -> str:
    """Bound tone movement relative to the preceding tone."""
    if not prev or prev == tag or prev not in _XY or tag not in _XY:
        return tag or DEFAULT
    (x0, y0), (x1, y1) = _XY[prev], _XY[tag]
    dx, dy = x1 - x0, y1 - y0
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= max_step:
        return tag
    k = max_step / dist
    tx, ty = x0 + dx * k, y0 + dy * k
    return min(_XY, key=lambda t: (_XY[t][0] - tx) ** 2 + (_XY[t][1] - ty) ** 2)

def split(text: str) -> tuple[str, str]:
    """Return the leading tone and plain text, stripping recognized control syntax."""
    m = _TAG.match(text or "")
    if not m:
        return "", text
    if m.group(1) is not None:
        return _from_head(m.group(1)), text[m.end():]
    tag = (m.group(2) or m.group(3) or "").strip()
    return (tag, text[m.end():]) if tag in TONES else ("", text)

def instruction(tag: str, base: str = "") -> str:
    """Render a tone instruction, using the default tone for unknown tags."""
    tone = TONES.get(tag) or TONES[DEFAULT]
    return f"{base}{tone}" if base else tone

def prompt_rule(lang: str = "zh") -> str:
    """Actual voice-control protocol, editable in root prompt/."""
    lang = "en" if str(lang).lower().startswith("en") else "zh"
    return read_prompt(f"llm_tone_rule_{lang}.md") + "\n"
