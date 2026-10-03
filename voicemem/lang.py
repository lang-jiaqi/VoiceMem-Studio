"""Chinese/English memory defaults and isolated operation language.

Stored Space language supplies a fallback, not a restriction on factual text.
Studio follows each input for generated descriptions, while canonical slot and
emotion keys retain their existing values. Request scopes and explicit context
capture prevent concurrent Spaces or deferred workers from changing one another.
The process setter remains available for legacy callers and startup defaults.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps

# Default language for callers without an instance or request scope.
ENV = "VOICEMEM_MEMORY_LANGUAGE"
SUPPORTED = ("en", "zh")
DEFAULT = "en"

_override: str | None = None
_request_language: ContextVar[str | None] = ContextVar("voicemem_language", default=None)


def detect_language(text: str, fallback: str = DEFAULT, *, keep_short: bool = True) -> str:
    """Infer the main Chinese/English prose language, retaining ambiguous short inputs."""
    prose = re.sub(r"```[\s\S]*?(?:```|$)|`[^`]*`|https?://\S+", " ", text or "")
    prose = re.sub(r"\$\$[\s\S]*?\$\$|\$[^$\n]*\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)", " ", prose)
    han = len(re.findall(r"[\u3400-\u9fff]", prose))
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", prose.lower())
    if not keep_short:
        if han and not words:
            return "zh"
        if words and not han:
            return "en"
    signals = {"i", "you", "my", "your", "we", "he", "she", "they", "it", "the", "a", "an",
               "is", "are", "was", "were", "do", "does", "can", "could", "would", "will",
               "what", "how", "why", "when", "where", "which", "who", "please", "tell",
               "explain", "remember", "about", "and", "but", "because", "have", "has"}
    signals.update({"i'm", "you're", "it's", "that's", "i've", "we're", "don't"})
    evidence = signals.intersection(words)
    if han:
        evidence -= {"a", "an", "and", "but", "because", "about"}
    english = len(words) >= 2 and bool(evidence)
    if han >= 2:
        return "en" if english and len(words) >= han else "zh"
    if english or (not han and len(words) >= 3) or (not han and prose.strip().lower().rstrip(".!?") in
                   {"hello", "hi", "good morning", "good evening", "thank you"}):
        return "en"
    return _check(fallback)


@contextmanager
def language_scope(language: str):
    """Bind memory prompt language to one task/thread and restore it on exit."""
    token = _request_language.set(_check(language))
    try:
        yield
    finally:
        _request_language.reset(token)


def contextualize(function):
    """Capture the caller's context for one subsequent worker invocation."""
    context = copy_context()
    return lambda *args, **kwargs: context.run(function, *args, **kwargs)


def scoped_operation(function):
    """Use the originating memory instance and input language for an operation."""
    @wraps(function)
    def run(self, *args, **kwargs):
        fallback = _request_language.get() or getattr(self, "memory_language", memory_language())
        language = kwargs.get("language") or fallback
        if not kwargs.get("language") and _request_language.get() is None and getattr(self, "follow_input_language", False):
            text = args[0] if args else kwargs.get("text", kwargs.get("query", ""))
            language = detect_language(text, fallback)
        with language_scope(language):
            return function(self, *args, **kwargs)
    return run


def _check(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in SUPPORTED:
        raise ValueError(f"memory_language 只能是 {' / '.join(SUPPORTED)}，收到 {value!r}")
    return v


def _set(lang: str) -> None:
    global _override
    _override = lang


def set_memory_language(value: str | None) -> None:
    """进程级设置。None / "" 表示清掉覆盖，回到 env / 默认。"""
    global _override
    _override = _check(value) if (value or "").strip() else None


def memory_language() -> str:
    if scoped := _request_language.get():
        return scoped
    if _override:
        return _override
    env = (os.environ.get(ENV, "") or "").strip()
    return _check(env) if env else DEFAULT


def resolve_for_space(memory_root, explicit: str | None = None) -> str:
    """Persist an instance fallback without mutating process or request language."""
    import json as _json
    from voicemem.utils.common import space as _space
    try:
        path = _space.json_path(memory_root)
    except Exception:
        return _check(explicit) if explicit else memory_language()

    stored = ""
    try:
        if path.exists():
            stored = (_json.loads(path.read_text(encoding="utf-8"))
                      .get("space", {}).get("language", "") or "")
    except Exception:
        stored = ""

    if explicit:
        lang = _check(explicit)
    elif stored:
        lang = _check(stored)
    else:
        env = (os.environ.get(ENV, "") or "").strip()
        lang = _check(env) if env else DEFAULT

    if lang != stored:
        try:
            doc = (_json.loads(path.read_text(encoding="utf-8"))
                   if path.exists() else {})
            doc.setdefault("space", {})["language"] = lang
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        except Exception as e:
            print(f"[lang] 写空间语言失败（不影响使用）：{e}", flush=True)

    return lang


def is_zh() -> bool:
    return memory_language() == "zh"


def label_rule() -> str:
    """Select description language from the current operation's captured scope."""
    lang = "Chinese" if is_zh() else "English"
    return (f"Write every label in {lang}, whatever language the speaker used. "
            f"Do not mix in any other language.")


#: 8 个规范情绪（内部值一律中文，见 anchor_router._CANONICAL_EMOTIONS）→ 展示词。
#:
#: 内部值不能翻译：右脑的锚点匹配、配额、脑图聚类都按它做键。但**存进记忆、给
#: 用户看的那一份**要跟库语言一致，否则英文库里会冒出「开心」「平静」。
#: 这里挑的英文词都在 anchor_router._EMOTION_KEYWORDS_EN 里，所以英文标签再被
#: 读回来时能正确归一回同一个内部值，不会丢。
_EMOTION_EN = {
    "焦虑": "anxious", "悲伤": "sad", "委屈": "wronged", "孤独": "lonely",
    "纠结": "conflicted", "平静": "calm", "开心": "happy", "疲惫": "tired",
}


def display_emotion(canonical: str) -> str:
    """规范情绪 → 当前库语言下的写法。不认识的原样返回。"""
    if is_zh():
        return canonical
    return _EMOTION_EN.get(canonical, canonical)
