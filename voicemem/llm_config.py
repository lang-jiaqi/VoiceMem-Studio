"""Resolve process-wide model roles and OpenAI-compatible endpoint settings.

Model precedence: explicit parameter, MODELS override, VOICEMEM role environment
variable, legacy OPENAI role variable, then the role default. Values are resolved
at access time. MODELS overrides are shared by all instances in this process.
OPENAI_API_KEY and OPENAI_BASE_URL retain their existing protocol-level meaning."""

from __future__ import annotations

import os
from dataclasses import dataclass

# A None role default inherits the chat model.
_FOLLOW_CHAT = None
# Distinguish an omitted fallback from an explicit None fallback.
_UNSET = object()


def env_name(role: str) -> str:
    """Return the VOICEMEM environment variable name for the model role."""
    return f"VOICEMEM_{role.upper()}_MODEL"


@dataclass(frozen=True)
class Role:
    default: str | None
    # Keep the existing environment variable names for each model role.
    legacy: str = ""


ROLES: dict[str, Role] = {
    "chat":      Role("gpt-4o-mini",            legacy="OPENAI_MODEL"),
    "reply":     Role(_FOLLOW_CHAT,             legacy="OPENAI_CHAT_MODEL"),
    "embedding": Role("text-embedding-3-small", legacy="OPENAI_EMBEDDING_MODEL"),
    "tts":       Role("gpt-4o-mini-tts",        legacy="OPENAI_TTS_MODEL"),
    "realtime":  Role("gpt-realtime",           legacy="OPENAI_REALTIME_MODEL"),
}


class Models:
    """Process-wide role overrides with effective values resolved on each access."""

    def __init__(self) -> None:
        self._over: dict[str, str] = {}

    def update(self, mapping: dict[str, str] | None = None, **kw: str) -> "Models":
        """Set role overrides; reject unknown roles rather than silently ignoring them."""
        for role, name in {**(mapping or {}), **kw}.items():
            if role not in ROLES:
                raise ValueError(f"未知的模型角色 {role!r}，可选：{', '.join(ROLES)}")
            if not name:
                continue
            name = str(name).strip()
            prev = self._over.get(role)
            if prev and prev != name:
                print(f"[models] {role}: {prev} → {name}。模型表是进程级的，"
                      f"这次覆盖对已经建好的 VoiceMem 同样生效。", flush=True)
            self._over[role] = name
        return self

    def get(self, role: str = "chat", explicit: str | None = None,
            default: str | None = _UNSET) -> str | None:
        """Resolve the role, using an explicit default when supplied."""
        if role not in ROLES:
            raise ValueError(f"未知的模型角色 {role!r}，可选：{', '.join(ROLES)}")
        spec = ROLES[role]
        name = (explicit or self._over.get(role)
                or os.environ.get(env_name(role), "").strip()
                or (os.environ.get(spec.legacy, "").strip() if spec.legacy else ""))
        if name:
            return name.strip()
        if default is not _UNSET:
            return default
        return spec.default if spec.default is not _FOLLOW_CHAT else self.get("chat")

    def as_dict(self) -> dict[str, str]:
        """Return the currently effective role-to-model mapping."""
        return {role: self.get(role) for role in ROLES}

    def explain(self) -> str:
        """Describe effective role models and their supported environment variable names."""
        return "\n".join(f"{role:<10} {self.get(role):<24} {env_name(role)}"
                          for role in ROLES)

    chat = property(lambda self: self.get("chat"))
    reply = property(lambda self: self.get("reply"))
    embedding = property(lambda self: self.get("embedding"))
    tts = property(lambda self: self.get("tts"))
    realtime = property(lambda self: self.get("realtime"))


# One registry owns process-wide model overrides for all VoiceMem instances.
MODELS = Models()


def resolve_model(explicit: str | None = None, role: str = "chat",
                  default: str | None = _UNSET) -> str | None:
    """Resolve an explicit model, then override, environment and role-default values."""
    return MODELS.get(role, explicit, default)


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the explicit key or OPENAI_API_KEY; return None if absent."""
    return explicit or os.environ.get("OPENAI_API_KEY")


def resolve_base_url(explicit: str | None = None) -> str | None:
    """Resolve the explicit URL or OPENAI_BASE_URL; None keeps the SDK default."""
    return explicit or os.environ.get("OPENAI_BASE_URL") or None


# Preserve the original public default name.
CHAT_MODEL_DEFAULT = ROLES["chat"].default
