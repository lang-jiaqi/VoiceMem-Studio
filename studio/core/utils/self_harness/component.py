"""Validate and apply the conversation-scoped Self Harness overlay."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

from studio.harness.self_harness.policy import (
    MAX_CHANGES_PER_TURN,
    PROFILE_SCHEMA,
    RECENT_CHANGE_TURNS,
)

CONTROL_OPEN = "<self_harness>"
CONTROL_CLOSE = "</self_harness>"
PROMPT_SCHEMA = {
    "persona": {"default": "", "max_length": 2000},
}


@dataclass(frozen=True)
class ControlPrefix:
    """A private model control header and the remaining spoken output."""

    resolved: bool
    update: dict[str, dict[str, str]]
    rest: str
    error: str = ""


def default_profile() -> dict[str, dict[str, str]]:
    """Return a fresh profile containing only immutable-policy defaults."""
    return {
        domain: {name: spec["default"] for name, spec in fields.items()}
        for domain, fields in PROFILE_SCHEMA.items()
    }


def public_schema() -> dict[str, dict[str, dict]]:
    """Return the user-facing enum metadata without private prompt instructions."""
    return {
        domain: {
            name: {
                "default": spec["default"],
                "values": list(spec["values"]),
                "labels": dict(spec["labels"]),
            }
            for name, spec in fields.items()
        }
        for domain, fields in PROFILE_SCHEMA.items()
    }


def public_prompt_schema() -> dict[str, dict]:
    """Return the explicit, session-only prompt fields exposed by settings."""
    return {name: dict(spec) for name, spec in PROMPT_SCHEMA.items()}


def _backchannel_profile_curve(mode: str) -> list[float]:
    from studio.harness.turn_taking.policy import SessionFrequencyCurve
    curve = SessionFrequencyCurve()
    distributions = (
        curve.opening_counts, curve.middle_counts,
        curve.steady_counts, curve.late_counts,
    )
    values = []
    for distribution in distributions:
        expected = 0.0
        for count, probability in distribution:
            if mode == "off":
                selected = 0
            elif mode == "less":
                selected = min(count, 1)
            elif mode == "more":
                selected = min(3, max(1, count + 1))
            else:
                selected = count
            expected += selected * probability
        values.append(round(expected, 1))
    return values


def public_backchannel_curve_schema() -> dict:
    """Return the four real quota phases used by the curve editor."""
    return {
        "min": 0.0,
        "max": 3.0,
        "step": 0.1,
        "phases": [
            {"key": "opening", "label": "0–3s"},
            {"key": "middle", "label": "3–6s"},
            {"key": "steady", "label": "6–10s"},
            {"key": "late", "label": "10s+"},
        ],
        "profiles": {
            mode: _backchannel_profile_curve(mode)
            for mode in ("auto", "off", "less", "more")
        },
        "early_cap": 2,
    }


def validate_backchannel_curve(value) -> list[float] | None:
    """Validate an explicit four-phase expected-quota curve."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("backchannel curve must contain four phase quotas")
    result = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("backchannel curve quotas must be numbers")
        selected = float(raw)
        if not math.isfinite(selected) or not 0.0 <= selected <= 3.0:
            raise ValueError("backchannel curve quotas must be between 0 and 3")
        result.append(round(selected, 1))
    return result


def normalize_profile(value=None) -> dict[str, dict[str, str]]:
    """Copy valid values onto a fresh default profile."""
    profile = default_profile()
    if not isinstance(value, dict):
        return profile
    for domain, fields in value.items():
        domain_spec = PROFILE_SCHEMA.get(domain)
        if not domain_spec or not isinstance(fields, dict):
            continue
        for name, selected in fields.items():
            spec = domain_spec.get(name)
            if spec and selected in spec["values"]:
                profile[domain][name] = selected
    return profile


def validate_update(value) -> dict[str, dict[str, str]]:
    """Reject unknown structure, invalid enum values, and oversized changes."""
    if not isinstance(value, dict):
        raise ValueError("Self Harness control must be a JSON object")
    update: dict[str, dict[str, str]] = {}
    changes = 0
    for domain, fields in value.items():
        domain_spec = PROFILE_SCHEMA.get(domain)
        if domain_spec is None:
            raise ValueError(f"unknown Self Harness domain: {domain}")
        if not isinstance(fields, dict):
            raise ValueError(f"Self Harness domain must be an object: {domain}")
        selected_fields = {}
        for name, selected in fields.items():
            spec = domain_spec.get(name)
            if spec is None:
                raise ValueError(f"unknown Self Harness field: {domain}.{name}")
            if selected not in spec["values"]:
                raise ValueError(f"invalid {domain}.{name}: {selected}")
            selected_fields[name] = selected
            changes += 1
        if selected_fields:
            update[domain] = selected_fields
    if changes > MAX_CHANGES_PER_TURN:
        raise ValueError(
            f"Self Harness permits at most {MAX_CHANGES_PER_TURN} changes per turn")
    return update


def merge_update(profile, update) -> dict[str, dict[str, str]]:
    """Apply an already validated update to a profile in place."""
    for domain, fields in update.items():
        profile[domain].update(fields)
    return profile


def split_control_prefix(value: str, *, final: bool = False) -> ControlPrefix:
    """Parse one streaming control header without exposing malformed metadata."""
    text = value or ""
    candidate = text.lstrip()
    if not candidate:
        if not final:
            return ControlPrefix(False, {}, "")
        return ControlPrefix(True, {}, text)
    if candidate and CONTROL_OPEN.startswith(candidate) and candidate != CONTROL_OPEN:
        if not final:
            return ControlPrefix(False, {}, "")
        return ControlPrefix(True, {}, "", "unterminated Self Harness control")
    if not candidate.startswith(CONTROL_OPEN):
        return ControlPrefix(True, {}, text)
    end = candidate.find(CONTROL_CLOSE, len(CONTROL_OPEN))
    if end < 0:
        if not final:
            return ControlPrefix(False, {}, "")
        return ControlPrefix(True, {}, "", "unterminated Self Harness control")
    raw = candidate[len(CONTROL_OPEN):end]
    rest = candidate[end + len(CONTROL_CLOSE):]
    try:
        update = validate_update(json.loads(raw))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return ControlPrefix(True, {}, rest, str(exc))
    return ControlPrefix(True, update, rest)


def _snapshot_parts(value):
    if isinstance(value, dict) and isinstance(value.get("profile"), dict):
        return normalize_profile(value["profile"]), tuple(value.get("recent") or ())
    return normalize_profile(value), ()


def normalize_snapshot(value=None) -> dict:
    """Return a detached snapshot suitable for one in-flight reply."""
    profile, recent = _snapshot_parts(value)
    pending = value.get("pending", {}) if isinstance(value, dict) else {}
    prompts = value.get("prompts", {}) if isinstance(value, dict) else {}
    return {
        "profile": profile,
        "recent": list(recent),
        "pending": dict(pending) if isinstance(pending, dict) else {},
        "prompts": {
            name: str(prompts.get(name, spec["default"]))[:spec["max_length"]]
            if isinstance(prompts, dict) else spec["default"]
            for name, spec in PROMPT_SCHEMA.items()
        },
        "backchannel_curve": (
            list(value.get("backchannel_curve"))
            if isinstance(value, dict)
            and isinstance(value.get("backchannel_curve"), (list, tuple))
            else None
        ),
    }


def profile_context(value=None) -> str:
    """Render current values and conservative stability hints for the reply model."""
    profile, recent = _snapshot_parts(value)
    pending = value.get("pending", {}) if isinstance(value, dict) else {}
    prompts = value.get("prompts", {}) if isinstance(value, dict) else {}
    rows = []
    overlays = []
    for domain, fields in profile.items():
        for name, selected in fields.items():
            spec = PROFILE_SCHEMA[domain][name]
            rows.append(f"- {domain}.{name}: {selected} ({spec['labels'][selected]})")
            prompt = spec.get("prompt", {}).get(selected, "")
            if prompt:
                overlays.append(f"- {prompt}")
    length = profile["speaking_style"]["reply_length"]
    length_note = {
        "concise": "后续回复尽量简短；安全性或正确性需要时可以补充必要细节。",
        "auto": "根据当前对话需要决定回复长度。",
        "detailed": "后续回复提供更多解释和有用细节。",
    }[length]
    parts = [
        "【当前 Self Harness（内部状态，不要念出）】",
        "以下是原始默认策略之上的会话差量；本轮正文和语音必须遵守。",
        *rows,
        length_note,
    ]
    try:
        backchannel_curve = validate_backchannel_curve(
            value.get("backchannel_curve") if isinstance(value, dict) else None)
    except ValueError:
        backchannel_curve = None
    if backchannel_curve is not None:
        parts.append(
            "用户已在设置中明确指定四阶段垫话配额："
            + " / ".join(f"{quota:.1f}" for quota in backchannel_curve)
            + "；不要主动覆盖。")
    if overlays:
        parts.extend(("当前行为覆盖：", *overlays))
    if recent:
        parts.append(
            "最近两轮刚改过、再次冲突时需要用户二次确认：" + "、".join(recent))
    if pending:
        parts.append("等待用户再次确认的值：" + "、".join(
            f"{path}={selected}" for path, selected in sorted(pending.items())))
    persona_prompt = str(prompts.get("persona", "")).strip() if isinstance(prompts, dict) else ""
    if persona_prompt:
        parts.extend((
            "【用户设置的当前对话 Persona 补充】",
            "在不覆盖系统安全边界和真实性要求的前提下遵守：",
            persona_prompt,
        ))
    return "\n".join(parts)


def speech_rate_instruction(base: str, value=None) -> str:
    profile, _ = _snapshot_parts(value)
    selected = profile["speaking_style"]["speech_rate"]
    instruction = PROFILE_SCHEMA["speaking_style"]["speech_rate"]["tts"][selected]
    return f"{base}{instruction}" if instruction else base


def fixed_tone(value=None) -> str:
    profile, _ = _snapshot_parts(value)
    tone = profile["speaking_style"]["tone"]
    return "" if tone == "auto" else tone


def reasoning_preference(value=None) -> str:
    profile, _ = _snapshot_parts(value)
    return profile["reply_modes"]["reasoning_depth"]


class SelfHarnessState:
    """Own one conversation's typed overlay and recent-change window."""

    def __init__(self, on_change=None):
        self.profile = default_profile()
        self.prompts = {
            name: spec["default"] for name, spec in PROMPT_SCHEMA.items()
        }
        self.backchannel_curve: list[float] | None = None
        self.turn = 0
        self.last_changed: dict[str, int] = {}
        self.pending: dict[str, tuple[str, int]] = {}
        self.on_change = on_change

    def snapshot(self) -> dict:
        recent = sorted(
            path for path, changed_at in self.last_changed.items()
            if self.turn - changed_at < RECENT_CHANGE_TURNS)
        return {
            "profile": {domain: dict(fields)
                        for domain, fields in self.profile.items()},
            "recent": recent,
            "pending": {path: selected
                        for path, (selected, _) in self.pending.items()},
            "prompts": dict(self.prompts),
            "backchannel_curve": (
                list(self.backchannel_curve)
                if self.backchannel_curve is not None else None),
        }

    def preview(self, value) -> dict[str, dict[str, str]]:
        """Return the part that would activate now, without changing state."""
        update = validate_update(value)
        effective: dict[str, dict[str, str]] = {}
        for domain, fields in update.items():
            for name, selected in fields.items():
                path = f"{domain}.{name}"
                if self.profile[domain][name] == selected:
                    continue
                changed_at = self.last_changed.get(path)
                recently_changed = (
                    changed_at is not None
                    and self.turn - changed_at < RECENT_CHANGE_TURNS)
                pending = self.pending.get(path)
                if pending is not None and pending[0] != selected:
                    continue
                if recently_changed and pending is None:
                    continue
                effective.setdefault(domain, {})[name] = selected
        return effective

    def apply(self, value) -> dict[str, dict[str, str]]:
        """Advance one reply, staging rapid conflicts until they are repeated."""
        update = validate_update(value)
        effective = self.preview(update)
        next_turn = self.turn + 1
        for domain, fields in update.items():
            for name, selected in fields.items():
                path = f"{domain}.{name}"
                if self.profile[domain][name] == selected:
                    self.pending.pop(path, None)
                    if path == "turn_taking.backchannel":
                        self.backchannel_curve = None
                elif name in effective.get(domain, {}):
                    self.profile[domain][name] = selected
                    self.last_changed[path] = next_turn
                    self.pending.pop(path, None)
                    if path == "turn_taking.backchannel":
                        self.backchannel_curve = None
                else:
                    self.pending[path] = (selected, next_turn)
        self.turn = next_turn
        self.pending = {
            path: pending for path, pending in self.pending.items()
            if self.turn - pending[1] < RECENT_CHANGE_TURNS
        }
        if self.on_change is not None:
            self.on_change(self.snapshot())
        return effective

    def set_explicit(self, value) -> dict[str, dict[str, str]]:
        """Apply an explicit UI choice immediately while retaining model hysteresis."""
        update = validate_update(value)
        for domain, fields in update.items():
            for name, selected in fields.items():
                path = f"{domain}.{name}"
                self.profile[domain][name] = selected
                self.last_changed[path] = self.turn
                self.pending.pop(path, None)
                if path == "turn_taking.backchannel":
                    self.backchannel_curve = None
        if self.on_change is not None:
            self.on_change(self.snapshot())
        return update

    def set_prompt(self, name: str, value: str) -> str:
        """Set one explicit user prompt for this conversation only."""
        spec = PROMPT_SCHEMA.get(name)
        if spec is None:
            raise ValueError(f"unknown Harness prompt: {name}")
        if not isinstance(value, str):
            raise ValueError("Harness prompt must be text")
        value = value.strip()
        if len(value) > spec["max_length"]:
            raise ValueError(
                f"Harness prompt exceeds {spec['max_length']} characters")
        self.prompts[name] = value
        if self.on_change is not None:
            self.on_change(self.snapshot())
        return value

    def set_backchannel_curve(self, value) -> list[float] | None:
        """Apply or reset the explicit phase quotas for this conversation."""
        selected = validate_backchannel_curve(value)
        self.backchannel_curve = selected
        self.profile["turn_taking"]["backchannel"] = "auto"
        self.pending.pop("turn_taking.backchannel", None)
        self.last_changed["turn_taking.backchannel"] = self.turn
        if self.on_change is not None:
            self.on_change(self.snapshot())
        return list(selected) if selected is not None else None


class StagedSelfHarnessUpdate:
    """Commit a speculative reply's profile observation at most once."""

    def __init__(self, target: SelfHarnessState):
        self.target = target
        self.update = None
        self.committed = False
        self.applied = False

    def stage(self, update) -> dict[str, dict[str, str]]:
        self.update = validate_update(update)
        if self.committed:
            return self._apply_if_ready()
        return self.target.preview(self.update)

    def commit(self) -> dict[str, dict[str, str]]:
        self.committed = True
        return self._apply_if_ready()

    def _apply_if_ready(self) -> dict[str, dict[str, str]]:
        if self.committed and self.update is not None and not self.applied:
            self.applied = True
            return self.target.apply(self.update)
        return {}


def apply_turn_taking_profile(machine, value=None) -> None:
    """Apply turn-taking overlays without mutating global policy defaults."""
    profile, _ = _snapshot_parts(value)
    backchannel = profile["turn_taking"]["backchannel"]
    machine.backchannel.enabled = backchannel != "off"
    machine.backchannel.frequency = backchannel
    custom_curve = value.get("backchannel_curve") if isinstance(value, dict) else None
    machine.backchannel.custom_phase_quotas = (
        tuple(validate_backchannel_curve(custom_curve))
        if custom_curve is not None else None)
    filler = profile["turn_taking"]["work_filler"]
    baseline = getattr(machine, "_default_work_filler_probability",
                       machine.work_filler_probability)
    machine.work_filler_probability = {
        "auto": baseline,
        "silent": 0.0,
        "reassuring": max(0.75, baseline),
    }[filler]
