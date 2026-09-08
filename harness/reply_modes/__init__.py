"""Stable contracts for selecting how a Studio reply is produced."""

MEMORY_COT = "memory_cot"
MEMORY = "memory"
DIRECT = "direct"

AVAILABLE_MODES = (MEMORY_COT, MEMORY, DIRECT)

from .thinking import (
    FAST,
    MEDIUM,
    SLOW,
    THINKING_LEVELS,
    QwenThinkingRouter,
    ThinkingDecision,
    parse_level,
    thinking_router,
)

__all__ = [
    "AVAILABLE_MODES",
    "DIRECT",
    "FAST",
    "MEDIUM",
    "MEMORY",
    "MEMORY_COT",
    "SLOW",
    "THINKING_LEVELS",
    "QwenThinkingRouter",
    "ThinkingDecision",
    "parse_level",
    "thinking_router",
]
