"""Initialize the local three-way router once per process."""
from functools import lru_cache
from studio.harness.reply_modes.policy import AVAILABLE_MODES, DIRECT, MEMORY, MEMORY_COT
from .component import FAST, MEDIUM, SLOW, THINKING_LEVELS, QwenThinkingRouter, ThinkingDecision, parse_level

@lru_cache(maxsize=1)
def thinking_router():
    """Return the process router with fixed local model selection."""
    return QwenThinkingRouter()
