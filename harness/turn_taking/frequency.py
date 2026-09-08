"""Turn-count probability curve for within-speech backchannels."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionFrequencyCurve:
    """Return the baseline chance for one eligible pause in a session."""

    opening_turns: int = 3
    recovery_turn: int = 6
    opening_probability: float = 0.50
    middle_probability: float = 0.10
    steady_probability: float = 0.30

    def __post_init__(self) -> None:
        if self.opening_turns < 0 or self.recovery_turn < self.opening_turns:
            raise ValueError("session frequency turn boundaries are invalid")
        for value in (
            self.opening_probability,
            self.middle_probability,
            self.steady_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("session frequency probabilities must be between 0 and 1")

    def probability(self, completed_turns: int) -> float:
        completed = max(0, int(completed_turns))
        if completed < self.opening_turns:
            return self.opening_probability
        if completed < self.recovery_turn:
            return self.middle_probability
        return self.steady_probability
