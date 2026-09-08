"""Reserved contracts for selecting how a reply is produced.

The runtime does not select among these modes yet. Keeping the identifiers
stable lets the future router and turn-taking layer agree on mode names.
"""

MEMORY_COT = "memory_cot"
MEMORY = "memory"
DIRECT = "direct"

AVAILABLE_MODES = (MEMORY_COT, MEMORY, DIRECT)

__all__ = ["AVAILABLE_MODES", "DIRECT", "MEMORY", "MEMORY_COT"]
