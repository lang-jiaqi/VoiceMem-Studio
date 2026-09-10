"""Compatibility imports for the Studio dialogue policies."""
from studio.harness.persona.policy import SYSTEM_PROMPT
from studio.core.utils.speaking_style.component import CONTEXT, TONE_RULE
from studio.core.utils.turn_taking.initialize import CONTROLS
from studio.core.utils.turn_taking.pause import (PauseGate, is_unfinished, backchannel_policy, backchannel_policy_summary)
from studio.core.utils.prompts.component import system_prompt
