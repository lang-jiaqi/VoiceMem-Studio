"""Assemble the four Studio policies into reply context."""
from studio.harness.persona.policy import SYSTEM_PROMPT
from studio.core.utils.speaking_style.component import (
    CONTEXT, TONE_RULE, prompt_rule as speaking_style_prompt,
)
from studio.harness.self_harness.policy import CONTROL_RULE
from studio.harness.prompts import resolve

def system_prompt(lang=None, *, tagged=False, reply=None, overrides=None):
    parts = [resolve("main_system", SYSTEM_PROMPT, overrides),
             resolve("speaking_style", speaking_style_prompt(), overrides)]
    if tagged:
        parts.extend((resolve("self_harness", CONTROL_RULE, overrides),
                      resolve("tone_rule", TONE_RULE, overrides)))
        from studio.core.utils.speaking_style.component import is_qwen36
        if is_qwen36(reply):
            from studio.harness.speaking_style.policy import QWEN36_PROMPT
            parts.append(QWEN36_PROMPT)
    return "\n\n".join(parts)
