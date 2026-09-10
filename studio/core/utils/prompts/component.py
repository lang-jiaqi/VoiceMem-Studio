"""Assemble the four Studio policies into reply context."""
from studio.harness.persona.policy import SYSTEM_PROMPT
from studio.core.utils.speaking_style.component import CONTEXT, TONE_RULE, prompt_rule as speaking_style_prompt

def system_prompt(lang=None, *, tagged=False, reply=None):
    parts = [SYSTEM_PROMPT, speaking_style_prompt()]
    if tagged:
        parts.append(TONE_RULE)
        from studio.core.utils.speaking_style.component import is_qwen36
        if is_qwen36(reply):
            from studio.harness.speaking_style.policy import QWEN36_PROMPT
            parts.append(QWEN36_PROMPT)
    return "\n\n".join(parts)
