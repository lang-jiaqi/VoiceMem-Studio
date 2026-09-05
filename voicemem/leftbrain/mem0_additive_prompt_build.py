"""Mem0 OSS V3 additive 用户侧 prompt 拼装逻辑。

源代码对齐：mem0/configs/prompts.py（``generate_additive_extraction_prompt`` 及辅助函数）。
System 侧长文本见 ``data/additive_extraction_prompt.txt``（同源仓库摘出）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PAST_MESSAGE_TRUNCATION_LIMIT = 300


def _load(name: str) -> str:
    p = Path(__file__).resolve().parent / "data" / name
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"System prompt missing or empty: {p}")
    return text


def load_additive_system_prompt() -> str:
    """载入与 Mem0 ``ADDITIVE_EXTRACTION_PROMPT`` 一致的 system 正文。"""
    return _load("additive_extraction_prompt.txt")


def load_three_stage_prompt() -> str:
    """三步版 system 正文：先判存不存 → 再判左脑右脑 → 最后各按各的形状写。

    换掉 mem0 那份 474 行的原因不是它写得差，是它**不是为这件事写的**：它假设输入
    是打字聊天，没有右脑这个概念，于是 voicemem 需要的三件事全靠往后面贴 addendum
    ——「什么不值得记」贴在 system 末尾第四段，「左右脑分流」压根没有，右脑的字段
    只能贴到用户消息里。结果是最要紧的那条约束躺在最长的 prompt 的尾巴上。实测就
    是不生效：库里躺着「用户用中文问候，内容是"你好"」「助手用中文回应用户的问候」
    这种条目，两条都是那段 addendum 明令禁止的形状，而它当时已经在 prompt 里了。

    所以重排成三步，把判断放在最前面：不值得存的在第一步就停住，根本走不到抽取。
    ``VOICEMEM_EXTRACTION_PROMPT=upstream`` 换回原来那份。
    """
    return _load("three_stage_extraction_prompt.txt")


def _truncate_content(text: str, limit: int = PAST_MESSAGE_TRUNCATION_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_summary(summary: object | None) -> str:
    if isinstance(summary, dict):
        return str((summary.get("summary") or "")).strip()
    return str(summary or "").strip()


def _format_conversation_history(messages: list[dict[str, object]] | None) -> str:
    if not messages:
        return ""
    result = ""
    for msg in messages:
        role = str(msg.get("role", "") or "")
        content = msg.get("message") or msg.get("content") or ""
        content = str(content).strip()
        if role and content:
            result += f"{role}: {_truncate_content(content)}\n"
    return result


def _serialize_memories(memories: object | None) -> str:
    return json.dumps(memories or [], ensure_ascii=False)


def _format_new_messages(new_messages: object | None) -> str:
    if isinstance(new_messages, str):
        return new_messages.strip()
    return json.dumps(new_messages or [], ensure_ascii=False)


def _resolve_dates(
    current_date: str | None = None,
    observation_date: str | None = None,
) -> tuple[str, str]:
    if current_date is None:
        current_date = datetime.now(timezone.utc).date().isoformat()
    if observation_date is None:
        observation_date = current_date
    return current_date, observation_date


def generate_additive_extraction_prompt(
    summary: object | None = None,
    recently_extracted_memories: list[object] | None = None,
    existing_memories: list[object] | None = None,
    new_messages: object | None = None,
    *,
    last_k_messages: list[dict[str, object]] | None = None,
    current_date: str | None = None,
    timestamp: str | None = None,
    custom_instructions: str | None = None,
    use_input_language: bool = False,
) -> str:
    """与 Mem0 官方同名函数一致参数语义；``timestamp`` 作为 Observation Date 传入时使用。"""
    cur, obs = _resolve_dates(current_date, timestamp)

    sections: list[str] = []
    sections.append(f"## Summary\n{_format_summary(summary)}")
    sections.append(f"## Last k Messages\n{_format_conversation_history(last_k_messages)}")
    sections.append(f"## Recently Extracted Memories\n{_serialize_memories(recently_extracted_memories)}")
    sections.append(f"## Existing Memories\n{_serialize_memories(existing_memories)}")
    sections.append(f"## New Messages\n{_format_new_messages(new_messages)}")
    sections.append(f"## Observation Date\n{obs}")
    sections.append(f"## Current Date\n{cur}")

    if custom_instructions:
        sections.append(f"## Custom Instructions\n{custom_instructions}")

    if use_input_language:
        sections.append(
            "## Language Requirement\n"
            "CRITICAL: Respond in the SAME LANGUAGE and SCRIPT as the input messages.\n"
            "1. Match the language of the user's messages exactly — if they write in Korean, extract in Korean; Japanese in Japanese; etc.\n"
            "2. Preserve the exact script/alphabet of the input.\n"
            "3. Do NOT translate or transliterate into English unless the input is already in English.\n"
            "4. Maintain all quality standards (contextual richness, temporal grounding, etc.) regardless of language.\n"
            "5. Technical terms, proper nouns, and brand names should be preserved in their original form as used in the input.\n"
            "6. If the input mixes languages (e.g., Hinglish), preserve both the mixed language style AND the script.\n"
            "7. For Japanese: explicitly resolve omitted subjects using conversation context.\n"
            "8. For CJK languages: maintain appropriate formality level from the source text."
        )
    else:
        sections.append(
            "## Language Requirement\n"
            "CRITICAL: Always write all extracted memory text in English, regardless of the language of the input messages.\n"
            "Translate any non-English content into English when extracting memories."
        )

    sections.append("# Output:")
    return "\n\n".join(sections)
