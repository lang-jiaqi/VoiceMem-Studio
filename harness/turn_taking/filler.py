"""Reusable timing primitives for short acknowledgements and work fillers."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Sequence


@dataclass(frozen=True)
class FillerPlan:
    """Describe filler duration and when the main reply should start."""

    target_seconds: float
    lead_seconds: float
    purpose: str

    @property
    def main_start_seconds(self) -> float:
        return max(0.0, self.target_seconds - self.lead_seconds)


SHORT_ACK = FillerPlan(target_seconds=0.35, lead_seconds=0.10, purpose="acknowledgement")
LONG_WORK_FILLER = FillerPlan(target_seconds=4.0, lead_seconds=0.20, purpose="tool_or_reasoning")


def short_ack_plan(clip_seconds: float) -> FillerPlan:
    """Start main work 100 ms before this acknowledgement clip ends."""
    return FillerPlan(
        target_seconds=max(0.0, float(clip_seconds)),
        lead_seconds=0.10,
        purpose="acknowledgement",
    )


@lru_cache(maxsize=2)
def _prompt_template(lang: str) -> str:
    code = "en" if str(lang).lower().startswith("en") else "zh"
    return Path(__file__).with_name(f"filler_prompt_{code}.md").read_text(
        encoding="utf-8").strip()


def generation_prompt(task_context: str, lang: str = "zh") -> str:
    """Build an LLM prompt for a spoken filler, not the final answer."""
    context = (task_context or "").strip()
    return _prompt_template(lang).format(task_context=context)


async def generate_filler(
    reply_stream: Callable[[str, str, Sequence[dict]], AsyncIterator[str]],
    task_context: str,
    *,
    history: Sequence[dict] = (),
    lang: str = "zh",
    max_chars: int = 36,
) -> str:
    """Generate one short spoken bridge with the configured reply model.

    The caller decides whether the turn is actually doing long-running work.
    This function intentionally does not infer tool or reasoning state from user
    text, because a false filler is more disruptive than silence.
    """
    chunks: list[str] = []
    size = 0
    stream = reply_stream(generation_prompt(task_context, lang), "", history)
    try:
        async for chunk in stream:
            if not chunk:
                continue
            chunks.append(chunk)
            size += len(chunk)
            text = "".join(chunks).strip()
            if size >= max_chars or any(mark in text for mark in "。！？!?\n"):
                break
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
    text = "".join(chunks).strip()
    # The normal llm_tts system prompt asks for a private leading tone tag.
    # Fillers are public speech too, so remove that transport-only prefix here.
    from voicemem import tts_control
    tag, spoken = tts_control.split(text)
    return spoken.strip() if tag else text


async def run_overlapped_handoff(
    play_filler: Callable[[], Awaitable[None]],
    start_main: Callable[[], Awaitable[None]],
    plan: FillerPlan,
) -> None:
    """Play filler now and start main work shortly before its expected end."""
    filler = asyncio.create_task(play_filler())
    try:
        await asyncio.sleep(plan.main_start_seconds)
        await start_main()
        await filler
    except BaseException:
        if not filler.done():
            filler.cancel()
        await asyncio.gather(filler, return_exceptions=True)
        raise
