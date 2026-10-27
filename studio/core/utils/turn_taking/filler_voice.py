"""Prepare reusable waiting speech outside the conversation's critical path."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import hashlib
import os
from pathlib import Path
import random
import tempfile

from studio.harness.turn_taking.policy import WORK_FILLER_PHRASES
from voicemem.audio_timing import TimedAudioChunk


def choose_phrase(language, recent=(), rng=None):
    """Choose a neutral waiting phrase in the spoken language without repetition."""
    bank = WORK_FILLER_PHRASES['en' if language == 'en' else 'zh']
    return (rng or random).choice([text for text in bank if text not in recent] or bank)


def instruction(agent, language, snapshot=None):
    """Keep waiting speech neutral while honoring captured explicit voice preferences."""
    from studio.harness.prompts import effective_prompts, resolve
    from studio.core.utils.self_harness.component import fixed_tone, speech_rate_instruction
    from studio.core.utils.tts.control import instruction as tone_instruction
    prompts = effective_prompts(snapshot)
    base = agent._speak_instruction('', language)
    if f'tts_{language}' in prompts:
        base = resolve(f'tts_{language}', '', prompts)
    tone = fixed_tone(snapshot)
    if tone or f'tts_{language}' in prompts:
        base = tone_instruction(tone or '平静', base, language)
    return speech_rate_instruction(base, snapshot, language=language)


async def synthesize(tts, text, style):
    """Collect complete PCM16 speech and close its provider on failure or cancellation."""
    chunks = bytearray()
    try:
        source = tts.stream(text, style)
    except TypeError:
        source = tts.stream(text)
    async with aclosing(source) as stream:
        async for chunk in stream:
            if isinstance(chunk, TimedAudioChunk):
                if chunk.sample_rate != 24000:
                    raise ValueError('Waiting speech must use 24000 Hz')
                chunk = chunk.pcm
            if not isinstance(chunk, bytes) or len(chunk) % 2:
                raise ValueError('Waiting speech must contain complete PCM16 samples')
            chunks.extend(chunk)
    if not chunks:
        raise RuntimeError('Waiting speech returned no audio')
    return bytes(chunks)


class WorkFillerVoice:
    """Own one voice/language/instruction bank; publish only a complete warmup."""

    def __init__(self, tts, language, style, *, root=None):
        from studio.paths import ROOT
        self.tts, self.language, self.style = tts, language, style
        self.voice_id = getattr(tts, 'cache_voice_id', '')
        self.root = Path(root) if root is not None else ROOT / 'cache/work-filler'
        self.clips = {}

    def matches(self, tts, language, style):
        """Require a declared stable voice identity or the exact provider object."""
        same_voice = (self.voice_id == getattr(tts, 'cache_voice_id', '')
                      if self.voice_id else self.tts is tts)
        return same_voice and self.language == language and self.style == style

    def _path(self, text):
        if not self.voice_id:
            return None
        key = f'1|pcm16|24000|{self.voice_id}|{self.language}|{self.style}|{text}'
        return self.root / (hashlib.sha256(key.encode()).hexdigest() + '.pcm')

    @staticmethod
    def _read(path):
        if path is None or not path.is_file():
            return None
        pcm = path.read_bytes()
        return pcm if pcm and not len(pcm) % 2 else None

    @staticmethod
    def _write(path, pcm):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(pcm)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    async def prime(self):
        """Load or synthesize all phrases before serving; failures remain visible."""
        prepared = {}
        for text in WORK_FILLER_PHRASES[self.language]:
            path = self._path(text)
            pcm = await asyncio.to_thread(self._read, path)
            if pcm is None:
                pcm = await synthesize(self.tts, text, self.style)
                await asyncio.to_thread(self._write, path, pcm)
            prepared[text] = pcm
        self.clips = prepared
        return len(prepared)

    def choose(self, recent=(), rng=None):
        """Return only warmed memory-resident speech; never perform I/O here."""
        if not self.clips:
            return None
        text = choose_phrase(self.language, recent, rng)
        return text, self.clips[text]
