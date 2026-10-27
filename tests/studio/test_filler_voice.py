"""Prepared waiting speech, cache ownership and request-scoped preference checks."""
import asyncio
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from studio.core.utils.turn_taking.filler_voice import WorkFillerVoice, instruction, synthesize
from studio.harness.turn_taking.policy import WORK_FILLER_PHRASES
from tests.helpers.conversation import ConversationFixture
from voicemem.audio_timing import TimedAudioChunk


class FillerVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calls = []

        async def stream(text, style):
            self.calls.append((text, style))
            yield bytes(480)

        self.tts = SimpleNamespace(stream=stream, cache_voice_id='synthetic-voice-v1')

    async def test_complete_bank_reuses_disk_without_synthesis_and_avoids_last_three(self):
        bank = WorkFillerVoice(self.tts, 'zh', 'Synthetic style', root=self.directory.name)
        self.assertEqual(await bank.prime(), 4)
        second = WorkFillerVoice(self.tts, 'zh', 'Synthetic style', root=self.directory.name)
        await second.prime()
        self.assertEqual(len(self.calls), 4)
        recent, rng = [], random.Random(8)
        for _ in range(20):
            text, pcm = second.choose(recent, rng)
            self.assertNotIn(text, recent)
            self.assertEqual(pcm, bytes(480))
            recent = (recent + [text])[-3:]

    async def test_voice_instruction_and_language_changes_cannot_reuse_old_audio(self):
        bank = WorkFillerVoice(self.tts, 'zh', 'First style', root=self.directory.name)
        await bank.prime()
        self.assertFalse(bank.matches(self.tts, 'en', 'First style'))
        self.assertFalse(bank.matches(self.tts, 'zh', 'Second style'))
        self.assertFalse(bank.matches(SimpleNamespace(cache_voice_id='other'), 'zh', 'First style'))
        for language, style in (('en', 'First style'), ('zh', 'Second style')):
            other = WorkFillerVoice(self.tts, language, style, root=self.directory.name)
            await other.prime()
        self.assertEqual(len(self.calls), 12)

    async def test_unknown_voice_has_only_object_scoped_memory_cache(self):
        self.tts.cache_voice_id = ''
        bank = WorkFillerVoice(self.tts, 'zh', 'Style', root=self.directory.name)
        await bank.prime()
        self.assertTrue(bank.matches(self.tts, 'zh', 'Style'))
        self.assertFalse(bank.matches(SimpleNamespace(), 'zh', 'Style'))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    async def test_corrupt_cache_is_rebuilt_and_partial_warmup_is_never_ready(self):
        bank = WorkFillerVoice(self.tts, 'zh', 'Style', root=self.directory.name)
        bank._path(WORK_FILLER_PHRASES['zh'][0]).write_bytes(b'x')
        await bank.prime()
        self.assertEqual(len(self.calls), 4)

        async def failing(text, style):
            if text == WORK_FILLER_PHRASES['en'][1]:
                raise RuntimeError('Synthetic failure')
            yield bytes(480)

        broken = WorkFillerVoice(SimpleNamespace(stream=failing), 'en', 'Style', root=self.directory.name)
        with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
            await broken.prime()
        self.assertIsNone(broken.choose())

    async def test_cancelled_preparation_closes_provider_and_keeps_bank_private(self):
        entered, closed = asyncio.Event(), asyncio.Event()

        async def stream(*_):
            try:
                yield bytes(480)
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        bank = WorkFillerVoice(SimpleNamespace(stream=stream), 'en', 'Style', root=self.directory.name)
        task = asyncio.create_task(bank.prime())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        self.assertIsNone(bank.choose())

    async def test_invalid_or_empty_provider_audio_cannot_enter_cache(self):
        for chunk in (b'x', TimedAudioChunk(bytes(480), sample_rate=16000)):
            async def stream(*_):
                yield chunk
            with self.assertRaises(ValueError):
                await synthesize(SimpleNamespace(stream=stream), 'Synthetic text', 'Style')
        async def empty(*_):
            if False:
                yield b''
        with self.assertRaises(RuntimeError):
            await synthesize(SimpleNamespace(stream=empty), 'Synthetic text', 'Style')

    async def test_legacy_single_argument_provider_still_returns_complete_speech(self):
        async def stream(text):
            self.assertEqual(text, 'Synthetic text')
            yield bytes(480)
            yield bytes(960)
        pcm = await synthesize(SimpleNamespace(stream=stream), 'Synthetic text', 'Style')
        self.assertEqual(pcm, bytes(1440))


class ConversationFillerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = ConversationFixture()
        self.f.setUp()
        self.addAsyncCleanup(self.f.asyncTearDown)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    async def test_warm_bank_calls_neither_text_model_nor_tts_and_preserves_playback_ack(self):
        calls = []

        async def stream(*args):
            calls.append(args)
            yield bytes(480)

        tts = SimpleNamespace(stream=stream, cache_voice_id='synthetic')
        self.f.memory.utils = {'tts': tts}
        snapshot = self.f.session.self_harness.snapshot()
        bank = WorkFillerVoice(tts, 'zh', instruction(self.f.agent, 'zh', snapshot), root=self.directory.name)
        await bank.prime()
        self.f.agent._WORK_FILLER_VOICES = [bank]
        pending = self.f.pending(language='zh')
        pending.self_harness_profile = snapshot
        with patch('studio.core.utils.conversation.component.generate_local_filler', new_callable=AsyncMock) as model:
            for _ in range(8):
                text, pcm = await self.f.session.synthesize_work_filler(pending, self.f.memory, 'fixture')
                self.assertNotIn(text, self.f.session.turn_taking.backchannel._recent)
                await self.f.session.emit_filler(text, pcm, filler_id='synthetic')
            model.assert_not_awaited()
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(message.get('filler_id') == 'synthetic' for message in self.f.messages))

    async def test_changed_rate_uses_prepared_text_with_current_voice_instead_of_stale_pcm(self):
        styles = []

        async def stream(text, style):
            styles.append(style)
            yield bytes(480)

        tts = SimpleNamespace(stream=stream, cache_voice_id='synthetic')
        self.f.memory.utils = {'tts': tts}
        fast = self.f.session.self_harness.snapshot()
        bank = WorkFillerVoice(tts, 'en', instruction(self.f.agent, 'en', fast), root=self.directory.name)
        await bank.prime()
        self.f.agent._WORK_FILLER_VOICES = [bank]
        self.f.session.self_harness.set_explicit({'speaking_style': {'speech_rate': 'slow'}})
        pending = self.f.pending(language='en')
        pending.self_harness_profile = self.f.session.self_harness.snapshot()
        with patch('studio.core.utils.conversation.component.generate_local_filler', new_callable=AsyncMock) as model:
            text, _ = await self.f.session.synthesize_work_filler(pending, self.f.memory, 'fixture')
            model.assert_not_awaited()
        self.assertIn(text, WORK_FILLER_PHRASES['en'])
        self.assertEqual(len(styles), 5)
        self.assertIn('more slowly', styles[-1])

    async def test_custom_filler_prompt_still_uses_its_captured_model_instructions(self):
        async def stream(*_):
            yield bytes(480)

        self.f.memory.utils = {'tts': SimpleNamespace(stream=stream)}
        pending = self.f.pending(language='zh')
        pending.self_harness_profile = {'prompt_overrides': {'filler_system': 'Synthetic {task_context}'}}
        with patch('studio.core.utils.conversation.component.generate_local_filler',
                   new=AsyncMock(return_value='合成的自定义等待语。')) as model:
            text, _ = await self.f.session.synthesize_work_filler(pending, self.f.memory, 'fixture')
        self.assertEqual(text, '合成的自定义等待语。')
        self.assertEqual(model.call_args.kwargs['prompts']['filler_system'], 'Synthetic {task_context}')
