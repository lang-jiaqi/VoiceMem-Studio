"""Local bilingual readiness and unchanged custom-provider request ownership."""
import asyncio
from contextlib import ExitStack, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from studio.core.utils.startup.component import Startup


class SpeechWarmupTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, language='zh', provider='recommended', failure=None):
        calls, closed = [], []

        async def stream(text, style=None):
            calls.append((text, style))
            try:
                if failure == 'empty':
                    return
                yield b'\x01\x00' * 240
                if failure == 'late':
                    raise RuntimeError('synthetic codec failure')
                if failure == 'cancel':
                    raise asyncio.CancelledError
                if failure == 'odd':
                    yield b'\x01'
            finally:
                closed.append(text)

        tts = SimpleNamespace(stream=stream, aclose=AsyncMock())
        agent = Startup()
        agent.MODE = 'llm_tts'
        agent.ACTIVE_SPACE = 'synthetic-space'
        agent.ARGS = SimpleNamespace(backchannel=False)
        agent.vm = SimpleNamespace(utils={'tts': tts})
        agent.space_language = lambda _: language
        stack = ExitStack()
        stack.enter_context(patch('studio.core.utils.component_config.component.selection',
                                  return_value={'provider': provider}))
        stack.enter_context(patch('studio.core.utils.self_harness.component.SelfHarnessState'))
        style = stack.enter_context(patch('studio.core.utils.turn_taking.filler_voice.instruction',
                                         side_effect=lambda _, lang, snapshot: f'synthetic-{lang}'))
        bank = Mock(prime=AsyncMock(return_value=4))
        banks = stack.enter_context(patch('studio.core.utils.turn_taking.filler_voice.WorkFillerVoice',
                                         return_value=bank))
        output = stack.enter_context(redirect_stdout(io.StringIO()))
        self.addCleanup(stack.close)
        return agent, tts, calls, closed, banks, bank, style, output

    async def test_both_languages_complete_with_preferences_and_active_language_last(self):
        for language, order in (('zh', ['en', 'zh']), ('en', ['zh', 'en'])):
            with self.subTest(language=language):
                agent, tts, calls, closed, banks, bank, _, output = self.fixture(language)
                await agent._warm_speech()
                self.assertEqual([style for _, style in calls], [f'synthetic-{lang}' for lang in order])
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(closed), 2)
                self.assertTrue(all(len(text) > 16 for text, _ in calls))
                self.assertEqual(bank.prime.await_count, 2)
                self.assertEqual([call.args[1] for call in banks.call_args_list], ['zh', 'en'])
                self.assertEqual(len(agent._WORK_FILLER_VOICES), 2)
                self.assertEqual(output.getvalue().count('[startup-tts]'), 2)
                tts.aclose.assert_not_awaited()

    async def test_custom_and_api_providers_keep_one_probe_and_existing_cleanup(self):
        for provider in ('adapter', 'openai'):
            with self.subTest(provider=provider):
                agent, tts, calls, _, _, _, _, _ = self.fixture('en', provider)
                await agent._warm_speech()
                self.assertEqual(calls, [('Hello, I am here.', None)])
                self.assertEqual(tts.aclose.await_count, int(provider == 'openai'))

    async def test_empty_malformed_failed_or_cancelled_speech_cannot_publish_readiness(self):
        for failure, exception in (('empty', RuntimeError), ('odd', ValueError),
                                   ('late', RuntimeError), ('cancel', asyncio.CancelledError)):
            with self.subTest(failure=failure):
                agent, _, _, closed, banks, _, _, _ = self.fixture(failure=failure)
                with self.assertRaises(exception):
                    await agent._warm_speech()
                self.assertEqual(len(closed), 1)
                banks.assert_not_called()
                self.assertFalse(hasattr(agent, '_WORK_FILLER_VOICES'))

    async def test_openai_failure_and_cancellation_still_release_loop_bound_client(self):
        for failure, exception in (('late', RuntimeError), ('cancel', asyncio.CancelledError)):
            with self.subTest(failure=failure):
                agent, tts, _, _, _, _, _, _ = self.fixture(provider='openai', failure=failure)
                with self.assertRaises(exception):
                    await agent._warm_speech()
                tts.aclose.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
