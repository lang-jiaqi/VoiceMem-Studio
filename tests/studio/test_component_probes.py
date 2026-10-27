"""Disposable startup checks never use user input or persist synthetic memory."""
import asyncio
from contextlib import aclosing
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from studio.core.utils.component_config.probe import probe_llm, validate_changed_llms
from studio.core.utils.startup.component import Startup


class LLMProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_uses_selected_role_and_closes_stream_and_client(self):
        closed = []
        async def reply(text, memory_context, history):
            self.assertEqual((text, memory_context, history), ('Hello', '', []))
            try:
                yield ''
                yield 'connected'
                raise AssertionError('Probe must stop at the first nonblank text delta')
            finally:
                closed.append('stream')
        reply.aclose = AsyncMock()
        with patch('studio.core.utils.llm.initialize.create', return_value=reply) as create, \
                patch('studio.core.utils.llm.initialize.credential', return_value='fixture') as credential:
            await probe_llm('memory', 'openai')
        self.assertEqual(closed, ['stream'])
        credential.assert_called_once_with('openai', 'memory')
        self.assertEqual(create.call_args.kwargs, {'purpose':'memory'})
        reply.aclose.assert_awaited_once_with()

    async def test_empty_or_cancelled_probe_closes_provider(self):
        for cancel in (False, True):
            waiting = asyncio.Event()
            closed = []
            async def reply(*args):
                try:
                    if cancel:
                        waiting.set()
                        await asyncio.Event().wait()
                    yield ''
                finally:
                    closed.append(True)
            reply.aclose = AsyncMock()
            with patch('studio.core.utils.llm.initialize.create',return_value=reply), \
                    patch('studio.core.utils.llm.initialize.credential',return_value='fixture'):
                if cancel:
                    task = asyncio.create_task(probe_llm('reply','openai'))
                    await asyncio.wait_for(waiting.wait(),1)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    with self.assertRaisesRegex(RuntimeError,'未输出'):
                        await probe_llm('reply','openai')
            self.assertEqual(closed,[True])
            reply.aclose.assert_awaited_once_with()


class SelectedProbeTests(unittest.TestCase):
    def test_default_startup_does_not_probe_remote_llms(self):
        with patch.dict(os.environ,{},clear=True), \
                patch('studio.core.utils.component_config.probe.probe_llm',new_callable=AsyncMock) as probe:
            validate_changed_llms(SimpleNamespace())
        probe.assert_not_called()

    def test_only_changed_remote_roles_are_probed_and_failure_is_sanitized(self):
        args = SimpleNamespace(memory_llm='qwen',llm='local')
        with patch.dict(os.environ,{'STUDIO_VALIDATE_SERVICES':'["memory","reply","memory"]'},clear=True), \
                patch('studio.core.utils.component_config.probe.probe_llm',new_callable=AsyncMock) as probe:
            validate_changed_llms(args)
            probe.assert_awaited_once_with('memory','qwen')
            probe.side_effect = RuntimeError('private-vendor-payload')
            with self.assertRaises(RuntimeError) as error:
                validate_changed_llms(args)
            self.assertNotIn('private-vendor-payload',str(error.exception))
        with patch.dict(os.environ,{'STUDIO_VALIDATE_SERVICES':'["unknown"]'},clear=True):
            with self.assertRaises(ValueError):
                validate_changed_llms(args)


class SpeechWarmupTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_use_reviewed_cache_and_custom_voice_primes_two_clips(self):
        for provider in ('recommended','openai','adapter'):
            async def stream(text, instruction=None):
                yield b'\0\0'
            tts = SimpleNamespace(stream=stream,aclose=AsyncMock())
            voice = SimpleNamespace(prime=AsyncMock(),ready=True)
            agent = Startup()
            agent.MODE = 'llm_tts'
            agent._speak_instruction = lambda *args: 'Synthetic voice style'
            agent.vm = SimpleNamespace(utils={'tts':tts})
            agent.ARGS = SimpleNamespace(backchannel=True)
            agent.ACTIVE_SPACE = 'synthetic'
            agent.space_language = lambda _: 'zh'
            agent._BC_VOICE = {}
            with patch('studio.core.utils.component_config.component.selection',return_value={'provider':provider}), \
                    patch('studio.core.utils.turn_taking.initialize.BackchannelVoice',return_value=voice):
                await agent._warm_speech()
            if provider == 'recommended':
                voice.prime.assert_awaited_once_with(cache_only=True)
            else:
                voice.prime.assert_awaited_once_with(tokens=('嗯','明白'),variants=1)
            self.assertIs(agent._BC_VOICE['obj'],voice)
            self.assertEqual(tts.aclose.await_count,int(provider=='openai'))

    async def test_failed_synthesis_and_backchannel_cache_close_api_client(self):
        for audio in (False,True):
            async def stream(text):
                if audio:
                    yield b'\0\0'
            tts = SimpleNamespace(stream=stream,aclose=AsyncMock())
            agent = Startup()
            agent.vm = SimpleNamespace(utils={'tts':tts})
            agent.ARGS = SimpleNamespace(backchannel=True)
            agent.ACTIVE_SPACE = 'synthetic'
            agent.space_language = lambda _: 'en'
            agent._BC_VOICE = {}
            voice = SimpleNamespace(prime=AsyncMock(),ready=False)
            with patch('studio.core.utils.component_config.component.selection',return_value={'provider':'openai'}), \
                    patch('studio.core.utils.turn_taking.initialize.BackchannelVoice',return_value=voice):
                with self.assertRaises(RuntimeError):
                    await agent._warm_speech()
            tts.aclose.assert_awaited_once_with()
            self.assertNotIn('obj',agent._BC_VOICE)


if __name__ == '__main__':
    unittest.main()
