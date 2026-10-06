"""SDK speech stream lifecycle checks without models, network or audio devices."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from voicemem.audio_timing import TimedAudioChunk
from voicemem.tts import speak_stream


async def text_deltas(*parts):
    for part in parts:
        yield part


async def collect(stream):
    return [chunk async for chunk in stream]


class SpeakStreamTests(unittest.IsolatedAsyncioTestCase):
    async def complete(self, stream):
        # A timeout makes a stranded consumer fail instead of hanging the suite.
        return await asyncio.wait_for(collect(stream), 1)

    async def test_preserves_segment_order_metadata_instruction_and_text_callback(self):
        calls, observed = [], []
        timed = TimedAudioChunk(pcm=b'\x01\x00')

        async def synth(text, instruction):
            calls.append((text, instruction))
            yield timed if len(calls) == 1 else b'\x02\x00'

        output = await self.complete(speak_stream(
            text_deltas('Hello', '!', ' trailing text '),
            tts=SimpleNamespace(stream=synth), instruction='calm',
            on_delta=observed.append))
        self.assertEqual(calls, [('Hello!', 'calm'), ('trailing text', 'calm')])
        self.assertEqual(observed, ['Hello', '!', ' trailing text '])
        self.assertEqual(output, [timed, b'\x02\x00'])
        self.assertIs(output[0], timed)

    async def test_empty_and_whitespace_input_do_not_synthesize(self):
        def unexpected(*args):
            self.fail('Empty text must not reach synthesis')

        for parts in ((), (' ', '\n')):
            with self.subTest(parts=parts):
                self.assertEqual(await self.complete(speak_stream(
                    text_deltas(*parts), tts=SimpleNamespace(stream=unexpected))), [])

    async def test_synthesis_failure_before_audio_propagates_and_cancels_input(self):
        closed = asyncio.Event()
        error = RuntimeError('synthesis failed')

        async def source():
            try:
                yield 'Hello!'
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def synth(*args):
            raise error
            yield b''

        with self.assertRaises(RuntimeError) as caught:
            await self.complete(speak_stream(source(), tts=SimpleNamespace(stream=synth)))
        self.assertIs(caught.exception, error)
        self.assertTrue(closed.is_set())

    async def test_synthesis_failure_after_partial_audio_is_not_normal_eof(self):
        error = RuntimeError('failure after PCM')

        async def synth(*args):
            yield b'\x01\x00'
            raise error

        stream = speak_stream(text_deltas('Hello!'), tts=SimpleNamespace(stream=synth))
        self.assertEqual(await asyncio.wait_for(anext(stream), 1), b'\x01\x00')
        with self.assertRaises(RuntimeError) as caught:
            await asyncio.wait_for(anext(stream), 1)
        self.assertIs(caught.exception, error)

    async def test_synchronous_provider_setup_failure_propagates(self):
        def synth(*args):
            raise ValueError('invalid provider configuration')

        with self.assertRaisesRegex(ValueError, 'invalid provider configuration'):
            await self.complete(speak_stream(
                text_deltas('Hello!'), tts=SimpleNamespace(stream=synth)))

    async def test_input_failure_before_text_propagates(self):
        async def source():
            raise LookupError('reply source failed')
            yield ''

        with self.assertRaisesRegex(LookupError, 'reply source failed'):
            await self.complete(speak_stream(source()))

    async def test_input_failure_cancels_stalled_synthesis(self):
        started, closed = asyncio.Event(), asyncio.Event()
        error = RuntimeError('reply failed during synthesis')

        async def source():
            yield 'Hello!'
            await started.wait()
            raise error

        async def synth(*args):
            try:
                started.set()
                await asyncio.Event().wait()
                yield b''
            finally:
                closed.set()

        with self.assertRaises(RuntimeError) as caught:
            await self.complete(speak_stream(source(), tts=SimpleNamespace(stream=synth)))
        self.assertIs(caught.exception, error)
        self.assertTrue(closed.is_set())

    async def test_callback_failure_propagates(self):
        def observer(delta):
            raise ValueError('observer failed')

        with self.assertRaisesRegex(ValueError, 'observer failed'):
            await self.complete(speak_stream(text_deltas('Hello!'), on_delta=observer))

    async def test_provider_cancellation_propagates_and_cancels_input(self):
        closed = asyncio.Event()

        async def source():
            try:
                yield 'Hello!'
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def synth(*args):
            raise asyncio.CancelledError()
            yield b''

        with self.assertRaises(asyncio.CancelledError):
            await self.complete(speak_stream(source(), tts=SimpleNamespace(stream=synth)))
        self.assertTrue(closed.is_set())

    async def test_consumer_cancellation_awaits_both_background_tasks(self):
        started = asyncio.Event()
        input_closed, synth_closed = asyncio.Event(), asyncio.Event()

        async def source():
            try:
                yield 'Hello!'
                await asyncio.Event().wait()
            finally:
                input_closed.set()

        async def synth(*args):
            try:
                started.set()
                await asyncio.Event().wait()
                yield b''
            finally:
                synth_closed.set()

        consumer = asyncio.create_task(collect(speak_stream(
            source(), tts=SimpleNamespace(stream=synth))))
        try:
            await asyncio.wait_for(started.wait(), 1)
        finally:
            consumer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(consumer, 1)
        self.assertTrue(input_closed.is_set())
        self.assertTrue(synth_closed.is_set())

    async def test_explicit_close_after_audio_cancels_pending_work(self):
        input_closed, synth_closed = asyncio.Event(), asyncio.Event()

        async def source():
            try:
                yield 'Hello!'
                await asyncio.Event().wait()
            finally:
                input_closed.set()

        async def synth(*args):
            try:
                yield b'\x01\x00'
                await asyncio.Event().wait()
            finally:
                synth_closed.set()

        stream = speak_stream(source(), tts=SimpleNamespace(stream=synth))
        try:
            self.assertEqual(await asyncio.wait_for(anext(stream), 1), b'\x01\x00')
        finally:
            await asyncio.wait_for(stream.aclose(), 1)
        self.assertTrue(input_closed.is_set())
        self.assertTrue(synth_closed.is_set())

    async def test_configured_provider_still_uses_existing_selection(self):
        async def synth(text, instruction):
            self.assertEqual((text, instruction), ('Hello!', 'friendly'))
            yield b'\x01\x00'

        config = {'tts': {'provider': 'openai', 'config': {'voice': 'alloy'}}}
        with patch('voicemem.tts.make_tts', return_value=SimpleNamespace(stream=synth)) as factory:
            self.assertEqual(await self.complete(speak_stream(
                text_deltas('Hello!'), reply=config, instruction='friendly')), [b'\x01\x00'])
        factory.assert_called_once_with('openai', voice='alloy')


if __name__ == '__main__':
    unittest.main()
