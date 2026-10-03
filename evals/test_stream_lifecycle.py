"""Connection-owned decoders and worker shutdown without models or recordings."""
import asyncio
import gc
import threading
import types
import unittest
import weakref
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

import numpy as np

from studio.core.utils.capture.component import Capture
from studio.core.utils.asr import initialize
from studio.core.utils.realtime_session.component import RealtimeSession
from studio.core.utils.contracts.component import Pending
from voicemem.stream import VoiceStream, _AsrWorker
from voicemem.utils.audio.asr import FunASRStreamingASR


def prototype():
    asr = object.__new__(FunASRStreamingASR)
    asr.model = object()
    asr.reset()
    return asr


class DecoderOwnershipTests(unittest.TestCase):
    def test_two_connections_share_only_weights_not_audio_text_or_decoder_cache(self):
        model = prototype()
        vm = types.SimpleNamespace(utils=types.SimpleNamespace(get=lambda _: model))
        first, second = VoiceStream(vm), VoiceStream(vm)
        a = first.asr
        a._buf = np.ones(320, dtype=np.float32)
        a._text, a._cache['state'] = 'synthetic partial', [1]
        b = second.asr
        self.assertIsNot(a, b)
        self.assertIs(a.model, b.model)
        self.assertEqual(len(a._buf), 320)
        self.assertEqual(a._text, 'synthetic partial')
        self.assertEqual(b._text, '')
        self.assertIsNot(a._cache, b._cache)
        b.reset()
        self.assertEqual(a._cache, {'state': [1]})

    def test_parallel_account_loads_construct_the_model_once(self):
        initialize._streaming_model.cache_clear()
        self.addCleanup(initialize._streaming_model.cache_clear)
        with patch.object(initialize, 'FunASRStreamingASR', side_effect=lambda **_: prototype()) as load, \
                patch('voicemem.utils.audio.asr.pick_device', return_value='cpu'), \
                patch.dict('os.environ', {'STUDIO_BACKEND': 'mlx', 'VOICEMEM_ASR_DEVICE': 'cpu'}):
            with ThreadPoolExecutor(max_workers=4) as workers:
                models = list(workers.map(lambda _: initialize.streaming(), range(4)))
        self.assertEqual(load.call_count, 1)
        self.assertEqual(len({id(model) for model in models}), 4)
        self.assertEqual(len({id(model.model) for model in models}), 1)


class StreamLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_discards_queued_audio_resolves_flush_and_releases_worker(self):
        entered, release = threading.Event(), threading.Event()

        class ASR:
            def feed(self, _):
                entered.set()
                release.wait(2)
                return 'obsolete'

        worker = _AsrWorker(ASR())
        try:
            worker.push(np.ones(160))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            worker.push(np.ones(160))
            flushed = worker.flush()
            worker.close()
            self.assertEqual(await asyncio.wait_for(flushed, 1), '')
            self.assertEqual(await worker.flush(), '')
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                worker.push(np.ones(160))
        finally:
            release.set()
            worker.close()
            await asyncio.to_thread(worker.t.join, 1)
        self.assertFalse(worker.t.is_alive())
        self.assertIsNone(worker.asr)
        ref = weakref.ref(worker)
        del worker
        gc.collect()
        self.assertIsNone(ref())

    async def test_audio_initialization_runs_off_loop_and_vad_state_is_private(self):
        loop_thread = threading.get_ident()
        called = []

        class VAD:
            def new_stream(self):
                return object()

        model, vad = prototype(), VAD()

        def get(name):
            called.append(threading.get_ident())
            return model if name == 'asr' else vad

        vm = types.SimpleNamespace(utils=types.SimpleNamespace(get=get))
        a, b = VoiceStream(vm), VoiceStream(vm)
        await asyncio.gather(a.prepare_audio(), b.prepare_audio())
        self.assertTrue(all(thread != loop_thread for thread in called))
        self.assertIsNot(a._vad, b._vad)
        self.assertIsNot(a._asr, b._asr)
        await a.aclose()
        await b.aclose()
        self.assertIsNone(a._asr)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            _ = a.asr_worker
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            await a.feed_text('synthetic input')

    async def test_capture_generator_close_releases_stream_and_background_task(self):
        capture = Capture()
        capture.vm = object()
        capture.SPEC_MIN_CHARS, capture.GAMBLE_S, capture.CONFIRM_S = 6, .2, .2
        capture._eot = lambda: None
        task = None

        async def turns(_, stream, pause, background, **kwargs):
            nonlocal task
            self.assertIs(kwargs['on_frame'], on_frame)
            task = asyncio.create_task(asyncio.Event().wait())
            background.add(task)
            yield 'synthetic turn'

        capture._capture_turns = turns
        stream = types.SimpleNamespace(aclose=AsyncMock())
        on_frame = AsyncMock()
        with patch('studio.core.utils.capture.component.open_stream', return_value=stream):
            iterator = capture.anticipate(object(), on_frame)
            self.assertEqual(await anext(iterator), 'synthetic turn')
            await iterator.aclose()
        stream.aclose.assert_awaited_once()
        self.assertTrue(task.cancelled())

    async def test_realtime_capture_closes_on_reply_error_and_cancellation(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                closed, entered = asyncio.Event(), asyncio.Event()
                agent = RealtimeSession()
                agent.REPLY, agent.BARGE_DEBUG, agent.MIC_RATE = {}, False, 24000
                agent.ACTIVE_SPACE, agent.vm = 'synthetic-space', object()
                agent._turn_detection = lambda: {}
                agent._is_backchannel = lambda _: False
                agent._push_history = lambda *args, **kwargs: 'synthetic-history'
                agent.queue_remember_turn = unittest.mock.Mock()
                agent._SESSION_CONTEXT = types.SimpleNamespace(clear_session=unittest.mock.Mock())

                async def capture(*args, **kwargs):
                    try:
                        yield Pending('synthetic input', '', None)
                    finally:
                        closed.set()

                async def start(*args, **kwargs):
                    entered.set()
                    if cancel:
                        await asyncio.Event().wait()
                    raise RuntimeError('synthetic provider error')

                class Connection:
                    session = types.SimpleNamespace(update=AsyncMock())

                    async def __aiter__(self):
                        await asyncio.Event().wait()
                        yield None

                @asynccontextmanager
                async def connect(_):
                    yield Connection()

                agent.anticipate, agent.start_realtime_turn = capture, start
                socket = types.SimpleNamespace(send_json=AsyncMock(), send_bytes=AsyncMock())
                with patch('studio.core.utils.realtime_session.component.utils.realtime_connect', connect):
                    task = asyncio.create_task(agent.realtime_session(socket))
                    await asyncio.wait_for(entered.wait(), 1)
                    if cancel:
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else RuntimeError):
                        await task
                self.assertTrue(closed.is_set())
                agent._SESSION_CONTEXT.clear_session.assert_called_once()


if __name__ == '__main__':
    unittest.main()
