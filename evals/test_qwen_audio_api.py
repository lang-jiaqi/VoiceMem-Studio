"""Offline contract checks for the Qwen-Audio-TTS Studio adapter."""
import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from studio.core.utils.tts import qwen_audio_api as qwen
from studio.core.utils.tts import initialize as speech_factory


class FakeSocket:
    def __init__(self, *, audio=b"\0\x01", fail=False):
        self.events = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.audio = audio
        self.fail = fail

    async def send(self, data):
        event = json.loads(data)
        self.sent.append(event)
        action = event["header"]["action"]
        task_id = event["header"]["task_id"]
        if action == "run-task":
            self.events.put_nowait(json.dumps({"header": {
                "task_id": task_id, "event": "task-started"}}))
        elif action == "finish-task":
            if self.fail:
                self.events.put_nowait(json.dumps({"header": {
                    "task_id": task_id, "event": "task-failed",
                    "error_code": "InvalidParameter", "error_message": "bad voice"}}))
            else:
                if self.audio:
                    self.events.put_nowait(self.audio)
                self.events.put_nowait(json.dumps({"header": {
                    "task_id": task_id, "event": "task-finished"}}))

    async def recv(self):
        return await self.events.get()

    async def close(self):
        self.closed = True


class QwenAudioContract(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "STUDIO_QWEN_TTS_API_KEY": "test-key",
            "STUDIO_QWEN_TTS_WORKSPACE_ID": "test-workspace",
            "STUDIO_QWEN_TTS_VOICE_ID": "old-cloned-voice",
            "STUDIO_QWEN_TTS_VOICE": "",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    async def test_instruction_and_pcm_stream_reuse(self):
        socket = FakeSocket()
        async def connect(*args, **kwargs):
            return socket
        with patch("websockets.connect", side_effect=connect) as opening:
            tts = qwen.QwenAudioAPI()
            self.assertEqual([chunk async for chunk in tts.stream("第一句", "温柔地说")], [b"\0\x01"])
            self.assertEqual([chunk async for chunk in tts.stream("第二句", "开心地说")], [b"\0\x01"])
            self.assertEqual(opening.call_count, 1)
            self.assertEqual([item["header"]["action"] for item in socket.sent], [
                "run-task", "continue-task", "finish-task",
                "run-task", "continue-task", "finish-task"])
            first = socket.sent[0]
            second = socket.sent[3]
            self.assertEqual(first["payload"]["parameters"]["instruction"], "温柔地说")
            self.assertEqual(second["payload"]["parameters"]["instruction"], "开心地说")
            self.assertEqual(first["payload"]["parameters"]["sample_rate"], 24000)
            self.assertEqual(first["payload"]["parameters"]["format"], "pcm")
            self.assertEqual(first["payload"]["parameters"]["voice"], "longanlingxi")
            self.assertEqual(first["payload"]["model"], qwen.MODEL)
            self.assertNotEqual(first["header"]["task_id"], second["header"]["task_id"])
            await tts.aclose()
            self.assertTrue(socket.closed)

    async def test_preconnect_does_not_start_synthesis(self):
        socket = FakeSocket()
        async def connect(*args, **kwargs):
            return socket
        with patch("websockets.connect", side_effect=connect) as opening:
            tts = qwen.QwenAudioAPI()
            await tts.preconnect()
            self.assertEqual(opening.call_count, 1)
            self.assertEqual(socket.sent, [])
            await tts.aclose()

    async def test_partial_stream_closes_socket(self):
        socket = FakeSocket()
        async def connect(*args, **kwargs):
            return socket
        with patch("websockets.connect", side_effect=connect):
            tts = qwen.QwenAudioAPI()
            stream = tts.stream("中断")
            self.assertEqual(await anext(stream), b"\0\x01")
            await stream.aclose()
            self.assertTrue(socket.closed)
            self.assertIsNone(tts._ws)

    async def test_task_failure_closes_socket(self):
        socket = FakeSocket(fail=True)
        async def connect(*args, **kwargs):
            return socket
        with patch("websockets.connect", side_effect=connect):
            tts = qwen.QwenAudioAPI()
            with self.assertRaisesRegex(RuntimeError, "bad voice"):
                [chunk async for chunk in tts.stream("失败")]
            self.assertTrue(socket.closed)

    async def test_empty_response_is_reported(self):
        socket = FakeSocket(audio=b"")
        async def connect(*args, **kwargs):
            return socket
        with patch("websockets.connect", side_effect=connect):
            tts = qwen.QwenAudioAPI()
            with self.assertRaisesRegex(RuntimeError, "without audio"):
                [chunk async for chunk in tts.stream("静音")]
            self.assertTrue(socket.closed)

    def test_system_voice_ignores_old_clone_id(self):
        self.assertEqual(qwen.settings()[2], "longanlingxi")

    def test_system_voice_can_be_selected(self):
        with patch.dict(os.environ, {"STUDIO_QWEN_TTS_VOICE": "longanfengyue"}):
            self.assertEqual(qwen.settings()[2], "longanfengyue")

    def test_invalid_system_voice_is_rejected(self):
        with patch.dict(os.environ, {"STUDIO_QWEN_TTS_VOICE": "two words"}):
            with self.assertRaisesRegex(ValueError, "系统音色格式无效"):
                qwen.settings()

    def test_demo_selection_and_desktop_override(self):
        with patch.dict(os.environ, {"STUDIO_PUBLIC_DEMO": "1", "STUDIO_TTS_PROVIDER": ""}):
            self.assertTrue(qwen.selected())
            with patch.dict(os.environ, {"STUDIO_TTS_PROVIDER": "breeze"}):
                self.assertFalse(qwen.selected())

    def test_voice_from_other_model_is_rejected(self):
        with patch.dict(os.environ, {"STUDIO_QWEN_TTS_VOICE": "qwen3-tts-vc-realtime-voice"}):
            with self.assertRaisesRegex(ValueError, "不支持"):
                qwen.settings()


class DemoSpeechSessions(unittest.IsolatedAsyncioTestCase):
    setUp = QwenAudioContract.setUp

    def agent(self, provider, *, public_demo=True):
        return SimpleNamespace(PUBLIC_DEMO=public_demo, UI_LANG='zh',
                               vm=SimpleNamespace(utils=SimpleNamespace(get=Mock(return_value=provider))))

    async def test_same_account_sessions_overlap_reuse_and_interrupt_independently(self):
        sockets = [FakeSocket(audio=b'\x01\x00'), FakeSocket(audio=b'\x02\x00')]
        pool = qwen.QwenDemoTTS(max_sessions=2)
        agent = self.agent(pool)
        with patch('websockets.connect', new=AsyncMock(side_effect=sockets)) as opening:
            await pool.preconnect()
            async with speech_factory.conversation_speech(agent) as first, \
                    speech_factory.conversation_speech(agent) as second:
                self.assertIsNot(first, second)
                self.assertIsNot(first._lock, second._lock)
                self.assertEqual(first.api_key, second.api_key)
                self.assertEqual(first.workspace, second.workspace)
                self.assertEqual(first.voice, second.voice)
                self.assertIsNone(pool._ws)
                self.assertEqual(opening.await_count, 2)
                for call in opening.await_args_list:
                    self.assertIn(first.workspace, call.args[0])
                    self.assertEqual(call.kwargs['additional_headers'],
                                     {'Authorization': f'Bearer {first.api_key}'})
                a, b = first.stream('first', 'warm'), second.stream('second', 'serious')
                try:
                    self.assertEqual(await anext(a), b'\x01\x00')
                    # The first generator still owns its lock when the second speaks.
                    self.assertEqual(await asyncio.wait_for(anext(b), 1), b'\x02\x00')
                    await a.aclose()
                    self.assertTrue(sockets[0].closed)
                    self.assertFalse(sockets[1].closed)
                    with self.assertRaises(StopAsyncIteration):
                        await anext(b)
                    self.assertEqual([x async for x in second.stream('next turn')], [b'\x02\x00'])
                    self.assertEqual(opening.await_count, 2)
                finally:
                    await a.aclose()
                    await b.aclose()
            self.assertTrue(all(socket.closed for socket in sockets))
            self.assertEqual(pool._sessions, set())

    async def test_limit_rejects_immediately_and_exit_releases_slot(self):
        pool = qwen.QwenDemoTTS(max_sessions=1)
        with patch('websockets.connect', new=AsyncMock(side_effect=[FakeSocket(), FakeSocket()])) as opening:
            async with pool.session():
                with self.assertRaises(qwen.DemoSpeechBusy):
                    async with pool.session():
                        self.fail('full session must not enter')
                self.assertEqual(opening.await_count, 1)
            async with pool.session():
                self.assertEqual(opening.await_count, 2)
            self.assertEqual(pool._sessions, set())

    async def test_cancelled_connect_counts_toward_limit_then_releases_slot(self):
        pool = qwen.QwenDemoTTS(max_sessions=1)
        entered = asyncio.Event()

        async def connect(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        async def session():
            async with pool.session():
                self.fail('cancelled connection must not enter')

        with patch('websockets.connect', side_effect=connect):
            task = asyncio.create_task(session())
            try:
                await asyncio.wait_for(entered.wait(), 1)
                with self.assertRaises(qwen.DemoSpeechBusy):
                    async with pool.session():
                        pass
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(pool._sessions, set())
        with patch('websockets.connect', new=AsyncMock(return_value=FakeSocket())):
            async with pool.session():
                pass

    async def test_failed_connect_releases_slot_and_next_session_can_retry(self):
        pool = qwen.QwenDemoTTS(max_sessions=1)
        with patch('websockets.connect', new=AsyncMock(side_effect=[OSError('synthetic'), FakeSocket()])):
            with self.assertRaises(qwen.DemoSpeechUnavailable):
                async with pool.session():
                    pass
            self.assertEqual(pool._sessions, set())
            async with pool.session():
                pass

    async def test_shutdown_closes_leases_and_unused_startup_socket(self):
        pool = qwen.QwenDemoTTS(max_sessions=2)
        sockets = [FakeSocket(), FakeSocket()]
        with patch('websockets.connect', new=AsyncMock(side_effect=sockets)):
            async with pool.session(), pool.session():
                await pool.aclose()
                self.assertTrue(all(socket.closed for socket in sockets))
                with self.assertRaises(qwen.DemoSpeechUnavailable):
                    async with pool.session():
                        pass
        self.assertEqual(pool._sessions, set())
        unused, socket = qwen.QwenDemoTTS(), FakeSocket()
        with patch('websockets.connect', new=AsyncMock(return_value=socket)):
            await unused.preconnect()
            await unused.aclose()
        self.assertTrue(socket.closed)

    async def test_app_does_not_lease_or_close_its_shared_provider(self):
        shared = qwen.QwenAudioAPI()
        agent = self.agent(shared, public_demo=False)
        async with speech_factory.conversation_speech(agent) as speech:
            self.assertIsNone(speech)
        agent.vm.utils.get.assert_not_called()

    async def test_shutdown_finishes_other_closes_before_reporting_one_failure(self):
        pool = qwen.QwenDemoTTS(max_sessions=2)
        sockets = [FakeSocket(), FakeSocket()]
        with patch('websockets.connect', new=AsyncMock(side_effect=sockets)):
            async with pool.session() as first, pool.session():
                with patch.object(first, 'aclose', new=AsyncMock(side_effect=OSError('synthetic close failure'))):
                    with self.assertRaisesRegex(OSError, 'synthetic close failure'):
                        await pool.aclose()
                    self.assertTrue(sockets[1].closed)
            self.assertTrue(all(socket.closed for socket in sockets))

    async def test_local_demo_provider_keeps_existing_factory(self):
        shared = SimpleNamespace(stream=AsyncMock(), aclose=AsyncMock())
        async with speech_factory.conversation_speech(self.agent(shared)) as speech:
            self.assertIsNone(speech)
        shared.stream.assert_not_called()
        shared.aclose.assert_not_called()

    def test_factory_switch_is_demo_only_and_app_reuse_is_preserved(self):
        self.addCleanup(speech_factory._create.cache_clear)
        with patch.dict(os.environ, {'STUDIO_TTS_PROVIDER': 'qwen_audio_api',
                                    'STUDIO_PUBLIC_DEMO': '0',
                                    'STUDIO_DEMO_MAX_TTS_SESSIONS': 'invalid'}):
            speech_factory._create.cache_clear()
            app = speech_factory.create()
            self.assertIs(type(app), qwen.QwenAudioAPI)
            self.assertIs(speech_factory.create(), app)
        with patch.dict(os.environ, {'STUDIO_TTS_PROVIDER': 'qwen_audio_api',
                                    'STUDIO_PUBLIC_DEMO': '1',
                                    'STUDIO_DEMO_MAX_TTS_SESSIONS': '3'}):
            speech_factory._create.cache_clear()
            demo = speech_factory.create()
            self.assertIsInstance(demo, qwen.QwenDemoTTS)
            self.assertEqual(demo.max_sessions, 3)
            self.assertIs(speech_factory.create(), demo)

    def test_limit_configuration_validation(self):
        with patch.dict(os.environ, {'STUDIO_DEMO_MAX_TTS_SESSIONS': '2'}):
            self.assertEqual(qwen.demo_session_limit(), 2)
        for value in ('0', '-1', '1.5', 'invalid', ''):
            with self.subTest(value=value), patch.dict(os.environ, {'STUDIO_DEMO_MAX_TTS_SESSIONS': value}):
                with self.assertRaises(ValueError):
                    qwen.demo_session_limit()

    async def test_conversation_cleanup_precedes_connection_release_on_exit_and_cancel(self):
        from studio.core.core import converse
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                pool, socket = qwen.QwenDemoTTS(), FakeSocket()
                entered = asyncio.Event()

                async def listen():
                    entered.set()
                    if cancel:
                        await asyncio.Event().wait()
                    if False:
                        yield None

                async def cleanup():
                    self.assertFalse(socket.closed)
                    self.assertIs(session.speech_provider._ws, socket)

                session = SimpleNamespace(listen=listen, close_session=AsyncMock(side_effect=cleanup))
                with patch('studio.core.utils.conversation.component.Conversation', return_value=session), \
                        patch('websockets.connect', new=AsyncMock(return_value=socket)):
                    task = asyncio.create_task(converse(self.agent(pool), Mock()))
                    await asyncio.wait_for(entered.wait(), 1)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await asyncio.wait_for(task, 1)
                session.close_session.assert_awaited_once()
                self.assertTrue(socket.closed)
                self.assertEqual(pool._sessions, set())

    async def test_busy_or_failed_session_reports_browser_error_before_listening(self):
        from studio.core.core import converse
        for busy in (False, True):
            with self.subTest(busy=busy):
                pool = qwen.QwenDemoTTS(max_sessions=1)
                if busy:
                    pool._sessions.add(object())
                socket = SimpleNamespace(send_json=AsyncMock(), close=AsyncMock())
                session = SimpleNamespace(listen=Mock(), close_session=AsyncMock())
                with patch('studio.core.utils.conversation.component.Conversation', return_value=session), \
                        patch('websockets.connect', new=AsyncMock(side_effect=OSError('private fixture detail'))):
                    await converse(self.agent(pool), socket)
                session.listen.assert_not_called()
                message = socket.send_json.call_args.args[0]
                self.assertEqual(message['type'], 'error')
                self.assertEqual(message['code'], 'demo_speech_busy' if busy else 'demo_speech_unavailable')
                self.assertNotIn('private fixture detail', message['message'])
                self.assertIn('人数已满' if busy else '连接失败', message['message'])
                socket.close.assert_awaited_once_with(code=1013)


if __name__ == "__main__":
    unittest.main()
