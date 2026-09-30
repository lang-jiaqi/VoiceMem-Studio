"""Offline contract checks for the Qwen-Audio-TTS Studio adapter."""
import asyncio
import json
import os
import unittest
from unittest.mock import patch

from studio.core.utils.tts import qwen_audio_api as qwen


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


if __name__ == "__main__":
    unittest.main()
