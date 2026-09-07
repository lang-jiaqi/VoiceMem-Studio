"""Local journals and concise console: no real models, API calls or user data."""
import asyncio
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "web")]
from logging_utils import _Tee
from voicemem import prompt_trace as trace
from voicemem.breeze_tts import BreezeMLXTTS
from voicemem.reply import deepseek_reply


class ConsoleTests(unittest.TestCase):
    def test_split_prints_filtered_console_but_full_file_preserved(self):
        console, logfile = io.StringIO(), io.StringIO()
        tee = _Tee(console, logfile, "stdout", threading.RLock(), concise=True)
        for piece in ("[asr] verbose\n", "[tts-prompt] ", '认真 "语速平稳"', "\n",
                      "[lat] 闭嘴→首帧 900ms｜后续中位 888ms\n",
                      "INFO: GET /api/memories 200 OK\n", "[web] 合成失败：test\n"):
            tee.write(piece)
        self.assertNotIn("verbose", console.getvalue())
        self.assertNotIn("GET", console.getvalue())
        self.assertNotIn("中位", console.getvalue())
        self.assertIn("[tts-prompt] 认真", console.getvalue())
        self.assertIn("900ms", console.getvalue())
        self.assertIn("失败", console.getvalue())
        self.assertIn("verbose", logfile.getvalue())
        self.assertIn("GET", logfile.getvalue())
        self.assertIn("中位", logfile.getvalue())

    def test_traceback_and_verbose_mode_survive(self):
        for concise in (True, False):
            console = io.StringIO()
            tee = _Tee(console, io.StringIO(), "stderr", threading.RLock(), concise=concise)
            error = 'Traceback (most recent call last):\n  File "a.py", line 1\n    run()\nValueError: invalid\n'
            tee.write(error)
            self.assertEqual(console.getvalue(), error)


class JournalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.writer = trace.PromptWriter(self.tmp.name)
        self.patch = patch.object(trace, "_writer", self.writer)
        self.patch.start()

    async def asyncTearDown(self):
        self.writer.flush()
        self.patch.stop()
        self.writer.queue.put(None)
        self.writer.thread.join(1)
        self.tmp.cleanup()

    def rows(self):
        self.writer.flush()
        return [[json.loads(line) for line in p.read_text().splitlines()]
                for p in self.writer.directory.glob("*.jsonl")]

    async def test_concurrent_turns_keep_llm_and_all_tts_segments_together(self):
        async def turn(label):
            with trace.prompt_scope(output_id=label):
                req = {"messages": [{"role": "user", "content": label}]}
                trace.record_request("llm", "test", req)
                req["messages"][0]["content"] = "mutated"
                await asyncio.sleep(0)
                async def child():
                    trace.record_request("tts", "test", {"text": label, "instruct": "温和"})
                await asyncio.create_task(child())
                trace.record_request("tts", "test", {"text": "second segment"})
        await asyncio.gather(turn("one"), turn("two"))
        files = self.rows()
        self.assertEqual(len(files), 2)
        for rows in files:
            self.assertEqual([r["kind"] for r in rows], ["llm", "tts", "tts"])
            self.assertEqual(len({r["output_id"] for r in rows}), 1)
            self.assertEqual(rows[0]["request"]["messages"][0]["content"], rows[0]["output_id"])

    async def test_deepseek_records_exact_wire_payload_without_credentials(self):
        import httpx
        original, wire = httpx.AsyncClient, []
        def handle(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, text='data: [DONE]\n\n')
        with patch("httpx.AsyncClient", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            provider = deepseek_reply(api_key="test-secret-not-a-real-key", system="完整人设")
            try:
                with trace.prompt_scope(output_id="test"):
                    self.assertEqual([s async for s in provider(
                        "当前问题", "干净记忆", [{"role": "assistant", "content": "历史"}])], [])
            finally:
                await provider.aclose()
        rows = self.rows()[0]
        self.assertEqual(rows[0]["request"], wire[0])
        self.assertNotIn("test-secret", json.dumps(rows))
        self.assertNotIn("Authorization", json.dumps(rows))

    async def test_breeze_records_effective_instruction_and_reference_without_gpu(self):
        tts = BreezeMLXTTS.__new__(BreezeMLXTTS)
        tts.__dict__.update(_lock=asyncio.Lock(), model_name="test-model", instruction="默认情绪",
                            ref_audio="/test/reference.wav", ref_text="参考原文", cfg_scale=1,
                            seed=42, max_tokens=750)
        out = queue.Queue()
        out.put(None)
        job = types.SimpleNamespace(out=out, cancel=lambda: None)
        loop = types.SimpleNamespace(iter=lambda *a, **kw: job)
        with patch("voicemem.utils.gpu_loop.gpu_loop", return_value=loop):
            with trace.prompt_scope(output_id="breeze"):
                self.assertEqual([x async for x in tts.stream("第一段", "语速平稳")], [])
        request = self.rows()[0][0]["request"]
        self.assertEqual(request["instruct"], "语速平稳")
        self.assertEqual(request["text"], "第一段")
        self.assertEqual(request["ref_text"], "参考原文")

    async def test_unscoped_synthesis_and_disabled_trace(self):
        trace.record_request("tts", "breeze_mlx", {"text": "嗯嗯"})
        self.assertEqual(self.rows()[0][0]["purpose"], "unscoped")
        with patch.object(trace, "_writer", None):
            trace.record_request("llm", "test", {"messages": []})
        self.assertEqual(len(self.rows()), 1)


if __name__ == "__main__":
    unittest.main()
