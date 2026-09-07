"""Offline DeepSeek SSE/config regressions: never contacts the network."""
import ast
import asyncio
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicemem.reply import deepseek_reply


class Body(httpx.AsyncByteStream):
    def __init__(self, payload, wait=False):
        self.payload, self.wait, self.closed = payload, wait, False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        # Split UTF-8 bytes and SSE records across transport chunks.
        for i in range(0, len(self.payload), 7):
            yield self.payload[i:i+7]
        if self.wait:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def event(data):
    return ("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode()


class DeepSeekTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_is_non_thinking_stream_and_preserves_history(self):
        body = Body(b": keepalive\n\n" + event({"choices": []}) +
                    event({"choices": [{"delta": {"reasoning_content": "not spoken"}}]}) +
                    event({"choices": [{"delta": {"content": "温和|你好。"}}]}) +
                    b"data: [DONE]\n\n")
        requests, clients = [], []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, stream=body)
        original = httpx.AsyncClient
        def client(**kw):
            c = original(transport=httpx.MockTransport(handle), **kw)
            clients.append(c)
            return c
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unrelated", "OPENAI_BASE_URL": "https://invalid.test"}), \
                patch("httpx.AsyncClient", side_effect=client):
            provider = deepseek_reply(api_key="test-only", system="persona")
            history = [{"role": "assistant", "content": "earlier"}]
            try:
                self.assertEqual([x async for x in provider("hello", "memory", history)], ["温和|你好。"])
                self.assertTrue(body.closed)
                payload = json.loads(requests[0].content)
                self.assertEqual(str(requests[0].url), "https://api.deepseek.com/chat/completions")
                self.assertEqual(requests[0].headers["authorization"], "Bearer test-only")
                self.assertEqual(payload["model"], "deepseek-v4-flash")
                self.assertEqual(payload["thinking"], {"type": "disabled"})
                self.assertTrue(payload["stream"])
                self.assertEqual(payload["max_tokens"], 512)
                self.assertEqual(payload["messages"], [
                    {"role": "system", "content": "persona"}, *history,
                    {"role": "user", "content": "memory\n\nhello"}])
                self.assertEqual(len(history), 1)
            finally:
                await provider.aclose()
        self.assertTrue(clients[0].is_closed)

    async def test_early_close_and_cancel_release_http_response(self):
        original = httpx.AsyncClient
        for cancel in (False, True):
            body = Body(event({"choices": [{"delta": {"content": "你好"}}]}), wait=True)
            transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
            with patch("httpx.AsyncClient", side_effect=lambda **kw: original(transport=transport, **kw)):
                provider = deepseek_reply(api_key="test-only")
                gen = provider("hello")
                try:
                    self.assertEqual(await anext(gen), "你好")
                    if cancel:
                        task = asyncio.create_task(anext(gen))
                        await asyncio.wait_for(body.waiting.wait(), 1)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await gen.aclose()
                    self.assertTrue(body.closed)
                finally:
                    await gen.aclose()
                    await provider.aclose()

    async def test_http_error_fails_without_retry(self):
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(401)
        original = httpx.AsyncClient
        with patch("httpx.AsyncClient", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            provider = deepseek_reply(api_key="test-only")
            try:
                with self.assertRaises(httpx.HTTPStatusError):
                    await anext(provider("hello"))
            finally:
                await provider.aclose()
        self.assertEqual(len(calls), 1)

    def test_no_implicit_openai_key_fallback_and_factory_support(self):
        from voicemem.config import _reply_factory
        with patch.dict(os.environ, {"OPENAI_API_KEY": "wrong-provider"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                deepseek_reply()
            self.assertTrue(callable(_reply_factory("deepseek", {"api_key": "test-only"})))

    def test_demo_cli_supports_deepseek_without_loading_models(self):
        import argparse
        tree = ast.parse((ROOT / "web/run.py").read_text())
        fn = next(n for n in tree.body if getattr(n, "name", "") == "_parse")
        ns = {"argparse": argparse, "os": os}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "web/run.py", "exec"), ns)
        args = ns["_parse"](["--mode", "llm_tts", "--llm", "deepseek"])
        self.assertEqual(args.llm, "deepseek")


if __name__ == "__main__":
    unittest.main()
