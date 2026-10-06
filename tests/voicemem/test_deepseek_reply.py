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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from voicemem.reply import deepseek_reply, reply_request_options


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
    async def test_qwen_stream_uses_dashscope_options(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            self.assertEqual(str(request.url), "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")
            return httpx.Response(200, stream=Body(event({"choices": [{"delta": {"content": "好"}}]}) + b"data: [DONE]\n\n"))
        original = httpx.AsyncClient
        with patch("httpx.AsyncClient", side_effect=lambda **kw: original(transport=httpx.MockTransport(handle), **kw)):
            provider = deepseek_reply(model="qwen3.6-flash", api_key="fixture", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", protocol="qwen")
            try:
                for effort in ("none", "high"):
                    with reply_request_options(reasoning_effort=effort):
                        self.assertEqual([x async for x in provider("你好")], ["好"])
                    self.assertEqual(requests[-1]["enable_thinking"], effort == "high")
                    self.assertNotIn("thinking", requests[-1])
                    self.assertNotIn("reasoning_effort", requests[-1])
                    self.assertTrue(requests[-1]["stream"])
            finally:
                await provider.aclose()

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

    async def test_thinking_effort_is_dynamic_and_reasoning_is_not_spoken(self):
        body = Body(
            event({"choices": [{"delta": {"reasoning_content": "private"}}]})
            + event({"choices": [{"delta": {"content": "答案"}}]})
            + b"data: [DONE]\n\n")
        requests = []
        original = httpx.AsyncClient

        def handle(request):
            requests.append(request)
            return httpx.Response(200, stream=body)

        with patch("httpx.AsyncClient", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            provider = deepseek_reply(api_key="test-only")
            try:
                with reply_request_options(reasoning_effort="low"):
                    result = [item async for item in provider("需要解释的问题")]
            finally:
                await provider.aclose()

        self.assertEqual(result, ["答案"])
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_tokens"], 1024)

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

    async def test_first_token_timeout_retries_with_fresh_connection(self):
        bodies = [Body(b": keepalive\n\n", wait=True),
                  Body(event({"choices": [{"delta": {"content": "好了"}}]}) +
                       b"data: [DONE]\n\n")]
        clients = []
        original = httpx.AsyncClient

        def client(**kw):
            body = bodies[len(clients)]
            transport = httpx.MockTransport(
                lambda _: httpx.Response(200, stream=body))
            made = original(transport=transport, **kw)
            clients.append(made)
            return made

        with patch.dict(os.environ,
                        {"VOICEMEM_DEEPSEEK_FIRST_TOKEN_TIMEOUT": "0.02"}), \
                patch("httpx.AsyncClient", side_effect=client):
            provider = deepseek_reply(api_key="test-only")
            try:
                self.assertEqual([x async for x in provider("hello")], ["好了"])
            finally:
                await provider.aclose()
        self.assertEqual(len(clients), 2)
        self.assertTrue(all(c.is_closed for c in clients))
        self.assertTrue(all(b.closed for b in bodies))

    async def test_two_first_token_timeouts_fail_in_bounded_time(self):
        bodies = [Body(b": keepalive\n\n", wait=True),
                  Body(b": keepalive\n\n", wait=True)]
        original = httpx.AsyncClient
        made = []

        def client(**kw):
            body = bodies[len(made)]
            c = original(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=body)), **kw)
            made.append(c)
            return c

        with patch.dict(os.environ,
                        {"VOICEMEM_DEEPSEEK_FIRST_TOKEN_TIMEOUT": "0.02"}), \
                patch("httpx.AsyncClient", side_effect=client):
            provider = deepseek_reply(api_key="test-only")
            try:
                with self.assertRaisesRegex(TimeoutError, "连续两次"):
                    await asyncio.wait_for(anext(provider("hello")), 0.5)
            finally:
                await provider.aclose()
        self.assertEqual(len(made), 2)
        self.assertTrue(all(b.closed for b in bodies))

    async def test_reasoning_only_completion_retries_without_reducing_effort(self):
        for protocol in ("deepseek", "qwen"):
            bodies = [Body(
                event({"choices": [{"delta": {"reasoning_content": "private"}}]})
                + event({"choices": [{"delta": {}, "finish_reason": "length"}]})
                + b"data: [DONE]\n\n"),
                Body(event({"choices": [{"delta": {"content": "答案"}}]})
                     + b"data: [DONE]\n\n")]
            requests, clients = [], []
            original = httpx.AsyncClient

            def client(**kw):
                body = bodies[len(clients)]
                def handle(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, stream=body)
                made = original(transport=httpx.MockTransport(handle), **kw)
                clients.append(made)
                return made

            with self.subTest(protocol=protocol), patch("httpx.AsyncClient", side_effect=client):
                provider = deepseek_reply(api_key="fixture", protocol=protocol)
                try:
                    with reply_request_options(reasoning_effort="high"):
                        self.assertEqual([x async for x in provider("推导", "context")], ["答案"])
                finally:
                    await provider.aclose()
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(requests[1]["max_tokens"], 4096)
            self.assertTrue(all(c.is_closed for c in clients))
            self.assertTrue(all(b.closed for b in bodies))

    async def test_repeated_reasoning_only_response_is_an_error(self):
        bodies, clients = [], []
        original = httpx.AsyncClient
        def client(**kw):
            body = Body(event({"choices": [{"delta": {"reasoning_content": "private"}}]})
                        + event({"choices": [{"delta": {}, "finish_reason": "length"}]})
                        + b"data: [DONE]\n\n")
            bodies.append(body)
            made = original(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=body)), **kw)
            clients.append(made)
            return made
        with patch("httpx.AsyncClient", side_effect=client):
            provider = deepseek_reply(api_key="fixture")
            try:
                with self.assertRaisesRegex(RuntimeError, "no answer content.*finish_reason=length"):
                    await anext(provider("推导"))
            finally:
                await provider.aclose()
        self.assertEqual(len(clients), 2)
        self.assertTrue(all(b.closed for b in bodies))

    async def test_empty_done_is_not_success(self):
        calls = []
        original = httpx.AsyncClient
        def handle(request):
            calls.append(request)
            return httpx.Response(200, stream=Body(b"data: [DONE]\n\n"))
        with patch("httpx.AsyncClient", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            provider = deepseek_reply(api_key="fixture")
            try:
                with self.assertRaisesRegex(RuntimeError, "空回复"):
                    await anext(provider("hello"))
            finally:
                await provider.aclose()
        self.assertEqual(len(calls), 2)

    async def test_partial_answer_failure_does_not_duplicate_output(self):
        body = Body(event({"choices": [{"delta": {"content": "部分答案"}}]}))
        calls = []
        original = httpx.AsyncClient
        def handle(request):
            calls.append(request)
            return httpx.Response(200, stream=body)
        with patch("httpx.AsyncClient", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            provider = deepseek_reply(api_key="fixture")
            stream = provider("hello")
            try:
                self.assertEqual(await anext(stream), "部分答案")
                with self.assertRaisesRegex(RuntimeError, "before.*DONE"):
                    await anext(stream)
            finally:
                await stream.aclose()
                await provider.aclose()
        self.assertEqual(len(calls), 1)
        self.assertTrue(body.closed)

    async def test_cancel_during_reasoning_closes_without_retry(self):
        body = Body(event({"choices": [{"delta": {"reasoning_content": "private"}}]}), wait=True)
        calls = []
        original = httpx.AsyncClient
        def handle(request):
            calls.append(request)
            return httpx.Response(200, stream=body)
        with patch("httpx.AsyncClient", side_effect=lambda **kw: original(
                transport=httpx.MockTransport(handle), **kw)):
            provider = deepseek_reply(api_key="fixture")
            stream = provider("推导")
            task = asyncio.create_task(anext(stream))
            try:
                await asyncio.wait_for(body.waiting.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                await stream.aclose()
                await provider.aclose()
        self.assertEqual(len(calls), 1)
        self.assertTrue(body.closed)

    def test_no_implicit_openai_key_fallback_and_factory_support(self):
        from voicemem.config import _reply_factory
        with patch.dict(os.environ, {"OPENAI_API_KEY": "wrong-provider"}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
                deepseek_reply()
            self.assertTrue(callable(_reply_factory("deepseek", {"api_key": "test-only"})))

    def test_demo_cli_supports_deepseek_without_loading_models(self):
        from studio.core.utils.cli.component import parse_args
        args = parse_args(["--mode", "llm_tts", "--llm", "deepseek"])
        self.assertEqual(args.llm, "deepseek")


if __name__ == "__main__":
    unittest.main()
