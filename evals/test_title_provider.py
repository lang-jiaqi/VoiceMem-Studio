"""Title generation must follow the configured Studio reply provider."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
import utils as web_utils


class _Completions:
    def __init__(self, calls):
        self.calls = calls

    async def create(self, **request):
        self.calls.append(request)
        message = types.SimpleNamespace(content="路由模型配置")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)])


class TitleProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_title_uses_deepseek_credentials_and_model(self):
        calls = []
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions(calls)))
        reply = {"llm": {"provider": "deepseek", "config": {
            "model": "deepseek-v4-flash",
        }}}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch.object(web_utils, "AsyncOpenAI", return_value=client) as factory:
            title = await web_utils.make_title_generator(reply)("测试一下路由模型")

        self.assertEqual(title, "路由模型配置")
        factory.assert_called_once_with(
            api_key="test-key", base_url="https://api.deepseek.com")
        self.assertEqual(calls[0]["model"], "deepseek-v4-flash")
        self.assertEqual(
            calls[0]["extra_body"], {"thinking": {"type": "disabled"}})

    async def test_openai_title_keeps_the_existing_client(self):
        calls = []
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions(calls)))
        with patch.object(web_utils, "_openai_client", return_value=client):
            title = await web_utils.make_title_generator()("普通对话")

        self.assertEqual(title, "路由模型配置")
        self.assertEqual(calls[0]["model"], web_utils.CHAT_MODEL)
        self.assertNotIn("extra_body", calls[0])

    async def test_openai_title_uses_the_studio_key_after_memory_configuration(self):
        calls = []
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions(calls)))
        reply = {"llm": {"provider": "openai", "config": {
            "model": "gpt-4o", "base_url": "https://api.openai.com/v1",
        }}}
        with patch.dict(os.environ, {
            "VOICEMEM_STUDIO_API_KEY": "studio-key",
            "OPENAI_API_KEY": "memory-key",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
        }, clear=True), patch.object(web_utils, "AsyncOpenAI", return_value=client) as factory:
            await web_utils.make_title_generator(reply)("测试隔离")
        factory.assert_called_once_with(
            api_key="studio-key", base_url="https://api.openai.com/v1")

    async def test_local_reply_title_uses_the_memory_provider_fallback(self):
        calls = []
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions(calls)))
        reply = {"llm": {"provider": "local", "config": {}}}
        memory = {"provider": "deepseek", "config": {
            "model": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",
            "api_key": "memory-key",
        }}
        with patch.object(web_utils, "AsyncOpenAI", return_value=client) as factory:
            await web_utils.make_title_generator(reply, memory)("本地回复的标题")
        factory.assert_called_once_with(
            api_key="memory-key", base_url="https://api.deepseek.com")
        self.assertEqual(calls[0]["model"], "deepseek-v4-flash")


if __name__ == "__main__":
    unittest.main()
