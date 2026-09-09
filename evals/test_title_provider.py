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


if __name__ == "__main__":
    unittest.main()
