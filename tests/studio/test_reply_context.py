"""Offline checks for private Studio context and provider message boundaries."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from openai import AsyncOpenAI

from studio.core.utils.context.component import Context
from studio.core.utils.llm.initialize import create
from studio.core.utils.llm.local import LocalLLM
from voicemem.reply import compose_reply_messages


class ReplyContextTests(unittest.TestCase):
    def test_library_default_preserves_existing_provider_format(self):
        messages = compose_reply_messages('question', 'memory', system='persona')
        self.assertEqual(messages, [
            {'role': 'system', 'content': 'persona'},
            {'role': 'user', 'content': 'memory\n\nquestion'}])

    def test_system_context_preserves_history_and_separates_current_input(self):
        history = [{'role': 'assistant', 'content': 'earlier'}]
        messages = compose_reply_messages(
            'question', 'private instructions', history, system='persona',
            context_as_system=True)
        self.assertEqual(messages, [
            {'role': 'system', 'content': 'persona\n\nprivate instructions'}, *history,
            {'role': 'user', 'content': 'question'}])
        self.assertEqual(len(history), 1)
        self.assertEqual(compose_reply_messages(
            'question', '', history, system='persona', context_as_system=True),
            [{'role': 'system', 'content': 'persona'}, *history, messages[-1]])

    def test_stranger_context_never_contains_gate_decision_or_owner_memory(self):
        agent = Context()
        agent.SPACE_LANG = 'zh'
        agent._wants_sound = lambda _: False
        self.assertEqual(agent.build_reply_context(
            'owner memory', stranger=True, route='deep', text='算这个积分'), '')
        agent._rt_persona = lambda: 'persona'
        agent._SESSION_CONTEXT = SimpleNamespace(render=Mock())
        self.assertEqual(agent._realtime_instructions(
            'owner memory', stranger=True, context_session='fixture'), 'persona')
        agent._SESSION_CONTEXT.render.assert_not_called()

    def test_local_prewarm_and_generation_use_same_private_context_order(self):
        model = create('persona', 'local')
        self.assertIsInstance(model, LocalLLM)
        history = [{'role': 'assistant', 'content': 'earlier'}]
        context = 'private instructions'
        messages = model._msgs('question', context, history)
        self.assertEqual(messages, [
            {'role': 'system', 'content': f'persona\n\n{context}'}, *history,
            {'role': 'user', 'content': 'question'}])
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda rows, **_: ''.join(
                f"<{row['role']}>{row['content']}</>" for row in rows),
            encode=lambda text: list(text.encode()))
        model.load = lambda: (None, tokenizer)
        prefix = model._prefix_ids(context, history)
        generated_input = tokenizer.encode(model._render(messages))
        self.assertEqual(generated_input[:len(prefix)], prefix)


class StudioProviderContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_and_qwen_wire_context_is_not_user_speech(self):
        original = httpx.AsyncClient
        for provider in ('deepseek', 'qwen'):
            with self.subTest(provider=provider):
                requests = []

                def handle(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, text=(
                        'data: {"choices":[{"delta":{"content":"答案"}}]}\n\n'
                        'data: [DONE]\n\n'))

                with patch('httpx.AsyncClient', side_effect=lambda **kw: original(
                        transport=httpx.MockTransport(handle), **kw)):
                    model = create('persona', provider, 'test-only')
                    try:
                        self.assertEqual([part async for part in model(
                            '算这个积分', '不要复述后台上下文', [])], ['答案'])
                    finally:
                        await model.aclose()
                self.assertEqual(requests[0]['messages'], [
                    {'role': 'system', 'content': 'persona\n\n不要复述后台上下文'},
                    {'role': 'user', 'content': '算这个积分'}])

    async def test_openai_wire_context_is_not_user_speech(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=(
                'data: {"id":"fixture","object":"chat.completion.chunk","created":0,'
                '"model":"fixture","choices":[{"index":0,"delta":{"content":"答案"}}]}\n\n'
                'data: [DONE]\n\n'))

        client = AsyncOpenAI(api_key='test-only', http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handle)))
        try:
            with patch('openai.AsyncOpenAI', return_value=client):
                model = create('persona', 'openai', 'test-only')
                self.assertEqual([part async for part in model(
                    '算这个积分', '不要复述后台上下文', [])], ['答案'])
        finally:
            await client.close()
        self.assertEqual(requests[0]['messages'], [
            {'role': 'system', 'content': 'persona\n\n不要复述后台上下文'},
            {'role': 'user', 'content': '算这个积分'}])


if __name__ == '__main__':
    unittest.main()
