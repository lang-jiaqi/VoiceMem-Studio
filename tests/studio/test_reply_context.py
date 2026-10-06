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
from studio.core.utils.prompts.component import system_prompt
from studio.harness.persona.policy import memory_reply_context
from voicemem.memory_api import build_memory_context
from voicemem.reply import compose_reply_messages


class ReplyContextTests(unittest.TestCase):
    def test_current_memory_rules_are_shared_by_text_and_realtime_without_gate_leaks(self):
        agent = Context()
        agent.SPACE_LANG = 'zh'
        agent.ACTIVE_SPACE = 'fixture'
        agent._rt_persona = lambda: 'persona'
        agent._history_block = lambda *_: ''
        agent._tone_note = lambda *_: ''
        agent._wants_sound = lambda _: False
        for memory in ('factual memory: synthetic fact',
                       'user emotion & characteristics: synthetic internal note', '', '  '):
            with self.subTest(memory=memory):
                expected = memory_reply_context(memory)
                text_context = agent.build_reply_context(memory, route='deep')
                realtime_context = agent._realtime_instructions(memory)
                if memory.strip():
                    self.assertEqual(text_context, expected)
                    self.assertEqual(realtime_context, 'persona\n\n' + expected)
                    self.assertIn('emotion/traits do not establish missing details', expected)
                    self.assertIn('Do not invent when the user said it', expected)
                else:
                    self.assertEqual(expected, '')
                    self.assertNotIn('Memory use for this reply', text_context)
                    self.assertNotIn('Memory use for this reply', realtime_context)
                self.assertEqual(agent.build_reply_context(
                    memory, route='deep', stranger=True), '')
                self.assertEqual(agent._realtime_instructions(
                    memory, stranger=True), 'persona')

    def test_memory_evidence_policy_applies_to_both_reply_modes_and_languages(self):
        for language in ('zh', 'en'):
            for tagged in (False, True):
                with self.subTest(language=language, tagged=tagged):
                    prompt = system_prompt(language, tagged=tagged)
                    self.assertIn('不能据此否定本轮已提供的相关记忆', prompt)
                    self.assertIn('用户明确更新或纠正的事实优先于旧记忆', prompt)
                    self.assertIn('记忆不相关、缺少所问细节或彼此冲突且无法判断', prompt)
                    self.assertIn('没写的细节不要补', prompt)

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
    async def test_retrieved_fact_and_correction_policy_reach_reply_after_prior_denials(self):
        requests = []
        fact = '用户家孩子高考考了612分。'
        memory = build_memory_context(SimpleNamespace(
            hits=[SimpleNamespace(text=fact, observed_at='2026-09-12')], rb_hits=[]))
        history = [
            {'role': 'user', 'content': '你还记得我孩子高考考了多少分吗？'},
            {'role': 'assistant', 'content': '我没有这方面的记录。'},
            {'role': 'user', 'content': '我之前说过的。'},
            {'role': 'assistant', 'content': '我确实不记得，不能猜一个数。'},
        ]
        original_history = [dict(message) for message in history]
        agent = Context()
        agent.SPACE_LANG = 'zh'
        agent._wants_sound = lambda _: False
        original_client = httpx.AsyncClient

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, text=(
                'data: {"choices":[{"delta":{"content":"合成测试回复"}}]}\n\n'
                'data: [DONE]\n\n'))

        with patch('httpx.AsyncClient', side_effect=lambda **kw: original_client(
                transport=httpx.MockTransport(handle), **kw)):
            model = create(system_prompt(tagged=True), 'deepseek', 'test-only')
            try:
                for text, stranger in (
                    ('我家孩子高考之前考了多少分？', False),
                    ('之前说错了，他其实考了622分。现在他考了多少分？', False),
                    ('高考满分是多少？', True),
                ):
                    context = agent.build_reply_context(
                        memory, text=text, route='deep', stranger=stranger)
                    self.assertEqual([part async for part in model(
                        text, context, [] if stranger else history)], ['合成测试回复'])
            finally:
                await model.aclose()

        self.assertEqual(len(requests), 3)
        for request in requests[:2]:
            messages = request['messages']
            self.assertEqual(messages[0]['role'], 'system')
            self.assertIn(fact, messages[0]['content'])
            self.assertIn('Prior assistant claims of not remembering', messages[0]['content'])
            self.assertIn('不能据此否定本轮已提供的相关记忆', messages[0]['content'])
            self.assertEqual(messages[1:-1], original_history)
        self.assertEqual(requests[0]['messages'][-1], {
            'role': 'user', 'content': '我家孩子高考之前考了多少分？'})
        self.assertEqual(requests[1]['messages'][-1], {
            'role': 'user', 'content': '之前说错了，他其实考了622分。现在他考了多少分？'})
        self.assertEqual(history, original_history)
        self.assertEqual(len(requests[2]['messages']), 2)
        self.assertNotIn(fact, requests[2]['messages'][0]['content'])

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
