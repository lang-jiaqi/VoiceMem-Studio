"""Explicit prompt edits, provider snapshots, and confirmed retrieval delivery."""
import asyncio
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from studio.harness.prompts import catalog, effective_prompts, prompt_scope, resolve, validate_overrides
from studio.core.utils.self_harness.component import SelfHarnessState, normalize_snapshot
from studio.core.utils.prompts.component import system_prompt
from studio.core.utils.reply_modes.component import QwenThinkingRouter
from studio.core.utils.turn_taking.filler import generate_local_filler
from tests.helpers.conversation import ConversationFixture
from voicemem.reply import compose_reply_messages, reply_request_options


class PromptEditsTests(unittest.TestCase):
    def test_localized_defaults_validate_without_becoming_custom_overrides(self):
        for language in ('zh', 'en'):
            with self.subTest(language=language):
                defaults = {key: spec['default'] for key, spec in catalog(language).items()}
                self.assertEqual(validate_overrides(defaults, language=language), {})
                self.assertIn('是', defaults['router'])
                self.assertIn('否', defaults['router'])
                for key in ('self_harness', 'tone_rule'):
                    self.assertIn('<self_harness>', defaults[key])
                    self.assertIn('</self_harness>', defaults[key])
        self.assertNotEqual(catalog('en')['main_system']['default'], catalog()['main_system']['default'])
        for key in ('tts_zh', 'tts_en'):
            self.assertEqual(catalog('en')[key], catalog('zh')[key])
            self.assertNotIn(key, effective_prompts({'prompt_language': 'en'}))

    def test_language_and_custom_prompts_are_captured_independently(self):
        state = SelfHarnessState()
        state.set_prompt_overrides({'main_system': 'Synthetic custom identity'})
        chinese = normalize_snapshot(state.snapshot())
        state.prompt_language = 'en'
        english = normalize_snapshot(state.snapshot())
        self.assertEqual(effective_prompts(chinese), {'main_system': 'Synthetic custom identity'})
        self.assertEqual(effective_prompts(english)['main_system'], 'Synthetic custom identity')
        self.assertEqual(effective_prompts(english)['router'], catalog('en')['router']['default'])
        state.set_prompt_overrides({})
        self.assertEqual(effective_prompts(state.snapshot())['main_system'], catalog('en')['main_system']['default'])
        self.assertEqual(effective_prompts(english)['main_system'], 'Synthetic custom identity')

    def test_defaults_round_trip_and_empty_updates_restore_defaults(self):
        defaults = {key: spec['default'] for key, spec in catalog().items()}
        self.assertEqual(validate_overrides(defaults), {})
        state = SelfHarnessState()
        state.set_prompt_overrides({'main_system': 'Synthetic system', 'persona': 'Synthetic role'})
        snapshot = state.snapshot()
        state.set_prompt_overrides({})
        self.assertEqual(state.prompts['persona'], '')
        self.assertEqual(state.prompt_overrides, {})
        self.assertEqual(snapshot['prompt_overrides']['main_system'], 'Synthetic system')

    def test_invalid_atomic_update_does_not_replace_previous_prompts(self):
        state = SelfHarnessState()
        state.set_prompt_overrides({'main_system': 'Synthetic system'})
        for value in ({'main_system': 'Changed', 'unknown': 'text'},
                      {'main_system': ''}, {'router': 'wrong protocol'},
                      {'filler_input': '{history:999999999999}'},
                      {'filler_system': '{task_context.__class__}'},
                      {'filler_system': '{task_context!r}'},
                      {'tone_rule': 'No control markers'}, {'tts_en': 'x' * 20001},
                      {'persona': 'x' * (catalog()['persona']['max_length'] + 1)}):
            with self.subTest(value=list(value)):
                with self.assertRaises(ValueError):
                    state.set_prompt_overrides(value)
                self.assertEqual(state.prompt_overrides, {'main_system': 'Synthetic system'})

    def test_model_control_cannot_edit_free_text_prompts(self):
        state = SelfHarnessState()
        with self.assertRaises(ValueError):
            state.apply({'prompt_overrides': {'main_system': 'Changed'}})
        self.assertEqual(state.prompt_overrides, {})

    def test_system_assembly_replaces_only_selected_parts(self):
        base = system_prompt(tagged=True)
        edited = system_prompt(tagged=True, overrides={'main_system': 'Synthetic identity'})
        self.assertTrue(edited.startswith('Synthetic identity\n\n'))
        self.assertIn(catalog()['speaking_style']['default'], edited)
        self.assertIn(catalog()['self_harness']['default'], edited)
        self.assertEqual(system_prompt(tagged=True), base)

    def test_router_cache_does_not_reuse_another_prompt(self):
        router = QwenThinkingRouter()
        router._predict = Mock(side_effect=['否', '是'])
        self.assertEqual(router.classify('Synthetic request').level, 'fast')
        with prompt_scope({'router': 'Synthetic policy: 是 or 否'}):
            self.assertEqual(router.classify('Synthetic request').level, 'slow')
            self.assertEqual(router.classify('Synthetic request').level, 'slow')
        self.assertEqual(router.classify('Synthetic request').level, 'fast')
        self.assertEqual(router._predict.call_count, 2)

    def test_local_adapter_uses_request_system_without_mutating_shared_model(self):
        from studio.core.utils.llm.local import LocalLLM
        model = LocalLLM.__new__(LocalLLM)
        model.system, model.context_as_system = 'Original', True
        with reply_request_options(system_prompt='Edited'):
            messages = model._msgs('Synthetic request', 'Synthetic memory', [])
        self.assertEqual(messages[0]['content'], 'Edited\n\nSynthetic memory')
        self.assertEqual(model.system, 'Original')

    def test_new_conversations_default_to_fastest_speech(self):
        self.assertEqual(SelfHarnessState().profile['speaking_style']['speech_rate'], 'very_fast')


class AsyncPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_routing_uses_captured_template_language(self):
        from studio.core.utils.routing.component import Routing
        from studio.core.utils.reply_modes.initialize import FAST, ThinkingDecision
        from studio.core.utils.contracts.component import Pending
        from voicemem.stream import empty_result
        agent = Routing()
        agent._THINKING_ROUTER_ON = True
        agent._replay_id = lambda *_: ''
        agent.vm = object()
        systems = []

        async def classify(*_, **__):
            systems.append(resolve('router', 'Default'))
            return ThinkingDecision(FAST, '否')

        router = types.SimpleNamespace(classify_async=classify)
        pending = Pending('Synthetic question', '', empty_result(), route='shallow')
        pending.self_harness_profile = {'prompt_language': 'en'}
        with patch('studio.core.utils.routing.component.thinking_router', return_value=router):
            await agent.route_pending_thinking(pending)
        self.assertEqual(systems, [catalog('en')['router']['default']])
        self.assertEqual(resolve('router', 'Default'), 'Default')

    async def test_language_and_prompt_controls_are_ordered_before_opening(self):
        import json
        from tests.helpers.capture import anticipate_namespace
        ns = anticipate_namespace()
        ns['vm'] = types.SimpleNamespace(stream=lambda **_: types.SimpleNamespace())
        events = []

        async def language(value):
            events.append(('language', value))

        async def prompts(value, request_id, **kwargs):
            events.append(('prompts', value, request_id, kwargs))

        class Sock:
            def __init__(self):
                self.messages = iter([
                    {'text': json.dumps({'type': 'conversation_language', 'language': 'en'})},
                    {'text': json.dumps({'type': 'harness_prompts_update', 'prompts': {},
                                         'request_id': 'save', 'prompt_language': 'en'})},
                    {'text': json.dumps({'type': 'conversation_start'})},
                    {'type': 'websocket.disconnect'},
                ])

            async def receive(self):
                return next(self.messages)

        with patch('studio.core.utils.turn_taking.backchannel.emitting', return_value=False):
            turns = [turn async for turn in ns['anticipate'](
                Sock(), on_language_config=language, on_harness_prompts_update=prompts,
                on_conversation_start=lambda: events.append(('opening',)))]
        self.assertEqual(turns, [])
        self.assertEqual(events, [('language', 'en'), ('prompts', {}, 'save', {'prompt_language': 'en'}), ('opening',)])

    async def test_english_filler_templates_keep_actual_output_language(self):
        router = types.SimpleNamespace(generate_short_text_async=AsyncMock(return_value='让我认真想一想'))
        await generate_local_filler('Synthetic input', router=router, lang='zh',
                                    prompts=effective_prompts({'prompt_language': 'en'}))
        system, prompt = router.generate_short_text_async.call_args.args
        self.assertTrue(system.startswith('You are a voice assistant'))
        self.assertIn('Output language: 中文', prompt)
        self.assertIn('<input>Synthetic input</input>', prompt)

    async def test_request_scopes_are_independent_and_restore_after_cancellation(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def request(name):
            with reply_request_options(system_prompt=name), prompt_scope({'main_system': name}):
                entered.set()
                await release.wait()
                return (compose_reply_messages('Input', system='Default')[0]['content'],
                        await asyncio.to_thread(resolve, 'main_system', 'Default'))

        first, second = asyncio.create_task(request('First')), asyncio.create_task(request('Second'))
        cancelled = asyncio.create_task(request('Cancelled'))
        await entered.wait()
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        release.set()
        self.assertEqual(await asyncio.gather(first, second), [('First', 'First'), ('Second', 'Second')])
        self.assertEqual(compose_reply_messages('Input', system='Default')[0]['content'], 'Default')

    async def test_filler_uses_edited_templates(self):
        router = types.SimpleNamespace(generate_short_text_async=AsyncMock(return_value='让我认真想一想'))
        await generate_local_filler('Synthetic input', router=router, prompts={
            'filler_system': 'System {task_context}',
            'filler_input': '{language}\n{history}\nCurrent: {task_context}',
        })
        args = router.generate_short_text_async.call_args.args
        self.assertEqual(args[0], 'System ')
        self.assertIn('Current: Synthetic input', args[1])


class PreparationMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = ConversationFixture()
        self.f.setUp()
        self.addAsyncCleanup(self.f.asyncTearDown)

    async def test_language_switch_keeps_custom_text_and_rejects_stale_saves(self):
        await self.f.session.update_harness_prompts({'main_system': '自定义角色'})
        await self.f.session.configure_language('en')
        state = self.f.messages[-1]
        self.assertEqual(state['prompt_catalog'], catalog('en'))
        self.assertEqual(state['snapshot']['prompt_overrides'], {'main_system': '自定义角色'})
        await self.f.session.update_harness_prompts({}, 'stale', prompt_language='zh')
        self.assertTrue(self.f.messages[-1]['error'])
        self.assertEqual(self.f.messages[-1]['request_id'], 'stale')
        self.assertEqual(self.f.session.self_harness.prompt_overrides, {'main_system': '自定义角色'})
        await self.f.session.update_harness_prompts({}, 'reset', prompt_language='en')
        self.assertFalse(self.f.messages[-1]['error'])
        self.assertEqual(self.f.messages[-1]['snapshot']['prompt_overrides'], {})

    async def test_language_switch_discards_only_private_generation(self):
        await self.f.early()
        early = self.f.session.early['task']
        await self.f.session.configure_language('en')
        self.assertTrue(early.done())
        self.assertIsNone(self.f.session.early['task'])
        systems = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def model(*_):
            systems.append(compose_reply_messages('Input', system='Injected system')[0]['content'])
            entered.set()
            await release.wait()
            yield '平静|合成测试回复。'

        self.f.memory.reply_stream = model
        await self.f.start()
        await asyncio.wait_for(entered.wait(), 1)
        await self.f.session.configure_language('zh')
        release.set()
        await asyncio.wait_for(self.f.session.turn['task'], 1)
        await self.f.start()
        await asyncio.wait_for(self.f.session.turn['task'], 1)
        self.assertTrue(systems[0].startswith(catalog('en')['main_system']['default']))
        self.assertEqual(systems[1], 'Injected system')

    async def test_english_opening_uses_selected_language_despite_chinese_space_and_memory(self):
        systems = []

        async def model(*args):
            systems.append(compose_reply_messages('Input', system='Injected system')[0]['content'])
            self.f.model_args.append(args)
            yield '平静|Hello, ready to talk?'

        self.f.memory.reply_stream = model
        self.f.agent.build_reply_context = lambda context, **_: context
        await self.f.session.configure_language('en')
        with patch('studio.core.utils.conversation.component.greeting_memories', return_value=['用户喜欢电影']):
            self.f.session.start_opening()
            await asyncio.wait_for(self.f.session.turn['task'], 1)
        self.assertIn('Use English for this opening only', self.f.model_args[0][0])
        self.assertIn('用户喜欢电影', self.f.model_args[0][1])
        self.assertTrue(systems[0].startswith(catalog('en')['main_system']['default']))
        self.assertEqual(self.f.session.turn['pending'].language, 'en')
        self.assertFalse(any(m['type'] == 'user_transcript' for m in self.f.messages))
        self.f.agent.queue_remember_turn.assert_not_called()

    async def test_chinese_opening_captures_language_before_memory_lookup(self):
        self.f.agent.space_language = lambda _: 'en'
        await self.f.session.configure_language('zh')
        with patch('studio.core.utils.conversation.component.greeting_memories', return_value=[]):
            self.f.session.start_opening()
            pending = self.f.session.turn['pending']
            await self.f.session.configure_language('en')
            await asyncio.wait_for(self.f.session.turn['task'], 1)
        self.assertEqual(pending.language, 'zh')
        self.assertEqual(pending.self_harness_profile['prompt_language'], 'zh')
        self.assertIn('本次开场必须使用中文', self.f.model_args[0][0])

    async def test_template_language_does_not_choose_tts_language(self):
        await self.f.session.configure_language('en')
        await self.f.session.update_harness_prompts({'tts_zh': 'Synthetic Chinese voice',
                                                    'tts_en': 'Synthetic English voice'})
        for body, expected, excluded in (
            ('这是合成测试的中文回复。', 'Synthetic Chinese voice', 'Synthetic English voice'),
            ('This is a synthetic English reply.', 'Synthetic English voice', 'Synthetic Chinese voice'),
        ):
            async def model(*_):
                yield f'平静|{body}'
            self.f.memory.reply_stream = model
            self.f.tts_instructions.clear()
            await self.f.start()
            await asyncio.wait_for(self.f.session.turn['task'], 1)
            self.assertTrue(self.f.tts_instructions)
            self.assertTrue(all(expected in text and excluded not in text for text in self.f.tts_instructions))

    async def test_retrieved_memory_is_visible_while_model_is_still_waiting(self):
        release, entered = asyncio.Event(), asyncio.Event()

        async def model(*_):
            entered.set()
            await release.wait()
            yield '平静|Synthetic reply.'

        self.f.memory.reply_stream = model
        pending = self.f.pending(input_turn_id='synthetic-input')
        pending.result.hits = [types.SimpleNamespace(text='Synthetic memory', score=.9,
                                                     attributed_to='', memory_id='synthetic-id')]
        await self.f.start(pending)
        await asyncio.wait_for(entered.wait(), 1)
        events = self.f.messages
        self.assertEqual([event['type'] for event in events[:2]], ['reply_preparing', 'memory_hits'])
        self.assertFalse(any(event['type'] == 'answer_delta' for event in events))
        self.assertEqual(events[1]['left_brain'][0]['text'], 'Synthetic memory')
        self.assertEqual(events[1]['input_turn_id'], 'synthetic-input')
        self.assertFalse(self.f.audio)
        release.set()
        await asyncio.wait_for(self.f.session.turn['task'], 1)

    async def test_early_memory_remains_private_until_confirmed(self):
        await self.f.early()
        self.assertFalse(self.f.messages)
        pending = self.f.pending(early_ok=True)
        await self.f.session.commit_early(pending, asyncio.create_task(asyncio.sleep(0)))
        await asyncio.wait_for(self.f.session.turn['task'], 1)
        kinds = [event['type'] for event in self.f.messages]
        self.assertLess(kinds.index('memory_hits'), kinds.index('answer_start'))

    async def test_settings_ack_and_failed_update_keep_existing_snapshot(self):
        await self.f.session.update_harness_prompts({'main_system': 'Synthetic identity'}, 'save-one')
        ack = self.f.messages[-1]
        self.assertEqual(ack['request_id'], 'save-one')
        self.assertFalse(ack['error'])
        await self.f.session.update_harness_prompts({'main_system': ''}, 'save-two')
        self.assertTrue(self.f.messages[-1]['error'])
        self.assertEqual(self.f.session.self_harness.prompt_overrides['main_system'], 'Synthetic identity')

    async def test_prompt_edits_leave_active_reply_on_its_captured_snapshot(self):
        entered, release = asyncio.Event(), asyncio.Event()
        systems = []

        async def model(*_):
            systems.append(compose_reply_messages('Input', system='Default')[0]['content'])
            entered.set()
            await release.wait()
            yield '平静|Synthetic reply.'

        self.f.memory.reply_stream = model
        await self.f.session.update_harness_prompts({'main_system': 'First system'})
        await self.f.start()
        await asyncio.wait_for(entered.wait(), 1)
        await self.f.session.update_harness_prompts({'main_system': 'Second system'})
        release.set()
        await asyncio.wait_for(self.f.session.turn['task'], 1)
        await self.f.start()
        await asyncio.wait_for(self.f.session.turn['task'], 1)
        self.assertTrue(systems[0].startswith('First system\n'))
        self.assertTrue(systems[1].startswith('Second system\n'))

    async def test_local_prefix_warmup_captures_edited_prompt(self):
        systems = []

        def prewarm(*_, **kwargs):
            systems.append(compose_reply_messages('Input', system='Original')[0]['content'])

        self.f.agent._LOCAL_LLM = types.SimpleNamespace(prewarm=prewarm)
        await self.f.session.update_harness_prompts({'main_system': 'Synthetic system'})
        self.assertTrue(self.f.session.prewarm_local())
        await asyncio.wait_for(self.f.session.prewarm['task'], 1)
        self.assertTrue(systems[0].startswith('Synthetic system\n'))

    async def test_local_prefix_warmup_uses_same_localized_defaults_as_reply(self):
        systems = []

        def prewarm(*_, **kwargs):
            systems.append(compose_reply_messages('Input', system='Original')[0]['content'])

        self.f.agent._LOCAL_LLM = types.SimpleNamespace(prewarm=prewarm)
        await self.f.session.configure_language('en')
        self.assertTrue(self.f.session.prewarm_local())
        await asyncio.wait_for(self.f.session.prewarm['task'], 1)
        self.assertEqual(systems, [system_prompt(tagged=True, overrides=effective_prompts(
            self.f.session.self_harness.snapshot()))])

    async def test_tts_only_edit_preserves_injected_reply_system(self):
        systems = []

        async def model(*_):
            systems.append(compose_reply_messages('Input', system='Injected system')[0]['content'])
            yield '平静|Synthetic reply.'

        self.f.memory.reply_stream = model
        await self.f.session.update_harness_prompts({'tts_en': 'Synthetic voice instruction'})
        await self.f.start()
        await asyncio.wait_for(self.f.session.turn['task'], 1)
        self.assertEqual(systems, ['Injected system'])
        self.assertIn('Synthetic voice instruction', self.f.tts_instructions[0])

    async def test_discarded_speculation_and_stale_space_cannot_publish_memory(self):
        await self.f.early()
        await self.f.session.drop_early('Synthetic cancellation')
        self.assertFalse(any(message['type'] == 'memory_hits' for message in self.f.messages))
        from studio.core.utils.audio_timeline.component import AudioTimeline
        timeline = AudioTimeline()
        self.f.session.turn['timeline'] = timeline
        pending = self.f.pending()
        await self.f.session.publish_reply_memory(pending, timeline, object(), 'fixture')
        await self.f.session.publish_reply_memory(pending, timeline, self.f.memory, 'other')
        timeline.mark_interrupted()
        await self.f.session.publish_reply_memory(pending, timeline, self.f.memory, 'fixture')
        self.assertFalse(any(message['type'] == 'memory_hits' for message in self.f.messages))

    async def test_base_language_is_session_local(self):
        await self.f.session.configure_language('en')
        self.assertEqual(self.f.session.languages['fixture'], 'en')
        self.assertEqual(self.f.session.base_language, 'en')
        self.assertEqual(self.f.agent.space_language('fixture'), 'zh')
        await self.f.session.configure_language('invalid')
        await self.f.session.configure_language([])
        self.assertEqual(self.f.session.base_language, 'en')
