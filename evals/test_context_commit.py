"""Offline context ownership checks using real orchestration and fake providers."""
import asyncio
import ast
import base64
from pathlib import Path
import time
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from evals import test_reply_text
from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.conversation.component import Conversation
from studio.core.utils.contracts.component import Pending
from studio.core.utils.session_context.component import SessionBuffer
from studio.core.voicemem import greeting_memories
from studio.harness.persona.policy import opening_prompt
from studio.core.utils.tts import control as tts_control
from voicemem.stream import empty_result


class SessionBufferCleanupTests(unittest.TestCase):
    def test_clear_session_removes_recent_history_in_all_spaces(self):
        context = SessionBuffer()
        committed = context.add('closed', 'personal', '我妹妹叫安安', '记住了')
        context.add('closed', 'work', '明天开会', '好的')
        context.add('active', 'personal', '周末出行', '好的')
        context.mark_complete(committed, True)
        self.assertEqual(len(context.recent('closed', 'personal', 6)), 1)

        context.clear_session('closed')
        context.clear_session('closed')

        self.assertEqual(context.messages('closed', 'personal', window=6), [])
        self.assertEqual(context.messages('closed', 'work', window=6), [])
        self.assertFalse(any(key[0] == 'closed' for key in context._recent))
        self.assertEqual(len(context.recent('active', 'personal', 6)), 1)


class GreetingMemoryTests(unittest.TestCase):
    def test_opening_tone_is_selected_for_this_reply_only(self):
        prompt = opening_prompt('zh')
        self.assertIn('根据提供的记忆选择合适的语气', prompt)
        self.assertIn('不代表后续对话的固定偏好', prompt)
        self.assertNotIn('默认平静', prompt)

    def test_only_current_user_low_sensitivity_facts_are_selected(self):
        entries = [
            {'id': 'daily', 'text': '用户喜欢看电影', 'date': '2026-09-30', 'role': 'user'},
            {'id': 'other', 'text': '别人喜欢散步', 'date': '2026-09-29', 'role': 'user'},
            {'id': 'health', 'text': '用户有健康问题', 'date': '2026-09-28', 'role': 'user'},
            {'id': 'private', 'text': '用户有私人计划', 'date': '2026-09-27', 'role': 'user'},
            {'id': 'assistant', 'text': '我是助手', 'date': '2026-09-26', 'role': 'assistant'},
        ]
        records = {
            'daily': types.SimpleNamespace(user_id='owner', slot='daily_life', sensitivity=0),
            'other': types.SimpleNamespace(user_id='someone_else', slot='daily_life', sensitivity=0),
            'health': types.SimpleNamespace(user_id='owner', slot='health', sensitivity=0),
            'private': types.SimpleNamespace(user_id='owner', slot='goals', sensitivity=.8),
        }
        repo = types.SimpleNamespace(
            _vector_store=types.SimpleNamespace(list_entries=Mock(return_value=entries)),
            _cognitive_store=types.SimpleNamespace(get_memory_record=lambda mid: records.get(mid)))
        memory = types.SimpleNamespace(_o=types.SimpleNamespace(
            _user_id='owner', _get_repo=lambda: repo))
        self.assertEqual(greeting_memories(memory), ['用户喜欢看电影'])
        repo._vector_store.list_entries.assert_called_once_with(user_id='owner', limit=80)


class PlaybackAuthorityTests(unittest.TestCase):
    def timeline(self, strict=True):
        timeline = AudioTimeline(track_delivery=strict)
        timeline.append_text('这是已经播放的部分，后面还有没播放的内容。')
        segment = timeline.begin_segment(0, len(timeline.generated_text))
        timeline.append_audio(bytes(48000 * 4))
        return timeline, segment

    def test_generated_sent_and_rendered_are_distinct(self):
        timeline, segment = self.timeline()
        timeline.finish_segment(segment)
        self.assertEqual(timeline.generated_samples, 96000)
        self.assertEqual(timeline.sent_samples, 0)
        timeline._first_audio_at = time.monotonic() - 20
        self.assertEqual(timeline.heard_text(), '')
        timeline.mark_sent(96000)
        self.assertEqual(timeline.heard_text(), '')
        timeline.update_checkpoint(24000, 24000, 'paused')
        self.assertTrue(timeline.heard_text())
        self.assertLess(len(timeline.heard_text()), len(timeline.generated_text))

    def test_interrupt_freezes_text_even_if_clock_alignment_and_reports_change(self):
        for strict in (True, False):
            with self.subTest(strict=strict):
                timeline, segment = self.timeline(strict)
                if strict:
                    timeline.mark_sent(96000)
                    timeline.update_checkpoint(24000, 24000, 'paused')
                else:
                    timeline._first_audio_at = time.monotonic() - 1
                timeline.mark_interrupted()
                text, samples = timeline.heard_text(), timeline.rendered_cutoff_samples()
                self.assertTrue(text)
                timeline._first_audio_at -= 20
                timeline.finish_segment(segment)
                self.assertFalse(timeline.update_checkpoint(96000, 24000, 'drained'))
                timeline.append_text('后到的文本')
                self.assertEqual(timeline.heard_text(), text)
                self.assertEqual(timeline.rendered_cutoff_samples(), samples)

    def test_missing_completion_does_not_promote_unheard_audio(self):
        timeline, segment = self.timeline()
        timeline.finish_segment(segment)
        timeline.mark_sent(96000)
        timeline.update_checkpoint(12000, 24000, 'paused')
        heard = timeline.heard_text()
        timeline.assume_drained()
        self.assertEqual(timeline.heard_text(), heard)
        self.assertEqual(timeline.rendered_cutoff_samples(), 12000)

    def test_early_checkpoint_is_bounded_by_successful_delivery(self):
        timeline, segment = self.timeline()
        timeline.finish_segment(segment)
        timeline.update_checkpoint(24000, 24000, 'playing')
        self.assertEqual(timeline.heard_text(), '')
        timeline.mark_sent(24000)
        self.assertEqual(timeline.rendered_cutoff_samples(), 24000)

    def test_failed_segment_cannot_make_later_text_a_contiguous_heard_prefix(self):
        timeline = AudioTimeline(track_delivery=True)
        timeline.append_text('没合成的句子。后续已经合成的句子。')
        first = timeline.begin_segment(0, 7)
        timeline.finish_segment(first, complete=False)
        second = timeline.begin_segment(7, len(timeline.generated_text))
        timeline.append_audio(bytes(48000))
        timeline.finish_segment(second)
        timeline.mark_sent(24000)
        timeline.update_checkpoint(24000, 24000, 'drained')
        self.assertEqual(timeline.heard_text(), '')


class ContextCommitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        f = test_reply_text.ShortReplyTextTests()
        f.setUp()
        self.agent = f.agent
        self.tts_instructions = []
        self.agent.BC_ECHO_WINDOW_S = 4
        self.agent.MIC_RATE = 24000
        self.agent.space_language = lambda _: 'zh'
        self.agent.BARGE_GRACE_MS = 500
        self.agent._LOCAL_LLM = None
        self.agent.route_pending_thinking = AsyncMock()
        self.agent.voicemem_llm_tts = self.agent._voicemem_llm_tts
        self.context = self.agent._SESSION_CONTEXT = SessionBuffer(text_limit=200)
        self.agent._push_history = Mock(side_effect=lambda *a, **k: self.context.add(*a, **k))
        self.memory = self.agent.vm
        self.messages, self.audio = [], []
        self.audio_ready = asyncio.Event()
        self.model_waiting = asyncio.Event()
        self.release_model = asyncio.Event()
        self.auto_playback = True
        self.block_model = False
        self.model_calls = 0
        self.model_args = []
        self.model_closed = asyncio.Event()

        async def model(*_):
            self.model_calls += 1
            self.model_args.append(_)
            try:
                yield '认真|这是生成的回复，后面还有更多说明。'
                self.model_waiting.set()
                if self.block_model:
                    await self.release_model.wait()
            finally:
                self.model_closed.set()

        async def tts(_text, instruction=None):
            self.tts_instructions.append(instruction)
            yield bytes(48000)

        self.memory.reply_stream = model
        self.memory.utils = types.SimpleNamespace(get=lambda _: types.SimpleNamespace(stream=tts))
        self.sock = types.SimpleNamespace(send_json=self.send, send_bytes=self.send_audio)
        self.session = Conversation(self.agent, self.sock)
        self.session.turn_taking.work_filler_enabled = False

    async def asyncTearDown(self):
        await self.session.close_session()

    def pending(self, text='最终确认的输入', **kwargs):
        return Pending(text, '', empty_result(), route='shallow', transcript_managed=True, **kwargs)

    async def send(self, message):
        self.messages.append(message)
        if message['type'] == 'answer_done' and self.auto_playback:
            timeline = self.session.turn['timeline']
            await self.session.playback_checkpoint(dict(
                output_id=message['output_id'], rendered_samples=timeline.sent_samples,
                sample_rate=24000, state='drained'))

    async def send_audio(self, pcm):
        self.audio.append(pcm)
        self.audio_ready.set()

    async def start(self, value=None):
        value = value or self.pending()
        await self.session.start_reply(value, asyncio.create_task(asyncio.sleep(0)))
        return self.session.turn['task']

    async def early(self):
        st = types.SimpleNamespace(memory=empty_result(), route='shallow', eot_score=.99)
        await self.session.start_early('设备重起怎么检查？', st)
        sink, timeline = self.session.early['sink'], self.session.early['timeline']
        await asyncio.wait_for(sink.wait_for_audio(), 1)
        return sink, timeline, self.session.early['task']

    async def test_opening_speaks_once_without_user_input_or_memory_ingest(self):
        self.agent.build_reply_context = lambda context, **_: context
        with patch('studio.core.utils.conversation.component.greeting_memories',
                   return_value=['用户喜欢看电影']) as memories:
            self.session.start_opening()
            task = self.session.turn['task']
            self.session.start_opening()
            self.assertIs(self.session.turn['task'], task)
            await asyncio.wait_for(task, 1)
        memories.assert_called_once_with(self.memory)
        self.assertIn('用户喜欢看电影', self.model_args[0][1])
        self.assertEqual(sum(m['type'] == 'answer_start' for m in self.messages), 1)
        self.assertFalse(any(m['type'] in {'user_transcript', 'memory_hits'}
                             for m in self.messages))
        self.assertEqual(len(self.saved()), 1)
        self.assertEqual(self.saved()[0].user_text, '')
        self.assertTrue(self.saved()[0].assistant_text)
        self.agent.queue_remember_turn.assert_not_called()

    async def test_opening_interrupted_before_speech_leaves_no_context(self):
        self.block_model = True
        with patch('studio.core.utils.conversation.component.greeting_memories',
                   return_value=[]):
            self.session.start_opening()
            await asyncio.wait_for(self.model_waiting.wait(), 1)
            await self.session.stop_reply(force=True)
        self.assertFalse(self.saved())
        self.agent.queue_remember_turn.assert_not_called()

    async def test_opening_does_not_restart_after_text_chat_in_same_session(self):
        self.context.add(self.session.context_session, 'fixture', '你好', '你好呀')
        self.session.start_opening()
        self.session.start_opening()
        self.assertTrue(self.session.opening_started)
        self.assertIsNone(self.session.turn['task'])
        self.assertFalse(self.messages)

    async def test_opening_uses_its_own_tone_without_changing_later_turns(self):
        self.session.owner['tone'] = '俏皮'

        async def model(*_):
            yield '共情|我在，想聊什么都可以。'

        self.memory.reply_stream = model
        with patch('studio.core.utils.conversation.component.greeting_memories',
                   return_value=['用户最近遇到挫折']):
            self.session.start_opening()
            await asyncio.wait_for(self.session.turn['task'], 1)
        self.assertTrue(any(tts_control.TONES['共情'] in instruction
                            for instruction in self.tts_instructions))
        self.assertEqual(self.session.owner['tone'], '俏皮')
        self.assertEqual(self.session.self_harness.profile['speaking_style']['tone'], 'auto')

        async def next_model(*_):
            yield '认真|接下来回答你的问题。'

        self.memory.reply_stream = next_model
        await self.start()
        await asyncio.wait_for(self.session.turn['task'], 1)
        self.assertEqual(self.session.owner['tone'],
                         tts_control.smooth('俏皮', '认真'))

    async def test_opening_cannot_update_harness_preference(self):
        async def model(*_):
            yield ('<self_harness>{"speaking_style":{"tone":"平静"}}</self_harness>'
                   '平静|你好，想聊什么都可以。')

        self.memory.reply_stream = model
        with patch('studio.core.utils.conversation.component.greeting_memories',
                   return_value=[]):
            self.session.start_opening()
            await asyncio.wait_for(self.session.turn['task'], 1)
        self.assertEqual(self.session.self_harness.profile['speaking_style']['tone'], 'auto')

    async def test_opening_memory_tone_does_not_replace_fixed_conversation_tone(self):
        self.session.self_harness.set_explicit(
            {'speaking_style': {'tone': '轻快'}})

        async def model(*_):
            yield '共情|我在，想聊什么都可以。'

        self.memory.reply_stream = model
        with patch('studio.core.utils.conversation.component.greeting_memories',
                   return_value=['用户最近遇到挫折']):
            self.session.start_opening()
            await asyncio.wait_for(self.session.turn['task'], 1)
        self.assertTrue(any(tts_control.TONES['共情'] in instruction
                            for instruction in self.tts_instructions))
        self.assertEqual(self.session.self_harness.profile['speaking_style']['tone'], '轻快')

        self.tts_instructions.clear()
        await self.start()
        await asyncio.wait_for(self.session.turn['task'], 1)
        self.assertTrue(any(tts_control.TONES['轻快'] in instruction
                            for instruction in self.tts_instructions))

    def saved(self):
        return self.context.turns(self.session.context_session, 'fixture')

    def assert_once(self, value):
        self.assertEqual(len(self.saved()), 1)
        self.assertEqual(self.saved()[0].user_text, value.text)
        self.agent.queue_remember_turn.assert_called_once()
        self.assertIs(self.agent.queue_remember_turn.call_args.args[0], value)

    async def test_stranger_turn_keeps_original_background_speaker_tracking(self):
        value = self.pending(stranger=True)
        await self.start(value)
        await asyncio.wait_for(self.session.turn['task'], 1)
        self.assert_once(value)

    async def test_known_stranger_does_not_start_speculative_reply(self):
        self.agent.SPEAKER_GATE = True
        self.agent.STRANGER_MIN_TURNS = 1
        self.session.owner.update(id='owner', miss=1)
        st = types.SimpleNamespace(memory=empty_result(), route='shallow')
        await self.session.start_early('测试', st)
        self.assertIsNone(self.session.early['task'])
        self.assertEqual(self.model_calls, 0)

    async def test_final_stranger_discards_speculative_reply(self):
        await self.early()
        value = self.pending(stranger=True, early_ok=True)
        thinking_task = await self.session.route(value)
        await thinking_task
        self.assertIsNone(self.session.early['task'])
        self.assertFalse(await self.session.commit_early(value, thinking_task))
        self.assertEqual(self.messages, [])
        self.assertEqual(self.audio, [])

    async def test_cancel_unconfirmed_audio_has_no_history_or_ingest(self):
        self.block_model = True
        self.session.owner['tone'] = '俏皮'
        sink, timeline, task = await self.early()
        self.assertTrue(timeline.reply_tone)
        self.assertEqual(self.session.owner['tone'], '俏皮')
        timeline._first_audio_at = time.monotonic() - 20
        await self.session.drop_early()
        self.assertTrue(task.done())
        self.assertEqual(timeline.sent_samples, 0)
        self.assertEqual(timeline.heard_text(), '')
        self.assertEqual(self.messages, [])
        self.assertEqual(self.audio, [])
        self.assertEqual(self.saved(), [])
        self.assertEqual(self.session.owner['tone'], '俏皮')
        self.agent.queue_remember_turn.assert_not_called()

    async def test_tone_smoothing_belongs_to_the_conversation(self):
        other = Conversation(self.agent, self.sock)
        self.session.owner['tone'] = '俏皮'
        other.owner['tone'] = '共情'
        try:
            await asyncio.wait_for(await self.start(), 1)
            self.assertEqual(self.session.owner['tone'], tts_control.smooth('俏皮', '认真'))
            self.assertEqual(other.owner['tone'], '共情')
        finally:
            await other.close_session()

    async def test_completed_private_generation_never_waits_for_playback_or_saves(self):
        _, timeline, task = await self.early()
        await asyncio.wait_for(task, .5)
        self.assertTrue(timeline.generation_complete)
        self.assertEqual(timeline.sent_samples, 0)
        self.assertFalse(timeline.playback_done)
        self.assertFalse(self.saved())
        self.agent._kick_acoustic.assert_not_called()
        await self.session.drop_early()
        self.agent.queue_remember_turn.assert_not_called()

    async def test_discarded_early_reply_cannot_change_self_harness(self):
        async def model(*_):
            yield ('<self_harness>{"speaking_style":{"speech_rate":"slow",'
                   '"tone":"轻快"}}</self_harness>'
                   '轻快|好的，我说慢一点。')
        self.memory.reply_stream = model
        _, _, generation = await self.early()
        await asyncio.wait_for(generation, 1)
        self.assertEqual(
            self.session.self_harness.profile["speaking_style"]["speech_rate"],
            "normal")
        self.assertEqual(
            self.session.self_harness.profile["speaking_style"]["tone"], "auto")
        await self.session.drop_early()
        self.assertEqual(
            self.session.self_harness.profile["speaking_style"]["speech_rate"],
            "normal")

    async def test_accepted_early_reply_commits_self_harness(self):
        async def model(*_):
            yield ('<self_harness>{"speaking_style":{"speech_rate":"slow",'
                   '"tone":"轻快"}}</self_harness>'
                   '轻快|好的，我说慢一点。')
        self.memory.reply_stream = model
        await self.early()
        value = self.pending('设备重启怎么检查？', early_ok=True)
        routed = asyncio.get_running_loop().create_future()
        routed.set_result(value)
        self.assertTrue(await self.session.commit_early(value, routed))
        await asyncio.wait_for(self.session.turn['task'], 1)
        self.assertEqual(
            self.session.self_harness.profile["speaking_style"]["speech_rate"],
            "slow")
        self.assertEqual(
            self.session.self_harness.profile["speaking_style"]["tone"], "轻快")

    async def test_accept_completed_early_reply_saves_final_pending_once(self):
        _, _, generation = await self.early()
        await asyncio.wait_for(generation, 1)
        value = self.pending('设备重启怎么检查？', early_ok=True, audio_path='final-turn.wav', emotion='平静')
        routed = asyncio.get_running_loop().create_future()
        routed.set_result(value)
        self.assertTrue(await self.session.commit_early(value, routed))
        await asyncio.wait_for(self.session.turn['task'], 1)
        self.assert_once(value)
        self.assertTrue(self.saved()[0].assistant_text)
        self.agent._kick_acoustic.assert_called_once_with(self.sock.send_json, 'final-turn.wav')
        self.assertEqual(self.model_calls, 1)

    async def test_accepted_early_cancel_before_handoff_starts_reaps_generation(self):
        self.block_model = True
        _, _, generation = await self.early()
        value = self.pending(early_ok=True)
        routed = asyncio.get_running_loop().create_future()
        routed.set_result(value)
        await self.session.commit_early(value, routed)
        await self.session.stop_reply(force=True)
        self.assertTrue(generation.done())
        self.assertTrue(self.model_closed.is_set())
        self.assert_once(value)
        self.assertEqual(self.saved()[0].assistant_text, '')
        self.assertEqual(self.audio, [])

    async def test_normal_reply_saves_once_after_actual_playback(self):
        value = self.pending()
        await asyncio.wait_for(await self.start(value), 1)
        self.assert_once(value)
        self.assertEqual(self.saved()[0].assistant_text, self.session.turn['timeline'].generated_text)
        await self.session.close_session()
        self.assert_once(value)

    async def test_session_speech_reaches_normal_opening_early_and_followup_replies(self):
        for path in ('normal', 'opening', 'early', 'followup'):
            with self.subTest(path=path):
                self.setUp()
                spoken = []

                async def speech(text, instruction=None):
                    spoken.append(text)
                    yield bytes(48000)

                self.session.speech_provider = types.SimpleNamespace(stream=speech)
                self.memory.utils.get = Mock(side_effect=AssertionError('shared TTS must not be used'))
                try:
                    if path == 'normal':
                        await asyncio.wait_for(await self.start(), 1)
                    elif path == 'opening':
                        with patch('studio.core.utils.conversation.component.greeting_memories', return_value=[]):
                            self.session.start_opening()
                            await asyncio.wait_for(self.session.turn['task'], 1)
                    elif path == 'early':
                        _, _, generation = await self.early()
                        await asyncio.wait_for(generation, 1)
                    else:
                        value = self.pending('因为', speech_end=time.monotonic() - 10)
                        task = asyncio.create_task(self.session.run_unfinished_followup(value, self.memory, 'fixture'))
                        self.session.unfinished_wait['task'] = task
                        await asyncio.wait_for(task, 1)
                    self.assertTrue(spoken)
                    self.memory.utils.get.assert_not_called()
                finally:
                    await self.session.close_session()

    async def test_fast_deep_reply_keeps_generation_concurrent_without_filler(self):
        self.session.turn_taking.work_filler_enabled = True
        self.session.turn_taking.long_filler_after_s = 10
        self.session.synthesize_work_filler = AsyncMock()
        value = self.pending(reply_mode='memory_cot', speech_end=time.monotonic())
        await asyncio.wait_for(await self.start(value), 1)
        self.session.synthesize_work_filler.assert_not_awaited()
        self.assertEqual(self.model_calls, 1)
        self.assertTrue(self.audio)
        self.assert_once(value)

    async def test_accepted_ready_deep_reply_skips_filler_after_long_confirmation(self):
        async def route(pending, *_args, **_kwargs):
            pending.reply_mode = 'memory_cot'
        self.agent.route_pending_thinking = route
        self.session.turn_taking.work_filler_enabled = True
        self.session.synthesize_work_filler = AsyncMock()
        _, _, generation = await self.early()
        await asyncio.wait_for(generation, 1)
        value = self.pending(early_ok=True, reply_mode='memory_cot',
                             speech_end=time.monotonic() - 10)
        routed = asyncio.get_running_loop().create_future()
        routed.set_result(value)
        self.assertTrue(await self.session.commit_early(value, routed))
        await asyncio.wait_for(self.session.turn['task'], 1)
        self.session.synthesize_work_filler.assert_not_awaited()
        self.assertEqual(self.model_calls, 1)
        self.assertTrue(self.audio)
        self.assert_once(value)

    async def test_first_audio_does_not_wait_for_model_end_or_playback_reports(self):
        self.auto_playback = False
        self.block_model = True
        await self.start()
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        self.assertFalse(self.release_model.is_set())
        self.assertFalse(self.saved())
        self.assertGreater(self.session.turn['timeline'].sent_samples, 0)

    async def test_interrupt_with_checkpoint_uses_one_frozen_prefix_everywhere(self):
        self.block_model = True
        value = self.pending()
        await self.start(value)
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        timeline = self.session.turn['timeline']
        await self.session.playback_checkpoint(dict(output_id=timeline.output_id,
            rendered_samples=12000, sample_rate=24000, state='paused'))
        await self.session.stop_reply(force=True)
        self.assert_once(value)
        heard = self.saved()[0].assistant_text
        self.assertTrue(heard)
        self.assertNotEqual(heard, timeline.generated_text)
        interruption = next(m for m in self.messages if m['type'] == 'answer_interrupt')
        self.assertEqual(interruption['heard_text'], heard)
        timeline._first_audio_at -= 20
        timeline.update_checkpoint(24000, 24000, 'drained')
        self.assertEqual(timeline.heard_text(), heard)

    async def test_interrupt_without_playback_report_keeps_only_user(self):
        self.auto_playback = False
        self.block_model = True
        self.session.owner['tone'] = '俏皮'
        value = self.pending()
        await self.start(value)
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        self.session.turn['timeline']._first_audio_at -= 20
        await self.session.stop_reply(force=True)
        self.assert_once(value)
        self.assertEqual(self.saved()[0].assistant_text, '')
        self.assertEqual(self.session.owner['tone'], '俏皮')

    async def test_stop_before_reply_task_starts_preserves_confirmed_user_once(self):
        value = self.pending()
        await self.start(value)
        await self.session.stop_reply(force=True)
        self.assert_once(value)
        self.assertEqual(self.model_calls, 0)

    async def test_followup_has_one_context_and_one_ingest(self):
        value = self.pending('因为', speech_end=time.monotonic()-10)
        task = asyncio.create_task(self.session.run_unfinished_followup(value, self.memory, 'fixture'))
        self.session.unfinished_wait['task'] = task
        await asyncio.wait_for(task, 1)
        self.assertEqual(len(self.saved()), 1)
        self.agent.queue_remember_turn.assert_called_once()

    async def test_cancelled_followup_does_not_add_a_second_uninterrupted_copy(self):
        self.block_model = True
        value = self.pending('因为', speech_end=time.monotonic()-10)
        task = asyncio.create_task(self.session.run_unfinished_followup(value, self.memory, 'fixture'))
        self.session.unfinished_wait['task'] = task
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        await self.session.stop_reply(force=True)
        self.assertEqual(len(self.saved()), 1)
        self.assertTrue(self.saved()[0].interrupted)
        self.agent.queue_remember_turn.assert_called_once()

    async def test_failed_send_does_not_count_as_delivered_or_heard(self):
        self.sock.send_bytes = AsyncMock(side_effect=ConnectionError('synthetic send failure'))
        value = self.pending()
        await asyncio.wait_for(await self.start(value), 1)
        self.assert_once(value)
        self.assertEqual(self.session.turn['timeline'].sent_samples, 0)
        self.assertEqual(self.saved()[0].assistant_text, '')

    async def test_playback_timeout_keeps_last_report_not_whole_reply(self):
        self.auto_playback = False
        self.block_model = True
        value = self.pending()
        task = await self.start(value)
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        timeline = self.session.turn['timeline']
        timeline.update_checkpoint(12000, 24000, 'paused')
        timeline.wait_playback_done = AsyncMock(side_effect=asyncio.TimeoutError)
        self.release_model.set()
        await asyncio.wait_for(task, 1)
        self.assert_once(value)
        self.assertEqual(timeline.rendered_cutoff_samples(), 12000)
        self.assertNotEqual(self.saved()[0].assistant_text, timeline.generated_text)

    async def test_memory_space_is_captured_before_background_completion(self):
        self.block_model = True
        task = await self.start()
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        self.agent.ACTIVE_SPACE = 'other'
        self.agent.vm = object()
        self.release_model.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(len(self.saved()), 1)
        self.assertIs(self.agent.queue_remember_turn.call_args.kwargs['memory_vm'], self.memory)
        self.assertFalse(self.context.turns(self.session.context_session, 'other'))

    async def test_uncommitted_speculation_on_disconnect_does_not_save(self):
        self.block_model = True
        await self.early()
        await self.session.close_session()
        self.assertFalse(self.saved())
        self.agent.queue_remember_turn.assert_not_called()

    async def test_provider_error_finalizes_confirmed_user_only_once(self):
        async def fail(*_):
            yield '未完成的短句'
            raise RuntimeError('synthetic provider failure')
        self.memory.reply_stream = fail
        value = self.pending()
        task = await self.start(value)
        with self.assertRaisesRegex(RuntimeError, 'synthetic provider'):
            await task
        self.assert_once(value)
        self.assertEqual(self.saved()[0].assistant_text, '')

    async def test_provider_failure_after_filler_releases_handoff_and_reports_error(self):
        from studio.core.utils.reply.component import Reply
        self.agent.voicemem_llm_tts = types.MethodType(Reply.voicemem_llm_tts, self.agent)
        self.session.turn_taking.work_filler_enabled = True
        self.session.turn_taking.long_filler_after_s = .01
        self.session.synthesize_work_filler = AsyncMock(
            return_value=('嗯，让我想一想。', bytes(4800)))
        filler_sent, error_queued = asyncio.Event(), asyncio.Event()
        original_send = self.sock.send_json

        async def send(message):
            await original_send(message)
            if message['type'] == 'backchannel':
                filler_sent.set()
        self.sock.send_json = send

        async def fail(*_):
            await filler_sent.wait()
            error_queued.set()
            raise RuntimeError('synthetic provider failure')
            yield  # Make the fixture an async generator without producing text.
        self.memory.reply_stream = fail
        value = self.pending(reply_mode='memory_cot')
        task = await self.start(value)
        await asyncio.wait_for(error_queued.wait(), 1)
        filler = next(m for m in self.messages if m['type'] == 'backchannel')
        self.session.filler_done(filler['filler_id'])
        with self.assertRaisesRegex(RuntimeError, 'synthetic provider'):
            await asyncio.wait_for(task, 1)
        self.assertEqual(sum(m['type'] == 'error' for m in self.messages), 1)
        self.assertFalse(self.audio)
        self.assert_once(value)
        self.assertEqual(self.saved()[0].assistant_text, '')

    async def test_candidate_pause_and_resume_neither_save_nor_restart_generation(self):
        self.block_model = True
        await self.start()
        await asyncio.wait_for(self.audio_ready.wait(), 1)
        timeline = self.session.turn['timeline']
        await self.session.pause_candidate()
        await self.session.resume_candidate()
        self.assertEqual(self.model_calls, 1)
        self.assertIs(self.session.turn['timeline'], timeline)
        self.assertFalse(self.saved())
        self.assertIsNone(timeline._frozen_samples)
        self.agent.queue_remember_turn.assert_not_called()

    async def test_stale_output_cannot_send_or_advance_another_timeline(self):
        old = AudioTimeline(track_delivery=True)
        old.append_audio(bytes(480))
        current = AudioTimeline(track_delivery=True)
        self.session.turn['timeline'] = current
        await self.session.send_audio(bytes(480), old)
        self.assertEqual(self.audio, [])
        self.assertEqual(old.sent_samples, 0)
        self.assertEqual(current.sent_samples, 0)

    async def test_main_audio_held_behind_filler_is_not_saved_as_heard(self):
        self.session.turn_taking.work_filler_enabled = True
        self.session.turn_taking.long_filler_after_s = .01
        filler_sent = asyncio.Event()
        model_release = asyncio.Event()
        original_send = self.sock.send_json
        async def send(message):
            await original_send(message)
            if message['type'] == 'backchannel':
                filler_sent.set()
        self.sock.send_json = send
        async def model(*_):
            await model_release.wait()
            yield '认真|正文已经生成，但还在等待垫话结束。'
        self.memory.reply_stream = model
        self.session.synthesize_work_filler = AsyncMock(return_value=('让我想想。', bytes(48000*4)))
        value = self.pending(reply_mode='memory_cot')
        await self.start(value)
        await asyncio.wait_for(filler_sent.wait(), 1)
        model_release.set()
        timeline = self.session.turn['timeline']
        for _ in range(100):
            if timeline.generation_complete:
                break
            await asyncio.sleep(0)
        self.assertTrue(timeline.generation_complete)
        self.assertGreater(timeline.generated_samples, 0)
        self.assertEqual(timeline.sent_samples, 0)
        await self.session.stop_reply(force=True)
        self.assert_once(value)
        self.assertEqual(self.saved()[0].assistant_text, '')
        self.assertEqual(self.audio, [])


class RealtimePlaybackAuthorityTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, interrupted=False):
        source = Path(__file__).resolve().parents[1] / 'studio/core/utils/realtime_session/component.py'
        names = {'pump', 'close_turn', 'playback_checkpoint', 'playback_fallback'}
        nodes = [n for n in ast.walk(ast.parse(source.read_text()))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
        timeline = AudioTimeline(track_delivery=True)
        pending = Pending('合成输入', '', empty_result())
        turn = dict(live=True, reply='', pending=pending, timeline=timeline,
                    space='fixture', memory_vm=object(), until=0, t0=time.monotonic(),
                    first=False, provider_item_id='', response_done=False)
        agent = types.SimpleNamespace(BARGE_DEBUG=False, MIC_RATE=24000, ACTIVE_SPACE='fixture',
            _push_history=Mock(return_value='test'), queue_remember_turn=Mock(),
            truncate_provider_output=AsyncMock())
        async def events():
            yield types.SimpleNamespace(type='response.output_audio_transcript.delta', delta='这是已经播放的部分和后面的文字。')
            yield types.SimpleNamespace(type='response.output_audio.delta', delta=base64.b64encode(bytes(48000)).decode(), item_id='provider-output')
            if interrupted:
                timeline.update_checkpoint(12000, 24000, 'paused')
            yield types.SimpleNamespace(type='response.cancelled' if interrupted else 'response.done')
        sock = types.SimpleNamespace(send_json=AsyncMock(), send_bytes=AsyncMock())
        ns = dict(asyncio=asyncio, time=time, base64=base64, self=agent, sock=sock, owner={},
                  turn=turn, timelines={timeline.output_id:timeline}, context_session='fixture-session',
                  response_idle=asyncio.Event(), conn=events(), schedule_playback_fallback=Mock())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        return ns, agent, timeline

    async def test_audio_delivery_and_report_own_normal_realtime_context(self):
        ns, agent, timeline = self.fixture()
        await ns['pump']()
        self.assertEqual((timeline.generated_samples, timeline.sent_samples), (24000, 24000))
        agent._push_history.assert_not_called()
        await ns['playback_checkpoint'](dict(output_id=timeline.output_id,
            rendered_samples=24000, sample_rate=24000, state='drained'))
        agent._push_history.assert_called_once()
        self.assertEqual(agent._push_history.call_args.args[3], timeline.generated_text)

    async def test_realtime_fallback_does_not_invent_played_audio(self):
        ns, agent, timeline = self.fixture()
        await ns['pump']()
        ns['turn']['until'] = 0
        await ns['playback_fallback'](timeline)
        agent._push_history.assert_called_once()
        self.assertEqual(agent._push_history.call_args.args[3], '')

    async def test_realtime_interrupt_freezes_same_prefix_for_ui_and_context(self):
        ns, agent, timeline = self.fixture(interrupted=True)
        await ns['pump']()
        heard = agent._push_history.call_args.args[3]
        self.assertTrue(heard)
        self.assertNotEqual(heard, timeline.generated_text)
        messages = [c.args[0] for c in ns['sock'].send_json.await_args_list]
        self.assertEqual(messages[-1]['heard_text'], heard)
        timeline.update_checkpoint(24000, 24000, 'drained')
        self.assertEqual(timeline.heard_text(), heard)


if __name__ == '__main__':
    unittest.main()
