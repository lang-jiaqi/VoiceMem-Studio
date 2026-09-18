"""Real reply/tone parsing with fake providers; no models, APIs or memory writes."""
import asyncio
import types
import unittest
from unittest.mock import Mock

from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.contracts.component import Pending, ReplySink
from studio.core.utils.reply.component import Reply
from studio.core.utils.self_harness.component import SelfHarnessState
from studio.core.utils.tts.control import TONES
from voicemem.stream import empty_result


class ShortReplyTextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.messages, self.audio, self.tts_text = [], [], []
        self.tts_instructions = []
        self.timeline = AudioTimeline()
        self.agent = Reply()
        self.agent.ACTIVE_SPACE = 'fixture'
        self.agent._SESSION_CONTEXT = types.SimpleNamespace(messages=lambda *a, **k: [])
        self.agent.HISTORY_TURNS = 4
        self.agent._speak_instruction = lambda _: ''
        self.agent._speak_base_env = ''
        self.agent._SPEAK_BASE = {}
        self.agent._by_lang = lambda _: ''
        self.agent._LAST_TONE = {'tag': ''}
        self.agent.build_reply_context = lambda *a, **k: ''
        self.agent.BARGE_DEBUG = False
        self.agent._lat_note = lambda _: 'fixture'
        self.agent._mem_line = lambda: 'fixture'
        self.agent.hot_path_enter = lambda: None
        self.agent.hot_path_exit = lambda _: None
        self.agent._kick_acoustic = Mock()
        self.agent._push_history = Mock(return_value='fixture-history')
        self.agent.queue_remember_turn = Mock()

        async def tts(text, instruction=None):
            self.tts_text.append(text)
            self.tts_instructions.append(instruction)
            yield bytes(480)
        self.agent.vm = types.SimpleNamespace(
            utils=types.SimpleNamespace(get=lambda _: types.SimpleNamespace(stream=tts)))
        self.pending = Pending('合成测试问题', '', empty_result(),
                               route='shallow', reply_mode='direct', transcript_managed=True)

    async def send(self, message):
        self.messages.append(message)
        if message['type'] == 'answer_done':
            self.timeline.update_checkpoint(self.timeline.sent_samples, 24000, 'drained')

    async def send_audio(self, pcm):
        self.audio.append(pcm)

    def pipeline(self, model, sink=None, **kwargs):
        self.agent.vm.reply_stream = model
        return self.agent._voicemem_llm_tts(
            self.pending, sink.send if sink else self.send,
            sink.send_audio if sink else self.send_audio, {}, self.timeline,
            context_session='fixture-session', context_space='fixture',
            **kwargs)

    async def complete(self, chunks, **kwargs):
        async def model(*_):
            for chunk in chunks:
                yield chunk
        await asyncio.wait_for(self.pipeline(model, **kwargs), 1)

    def assert_body(self, text):
        displayed = ''.join(m['text'] for m in self.messages if m['type'] == 'answer_delta')
        self.assertEqual(displayed, text)
        self.assertEqual(''.join(self.tts_text), text)
        self.assertEqual(self.timeline.generated_text, text)
        self.assertTrue(self.audio)
        self.assertEqual(self.agent._push_history.call_args.args[3], text)
        self.assertEqual(sum(m['type'] == 'answer_done' for m in self.messages), 1)

    async def test_untagged_short_replies_are_not_lost_at_normal_eof(self):
        for text in ('4。', '四', '二加二等于四。', 'Yes.', '0', '3.14'):
            with self.subTest(text=text):
                self.setUp()
                await self.complete([text])
                self.assert_body(text)

    async def test_untagged_text_split_across_deltas_is_flushed_once(self):
        await self.complete(['二加二', '等于', '四。'])
        self.assert_body('二加二等于四。')
        self.assertEqual(sum(m['type'] == 'answer_delta' for m in self.messages), 1)

    async def test_plain_text_at_prefix_buffer_boundary_is_not_duplicated(self):
        for size in (25, 26, 27):
            with self.subTest(size=size):
                self.setUp()
                text = '答' * (size - 1) + '。'
                await self.complete(list(text))
                self.assert_body(text)

    async def test_recognized_split_tags_are_still_removed(self):
        for chunks in (['认', '真', '|', '四。'], ['【', '温和', '】', '四。'],
                       ['轻快', '｜', '四。'], ['serious', '|', '四。']):
            with self.subTest(chunks=chunks):
                self.setUp()
                await self.complete(chunks)
                self.assert_body('四。')

    async def test_private_harness_header_updates_preferences_without_leaking(self):
        updates = {}
        await self.complete([
            ' \n',
            '<self_',
            'harness>{"speaking_style":{"speech_rate":"slow",',
            '"tone":"鼓励"}}</self_harness>认',
            '真|好，我们慢一点。',
        ], self_harness_profile={}, on_self_harness_update=updates.update)
        self.assert_body('好，我们慢一点。')
        self.assertEqual(updates, {"speaking_style": {
            "speech_rate": "slow", "tone": "鼓励"}})
        self.assertTrue(any("语速稍慢" in instruction
                            for instruction in self.tts_instructions))
        self.assertTrue(any(TONES["鼓励"] in instruction
                            for instruction in self.tts_instructions))
        self.assertFalse(any(TONES["认真"] in instruction
                             for instruction in self.tts_instructions))
        self.assertFalse(any("<self_harness>" in message.get("text", "")
                             for message in self.messages))

    async def test_invalid_harness_header_is_removed_but_never_applied(self):
        updates = {}
        await self.complete([
            '<self_harness>{"temperature":2}</self_harness>',
            '温和|这部分仍然正常回答。',
        ], self_harness_profile={}, on_self_harness_update=updates.update)
        self.assert_body('这部分仍然正常回答。')
        self.assertEqual(updates, {})

    async def test_unconfirmed_conflict_does_not_change_this_reply_tts(self):
        state = SelfHarnessState()
        state.apply({"speaking_style": {"tone": "轻快"}})
        await self.complete([
            '<self_harness>{"speaking_style":{"tone":"认真"}}'
            '</self_harness>认真|先保持当前方式，请你再确认一次。',
        ], self_harness_profile=state.snapshot(),
            on_self_harness_update=state.apply)

        self.assert_body('先保持当前方式，请你再确认一次。')
        self.assertEqual(state.profile["speaking_style"]["tone"], "轻快")
        self.assertEqual(state.snapshot()["pending"], {
            "speaking_style.tone": "认真"})
        self.assertTrue(any(TONES["轻快"] in instruction
                            for instruction in self.tts_instructions))
        self.assertFalse(any(TONES["认真"] in instruction
                             for instruction in self.tts_instructions))

    async def test_empty_whitespace_and_recognized_tag_only_stay_silent(self):
        for chunks in ([], [''], ['  ', '\n'], ['温和|'], ['【认真】']):
            with self.subTest(chunks=chunks):
                self.setUp()
                await self.complete(chunks)
                self.assertFalse(self.audio)
                self.assertFalse(self.tts_text)
                self.assertFalse(any(m['type'] == 'answer_delta' for m in self.messages))

    async def test_cancellation_never_flushes_pending_short_text(self):
        entered = asyncio.Event()
        async def model(*_):
            yield '未完成的短句'
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(self.pipeline(model))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(task, 1)
        self.assertFalse(self.audio)
        self.assertFalse(self.tts_text)
        self.assertEqual([m['type'] for m in self.messages], ['answer_start'])
        saved = self.agent._push_history.call_args
        self.assertEqual(saved.args[3], '')
        self.assertTrue(saved.kwargs['interrupted'])

    async def test_provider_error_never_flushes_pending_short_text(self):
        async def model(*_):
            yield '错误前的短句'
            raise RuntimeError('synthetic provider failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic provider'):
            await asyncio.wait_for(self.pipeline(model), 1)
        self.assertFalse(self.audio)
        self.assertFalse(self.tts_text)
        self.assertEqual([m['type'] for m in self.messages], ['answer_start'])

    async def test_completed_short_speculation_stays_private_until_commit(self):
        sink = ReplySink(self.send, self.send_audio)
        async def model(*_):
            yield '四。'
        task = asyncio.create_task(self.pipeline(model, sink))
        try:
            await asyncio.wait_for(sink.wait_for_audio(), .5)
            self.assertEqual(self.messages, [])
            self.assertEqual(self.audio, [])
            await sink.commit()
            await asyncio.wait_for(task, 1)
            self.assert_body('四。')
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancelled_short_speculation_never_reaches_browser(self):
        sink = ReplySink(self.send, self.send_audio)
        entered = asyncio.Event()
        async def model(*_):
            yield '旧的短回复'
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(self.pipeline(model, sink))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.messages, [])
        self.assertEqual(self.audio, [])
        self.assertEqual(self.tts_text, [])
        self.assertEqual(sink.buffered_ms, 0)


if __name__ == '__main__':
    unittest.main()
