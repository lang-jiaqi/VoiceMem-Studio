"""Real reply/tone parsing with fake providers; no models, APIs or memory writes."""
import asyncio
import types
import unittest
from unittest.mock import Mock
from tests.helpers.reply import ReplyFixture

from studio.core.utils.contracts.component import ReplySink
from studio.core.utils.self_harness.component import SelfHarnessState
from studio.core.utils.tts.control import TONES


class ShortReplyTextTests(ReplyFixture, unittest.IsolatedAsyncioTestCase):
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

    async def test_injected_session_speech_bypasses_shared_tts(self):
        spoken = []

        async def speech(text, instruction=None):
            spoken.append(text)
            yield bytes(480)

        self.agent.vm.utils.get = Mock(side_effect=AssertionError('shared TTS must not be used'))
        await self.complete(['认真|会话独立语音。'],
                            speech_provider=types.SimpleNamespace(stream=speech))
        self.assertEqual(''.join(spoken), '会话独立语音。')
        self.assertTrue(self.audio)

    async def test_stranger_reply_excludes_owner_session_history(self):
        self.pending.stranger = True
        self.agent._SESSION_CONTEXT.messages = Mock(return_value=[
            {"role": "user", "content": "主人的私人信息"}])
        received = []

        async def model(text, context, history):
            received.append(history)
            yield '认真|这道题等于零。'

        await asyncio.wait_for(self.pipeline(model), 1)
        self.assertEqual(received, [[]])
        self.agent._SESSION_CONTEXT.messages.assert_not_called()

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

    async def test_misplaced_private_header_never_reaches_any_reply_consumer(self):
        raw = '温和|<self_harness>{}</self_harness>合成测试回答。'
        for split in range(1, len(raw)):
            with self.subTest(split=split):
                self.setUp()
                await self.complete([raw[:split], raw[split:]], self_harness_profile={})
                self.assert_body('合成测试回答。')
        self.setUp()
        await self.complete(list(raw), self_harness_profile={})
        self.assert_body('合成测试回答。')

    async def test_private_headers_stay_private_when_harness_is_disabled(self):
        updates = {}
        await self.complete(list(
            '<self_harness>{"speaking_style":{"tone":"认真"}}'
            '</self_harness>温和|合成测试回答。'), on_self_harness_update=updates.update)
        self.assert_body('合成测试回答。')
        self.assertEqual(updates, {})

    async def test_body_control_is_removed_without_applying_a_second_update(self):
        updates = {}
        await self.complete(list(
            '<self_harness>{}</self_harness>温和|前半句。'
            '<self_harness>{"speaking_style":{"tone":"认真"}}'
            '</self_harness>后半句。</self_harness>'),
            self_harness_profile={}, on_self_harness_update=updates.update)
        self.assert_body('前半句。后半句。')
        self.assertEqual(updates, {})

    async def test_variant_body_headers_and_repeated_tones_never_reach_reply_consumers(self):
        for opening, closing, tone in (
            ('<self_harness >', '</self_harness >', '认真|'),
            ('<SELF_HARNESS>', '</SELF_HARNESS>', 'Serious｜'),
            ('< self_harness\n>', '< / self_harness >', '【认真】'),
        ):
            raw = ('<self_harness>{}</self_harness>温和|前半句。' + opening
                   + '{"speaking_style":{"tone":"认真"}}' + closing + tone + '后半句。')
            for chunks in ([raw], list(raw), [raw[:raw.index(tone)], raw[raw.index(tone):]]):
                with self.subTest(opening=opening, chunks=chunks):
                    self.setUp()
                    updates = {}
                    await self.complete(chunks, self_harness_profile={},
                                        on_self_harness_update=updates.update)
                    self.assert_body('前半句。后半句。')
                    self.assertEqual(updates, {})
                    self.assertTrue(all(TONES['温和'] in value for value in self.tts_instructions))
                    self.assertFalse(any(TONES['认真'] in value for value in self.tts_instructions))

    async def test_variant_leading_header_is_applied_once_and_never_displayed(self):
        updates = {}
        await self.complete(list(
            '< SELF_HARNESS >{"speaking_style":{"speech_rate":"slow"}}'
            '< / SELF_HARNESS >温和|正常回答。'),
            self_harness_profile={}, on_self_harness_update=updates.update)
        self.assert_body('正常回答。')
        self.assertEqual(updates, {'speaking_style': {'speech_rate': 'slow'}})

    async def test_incomplete_variant_header_and_repeated_tone_are_discarded_on_cancel_or_error(self):
        for tail in ('<SELF_HARNESS >{"unfinished":',
                     '<self_harness >{}</self_harness >认真'):
            for fail in (False, True):
                with self.subTest(tail=tail, provider_error=fail):
                    self.setUp()
                    entered = asyncio.Event()

                    async def model(*_):
                        yield '温和|前半句。' + tail
                        entered.set()
                        if fail:
                            raise RuntimeError('synthetic provider failure')
                        await asyncio.Event().wait()

                    task = asyncio.create_task(self.pipeline(model, self_harness_profile={}))
                    await asyncio.wait_for(entered.wait(), 1)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'synthetic provider'):
                            await asyncio.wait_for(task, 1)
                    else:
                        task.cancel()
                        await asyncio.wait_for(task, 1)
                    displayed = ''.join(m['text'] for m in self.messages if m['type'] == 'answer_delta')
                    self.assertEqual(displayed, '前半句。')
                    self.assertNotIn('认真', ''.join(self.tts_text))
                    self.assertNotIn('self_harness', self.timeline.generated_text.lower())

    async def test_unfinished_misplaced_control_stays_private_at_eof(self):
        await self.complete(list('温和|合成测试回答。<self_harness>{"unfinished":'),
                            self_harness_profile={})
        self.assert_body('合成测试回答。')

    async def test_misplaced_private_control_is_not_flushed_on_cancel_or_error(self):
        for fail in (False, True):
            with self.subTest(provider_error=fail):
                self.setUp()
                entered = asyncio.Event()

                async def model(*_):
                    yield '温和|<self_harness>{"unfinished":'
                    entered.set()
                    if fail:
                        raise RuntimeError('synthetic provider failure')
                    await asyncio.Event().wait()

                task = asyncio.create_task(self.pipeline(model, self_harness_profile={}))
                await asyncio.wait_for(entered.wait(), 1)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic provider'):
                        await asyncio.wait_for(task, 1)
                else:
                    task.cancel()
                    await asyncio.wait_for(task, 1)
                    self.assertEqual(self.agent._push_history.call_args.args[3], '')
                self.assertFalse(self.audio)
                self.assertFalse(self.tts_text)
                self.assertEqual([m['type'] for m in self.messages], ['answer_start'])

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


class SpeechFailureTests(unittest.IsolatedAsyncioTestCase):
    """Exercise Studio's actual public reply entry with isolated providers."""

    def setUp(self):
        from unittest.mock import AsyncMock
        self.case = ReplyFixture()
        self.case.setUp()
        self.case.timeline.context_managed = True
        self.case.agent._send_reply_display = AsyncMock()

    async def run_reply(self, speech, model=None):
        from types import SimpleNamespace
        async def default_model(*_):
            yield '认真|第一句话。第二句话。'
        case = self.case
        case.agent.vm.reply_stream = model or default_model
        return await asyncio.wait_for(case.agent.voicemem_llm_tts(
            case.pending, case.send, case.send_audio, {}, case.timeline,
            context_session='fixture-session', context_space='fixture',
            speech_provider=SimpleNamespace(stream=speech, SERIAL=True)), 1)

    def assert_failed(self):
        events = self.case.messages
        self.assertFalse(any(event['type'] == 'answer_done' for event in events))
        errors = [event for event in events if event['type'] == 'error']
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['code'], 'speech_failed')
        self.assertEqual(errors[0]['output_id'], self.case.timeline.output_id)
        self.assertFalse(self.case.timeline.generation_complete)
        self.assertTrue(self.case.timeline.interrupted)
        self.case.agent.queue_remember_turn.assert_not_called()

    async def test_failed_first_segment_is_an_error_instead_of_silent_success(self):
        from studio.core.utils.contracts.component import SpeechSynthesisError
        async def speech(*_):
            raise RuntimeError('private vendor error detail')
            yield b''
        with self.assertRaises(SpeechSynthesisError):
            await self.run_reply(speech)
        self.assertEqual(self.case.audio, [])
        self.assert_failed()
        self.assertNotIn('private vendor', str(self.case.messages))

    async def test_empty_synthesis_is_not_a_success(self):
        from studio.core.utils.contracts.component import SpeechSynthesisError
        async def speech(*_):
            yield b''
        with self.assertRaises(SpeechSynthesisError):
            await self.run_reply(speech)
        self.assert_failed()
        self.assertEqual(self.case.audio, [])

    async def test_partial_audio_failure_closes_the_generator(self):
        from studio.core.utils.contracts.component import SpeechSynthesisError
        closed = asyncio.Event()
        async def speech(*_):
            try:
                yield bytes(480)
                raise RuntimeError('synthetic partial failure')
            finally:
                closed.set()
        with self.assertRaises(SpeechSynthesisError):
            await self.run_reply(speech)
        self.assertTrue(closed.is_set())
        self.assertTrue(self.case.audio)
        self.assert_failed()

    async def test_failed_later_segment_is_not_skipped(self):
        from studio.core.utils.contracts.component import SpeechSynthesisError
        calls = []
        async def speech(text, *_):
            calls.append(text)
            if len(calls) == 2:
                raise RuntimeError('synthetic second segment failure')
            yield bytes(480)
        with self.assertRaises(SpeechSynthesisError):
            await self.run_reply(speech)
        self.assertEqual(len(calls), 2)
        self.assertTrue(self.case.audio)
        self.assert_failed()

    async def test_speech_failure_cancels_an_llm_waiting_for_more_tokens(self):
        from studio.core.utils.contracts.component import SpeechSynthesisError
        closed = asyncio.Event()
        async def model(*_):
            try:
                yield '认真|第一句话。'
                await asyncio.Event().wait()
            finally:
                closed.set()
        async def speech(*_):
            raise RuntimeError('synthetic failure')
            yield b''
        before = set(asyncio.all_tasks())
        with self.assertRaises(SpeechSynthesisError):
            await self.run_reply(speech, model)
        self.assertTrue(closed.is_set())
        self.assert_failed()
        self.assertFalse(set(asyncio.all_tasks()) - before)

    async def test_cancellation_closes_tts_and_llm_without_a_failure_notice(self):
        entered, speech_closed, model_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def model(*_):
            try:
                yield '认真|第一句话。'
                await asyncio.Event().wait()
            finally:
                model_closed.set()
        async def speech(*_):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield bytes(480)
            finally:
                speech_closed.set()
        task = asyncio.create_task(self.run_reply(speech, model))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(speech_closed.is_set())
        self.assertTrue(model_closed.is_set())
        self.assertTrue(self.case.timeline.interrupted)
        self.assertFalse(any(event['type'] in {'error', 'answer_done'} for event in self.case.messages))

    async def test_cancellation_during_text_delivery_closes_a_suspended_llm(self):
        from voicemem.reply import capture
        entered, closed = asyncio.Event(), asyncio.Event()
        async def provider():
            try:
                yield '认真|第一句话。'
                yield '第二句话。'
            finally:
                closed.set()
        def model(*_):
            return capture(provider(), lambda text: None)
        original_send = self.case.send
        async def blocked_send(event):
            if event['type'] == 'answer_delta':
                entered.set()
                await asyncio.Event().wait()
            await original_send(event)
        async def speech(*_):
            yield bytes(480)
        self.case.send = blocked_send
        task = asyncio.create_task(self.run_reply(speech, model))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(closed.is_set())
        self.assertFalse(any(event['type'] in {'error', 'answer_done'} for event in self.case.messages))


if __name__ == "__main__":
    unittest.main()
