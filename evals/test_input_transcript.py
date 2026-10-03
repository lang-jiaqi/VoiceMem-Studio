"""Offline input-display regressions with real orchestration and fake providers."""
import asyncio
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from studio.core.core import converse
from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.contracts.component import Pending, ReplySink, input_transcript_event
from studio.core.utils.conversation.component import Conversation
from studio.core.utils.session_context.component import SessionBuffer
from voicemem.stream import empty_result
from evals.test_reply_latency import display_namespace
from evals import test_short_turns as capture_fixtures


def pending(text="完整问题", input_id="input-one", **kwargs):
    return Pending(text, "", empty_result(), input_turn_id=input_id, **kwargs)


class InputIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_partials_and_confirmed_input_share_one_id(self):
        sent, turns, _ = await capture_fixtures.AnticipateTests().run_frames([
            (False, "", capture_fixtures.state("第一个问题")),
            (False, "", capture_fixtures.state("第一个完整问题", final=True)),
            (False, "", capture_fixtures.state("第二个问题")),
            (False, "", capture_fixtures.state("第二个完整问题", final=True)),
        ])
        self.assertEqual(len(turns), 2)
        self.assertNotEqual(turns[0].input_turn_id, turns[1].input_turn_id)
        partials = [m for m in sent if m["type"] == "partial_transcript"]
        for message in partials:
            expected = turns[0] if message['text'].startswith('第一') else turns[1]
            self.assertEqual(message['input_turn_id'], expected.input_turn_id)
        self.assertTrue(partials)

    def test_display_ids_do_not_become_output_or_memory_ids(self):
        message = input_transcript_event(pending(replace_input_turn_id="prefix"))
        self.assertEqual(message, {"type": "user_transcript", "text": "完整问题",
                                  "input_turn_id": "input-one", "replace_input_turn_id": "prefix"})
        self.assertEqual(input_transcript_event(types.SimpleNamespace(text="legacy")),
                         {"type": "user_transcript", "text": "legacy"})

    async def test_publication_does_not_call_context_or_memory_writers(self):
        session = types.SimpleNamespace(sock=types.SimpleNamespace(send_json=AsyncMock()),
                                        agent=types.SimpleNamespace(_push_history=Mock(), queue_remember_turn=Mock()))
        value = pending()
        await Conversation.publish_user_input(session, value)
        session.sock.send_json.assert_awaited_once_with(input_transcript_event(value))
        session.agent._push_history.assert_not_called()
        session.agent.queue_remember_turn.assert_not_called()
        self.assertTrue(value.transcript_managed)

    async def test_core_publishes_before_routing_even_when_routing_fails(self):
        value = pending()
        events = []
        async def listen():
            yield value
        async def send(message):
            events.append(message['type'])
        async def fail_route(_):
            self.assertTrue(value.transcript_managed)
            events.append('route')
            raise RuntimeError('synthetic routing failure')
        session = types.SimpleNamespace(
            sock=types.SimpleNamespace(send_json=send), listen=listen,
            ignore=lambda _:False, merge_continuation=AsyncMock(side_effect=lambda p:p),
            stop_prewarm=Mock(), cached_ack=lambda _:None,
            route=fail_route, close_session=AsyncMock())
        session.publish_user_input = types.MethodType(Conversation.publish_user_input, session)
        with patch('studio.core.utils.conversation.component.Conversation', return_value=session):
            with self.assertRaisesRegex(RuntimeError, 'synthetic routing'):
                await converse(object(), session.sock)
        self.assertEqual(events, ['user_transcript', 'route'])
        session.close_session.assert_awaited_once()

    async def test_ignored_input_is_not_published(self):
        async def listen():
            yield pending()
        session = types.SimpleNamespace(listen=listen, ignore=lambda _:True,
                                        publish_user_input=AsyncMock(), close_session=AsyncMock())
        with patch('studio.core.utils.conversation.component.Conversation', return_value=session):
            await converse(object(), object())
        session.publish_user_input.assert_not_awaited()

    async def test_unfinished_input_is_visible_before_delayed_followup(self):
        value = pending('因为')
        events = []
        async def listen():
            yield value
        async def send(message):
            events.append(message['type'])
        async def defer(_):
            events.append('defer')
        session = types.SimpleNamespace(listen=listen, ignore=lambda _:False,
            merge_continuation=AsyncMock(side_effect=lambda p:p), stop_prewarm=Mock(),
            sock=types.SimpleNamespace(send_json=send), drop_early=AsyncMock(),
            defer_unfinished_reply=defer, close_session=AsyncMock())
        session.publish_user_input = types.MethodType(Conversation.publish_user_input, session)
        with patch('studio.core.utils.conversation.component.Conversation', return_value=session):
            await converse(object(), session.sock)
        self.assertEqual(events, ['user_transcript', 'defer'])

    async def test_failed_publication_does_not_claim_the_transcript_was_managed(self):
        session = types.SimpleNamespace(sock=types.SimpleNamespace(send_json=AsyncMock(side_effect=ConnectionError)))
        value = pending()
        with self.assertRaises(ConnectionError):
            await Conversation.publish_user_input(session, value)
        self.assertFalse(value.transcript_managed)

    async def test_continuation_preserves_text_and_only_adds_display_replacement(self):
        memory = object()
        agent = types.SimpleNamespace(BC_ECHO_WINDOW_S=3, ACTIVE_SPACE='fixture', vm=memory,
                                      BARGE_DEBUG=False, space_language=lambda _: 'zh')
        sock = types.SimpleNamespace(send_json=AsyncMock())
        session = Conversation(agent, sock)
        base = pending('我想问一下', 'prefix')
        await session.publish_user_input(base)
        waiting = asyncio.create_task(asyncio.Event().wait())
        session.unfinished_wait = {'pending':base, 'task':waiting, 'space':'fixture', 'memory_vm':memory}
        session.turn['continuation_task'] = waiting
        value = pending('模型应该怎么选？', 'continuation', early_ok=True)
        result_before = value.result
        merged = await session.merge_continuation(value)
        self.assertEqual(merged.text, '我想问一下模型应该怎么选？')
        self.assertEqual(merged.input_turn_id, 'continuation')
        self.assertEqual(merged.replace_input_turn_id, 'prefix')
        self.assertIs(merged.result, result_before)
        self.assertFalse(merged.early_ok)
        self.assertFalse(merged.transcript_managed)
        self.assertTrue(waiting.cancelled())
        await session.publish_user_input(merged)
        self.assertEqual(sock.send_json.await_count, 2)


class PipelineDisplayTests(unittest.IsolatedAsyncioTestCase):
    def setup_pipeline(self):
        ns = display_namespace()
        self.entered = asyncio.Event()
        self.answer_started = asyncio.Event()
        self.inputs, self.sent, self.saved = [], [], []
        self.history = SessionBuffer()
        async def model(text, context, history):
            self.inputs.append(text)
            self.entered.set()
            await asyncio.Event().wait()
            yield 'unreached'
        async def tts(*args):
            yield bytes(480)
        async def send(message):
            self.sent.append(message)
            if message['type'] == 'answer_start':
                self.answer_started.set()
        async def audio(_):
            pass
        def keep(session, space, user, answer, **kwargs):
            self.saved.append((user, answer, kwargs.get('interrupted')))
            return self.history.add(session, space, user, answer, **kwargs)
        ns.update(_SESSION_CONTEXT=self.history, HISTORY_TURNS=6, _speak_instruction=lambda _, lang='': '',
            _speak_base_env='', _SPEAK_BASE={}, _by_lang=lambda _, lang='': '', _LAST_TONE={'tag':''},
            space_language=lambda _: 'zh',
            tts_control=types.SimpleNamespace(split=lambda text: ('',text), smooth=lambda a,b:b, instruction=lambda *a:''),
            build_reply_context=lambda *a,**k:'', BARGE_DEBUG=False, _lat_note=lambda _: '', _mem_line=lambda:'',
            TimedAudioChunk=type('TimedAudioChunk',(),{}), hot_path_enter=lambda:None, hot_path_exit=lambda _:None,
            _kick_acoustic=lambda *a:None, _push_history=keep, queue_remember_turn=Mock())
        memory = types.SimpleNamespace(reply_stream=model, utils=types.SimpleNamespace(get=lambda _:types.SimpleNamespace(stream=tts)))
        ns['vm'] = memory
        return ns, memory, send, audio

    async def test_confirmed_full_input_survives_reply_cancel_before_first_audio(self):
        ns, memory, send, audio = self.setup_pipeline()
        value = pending('请比较准确率、响应速度和维护成本。')
        session = types.SimpleNamespace(sock=types.SimpleNamespace(send_json=send))
        await Conversation.publish_user_input(session, value)
        sink = ReplySink(send, audio)
        task = asyncio.create_task(ns['_voicemem_llm_tts'](value, sink.send, sink.send_audio,
            {}, AudioTimeline(), context_session='fixture', context_space='fixture', memory_vm=memory))
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            self.assertEqual(self.inputs, [value.text])
            self.assertEqual(self.sent, [input_transcript_event(value)])
            self.assertFalse(any(kind == 'json' and msg.get('type') == 'user_transcript'
                                 for kind, msg in sink._buf))
        finally:
            task.cancel()
            await asyncio.wait_for(task, 1)
        self.assertEqual(self.sent, [input_transcript_event(value)])
        self.assertEqual(self.saved, [(value.text, '', True)])
        ns['queue_remember_turn'].assert_called_once()
        self.assertEqual(self.history.messages('fixture', 'fixture')[0]['content'], value.text)

    async def test_legacy_direct_pipeline_still_emits_transcript(self):
        ns, memory, send, audio = self.setup_pipeline()
        value = pending(input_id='')
        task = asyncio.create_task(ns['_voicemem_llm_tts'](value, send, audio, {}, AudioTimeline(), memory_vm=memory))
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            self.assertEqual(self.sent[0], {'type':'user_transcript', 'text':value.text})
        finally:
            task.cancel()
            await asyncio.wait_for(task, 1)

    async def test_interrupted_context_still_contains_only_heard_assistant_prefix(self):
        ns, memory, send, audio = self.setup_pipeline()
        value = pending('确认的用户输入')
        await Conversation.publish_user_input(types.SimpleNamespace(sock=types.SimpleNamespace(send_json=send)), value)
        timeline = AudioTimeline()
        generated = '已经播出的部分后面还有未播放的内容'
        timeline.append_text(generated)
        segment = timeline.begin_segment(0, len(generated))
        timeline.append_audio(bytes(48000))
        timeline.finish_segment(segment)
        timeline.update_checkpoint(9600, 24000, 'paused')
        heard = timeline.heard_text()
        self.assertTrue(heard)
        self.assertNotEqual(heard, generated)
        task = asyncio.create_task(ns['_voicemem_llm_tts'](value, send, audio, {}, timeline, memory_vm=memory))
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
        finally:
            task.cancel()
            await asyncio.wait_for(task, 1)
        self.assertEqual(self.saved, [(value.text, heard, True)])
        ns['queue_remember_turn'].assert_called_once()
        self.assertNotEqual(timeline.output_id, value.input_turn_id)

    async def test_early_snapshot_never_emits_final_user_transcript(self):
        ns, memory, send, audio = self.setup_pipeline()
        agent = types.SimpleNamespace(BC_ECHO_WINDOW_S=3, ACTIVE_SPACE='fixture', vm=memory,
            HISTORY_TURNS=6, _SESSION_CONTEXT=self.history, BARGE_DEBUG=False, space_language=lambda _: 'zh',
            route_pending_thinking=AsyncMock(side_effect=lambda p,*a,**k:p),
            voicemem_llm_tts=ns['_voicemem_llm_tts'])
        session = Conversation(agent, types.SimpleNamespace(send_json=send, send_bytes=audio))
        st = types.SimpleNamespace(memory=empty_result(), route='shallow', eot_score=.9)
        await session.start_early('临时快照', st)
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            self.assertTrue(session.early['pending'].transcript_managed)
            self.assertEqual(self.sent, [])
            self.assertFalse(any(kind == 'json' and msg.get('type') == 'user_transcript'
                                 for kind, msg in session.early['sink']._buf))
        finally:
            await session.drop_early()
        self.assertEqual(self.sent, [])
        self.assertEqual(self.saved, [])
        ns['queue_remember_turn'].assert_not_called()

    async def test_accepting_early_reply_cannot_republish_its_snapshot(self):
        ns, memory, send, audio = self.setup_pipeline()
        agent = types.SimpleNamespace(BC_ECHO_WINDOW_S=3, ACTIVE_SPACE='fixture', vm=memory,
            HISTORY_TURNS=6, _SESSION_CONTEXT=self.history, BARGE_DEBUG=False, MIC_RATE=24000, space_language=lambda _: 'zh',
            route_pending_thinking=AsyncMock(side_effect=lambda p,*a,**k:p),
            _push_history=ns['_push_history'], queue_remember_turn=ns['queue_remember_turn'],
            _kick_acoustic=ns['_kick_acoustic'],
            voicemem_llm_tts=ns['_voicemem_llm_tts'])
        session = Conversation(agent, types.SimpleNamespace(send_json=send, send_bytes=audio))
        st = types.SimpleNamespace(memory=empty_result(), route='shallow', eot_score=.9)
        await session.start_early('测试问题', st)
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            value = pending('测试问题。', early_ok=True)
            await session.publish_user_input(value)
            routed = asyncio.get_running_loop().create_future()
            routed.set_result(value)
            self.assertTrue(await session.commit_early(value, routed))
            await asyncio.wait_for(self.answer_started.wait(), 1)
            self.assertEqual([m for m in self.sent if m['type']=='user_transcript'], [input_transcript_event(value)])
            self.assertEqual(self.inputs, ['测试问题'])
        finally:
            await session.close_session()
        self.assertEqual([m for m in self.sent if m['type']=='user_transcript'], [input_transcript_event(value)])


class RealtimeTranscriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_retains_transcript_order_and_input_identity(self):
        from studio.core.utils.realtime_session.component import RealtimeSession
        agent = types.SimpleNamespace(audio_of=None, hit_cluster=None,
            fill_tags=lambda *a,**k:{}, _kick_acoustic=Mock(), _realtime_instructions=lambda *a,**k:'fixture')
        conn = types.SimpleNamespace(input_audio_buffer=types.SimpleNamespace(commit=AsyncMock()),
                                      response=types.SimpleNamespace(create=AsyncMock()))
        send = AsyncMock()
        value = pending(route='shallow')
        with patch('studio.core.utils.realtime_session.component.utils.hits_payload', return_value={}):
            await RealtimeSession.start_realtime_turn(agent, value, conn, send, AudioTimeline())
        self.assertEqual(send.await_args_list[0].args[0], input_transcript_event(value))
        self.assertEqual(send.await_args_list[-1].args[0]['type'], 'answer_start')
        conn.input_audio_buffer.commit.assert_awaited_once()
        conn.response.create.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
