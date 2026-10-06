"""Turn-owned acoustic updates with synthetic inference and no audio files."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from studio.core.utils.audio_timeline.component import AudioTimeline
from studio.core.utils.conversation.component import Conversation
from studio.core.utils.perception.component import Perception


class AcousticLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = Perception()
        self.agent.vm = object()
        self.agent.ACTIVE_SPACE = 'fixture'
        self.agent.BARGE_DEBUG = False
        self.agent.ACOUSTIC_MIN_SCORE = .92
        self.agent.ACOUSTIC_TRUST = {'开心'}
        self.agent.wait_idle = AsyncMock()
        self.agent._acoustic_emotion = Mock(return_value=('开心', .99))
        self.send = AsyncMock()

    def session(self):
        session = Conversation.__new__(Conversation)
        session.agent = self.agent
        session.sock = SimpleNamespace(send_json=self.send)
        session.prewarm = {'closed': False}
        session.acoustic_tasks = set()
        session.turn = {'timeline': None, 'task': None, 'generation_task': None,
                        'continuation_task': None, 'finalize': None}
        session.save_reply_context = Mock()
        session.stop_prewarm = Mock()
        session.drop_early = AsyncMock()
        timeline = AudioTimeline()
        timeline.mark_generation_complete()
        pending = SimpleNamespace(speech_end=0, audio_path='synthetic.wav')
        session.reset_output_state(pending, timeline, {}, memory_vm=self.agent.vm,
                                   context_space='fixture')
        return session, pending, timeline

    async def test_current_result_identifies_its_output(self):
        task = self.agent._kick_acoustic(self.send, 'synthetic.wav', output_id='one', is_current=lambda: True)
        await asyncio.wait_for(task, 1)
        self.send.assert_awaited_once_with({'type': 'tag_update', 'output_id': 'one',
                                          'emotion': '开心', 'emotion_from': 'acoustic'})

    async def test_stale_result_after_native_inference_is_discarded(self):
        entered, release = threading.Event(), threading.Event()
        current = [True]
        def infer(_):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test inference was not released')
            return '开心', .99
        self.agent._acoustic_emotion = infer
        task = self.agent._kick_acoustic(self.send, 'synthetic.wav', output_id='old', is_current=lambda: current[0])
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            current[0] = False
            release.set()
            await asyncio.wait_for(task, 1)
            self.send.assert_not_awaited()
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_owner_change_while_waiting_does_not_start_inference(self):
        current, entered, release = [True], asyncio.Event(), asyncio.Event()
        async def idle(_):
            entered.set()
            await release.wait()
        self.agent.wait_idle = idle
        task = self.agent._kick_acoustic(self.send, 'synthetic.wav', output_id='old', is_current=lambda: current[0])
        await asyncio.wait_for(entered.wait(), 1)
        current[0] = False
        release.set()
        await asyncio.wait_for(task, 1)
        self.agent._acoustic_emotion.assert_not_called()
        self.send.assert_not_awaited()

    async def test_next_output_cancels_the_previous_background_task(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def idle(_):
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        self.agent.wait_idle = idle
        session, pending, timeline = self.session()
        await session.wait_reply_playback(pending, timeline)
        old_tasks = tuple(session.acoustic_tasks)
        await asyncio.wait_for(entered.wait(), 1)
        session.reset_output_state(pending, AudioTimeline(), {})
        await asyncio.gather(*old_tasks, return_exceptions=True)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(session.acoustic_tasks, set())
        self.agent._acoustic_emotion.assert_not_called()
        self.send.assert_not_awaited()

    async def test_changed_space_before_scheduling_keeps_original_ownership(self):
        session, pending, timeline = self.session()
        self.agent.ACTIVE_SPACE = 'another'
        self.agent.vm = object()
        await session.wait_reply_playback(pending, timeline)
        await asyncio.gather(*tuple(session.acoustic_tasks))
        self.agent._acoustic_emotion.assert_not_called()
        self.send.assert_not_awaited()

    async def test_realtime_keeps_ownership_captured_before_provider_awaits(self):
        from studio.core.utils.contracts.component import Pending
        from studio.core.utils.realtime_session.component import RealtimeSession
        from voicemem.stream import empty_result
        self.agent.audio_of = self.agent.hit_cluster = None
        self.agent.fill_tags = lambda *args, **kwargs: {}
        self.agent._realtime_instructions = lambda *args, **kwargs: 'fixture'
        async def switch_space(**kwargs):
            self.agent.ACTIVE_SPACE = 'another'
            self.agent.vm = object()
        conn = SimpleNamespace(
            input_audio_buffer=SimpleNamespace(commit=AsyncMock()),
            response=SimpleNamespace(create=AsyncMock(side_effect=switch_space)))
        pending = Pending('synthetic question', '', empty_result(),
                          route='shallow', audio_path='synthetic.wav')
        timeline = AudioTimeline()
        with patch('studio.core.utils.realtime_session.component.utils.hits_payload', return_value={}):
            task = await RealtimeSession.start_realtime_turn(
                self.agent, pending, conn, self.send, timeline,
                context_space='fixture', is_current=lambda: True)
        await asyncio.wait_for(task, 1)
        self.agent._acoustic_emotion.assert_not_called()
        self.assertFalse(any(call.args[0]['type'] == 'tag_update'
                             for call in self.send.await_args_list))

    async def test_disconnect_cancels_tasks_without_waiting_for_native_inference(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def infer(_):
            entered.set()
            try:
                if not release.wait(2):
                    raise TimeoutError('test inference was not released')
                return '开心', .99
            finally:
                finished.set()
        self.agent._acoustic_emotion = infer
        session, pending, timeline = self.session()
        await session.wait_reply_playback(pending, timeline)
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            session.turn['finalize'] = None
            await asyncio.wait_for(session.close_session(), 1)
            self.assertFalse(release.is_set())
            self.assertEqual(session.acoustic_tasks, set())
            release.set()
            self.assertTrue(await asyncio.to_thread(finished.wait, 1))
            await asyncio.sleep(0)
            self.send.assert_not_awaited()
        finally:
            release.set()
