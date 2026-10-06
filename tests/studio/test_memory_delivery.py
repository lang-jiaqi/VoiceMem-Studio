"""Durable memory completion, account retention and stale WebSocket events."""
import asyncio
import types
import unittest
from unittest.mock import AsyncMock, Mock

from tests.helpers.conversation import ConversationFixture
from studio.core.utils.memory.component import Memory


class MemoryCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_completion_keeps_agent_pinned_after_background_task_finishes(self):
        callbacks, events = [], []
        agent = Memory()
        agent.BARGE_DEBUG = False
        agent._REMEMBER_LOCK, agent._REMEMBER_TASKS = asyncio.Lock(), set()
        agent.wait_idle = AsyncMock()
        agent._finish_history_turn = Mock()

        def ingest(text, **kwargs):
            callbacks.append(kwargs['on_complete'])
            return {'affect': '', 'speaker_id': ''}

        memory = types.SimpleNamespace(ingest=ingest)
        agent.vm = memory
        pending = types.SimpleNamespace(text='synthetic fact', audio_path='', language='en')
        agent.queue_remember_turn(pending, 'synthetic reply', {}, 'turn', memory, events.append)
        task = next(iter(agent._REMEMBER_TASKS))
        await asyncio.wait_for(task, 1)
        await asyncio.sleep(0)
        self.assertFalse(agent._REMEMBER_TASKS)
        self.assertTrue(agent._MEMORY_WRITES)
        self.assertEqual(events, [])
        result = {'persistent_memory_created': True, 'memory_ids': ['fixture-memory']}
        await asyncio.to_thread(callbacks[0], result)
        self.assertFalse(agent._MEMORY_WRITES)
        self.assertEqual(events, [result])
        agent._finish_history_turn.assert_called_once_with('turn', result)
        callbacks[0](result)
        self.assertEqual(events, [result])

    async def test_ingest_failure_releases_write_and_reports_failure(self):
        agent = Memory()
        agent.BARGE_DEBUG = False
        agent._finish_history_turn = Mock()
        memory = types.SimpleNamespace(ingest=Mock(side_effect=RuntimeError('synthetic failure')))
        pending = types.SimpleNamespace(text='synthetic fact', audio_path='', language='en')
        events = []
        await asyncio.to_thread(agent.remember_turn, pending, '', {}, 'turn', memory, events.append)
        self.assertFalse(agent._MEMORY_WRITES)
        self.assertEqual(events, [{'error': True, 'persistent_memory_created': False}])


class MemoryDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def prepare(self):
        fixture = ConversationFixture()
        fixture.setUp()
        self.addAsyncCleanup(fixture.asyncTearDown)
        pending = fixture.pending(input_turn_id='fixture-turn')
        await fixture.start(pending)
        await asyncio.wait_for(fixture.session.turn['task'], 1)
        return fixture, fixture.agent.queue_remember_turn.call_args.kwargs['on_complete']

    async def test_completion_event_is_scoped_to_captured_turn_and_excludes_backend_details(self):
        fixture, completed = await self.prepare()
        await asyncio.to_thread(completed, {'error': 'synthetic backend detail', 'memory_ids': []})
        await asyncio.sleep(0)
        events = [e for e in fixture.messages if e['type'] == 'memory_store_status']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['input_turn_id'], 'fixture-turn')
        self.assertEqual(events[0]['status'], 'error')
        self.assertNotIn('synthetic backend detail', str(events))

    async def test_completion_after_disconnect_or_space_change_is_not_sent(self):
        for change in ('disconnect', 'space', 'memory'):
            with self.subTest(change=change):
                fixture, completed = await self.prepare()
                if change == 'disconnect':
                    fixture.session.prewarm['closed'] = True
                elif change == 'space':
                    fixture.agent.ACTIVE_SPACE = 'other-space'
                else:
                    fixture.agent.vm = object()
                await asyncio.to_thread(completed, {'persistent_memory_created': True})
                await asyncio.sleep(0)
                self.assertFalse(any(e['type'] == 'memory_store_status' for e in fixture.messages))


if __name__ == '__main__':
    unittest.main()
