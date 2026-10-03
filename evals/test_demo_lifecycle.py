"""Account leases and idle eviction protect in-flight memory writes."""
import tempfile
import types
import unittest
import asyncio
import threading
from unittest.mock import AsyncMock
from pathlib import Path

from studio.web.demo_accounts import DemoAccounts
from studio.web.transport import build_app


class AccountLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.accounts = DemoAccounts(types.SimpleNamespace(lang='zh'), root=self.temp.name,
                                     agent_factory=lambda args, root: types.SimpleNamespace(
                                         root=Path(root), _REMEMBER_TASKS=set(), _MEMORY_WRITES=set()))

    def test_two_connections_keep_one_agent_pinned_until_both_close(self):
        first = self.accounts.acquire_agent('fixture')
        second = self.accounts.acquire_agent('fixture')
        self.assertIs(first, second)
        self.assertEqual(self.accounts.prune_idle(idle_seconds=0, max_idle_agents=0), 0)
        self.accounts.release_agent('fixture', first)
        self.assertEqual(self.accounts.prune_idle(idle_seconds=0, max_idle_agents=0), 0)
        self.accounts.release_agent('fixture', second)
        self.assertEqual(self.accounts.prune_idle(idle_seconds=0, max_idle_agents=0), 1)
        reopened = self.accounts.acquire_agent('fixture')
        self.assertIsNot(reopened, first)
        self.assertEqual(reopened.root, first.root)

    def test_native_memory_write_prevents_eviction_after_async_task_returns(self):
        agent = self.accounts.acquire_agent('fixture')
        agent._MEMORY_WRITES.add('pending-write')
        self.accounts.release_agent('fixture', agent)
        self.assertEqual(self.accounts.prune_idle(idle_seconds=0, max_idle_agents=0), 0)
        agent._MEMORY_WRITES.clear()
        self.assertEqual(self.accounts.prune_idle(idle_seconds=0, max_idle_agents=0), 1)

    def test_idle_capacity_and_timeout_leave_active_account_untouched(self):
        active = self.accounts.acquire_agent('active')
        for index in range(10):
            uid = f'fixture-{index}'
            self.accounts.release_agent(uid, self.accounts.acquire_agent(uid))
        self.assertEqual(len(self.accounts._agents), 9)
        self.assertEqual(self.accounts.prune_idle(max_idle_agents=2), 6)
        self.assertIs(self.accounts._agents['active'], active)
        self.assertEqual(self.accounts.prune_idle(now=float('inf')), 2)
        self.assertEqual(list(self.accounts._agents), ['active'])


class CancelledAccountRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_cold_connection_releases_lease_after_factory_returns(self):
        entered, resume, released = threading.Event(), threading.Event(), threading.Event()
        agent, release_calls = object(), []

        def acquire(_):
            entered.set()
            resume.wait(2)
            return agent

        def release(uid, value):
            release_calls.append((uid, value))
            released.set()

        accounts = types.SimpleNamespace(user=lambda _: ('fixture', 'Fixture'),
                                         acquire_agent=acquire, release_agent=release)
        app = build_app('llm_tts', lambda _: None, lambda _: None,
                        demo_accounts=accounts, demo_session=lambda *args: None)
        endpoint = next(route.endpoint for route in app.routes if route.path == '/ws')
        socket = types.SimpleNamespace(
            headers={'origin':'https://fixture.test','host':'fixture.test'},
            cookies={'vm_demo_session':'fixture-cookie'}, accept=AsyncMock())
        task = asyncio.create_task(endpoint(socket))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            socket.accept.assert_not_awaited()
        finally:
            resume.set()
        self.assertTrue(await asyncio.to_thread(released.wait, 1))
        self.assertEqual(release_calls, [('fixture', agent)])
        await asyncio.gather(*tuple(app.state.account_jobs), return_exceptions=True)


if __name__ == '__main__':
    unittest.main()
