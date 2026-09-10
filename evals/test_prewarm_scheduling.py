"""CPU-only regressions: background KV warming must yield to speech replies.

No demo import, GPU/model load, network or memory DB writes.
"""
import ast
import asyncio
import contextlib
import io
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.studio_helpers import studio_tree, studio_source, execute


def session_namespace():
    tree = studio_tree()
    names = {"prewarm_local", "stop_prewarm", "hearing", "close_session",
             "start_early", "drop_early"}
    functions = [n for n in tree.body if getattr(n, "name", "") in names]
    async def route_pending(pending, memory_vm=None, history=None):
        return pending

    ns = dict(
        asyncio=asyncio, threading=threading, time=time,
        prewarm={"task": None, "cancelled": None, "closed": False},
        turn={"task": None, "timeline": None, "until": 0.0},
        early={"task": None}, candidate_paused=False,
        _SESSION_CONTEXT=types.SimpleNamespace(messages=lambda *a, **kw: []),
        context_session="session", ACTIVE_SPACE="test", HISTORY_TURNS=6,
        gate=types.SimpleNamespace(DEEP="deep", needs_memory=lambda route: route == "deep"),
        route_pending_thinking=route_pending,
        DIRECT="direct", MEMORY="memory",
        build_memory_context=lambda _: "memory",
        build_reply_context=lambda *a, **kw: "context", _replay_id=lambda *a: "",
        BARGE_DEBUG=False, owner={}, speech_rate=None, vm=object(),
        Pending=lambda *a, **kw: types.SimpleNamespace(),
        ReplySink=lambda *a: types.SimpleNamespace(send=None, send_audio=None),
        AudioTimeline=lambda **kw: object(),
        sock=types.SimpleNamespace(send_json=None), send_audio=None)
    execute(functions, ns)
    ns["agent"] = ns["self"]
    return ns


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = session_namespace()
        self.calls = []
        def warm(hist, ctx, *, cancelled):
            self.calls.append((hist, ctx, cancelled))
            return 42
        self.ns["_LOCAL_LLM"] = types.SimpleNamespace(prewarm=warm)

    async def test_idle_can_warm_and_does_not_duplicate_pending_work(self):
        self.assertTrue(self.ns["prewarm_local"](object(), "你好"))
        self.assertFalse(self.ns["prewarm_local"]())
        await self.ns["prewarm"]["task"]
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][:2], ([], "context"))
        self.assertFalse(self.calls[0][2].is_set())

    async def test_early_generation_and_finished_uncommitted_audio_block_warming(self):
        task = asyncio.create_task(asyncio.sleep(0))
        self.ns["early"]["task"] = task
        self.assertFalse(self.ns["prewarm_local"]())
        await task
        self.assertFalse(self.ns["prewarm_local"]())
        self.ns["early"]["task"] = None
        self.assertTrue(self.ns["prewarm_local"]())
        await self.ns["prewarm"]["task"]
        self.assertEqual(len(self.calls), 1)

    async def test_formal_generation_and_remaining_browser_audio_block_warming(self):
        task = asyncio.create_task(asyncio.sleep(0))
        self.ns["turn"]["task"] = task
        self.assertFalse(self.ns["prewarm_local"]())
        await task
        self.ns["turn"]["until"] = time.monotonic() + 10
        self.assertFalse(self.ns["prewarm_local"]())
        self.assertEqual(self.calls, [])

    async def test_start_early_invalidates_pending_warm_without_waiting_for_worker(self):
        self.ns["prewarm_local"]()
        warm_task = self.ns["prewarm"]["task"]
        cancelled = self.ns["prewarm"]["cancelled"]
        entered = asyncio.Event()
        pending_text = []
        self.ns["Pending"] = lambda text, *a, **kw: (
            pending_text.append(text) or types.SimpleNamespace())
        async def reply(*a, **kw):
            entered.set()
            await asyncio.Event().wait()
        self.ns["voicemem_llm_tts"] = reply
        refined = asyncio.create_task(asyncio.sleep(0, result="离线复核文本"))
        try:
            await self.ns["start_early"]("你好", types.SimpleNamespace(
                memory=object(), route="deep", eot_score=.99), refined)
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(pending_text, ["离线复核文本"])
            self.assertTrue(cancelled.is_set())
            self.assertTrue(warm_task.cancelled())
            self.assertFalse(self.ns["prewarm_local"]())
        finally:
            await self.ns["close_session"]()

    async def test_close_cancels_warming_early_and_formal_tasks(self):
        self.ns["prewarm_local"]()
        warm_task = self.ns["prewarm"]["task"]
        early_task = asyncio.create_task(asyncio.Event().wait())
        formal_task = asyncio.create_task(asyncio.Event().wait())
        self.ns["early"]["task"] = early_task
        self.ns["turn"]["task"] = formal_task
        await self.ns["close_session"]()
        await asyncio.gather(warm_task, early_task, formal_task, return_exceptions=True)
        self.assertTrue(all(t.cancelled() for t in (warm_task, early_task, formal_task)))
        self.assertTrue(self.ns["prewarm"]["cancelled"].is_set())
        self.assertFalse(self.ns["prewarm_local"]())

    async def test_warm_failure_is_observed_and_does_not_disable_future_attempts(self):
        def fail(*a, **kw):
            raise RuntimeError("test failure")
        self.ns["_LOCAL_LLM"].prewarm = fail
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(self.ns["prewarm_local"]())
            await self.ns["prewarm"]["task"]
        self.assertIn("后台预热失败", output.getvalue())
        self.assertTrue(self.ns["prewarm_local"]())
        await self.ns["close_session"]()

    def test_normal_reply_invalidates_warm_before_early_commit_or_new_generation(self):
        tree = ast.parse((ROOT / "studio/core/core.py").read_text())
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFor))
        calls = [ast.unparse(n) for n in loop.body]
        stop = calls.index("session.stop_prewarm()")
        commit = next(i for i, text in enumerate(calls) if "session.commit_early(" in text)
        self.assertLess(stop, commit)


class QueuedWarmTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalidated_gpu_queue_entry_never_runs_model(self):
        from voicemem.local_llm import LocalLLM
        llm = LocalLLM.__new__(LocalLLM)
        calls = []
        llm._prewarm = lambda *a: calls.append(a) or 42
        queued, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        def call(fn):
            queued.set()
            if not release.wait(2):
                raise TimeoutError("test queue was not released")
            return fn()
        with patch("voicemem.local_llm.gpu_loop", return_value=types.SimpleNamespace(call=call)):
            task = asyncio.create_task(asyncio.to_thread(
                llm.prewarm, [], "memory", cancelled=cancelled))
            try:
                self.assertTrue(await asyncio.to_thread(queued.wait, 1))
                cancelled.set()
            finally:
                release.set()
            self.assertEqual(await task, 0)
        self.assertEqual(calls, [])

    def test_startup_and_uncancelled_warming_preserve_arguments_and_result(self):
        from voicemem.local_llm import LocalLLM
        llm = LocalLLM.__new__(LocalLLM)
        calls = []
        llm._prewarm = lambda *a: calls.append(a) or 42
        with patch("voicemem.local_llm.gpu_loop", return_value=types.SimpleNamespace(call=lambda f: f())):
            self.assertEqual(llm.prewarm(), 42)
            self.assertEqual(llm.prewarm(["history"], "memory", cancelled=threading.Event()), 42)
            cancelled = threading.Event()
            cancelled.set()
            self.assertEqual(llm.prewarm(cancelled=cancelled), 0)
        self.assertEqual(calls, [(None, ""), (["history"], "memory")])


if __name__ == "__main__":
    unittest.main()
