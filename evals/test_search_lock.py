"""CPU-only: real Search must never wait for a child under TORCH_LOCK."""
import asyncio
from pathlib import Path
import sys
import threading
import time
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicemem.stream import VoiceStream
from voicemem.orchestrator import Orchestrator
from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from voicemem.leftbrain.local_embedder import _LockedEncoder
from voicemem.utils.torch_lock import TORCH_LOCK


class SearchLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_speculation_and_search_allow_right_brain_to_encode(self):
        classification = QueryClassification(slots=["daily_life"], entities=[])
        acquired_by_child, encoded_on = [], []
        def encode(*args, **kwargs):
            encoded_on.append(threading.get_ident())
            return np.ones((1, 3))
        encoder = _LockedEncoder(types.SimpleNamespace(encode=encode))
        def right_search(*args):
            # Bounded acquisition: regression fails instead of hanging the suite.
            acquired = TORCH_LOCK.acquire(timeout=.3)
            acquired_by_child.append(acquired)
            if acquired:
                try:
                    encoder.encode(["right query"])
                finally:
                    TORCH_LOCK.release()
            return [], ""
        def left_search(*args):
            now = time.time()
            return dict(slot_mem_ids=set(), final_ids=set(), activated_names=[],
                        classification=classification, related_summaries={}, t0=now, t1=now, t2=now)
        def rank(*args, **kwargs):
            encoder.encode(["left query"])
            return []
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._left = types.SimpleNamespace(search=left_search, rank=rank)
        orchestrator._right = types.SimpleNamespace(search=right_search)
        orchestrator.last_agent_reply = lambda: ""
        orchestrator._record_subgraph_activation = lambda _: None
        orchestrator._user_id = "test"
        orchestrator._get_trigger_store = lambda: types.SimpleNamespace(get_last_scene=lambda _: None)
        vm = types.SimpleNamespace(classify=lambda _: classification,
            search=lambda text, **kw: orchestrator.Search(text, scene_filter="office", **kw))
        stream = VoiceStream(vm, gate=lambda _: "deep")
        result = await asyncio.wait_for(stream._speculate("我喜欢什么咖啡"), 2)
        self.assertEqual(acquired_by_child, [True])
        self.assertEqual(len(set(encoded_on)), 2)
        self.assertEqual(result.text, "我喜欢什么咖啡")
        self.assertEqual(result.result.classification, classification)

    async def test_asr_can_acquire_model_lock_while_search_waits(self):
        started, release = threading.Event(), threading.Event()
        result = types.SimpleNamespace(hits=[], timing={})
        def search(*args, **kwargs):
            started.set()
            release.wait(2)
            return result
        vm = types.SimpleNamespace(search=search, classify=lambda _: QueryClassification([], []))
        stream = VoiceStream(vm, gate=lambda _: "deep")
        task = asyncio.create_task(stream._speculate("测试"))
        def asr_probe():
            acquired = TORCH_LOCK.acquire(timeout=.3)
            if acquired:
                TORCH_LOCK.release()
            return acquired
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            self.assertTrue(await asyncio.to_thread(asr_probe))
        finally:
            release.set()
            await task

    async def test_model_encoder_stays_serial_across_threads(self):
        active, peak = 0, 0
        def encode(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            time.sleep(.01)
            active -= 1
            return "vector"
        encoder = _LockedEncoder(types.SimpleNamespace(encode=encode))
        results = await asyncio.gather(*[
            asyncio.to_thread(encoder.encode, ["test"]) for _ in range(8)])
        self.assertEqual(results, ["vector"] * 8)
        self.assertEqual(peak, 1)

    async def test_shallow_route_does_not_search(self):
        def unexpected(*args, **kwargs):
            raise AssertionError("shallow route must not access memory")
        stream = VoiceStream(types.SimpleNamespace(classify=unexpected, search=unexpected),
                             gate=lambda _: "shallow")
        result = await stream._speculate("你好")
        self.assertEqual(result.route, "shallow")
        self.assertEqual(result.result.hits, [])


if __name__ == "__main__":
    unittest.main()
