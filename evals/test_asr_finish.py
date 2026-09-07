"""CPU-only regression: final ASR must not wait behind obsolete streaming audio."""
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
from voicemem.stream import VoiceStream, _AsrWorker


class FinishTests(unittest.IsolatedAsyncioTestCase):
    def make_stream(self, transcribe):
        stream = VoiceStream(types.SimpleNamespace(), gate=lambda _: "shallow")
        stream._text = "partial"
        stream._final_asr = types.SimpleNamespace(transcribe=transcribe)
        self.flushed = asyncio.get_running_loop().create_future()
        self.resets = 0
        def reset():
            self.resets += 1
        stream._asr_w = types.SimpleNamespace(flush=lambda: self.flushed, reset=reset)
        return stream

    async def test_full_audio_result_bypasses_unfinished_stream_and_uses_full_coverage(self):
        pcm = np.arange(3200, dtype=np.float32)
        received = []
        def transcribe(audio):
            received.append(audio.copy())
            return "完整句子，包括最后的补充"
        stream = self.make_stream(transcribe)
        await asyncio.wait_for(stream._finish_asr(pcm), .5)
        self.assertEqual(stream._text, "完整句子，包括最后的补充")
        self.assertEqual(stream._raw_text, stream._text)
        self.assertTrue(self.flushed.cancelled())
        self.assertEqual(self.resets, 1)
        np.testing.assert_array_equal(received[0], pcm)

    async def test_valid_stream_still_waits_for_full_audio_refinement(self):
        release = threading.Event()
        def transcribe(_):
            release.wait(2)
            return "refined"
        stream = self.make_stream(transcribe)
        self.flushed.set_result("complete streaming")
        task = asyncio.create_task(stream._finish_asr(np.ones(100)))
        try:
            await asyncio.sleep(.02)
            self.assertFalse(task.done())
        finally:
            release.set()
        await task
        self.assertEqual(stream._text, "refined")
        self.assertEqual(stream._raw_text, "complete streaming")
        self.assertEqual(self.resets, 0)

    async def test_failed_empty_and_disabled_final_use_complete_stream_fallback(self):
        def fail(_):
            raise ValueError("failed")
        for transcribe in (lambda _: None, lambda _: "", lambda _: "   ", fail):
            stream = self.make_stream(transcribe)
            task = asyncio.create_task(stream._finish_asr(np.ones(100)))
            await asyncio.sleep(.01)
            self.assertFalse(task.done())
            self.flushed.set_result("complete fallback")
            await task
            self.assertEqual(stream._text, "complete fallback")
            self.assertEqual(self.resets, 0)
        stream = self.make_stream(lambda _: "unused")
        stream._final_asr = None
        self.flushed.set_result("disabled fallback")
        await stream._finish_asr(np.ones(100))
        self.assertEqual(stream._text, "disabled fallback")

    async def test_cancelled_offline_result_cannot_mutate_new_turn(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        def transcribe(_):
            started.set()
            release.wait(2)
            finished.set()
            return "obsolete"
        stream = self.make_stream(transcribe)
        task = asyncio.create_task(stream._finish_asr(np.ones(100)))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            stream._text = "new turn"
        finally:
            release.set()
        await asyncio.to_thread(finished.wait, 1)
        await asyncio.sleep(0)
        self.assertEqual(stream._text, "new turn")
        self.assertTrue(self.flushed.cancelled())

    async def test_audio_feed_returns_turn_before_stalled_worker_finishes(self):
        stream = self.make_stream(lambda _: "完整句子")
        stream.confirm_s, stream.spec_min_chars = .02, 999
        speech = iter([True, False])
        stream._vad = types.SimpleNamespace(is_speech=lambda _: next(speech))
        stream._asr_w.push = lambda _: "流式半句"
        stream._asr_w.report = lambda: "stalled"
        pcm = np.zeros(480, np.int16).tobytes()
        first = await stream.feed(pcm)
        result = await asyncio.wait_for(stream.feed(pcm), .5)
        self.assertEqual(result.turn.text, "完整句子")
        self.assertEqual(result.turn.raw_text, "完整句子")
        self.assertEqual(result.speech_end, first.speech_end)
        self.assertEqual(result._pcm.shape, (640,))


class EpochTests(unittest.IsolatedAsyncioTestCase):
    async def test_reset_skips_old_queue_and_serializes_reset_before_new_audio(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        active, peak = 0, 0
        def feed(frame):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            n = int(frame[0])
            calls.append(("feed", n))
            if n == 1:
                started.set()
                release.wait(2)
            active -= 1
            return str(n)
        worker = _AsrWorker(types.SimpleNamespace(
            feed=feed, flush=lambda: "fresh", reset=lambda: calls.append(("reset", 0))))
        try:
            worker.push(np.ones(320))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            worker.push(np.full(320, 2))
            old_flush = worker.flush()
            worker.reset()
            self.assertEqual(worker.push(np.full(320, 3)), "")
            new_flush = worker.flush()
            self.assertEqual(await asyncio.wait_for(old_flush, .5), "")
            release.set()
            self.assertEqual(await asyncio.wait_for(new_flush, 1), "fresh")
        finally:
            release.set()
        self.assertEqual(calls, [("feed", 1), ("reset", 0), ("feed", 3)])
        self.assertEqual(worker.text, "fresh")
        self.assertEqual(worker.backlog, 0)
        self.assertEqual(peak, 1)

    async def test_inflight_flush_cannot_publish_old_text_after_reset(self):
        started, release = threading.Event(), threading.Event()
        def flush():
            started.set()
            release.wait(2)
            return "old"
        worker = _AsrWorker(types.SimpleNamespace(
            feed=lambda _: "new", flush=flush, reset=lambda: None))
        old_flush = worker.flush()
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            worker.reset()
            release.set()
            self.assertEqual(await asyncio.wait_for(old_flush, 1), "")
            self.assertEqual(worker.text, "")
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
