"""CPU-only regressions; no demo import, model load, network or memory DB writes.

Run: python3 -m unittest discover -s evals -p test_reply_latency.py -v
"""
import ast
import asyncio
import contextlib
import io
from pathlib import Path
import queue
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def display_namespace():
    names = {"ReplySink", "_early_reply_compatible", "_send_reply_display",
             "voicemem_llm_tts", "_voicemem_llm_tts"}
    tree = ast.parse((ROOT / "web/run.py").read_text())
    code = ast.Module(body=[n for n in tree.body if getattr(n, "name", "") in names],
                      type_ignores=[])
    ns = dict(asyncio=asyncio, threading=threading, time=time,
              _REPLY_DISPLAY_LOCK=threading.Lock(), ACTIVE_SPACE="test", vm=object(),
              gate=types.SimpleNamespace(needs_memory=lambda _: True),
              note_hits=lambda _: None, audio_of=None, hit_cluster=None,
              utils=types.SimpleNamespace(hits_payload=lambda *a, **kw: {}),
              fill_tags=lambda *a, **kw: {"emotion": "平静"}, MIC_RATE=24000)
    exec(compile(code, str(ROOT / "web/run.py"), "exec"), ns)
    return ns


class DisplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = display_namespace()
        self.pending = types.SimpleNamespace(result=object(), route="deep", text="你好",
                                             audio_path="")
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    def start(self, ready):
        return asyncio.create_task(self.ns["_send_reply_display"](
            self.pending, self.send, ready, "output-1", "test", self.ns["vm"]))

    async def test_display_waits_for_audio_and_runs_off_event_loop(self):
        ready = asyncio.Event()
        calls = []
        self.ns["fill_tags"] = lambda *a, **kw: calls.append(threading.get_ident()) or {}
        task = self.start(ready)
        await asyncio.sleep(0)
        self.assertEqual(calls, [])
        ready.set()
        await asyncio.wait_for(task, 2)
        self.assertNotEqual(calls[0], threading.get_ident())
        self.assertEqual(self.sent[0]["output_id"], "output-1")

    async def test_cancellation_during_work_never_sends_stale_display(self):
        started, release = threading.Event(), threading.Event()
        def slow(*a, **kw):
            started.set()
            release.wait(2)
            return {}
        self.ns["fill_tags"] = slow
        ready = asyncio.Event()
        ready.set()
        task = self.start(ready)
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            release.set()
        await asyncio.sleep(0)
        self.assertEqual(self.sent, [])

    async def test_space_switch_discards_result(self):
        ready = asyncio.Event()
        task = self.start(ready)
        self.ns["ACTIVE_SPACE"] = "other"
        ready.set()
        await task
        self.assertEqual(self.sent, [])

    async def test_display_failure_does_not_fail_reply(self):
        def fail(*a, **kw):
            raise RuntimeError("display unavailable")
        self.ns["fill_tags"] = fail
        ready = asyncio.Event()
        ready.set()
        await self.start(ready)
        self.assertEqual(self.sent, [])

    async def test_wrapper_cleans_up_on_cancel_before_audio(self):
        entered = asyncio.Event()
        async def pipeline(*a, **kw):
            entered.set()
            await asyncio.Event().wait()
        self.ns["_voicemem_llm_tts"] = pipeline
        task = asyncio.create_task(self.ns["voicemem_llm_tts"](
            self.pending, self.send, self.send, {}, types.SimpleNamespace(output_id="x")))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.sent, [])
        self.assertFalse(any(t.get_coro().__name__ == "_send_reply_display"
                             for t in asyncio.all_tasks() if not t.done()))

    async def test_wrapper_reports_reply_failure_to_browser(self):
        async def pipeline(*a, **kw):
            raise TimeoutError("provider stalled")
        self.ns["_voicemem_llm_tts"] = pipeline
        with self.assertRaises(TimeoutError):
            await self.ns["voicemem_llm_tts"](
                self.pending, self.send, self.send, {},
                types.SimpleNamespace(output_id="x"))
        self.assertEqual(self.sent, [{
            "type": "error",
            "message": "回复服务刚才没有及时返回，已自动重试；请再说一次。",
        }])

    def test_pipeline_does_not_build_display_before_generation(self):
        tree = ast.parse((ROOT / "web/run.py").read_text())
        fn = next(n for n in tree.body if getattr(n, "name", "") == "_voicemem_llm_tts")
        called = {n.func.id for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertNotIn("fill_tags", called)
        self.assertNotIn("note_hits", called)
        self.assertIn("await send_audio(pcm)\n                    if first_audio_ready",
                      (ROOT / "web/run.py").read_text())

    async def test_real_pipeline_sends_audio_before_slow_display(self):
        ns = self.ns
        order = []
        displayed = asyncio.Event()
        release = threading.Event()
        async def stream(*a):
            order.append("llm")
            yield "你好"
        async def tts_stream(*a):
            yield b"\0\0" * 480
        def slow_tags(*a, **kw):
            release.wait(2)
            return {}
        async def audio(pcm):
            order.append("audio")
            release.set()
        async def send(msg):
            order.append(msg["type"])
            if msg["type"] == "memory_hits":
                displayed.set()
        ns.update(
            fill_tags=slow_tags, _SESSION_CONTEXT=types.SimpleNamespace(messages=lambda *a, **k: []),
            HISTORY_TURNS=6, _speak_instruction=lambda _: "", _speak_base_env="soft",
            _SPEAK_BASE={}, _by_lang=lambda _: "", _LAST_TONE={"tag": ""},
            tts_control=types.SimpleNamespace(split=lambda s: ("平静", s),
                                             smooth=lambda a, b: b, instruction=lambda *a: ""),
            build_reply_context=lambda *a, **kw: "original-memory", BARGE_DEBUG=False,
            _FIRST_MAX_CHARS=20, _cut_point=lambda *a, **k: False,
            _lat_note=lambda _: "test", _mem_line=lambda: "test",
            TimedAudioChunk=type("TimedAudioChunk", (), {}),
            hot_path_enter=lambda: {}, hot_path_exit=lambda _: None,
            _kick_acoustic=lambda *a: None, _push_history=lambda *a, **kw: "history",
            queue_remember_turn=lambda *a, **kw: None)
        ns["vm"] = types.SimpleNamespace(
            reply_stream=stream,
            utils=types.SimpleNamespace(get=lambda _: types.SimpleNamespace(stream=tts_stream)))
        timeline = types.SimpleNamespace(
            output_id="output-1", sample_rate=24000, sent_samples=480, context_saved=False,
            begin_segment=lambda *a: 0, append_audio=lambda _: None, append_text=lambda _: None,
            finish_segment=lambda *a, **kw: None, mark_generation_complete=lambda: None,
            wait_playback_done=lambda: asyncio.wait_for(displayed.wait(), 2))
        self.pending.__dict__.update(replay="", emotion="", memory_context="original-memory",
                                     stranger=False, speech_end=time.monotonic(), spoken=True)
        try:
            await ns["voicemem_llm_tts"](self.pending, send, audio, {}, timeline)
        finally:
            release.set()
        self.assertLess(order.index("llm"), order.index("memory_hits"))
        self.assertLess(order.index("audio"), order.index("memory_hits"))
        self.assertLess(order.index("answer_start"), order.index("audio"))
        self.assertTrue(timeline.context_saved)


class ReplySinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_commit_replaces_eot_snapshot_with_confirmed_transcript(self):
        sent = []

        async def send(message):
            sent.append(message)

        sink = display_namespace()["ReplySink"](send, lambda _: None)
        await sink.send({"type": "user_transcript", "text": "我想问"})
        await sink.send({"type": "answer_start", "output_id": "one"})
        await sink.commit(final_user_text="我想问你一个问题")
        self.assertEqual(sent[0]["text"], "我想问你一个问题")

    def test_material_continuation_never_reuses_the_eot_reply(self):
        compatible = display_namespace()["_early_reply_compatible"]
        self.assertFalse(compatible(
            "行那就行", "行那就行那明天我有什么安排你帮我查一下"))

    def test_punctuation_fillers_and_small_asr_repairs_keep_the_fast_path(self):
        compatible = display_namespace()["_early_reply_compatible"]
        self.assertTrue(compatible("明天我有什么安排", "明天我有什么安排？"))
        self.assertTrue(compatible("明天我有什么安徘", "明天我有什么安排"))
        self.assertTrue(compatible("明天我有什么安排", "明天我有什么安排呀"))

class LlmTimingTests(unittest.TestCase):
    def test_instrumentation_keeps_cache_text_and_generation_options(self):
        from voicemem import local_llm
        llm = local_llm.LocalLLM.__new__(local_llm.LocalLLM)
        llm._cache, llm._cached_prefix = "cache", [1, 2]
        llm._base, llm._base_ids = None, []
        llm.load = lambda: (object(), types.SimpleNamespace(encode=lambda _: [1, 2, 3, 4]))
        llm._render = lambda _: "prompt"
        llm._msgs = lambda *a: []
        llm._fork = lambda c: "forked-" + c
        options = []
        def generate(model, tok, prompt, **kw):
            options.append((prompt, kw))
            cb = kw.get("prompt_progress_callback")
            if cb:
                cb(0, 2); cb(1, 2); cb(2, 2)
            for text in ["", "你", "好"]:
                yield types.SimpleNamespace(text=text)
        fake_core = types.ModuleType("mlx.core")
        fake_core.clear_cache = lambda: None
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core
        fake_lm = types.ModuleType("mlx_lm")
        fake_lm.stream_generate = generate
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core,
                                      "mlx_lm": fake_lm}):
            for debug in [True, False]:
                output = io.StringIO()
                with patch.object(local_llm, "DEBUG", debug), contextlib.redirect_stdout(output):
                    self.assertEqual(list(llm._gen_iter("hi", "", [])), ["", "你", "好"])
                self.assertEqual(options[-1][0], [3, 4])
                self.assertEqual(options[-1][1]["prompt_cache"], "forked-cache")
                self.assertEqual(options[-1][1]["max_tokens"], 512)
                self.assertEqual("[llm-detail]" in output.getvalue(), debug)
                if debug:
                    self.assertIn("前置空文本token 1", output.getvalue())

    def test_delivery_unwraps_timestamp_and_cancels_job(self):
        from voicemem import local_llm
        llm = local_llm.LocalLLM.__new__(local_llm.LocalLLM)
        def tokens(*a):
            yield ""
            yield "你好"
        llm._gen_iter = tokens
        job = types.SimpleNamespace(out=queue.Queue(), cancelled=False)
        job.cancel = lambda: setattr(job, "cancelled", True)
        def submit(make):
            for item in make():
                job.out.put(("ok", item))
            job.out.put(None)
            return job
        async def run():
            return [text async for text in llm("hi")]
        with patch.object(local_llm, "gpu_loop", lambda: types.SimpleNamespace(iter=submit)):
            self.assertEqual(asyncio.run(run()), ["你好"])
        self.assertTrue(job.cancelled)


if __name__ == "__main__":
    unittest.main()
