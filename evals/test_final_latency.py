"""CPU-only regressions for final ASR and per-generation BPE state."""
import ast
import asyncio
import importlib.util
from importlib.metadata import distribution
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FinalAsrTests(unittest.IsolatedAsyncioTestCase):
    def stream(self, model):
        from voicemem.stream import VoiceStream
        stream = VoiceStream(types.SimpleNamespace())
        stream._final_asr = model
        stream._text = "partial"
        return stream

    async def test_refine_does_not_block_loop_and_preserves_raw_text(self):
        started, release = threading.Event(), threading.Event()
        threads = []
        def transcribe(pcm):
            threads.append(threading.get_ident())
            started.set()
            release.wait(2)
            return "final"
        stream = self.stream(types.SimpleNamespace(transcribe=transcribe))
        task = asyncio.create_task(stream._refine_async(np.ones(1600)))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            self.assertFalse(task.done())
            self.assertEqual(stream._text, "partial")
        finally:
            release.set()
        await task
        self.assertEqual(stream._text, "final")
        self.assertEqual(stream._raw_text, "partial")
        self.assertNotEqual(threads[0], threading.get_ident())

    async def test_cancelled_result_cannot_overwrite_next_turn(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        def transcribe(pcm):
            started.set()
            release.wait(2)
            finished.set()
            return "old result"
        stream = self.stream(types.SimpleNamespace(transcribe=transcribe))
        task = asyncio.create_task(stream._refine_async(np.ones(10)))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            stream._text = "new turn"
        finally:
            release.set()
        await asyncio.to_thread(finished.wait, 1)
        await asyncio.sleep(.01)
        self.assertEqual(stream._text, "new turn")

    async def test_shared_recognizer_is_serial(self):
        active, peak = 0, 0
        def transcribe(pcm):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(.02)
            active -= 1
            return "final"
        model = types.SimpleNamespace(transcribe=transcribe)
        streams = [self.stream(model) for _ in range(3)]
        await asyncio.gather(*(s._refine_async(np.ones(10)) for s in streams))
        self.assertEqual(peak, 1)
        self.assertTrue(all(s._text == "final" for s in streams))

    async def test_empty_missing_and_failed_recognizer_preserve_partial(self):
        def fail(pcm):
            raise ValueError("test")
        for model in [None, types.SimpleNamespace(transcribe=lambda _: ""),
                      types.SimpleNamespace(transcribe=fail)]:
            stream = self.stream(model)
            await stream._refine_async(np.ones(10))
            self.assertEqual(stream._text, "partial")

    async def test_audio_turn_waits_for_refinement_and_keeps_timestamp(self):
        from voicemem.stream import VoiceStream
        stream = VoiceStream(types.SimpleNamespace(), confirm_s=.02,
                             spec_min_chars=999, gate=lambda _: "shallow")
        stream._final_asr = types.SimpleNamespace(transcribe=lambda _: "完整文本")
        async def flush():
            return "流式文本"
        stream._asr_w = types.SimpleNamespace(push=lambda _: "流式文本", flush=flush,
                                             reset=lambda: None, report=lambda: "test")
        speech = iter([True, False])
        stream._vad = types.SimpleNamespace(is_speech=lambda _: next(speech))
        pcm = np.zeros(480, np.int16).tobytes()
        first = await stream.feed(pcm)
        final = await stream.feed(pcm)
        self.assertEqual(final.state, "turn_over")
        self.assertEqual(final.turn.text, "完整文本")
        self.assertEqual(final.turn.raw_text, "流式文本")
        self.assertEqual(final.speech_end, first.speech_end)
        self.assertEqual(stream._speech_end, 0)


class WarmupTests(unittest.TestCase):
    def test_warmup_decodes_and_respects_disabled_option(self):
        tree = ast.parse((ROOT / "web/run.py").read_text())
        fn = next(n for n in tree.body if getattr(n, "name", "") == "_warm_final_asr")
        ns = dict(os=os, time=time)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "warmup", "exec"), ns)
        calls = []
        model = types.SimpleNamespace(transcribe=lambda pcm: calls.append(len(pcm)))
        vm = types.SimpleNamespace(utils=types.SimpleNamespace(get=lambda name: model))
        with patch.dict(os.environ, {"VOICEMEM_FINAL_ASR": "1"}):
            ns["_warm_final_asr"](vm)
        self.assertEqual(calls, [16000])
        with patch.dict(os.environ, {"VOICEMEM_FINAL_ASR": "0"}):
            ns["_warm_final_asr"](vm)
        self.assertEqual(calls, [16000])


class BpeCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read installed Python decoder without importing mlx_lm/__init__ (GPU).
        path = distribution("mlx-lm").locate_file("mlx_lm/tokenizer_utils.py")
        spec = importlib.util.spec_from_file_location("cpu_tokenizers", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_real_decoder_interleaving_matches_uncached(self):
        from voicemem.detokenizer_cache import cache_bpe_vocabulary
        m = self.module
        m.BPEStreamingDetokenizer.make_byte_decoder()
        encoder = {v: k for k, v in m.BPEStreamingDetokenizer._byte_decoder.items()}
        vocab = {encoder[b]: b for b in range(256)}
        tok = m.TokenizerWrapper.__new__(m.TokenizerWrapper)
        tok._tokenizer = types.SimpleNamespace(vocab=vocab, clean_up_tokenization_spaces=False)
        tok._detokenizer_class = m.BPEStreamingDetokenizer
        left, right = "你好，世界！", "Hello world."
        def decode(decoder, text):
            segments = []
            for byte in text.encode():
                decoder.add_token(byte)
                segments.append(decoder.last_segment)
            decoder.finalize()
            segments.append(decoder.last_segment)
            return segments
        expected = [decode(tok.detokenizer, text) for text in [left, right]]
        with patch.dict(sys.modules, {"mlx_lm.tokenizer_utils": m}):
            self.assertTrue(cache_bpe_vocabulary(tok))
            self.assertFalse(cache_bpe_vocabulary(tok))
        a, b = tok.detokenizer, tok.detokenizer
        self.assertIs(a.tokenmap, b.tokenmap)
        self.assertIsInstance(a.tokenmap, tuple)
        self.assertIsNot(a.tokens, b.tokens)
        # Interleave two generations, including incomplete multibyte characters.
        outputs = [[], []]
        data = [left.encode(), right.encode()]
        for i in range(max(map(len, data))):
            for k, decoder in enumerate([a, b]):
                if i < len(data[k]):
                    decoder.add_token(data[k][i])
                    outputs[k].append(decoder.last_segment)
        for k, decoder in enumerate([a, b]):
            decoder.finalize()
            outputs[k].append(decoder.last_segment)
        self.assertEqual(outputs, expected)
        self.assertEqual(tok.detokenizer.tokens, [])

    def test_custom_decoder_unchanged(self):
        from voicemem.detokenizer_cache import cache_bpe_vocabulary
        m = self.module
        tok = m.TokenizerWrapper.__new__(m.TokenizerWrapper)
        factory = lambda _: object()
        tok._detokenizer_class = factory
        with patch.dict(sys.modules, {"mlx_lm.tokenizer_utils": m}):
            self.assertFalse(cache_bpe_vocabulary(tok))
        self.assertIs(tok._detokenizer_class, factory)


if __name__ == "__main__":
    unittest.main()
