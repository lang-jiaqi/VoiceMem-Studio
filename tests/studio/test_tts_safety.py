"""Local speech overrun handling with synthetic PCM and native-provider doubles."""
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from studio.core.utils.tts.component import BreezeMLXTTS
from studio.core.utils.tts.safety import SpeechGuard, speech_budget_seconds


class BudgetTests(unittest.TestCase):
    def test_normal_and_slow_multilingual_segments_fit_without_truncation(self):
        for text, seconds in (
            ('Hey, good to hear you!', 6.0),
            ('你好，很高兴见到你。', 7.0),
            ('The API request takes 480 milliseconds.', 9.0),
            ('请记住这个号码：1234567890。', 15.0),
            ('Let us explain this carefully, one step at a time. ' * 5, 50.0),
            ('我们慢慢地把这件事情说清楚，然后再继续。' * 3, 50.0),
        ):
            with self.subTest(text=text):
                guard = SpeechGuard(text, 24000)
                guard.accept(round(seconds * 24000))

    def test_budget_scales_for_numbers_acronyms_and_mixed_speech(self):
        self.assertGreater(speech_budget_seconds('123456789012'), speech_budget_seconds('1'))
        self.assertGreater(speech_budget_seconds('API GPU CUDA TTS'),
                           speech_budget_seconds('api gpu cuda tts'))
        self.assertGreater(speech_budget_seconds('我们讨论 API 和 streaming audio，编号 2026。'),
                           speech_budget_seconds('我们讨论。'))

    def test_exact_ceiling_is_allowed_but_excess_audio_fails(self):
        guard = SpeechGuard('Synthetic speech.', 24000)
        guard.accept(guard.limit_samples)
        with self.assertRaisesRegex(RuntimeError, 'speech budget'):
            guard.accept(1)


class MLXSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tts = BreezeMLXTTS(model='synthetic-model')
        self.cache = SimpleNamespace(install=Mock(), close=Mock())
        self.decoder = SimpleNamespace(reset_streaming_state=Mock())
        self.model = SimpleNamespace(sample_rate=24000, generate=Mock(),
                                     audio_tokenizer=SimpleNamespace(decoder=self.decoder))
        self.tts._load = lambda: self.model
        self.tts._reference_cache = self.cache
        self.tts._coalesce = 1
        self.tts.interval = .16
        self.closed = []
        self.addCleanup(patch.stopall)
        patch.dict('sys.modules', {'mlx': ModuleType('mlx'),
                                  'mlx.core': SimpleNamespace(float32=np.float32)}).start()

    def chunks(self, count):
        try:
            for _ in range(count):
                yield SimpleNamespace(audio=np.zeros(3840, dtype=np.float32),
                                      is_streaming_chunk=True, token_count=2, sample_rate=24000)
        finally:
            self.closed.append(True)

    def test_mlx_passes_repetition_control_and_preserves_normal_pcm(self):
        self.model.generate.side_effect = lambda **_: self.chunks(4)
        pcm = list(self.tts._segments('Hello there.', None))
        self.assertEqual([len(chunk) for chunk in pcm], [7680] * 4)
        self.assertEqual(self.model.generate.call_args.kwargs['repetition_penalty'], 1.1)
        self.assertEqual(self.model.generate.call_args.kwargs['seed'], 42)
        self.assertEqual(len(self.closed), 1)
        self.decoder.reset_streaming_state.assert_called_once_with()
        self.cache.close.assert_called_once_with()

    def test_runaway_generation_fails_closes_native_state_and_next_segment_recovers(self):
        self.model.generate.side_effect = lambda **_: self.chunks(400)
        produced = []
        with self.assertRaisesRegex(RuntimeError, 'speech budget'):
            for chunk in self.tts._segments('Hey, good to hear you!', None):
                produced.append(chunk)
        self.assertLessEqual(sum(map(len, produced)) / 48000,
                             speech_budget_seconds('Hey, good to hear you!'))
        self.assertEqual(len(self.closed), 1)
        self.decoder.reset_streaming_state.assert_called_once_with()
        self.cache.close.assert_called_once_with()
        self.model.generate.side_effect = lambda **_: self.chunks(4)
        self.assertEqual(len(list(self.tts._segments('Hello again.', None))), 4)
        self.assertEqual(len(self.closed), 2)
        self.assertEqual(self.cache.close.call_count, 2)

    def test_early_close_retains_native_cleanup(self):
        self.model.generate.side_effect = lambda **_: self.chunks(400)
        stream = self.tts._segments('Synthetic speech.', None)
        next(stream)
        stream.close()
        self.assertEqual(len(self.closed), 1)
        self.decoder.reset_streaming_state.assert_called_once_with()
        self.cache.close.assert_called_once_with()

    def test_repetition_changes_invalidate_synthesized_voice_identity(self):
        changed = BreezeMLXTTS(model='synthetic-model', repetition_penalty=1.0)
        self.assertNotEqual(self.tts.cache_voice_id, changed.cache_voice_id)
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BreezeMLXTTS(model='synthetic-model', repetition_penalty=value)


class CUDASafetyTests(unittest.TestCase):
    def test_cuda_overrun_closes_iterator_and_normal_followup_still_works(self):
        from studio.core.utils.tts.cuda import BreezeCUDATTS
        provider = BreezeCUDATTS.__new__(BreezeCUDATTS)
        provider.__dict__.update(
            device='cuda:0', ref_audio=None, ref_text=None, instruction='Synthetic style',
            cfg_scale=1.0, seed=42, first_frames=2, chunk_frames=2,
            _reference_cache=None, _tokenizer=object(), _model=object(), _codec=object())
        closed = []

        def chunks(count):
            try:
                for _ in range(count):
                    yield SimpleNamespace(audio=np.zeros(3840, dtype=np.float32),
                                          codec_frames=2, sample_rate=24000)
            finally:
                closed.append(True)

        runtime = SimpleNamespace(iter_audio_chunks=Mock(side_effect=lambda *a, **kw: chunks(400)))
        provider._runtime = runtime
        torch = SimpleNamespace(inference_mode=nullcontext,
                                cuda=SimpleNamespace(device=lambda _: nullcontext(), synchronize=Mock()))
        modules = {'torch': torch, 'breeze_infer': ModuleType('breeze_infer'),
                   'breeze_infer.runtime': SimpleNamespace(set_all_seeds=Mock()),
                   'breeze_infer.templates': SimpleNamespace(
                       get_template=lambda _: None, select_template_name=lambda _: None,
                       prepare_inputs=lambda *a, **kw: object())}
        with patch.dict('sys.modules', modules):
            with self.assertRaisesRegex(RuntimeError, 'speech budget'):
                list(provider._segments('Synthetic speech.', None, threading.Event()))
            self.assertEqual(len(closed), 1)
            torch.cuda.synchronize.assert_called_once_with('cuda:0')
            runtime.iter_audio_chunks.side_effect = lambda *a, **kw: chunks(4)
            self.assertEqual(len(list(provider._segments('Hello again.', None, threading.Event()))), 4)
            self.assertEqual(len(closed), 2)


if __name__ == '__main__':
    unittest.main()
