"""Streaming and acquisition contracts using synthetic audio and artifacts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from studio.core.utils.asr.artifacts import XASRModel
from studio.core.utils.asr.component import XASRStreamingASR
from studio.core.utils.asr.profile import (
    XASR, XASR_FILES, XASR_REPOSITORY, XASR_REVISION, XASR_SOURCE, local_backend)


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.native = SimpleNamespace(accept_waveform=Mock(), input_finished=Mock())
        self.recognizer = SimpleNamespace(
            create_stream=Mock(return_value=self.native), is_ready=lambda _: False,
            get_result=lambda _: '我 用 API ， 测 试 Hello world !')
        self.asr = XASRStreamingASR.from_recognizer(self.recognizer)

    def test_tail_is_padded_once_without_closing_and_resume_restores_context(self):
        self.assertEqual(self.asr.flush(), '我用 API，测试 Hello world!')
        self.assertEqual(len(self.native.accept_waveform.call_args.args[1]), 12800)
        self.asr.flush()
        self.assertEqual(self.native.accept_waveform.call_count, 1)
        self.native.input_finished.assert_not_called()
        self.asr.feed(np.ones(640, dtype=np.float32))
        self.asr.flush()
        self.assertEqual(self.native.accept_waveform.call_count, 3)
        self.assertEqual(len(self.native.accept_waveform.call_args.args[1]), 12800)

    def test_only_exact_zero_context_counts_and_reset_is_connection_owned(self):
        self.asr.feed(np.zeros(8000, dtype=np.float32))
        self.asr.flush()
        self.assertEqual(len(self.native.accept_waveform.call_args.args[1]), 4800)
        other = self.asr.new_stream()
        self.assertEqual(other._zero_tail, 0)
        self.assertEqual(self.asr._zero_tail, 12800)
        self.asr.feed(np.full(640, 1e-8, dtype=np.float32))
        self.asr.flush()
        self.assertEqual(len(self.native.accept_waveform.call_args.args[1]), 12800)
        self.asr.reset()
        self.assertFalse(self.asr._flushed)
        self.assertEqual(self.asr._zero_tail, 0)

    def test_existing_silence_is_not_padded_again(self):
        self.asr.feed(np.zeros(16000, dtype=np.float32))
        self.asr.flush()
        self.assertEqual(self.native.accept_waveform.call_count, 1)

    def test_profile_uses_one_cpu_thread_and_keeps_natural_casing(self):
        with patch('sherpa_onnx.OnlineRecognizer.from_transducer', return_value=self.recognizer) as make, \
                patch.object(XASRStreamingASR, '_pick', return_value='synthetic.onnx'):
            asr = XASRStreamingASR('synthetic')
        self.assertEqual(make.call_args.kwargs['num_threads'], 1)
        self.assertEqual(make.call_args.kwargs['model_type'], 'zipformer2')
        self.assertEqual(asr._format_text('API AND NASA'), 'API AND NASA')

    def test_zipformer_fallback_matches_manifest_and_runtime(self):
        from studio.core.utils.asr import initialize
        from studio.core.utils.asr.profile import ZIPFORMER
        from studio.core.utils.models.initialize import models
        args = SimpleNamespace(mode='realtime', backend='mlx', eot=False)
        self.addCleanup(initialize._streaming_model.cache_clear)
        with patch.dict('os.environ', {'VOICEMEM_ASR': 'sherpa'}, clear=True), \
                patch.object(initialize, 'StreamingASR') as make:
            initialize._streaming_model.cache_clear()
            model = next(m for m in models(args) if '流式 ASR' in m.name)
            self.assertEqual(model.directory, 'asr/' + ZIPFORMER)
            initialize.streaming('en')
            make.assert_called_once_with(str(initialize.MODELS / model.directory), prefer_int8=True)
        for value in ('', 'auto', 'bilingual', 'xasr', 'en', 'zh'):
            with self.subTest(value=value), patch.dict('os.environ', {'VOICEMEM_ASR': value}):
                self.assertEqual(local_backend(), 'xasr')


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = XASRModel('Synthetic ASR', 'asr/' + XASR, XASR_REPOSITORY, XASR_FILES)
        self.download = Mock(side_effect=self._download)
        self.convert = Mock(side_effect=self._convert)
        self.addCleanup(patch.stopall)
        patch('huggingface_hub.hf_hub_download', self.download).start()
        patch('studio.core.utils.asr.artifacts.subprocess.run', self.convert).start()

    def _download(self, *, filename, local_dir, **kwargs):
        self.assertEqual(kwargs['revision'], XASR_REVISION)
        self.assertEqual(kwargs['repo_id'], XASR_REPOSITORY)
        self.assertTrue(filename.startswith(XASR_SOURCE + '/'))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * 2048)
        return str(path)

    def _convert(self, argv, *, cwd, check):
        self.assertTrue(check)
        self.assertEqual(Path(argv[-1]).resolve(), Path(cwd).resolve())
        self.assertTrue((Path(argv[-2]) / 'encoder-480ms.onnx').is_file())
        for name in XASR_FILES[:3]:
            (Path(cwd) / name).write_bytes(b'q' * 2048)

    def acquire(self):
        return self.model.acquire(self.root, self.root / 'legacy')

    def test_pinned_build_is_cached_and_size_or_identity_damage_is_not_ready(self):
        target = self.acquire()
        self.assertTrue(self.model.ready(self.root))
        self.assertEqual(self.acquire(), target)
        self.assertEqual(self.convert.call_count, 1)
        self.assertEqual(self.download.call_count, 4)
        (target / XASR_FILES[0]).write_bytes(b'q' * 2050)
        self.assertFalse(self.model.ready(self.root))
        self.acquire()
        receipt = json.loads((target / 'prepared.json').read_text())
        receipt['revision'] = 'stale'
        (target / 'prepared.json').write_text(json.dumps(receipt))
        self.assertFalse(self.model.ready(self.root))
        (target / 'prepared.json').write_text('[]')
        self.assertFalse(self.model.ready(self.root))

    def test_failed_or_cancelled_conversion_never_publishes_and_can_retry(self):
        for error in (RuntimeError('synthetic conversion failure'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.convert.side_effect = error
                with self.assertRaises((RuntimeError, KeyboardInterrupt)):
                    self.acquire()
                self.assertFalse(self.model.ready(self.root))
                self.assertEqual(list((self.root / 'asr').glob('.xasr-build-*')), [])
        self.convert.side_effect = self._convert
        self.acquire()
        self.assertTrue(self.model.ready(self.root))

    def test_download_failure_and_partial_converter_output_are_visible(self):
        self.download.side_effect = OSError('synthetic download failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic download failure'):
            self.acquire()
        self.download.side_effect = self._download
        self.convert.side_effect = lambda *a, **kw: None
        with self.assertRaises(RuntimeError):
            self.acquire()
        self.assertFalse(self.model.ready(self.root))
        self.assertFalse((self.root / self.model.directory / 'prepared.json').exists())

    def test_parallel_acquisition_publishes_once(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            targets = list(pool.map(lambda _: self.acquire(), range(3)))
        self.assertEqual(len(set(targets)), 1)
        self.assertEqual(self.convert.call_count, 1)
        self.assertTrue(self.model.ready(self.root))

    def test_completed_legacy_profile_is_reused_without_download(self):
        legacy = self.root / 'legacy'
        self.model.acquire(legacy, self.root / 'absent')
        self.download.reset_mock()
        self.convert.reset_mock()
        target = self.model.acquire(self.root, legacy)
        self.assertTrue(target.is_symlink())
        self.assertTrue(self.model.ready(self.root))
        self.download.assert_not_called()
        self.convert.assert_not_called()

    def test_interrupted_publication_has_no_ready_receipt_and_can_retry(self):
        replace = Path.replace

        def fail_one(path, target):
            if path.name == XASR_FILES[1]:
                raise OSError('synthetic publish failure')
            return replace(path, target)

        with patch.object(Path, 'replace', fail_one):
            with self.assertRaisesRegex(RuntimeError, 'synthetic publish failure'):
                self.acquire()
        self.assertFalse(self.model.ready(self.root))
        self.assertFalse((self.root / self.model.directory / 'prepared.json').exists())
        self.acquire()
        self.assertTrue(self.model.ready(self.root))


if __name__ == '__main__':
    unittest.main()
