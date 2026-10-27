"""Observed audio provenance and group-split checks without models or recordings."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from finetune.decision_router.audio_data import observe, perturb, recognize
from finetune.decision_router.assemble import assemble, split_for
from finetune.decision_router.train import load_rows


class ObservedAudioTests(unittest.TestCase):
    def test_perturbation_is_reproducible_and_does_not_mutate_source_audio(self):
        pcm = np.full(16000, .1, dtype=np.float32)
        original = pcm.copy()
        first, second = perturb(pcm, 7), perturb(pcm, 7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(pcm, original)
        self.assertFalse(np.shares_memory(first, pcm))
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.array_equal(first, pcm))

    def test_acoustic_state_uses_actual_vad_and_eot_observations(self):
        pcm = np.r_[np.full(1024, .1), np.zeros(512)].astype(np.float32)
        vad = SimpleNamespace(is_speech=lambda frame: bool(np.max(frame) > .05))
        eot = SimpleNamespace(score=Mock(return_value=.8))
        state = observe(pcm, vad, eot)
        self.assertAlmostEqual(state['span_s'], 1024/16000)
        self.assertAlmostEqual(state['silence_s'], 512/16000)
        self.assertEqual(state['eot_probability'], .8)
        np.testing.assert_array_equal(eot.score.call_args.args[0], pcm)

    def test_recognition_preserves_distinct_observed_outputs_and_closes_stream(self):
        stream = SimpleNamespace(feed=Mock(return_value='streaming partial'),
                                 flush=Mock(return_value='streaming final'), close=Mock())
        offline = SimpleNamespace(transcribe=Mock(return_value='offline correction'))
        pcm = np.zeros(20000, dtype=np.float32)
        values = recognize(pcm, offline, lambda: stream)
        self.assertEqual(values, {'streaming': 'streaming final', 'final': 'offline correction'})
        stream.close.assert_called_once()
        self.assertEqual(stream.feed.call_count, 3)
        offline.transcribe.assert_called_once_with(pcm)

    def test_flush_failure_still_closes_decoder_and_empty_text_stays_empty(self):
        stream = SimpleNamespace(feed=lambda _: '', flush=Mock(side_effect=RuntimeError('fixture')), close=Mock())
        with self.assertRaises(RuntimeError):
            recognize(np.zeros(10), None, lambda: stream)
        stream.close.assert_called_once()
        stream.flush = lambda: ''
        values = recognize(np.zeros(10), SimpleNamespace(transcribe=lambda _: ''), lambda: stream)
        self.assertEqual(values, {'streaming': '', 'final': ''})


class GroupAssemblyTests(unittest.TestCase):
    def test_evaluation_only_file_keeps_schema_checks_without_requiring_training_rows(self):
        rows = [{'id': f'eval-{index}', 'group': f'eval-group-{index}', 'split': 'test',
                 'text': f'synthetic evaluation {index}', 'needs_deep': bool(index),
                 'source': 'synthetic-test'} for index in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evaluation.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'train must contain'):
                load_rows(path)
            actual = load_rows(path, require_splits=False)
        self.assertEqual(len(actual), 2)

    def rows(self):
        groups = {}
        for index in range(1000):
            name = f'group-{index}'
            groups.setdefault(split_for(name), name)
            if len(groups) == 4:
                break
        return [{'id': f'{group}-{int(label)}', 'group': group, 'split': 'train',
                 'text': f'fixture request {group} {label}', 'needs_deep': label,
                 'source': 'synthetic-test', 'audio_state': {'eot_probability': .8}}
                for group in groups.values() for label in (False, True)]

    def test_validation_and_calibration_are_separate_and_weak_labels_stay_training_only(self):
        rows = self.rows()
        rows.append({'id': 'weak-case', 'group': 'weak-group', 'split': 'test',
                     'text': 'weakly labelled fixture', 'needs_deep': False,
                     'source': 'public-fixture', 'label_provenance': 'local-qwen-and-laya-agreement'})
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory)/'source.jsonl', Path(directory)/'assembled.jsonl'
            src.write_text('\n'.join(json.dumps(row) for row in rows))
            report = assemble([src], dst)
            actual = load_rows(dst)
        self.assertEqual(set(report['splits']), {'train', 'validation', 'calibration', 'test'})
        self.assertEqual(next(row for row in actual if row['id'] == 'weak-case')['split'], 'train')
        groups = {}
        for row in actual:
            groups.setdefault(row['group'], set()).add(row['split'])
        self.assertTrue(all(len(values) == 1 for values in groups.values()))

    def test_conflicting_resume_identity_is_rejected(self):
        rows = self.rows()
        rows.append({**rows[0], 'text': 'different recognized input'})
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory)/'source.jsonl', Path(directory)/'assembled.jsonl'
            src.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'Conflicting repeated'):
                assemble([src], dst)


if __name__ == '__main__':
    unittest.main()
