"""Check held-out intent families and category-based model qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from finetune.decision_router.dialogue_data import audit, build
from finetune.decision_router.dialogue_eval import summarize, record_predictions, validation_score
from finetune.decision_router.assemble_v2 import assemble
from finetune.decision_router.intent_data import build as build_scope
from finetune.decision_router.laya_train import typed_row


class DialogueCoverageTests(unittest.TestCase):
    def test_all_partitions_include_context_and_both_sides_of_intent_families(self):
        rows = build()
        report = audit(rows)
        self.assertGreater(report['splits']['validation']['with_history'], 0)
        families = {}
        for row in rows:
            families.setdefault(row['group'], []).append(row)
        for family, cases in families.items():
            self.assertEqual(len({row['split'] for row in cases}), 1)
            if '/intent-' in family or '/followup-' in family:
                self.assertEqual({row['needs_deep'] for row in cases}, {True, False})

    def test_family_leakage_fails_even_for_different_wording(self):
        rows = build()
        changed = copy.deepcopy(rows[0])
        changed.update(id='new-fixture', text='an independent fixture sentence', split='test')
        with self.assertRaisesRegex(ValueError, 'wording family'):
            audit(rows + [changed])

    def test_contradictory_duplicate_cannot_hide_in_another_family(self):
        rows = build()
        changed = {**rows[0], 'id': 'other-fixture', 'group': 'different-family',
                   'needs_deep': not rows[0]['needs_deep']}
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            audit(rows + [changed])

    def test_macro_selection_exposes_a_small_failed_category(self):
        rows = [{'group': str(index), 'category': 'ordinary', 'needs_deep': False}
                for index in range(20)]
        rows.append({'group': 'rare', 'category': 'explicit_deep', 'needs_deep': True,
                     'history': [{'role': 'user', 'content': 'synthetic fixture'}]})
        measured = summarize(rows, [False]*21)
        self.assertGreater(measured['accuracy'], .95)
        self.assertEqual(measured['by_category']['explicit_deep']['deep_recall'], 0)
        self.assertEqual(validation_score(measured)[0], .5)
        self.assertEqual(measured['with_history']['accuracy'], 0)

    def test_canonical_logits_and_row_order_are_retained(self):
        records = [(0, np.array([.2, .8]), None, 2), (0, np.array([.7, .3]), None, 2)]
        self.assertEqual(record_predictions(records), [True, False])
        with self.assertRaises(ValueError):
            summarize([{'needs_deep': True}], [])

    def test_legacy_holdouts_and_weak_labels_cannot_enter_retraining(self):
        selected = {}
        for row in build():
            if row.get('history'):
                selected.setdefault((row['split'], row['category']), row)
        base = {'id': 'legacy-good', 'group': 'legacy-family', 'split': 'train',
                'text': 'an unrelated legacy fixture', 'needs_deep': False,
                'source': 'synthetic-test'}
        legacy = [base, {**base, 'id': 'legacy-frozen', 'group': 'frozen-family',
                         'split': 'test', 'text': 'frozen fixture'},
                  {**base, 'id': 'legacy-weak', 'group': 'weak-family', 'text': 'weak fixture',
                   'label_provenance': 'local-qwen-and-laya-agreement'}]
        with tempfile.TemporaryDirectory() as directory:
            authored, old, out = [Path(directory)/name for name in ('authored.jsonl', 'old.jsonl', 'out.jsonl')]
            authored.write_text('\n'.join(json.dumps(row) for row in selected.values()))
            old.write_text('\n'.join(json.dumps(row) for row in legacy))
            report = assemble(authored, old, [], out)
            actual = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(report['counts']['excluded_weak'], 1)
        self.assertEqual(report['counts']['legacy_frozen_not_reused'], 1)
        legacy_ids = {row['id'] for row in actual if row['id'].startswith('legacy-')}
        self.assertEqual(legacy_ids, {'legacy-good'})

    def test_scope_revision_excludes_the_previous_test_partition(self):
        selected = {}
        for row in build():
            selected.setdefault((row['split'], row['category']), row)
        source = list(selected.values())
        old_test = {row['id'] for row in source if row['split'] == 'test'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'previous.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in source))
            actual = build_scope(path)
        self.assertFalse(old_test & {row['id'] for row in actual})
        self.assertTrue(all(row['source'] == 'authored-scope-contrast-v3'
                            for row in actual if row['group'].startswith('intent-scope-v3/')))

    def test_soft_targets_are_opt_in_and_do_not_leak_labels_into_model_state(self):
        row = {'text': 'synthetic fixture', 'needs_deep': True, 'split': 'train'}
        softened = typed_row(row, .06)
        hard = typed_row(row)
        self.assertAlmostEqual(softened['gold']['depth']['probabilities']['deep'], .97)
        self.assertAlmostEqual(softened['gold']['depth']['probabilities']['ordinary'], .03)
        self.assertEqual(hard['gold']['depth']['probabilities']['deep'], 1)
        self.assertNotIn('needs_deep', softened['state'])
        self.assertNotIn('split', softened['state'])


if __name__ == '__main__':
    unittest.main()
