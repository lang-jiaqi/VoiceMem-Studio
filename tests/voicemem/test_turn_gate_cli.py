"""Offline CLI regressions; no models or real Memory Spaces are read."""
import contextlib
import io
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from evals import turn_gate


class TurnGateCliTests(unittest.TestCase):
    def run_main(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(turn_gate, '_route', return_value=turn_gate.DEEP), \
                patch.object(turn_gate.gate, 'semantic_margin', return_value=0):
            turn_gate.main(arguments)

    def test_default_evaluation_never_reads_a_memory_space(self):
        with patch.object(turn_gate, 'holdout') as holdout:
            self.run_main([])
            holdout.assert_not_called()

    def test_space_option_works_with_or_without_positional_prefix(self):
        for arguments in (['15', '--space', 'fixture'], ['--space', 'fixture'],
                          ['--space', 'fixture', '15']):
            with self.subTest(arguments=arguments), patch.object(turn_gate, 'holdout') as holdout:
                self.run_main(arguments)
                holdout.assert_called_once_with('fixture')

    def test_invalid_arguments_are_reported_before_evaluation(self):
        for arguments in (['--space'], ['--space', '../private'], ['0'], ['bad']):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(turn_gate, '_route') as route:
                with self.assertRaises(SystemExit) as stopped:
                    turn_gate.main(arguments)
                self.assertEqual(stopped.exception.code, 2)
                route.assert_not_called()

    def test_holdout_with_no_questions_is_read_only_and_does_not_divide_by_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / 'voicemem_memoryspace/fixture/fixture.sqlite'
            db_path.parent.mkdir(parents=True)
            with sqlite3.connect(db_path) as db:
                db.execute('CREATE TABLE rb_evidence (quote TEXT)')
                db.execute('INSERT INTO rb_evidence VALUES (?)', ('I enjoy taking walks.',))
            before = db_path.read_bytes()
            with patch.object(turn_gate, '__file__', str(root / 'evals/turn_gate.py')), \
                    patch.object(turn_gate, '_route', return_value=turn_gate.DEEP), \
                    contextlib.redirect_stdout(io.StringIO()):
                turn_gate.holdout('fixture')
            self.assertEqual(db_path.read_bytes(), before)
            self.assertEqual({p.name for p in db_path.parent.iterdir()}, {'fixture.sqlite'})

    def test_empty_space_without_evidence_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / 'voicemem_memoryspace/fixture/fixture.sqlite'
            db_path.parent.mkdir(parents=True)
            sqlite3.connect(db_path).close()
            with patch.object(turn_gate, '__file__', str(root / 'evals/turn_gate.py')), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                turn_gate.holdout('fixture')
            self.assertIn('没有可评测', output.getvalue())


if __name__ == '__main__':
    unittest.main()
