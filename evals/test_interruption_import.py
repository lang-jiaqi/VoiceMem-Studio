"""Exercise production echo detection without the old web import aliases."""
import sys
import unittest
from unittest.mock import patch

from studio.core.utils.interruption.component import Interruption


class InterruptionImportTests(unittest.TestCase):
    def test_echo_detection_works_without_top_level_echo_guard(self):
        guard = Interruption()
        guard.ECHO_WINDOW = 300
        guard.ECHO_RATIO = 0.6
        guard.ECHO_FUZZY_MIN = 4
        with patch.dict(sys.modules, {'echo_guard': None}):
            self.assertTrue(guard._is_echo('在呢，刚刚在整理', '在呢刚在整理'))
            self.assertFalse(guard._is_echo('项目其实上周就交了', '是不是最近那个项目压得慌'))


if __name__ == '__main__':
    unittest.main()
