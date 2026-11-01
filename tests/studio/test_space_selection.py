"""Selection publication and native preparation use isolated synthetic Spaces."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from studio.core.utils.spaces.component import Spaces, capture_space


class SpaceSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch('studio.core.utils.spaces.component._ROOT', Path(directory)))
        self.opened = self.enterContext(patch(
            'studio.core.utils.spaces.component.open_memory',
            side_effect=lambda cfg: SimpleNamespace(name=cfg['space'])))
        self.agent = Spaces()
        self.agent.ARGS = SimpleNamespace(space='initial', lang='en', llm='deepseek')
        self.agent.CONFIG = {'reply': None}
        self.agent._SPACES = {}
        self.agent._LOCAL_LLM = None
        self.agent.use_space('initial')

    def test_preparation_never_publishes_half_a_selection(self):
        entered, release, queued = threading.Event(), threading.Event(), threading.Event()
        language = self.agent.space_language

        def paused(name):
            if name == 'first':
                entered.set()
                if not release.wait(3):
                    raise TimeoutError('synthetic preparation timed out')
            return language(name)

        def second():
            queued.set()
            return self.agent.use_space('second')

        with patch.object(self.agent, 'space_language', side_effect=paused), ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.agent.use_space, 'first')
            try:
                self.assertTrue(entered.wait(2))
                following = pool.submit(second)
                self.assertTrue(queued.wait(2))
                space, memory = capture_space(self.agent)
                self.assertEqual((space, memory.name), ('initial', 'initial'))
                self.assertFalse(following.done())
            finally:
                release.set()
            self.assertEqual(first.result(3), 'first')
            self.assertEqual(following.result(3), 'second')
        space, memory = capture_space(self.agent)
        self.assertEqual((space, memory.name, self.agent.SPACE_LANG), ('second', 'second', 'en'))

    def test_failed_switch_preserves_previous_memory_and_language(self):
        previous = capture_space(self.agent)
        self.opened.side_effect = RuntimeError('synthetic load failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic load failure'):
            self.agent.use_space('failed')
        self.assertEqual(capture_space(self.agent), previous)
        self.assertEqual(self.agent.SPACE_LANG, 'en')
        self.assertNotIn('failed', self.agent._SPACES)

    def test_concurrent_open_reuses_one_native_instance(self):
        entered, release = threading.Event(), threading.Event()
        memory = object()

        def open_once(cfg):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('synthetic open timed out')
            return memory

        self.opened.reset_mock(side_effect=True)
        self.opened.side_effect = open_once
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.agent.get_space, 'shared')
            try:
                self.assertTrue(entered.wait(2))
                second = pool.submit(self.agent.get_space, 'shared')
            finally:
                release.set()
            self.assertIs(first.result(3), memory)
            self.assertIs(second.result(3), memory)
        self.opened.assert_called_once()

    def test_creation_preserves_explicit_language_and_does_not_switch(self):
        previous = capture_space(self.agent)
        self.assertEqual(self.agent.create_space('chinese', 'zh')['language'], 'zh')
        self.assertEqual(self.agent.space_language('chinese'), 'zh')
        self.assertEqual(capture_space(self.agent), previous)
        with self.assertRaises(FileExistsError):
            self.agent.create_space('chinese', 'en')
        self.assertEqual(self.agent.space_language('chinese'), 'zh')

    def test_local_reply_owner_follows_cached_selection_not_background_opening(self):
        self.agent.ARGS.llm = 'local'
        self.agent._rt_persona = lambda language: f'synthetic policy {language}'
        self.opened.side_effect = lambda cfg: SimpleNamespace(
            name=cfg['space'], _reply_src=SimpleNamespace(system=''))
        self.agent.use_space('local-first')
        first = self.agent._LOCAL_LLM
        second = self.agent.get_space('local-second')._reply_src
        self.assertIs(self.agent._LOCAL_LLM, first)
        self.agent.use_space('local-second')
        self.assertIs(self.agent._LOCAL_LLM, second)
        self.agent.use_space('local-first')
        self.assertIs(self.agent._LOCAL_LLM, first)
        self.assertIs(self.agent.vm._reply_src, first)


if __name__ == '__main__':
    unittest.main()
