"""Bounded session context preserves the end of detailed requests and replies."""
import unittest
import types
from unittest.mock import patch
from studio.core.utils.session_context.component import SessionBuffer


class SessionHistoryTests(unittest.TestCase):
    def test_interrupted_history_keeps_heard_content_and_private_state_separate(self):
        for question, heard in (('请解释这个问题。', '我先说第一点。'),
                                ('Please explain this.', 'Let me start with the first point.')):
            with self.subTest(question=question):
                buffer = SessionBuffer()
                turn = buffer.add('session', 'space', question, heard, interrupted=True)
                expected = [{'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': heard}]
                self.assertTrue(buffer.turns('session', 'space')[0].interrupted)
                self.assertEqual(buffer.messages('session', 'space'), expected)
                buffer.mark_complete(turn, True)
                self.assertEqual(buffer.messages('session', 'space'), [])
                self.assertEqual(buffer.messages('session', 'space', window=6), expected)
                self.assertTrue(buffer.recent('session', 'space', 6)[0].interrupted)

    def test_literal_interruption_text_is_preserved_without_an_added_annotation(self):
        buffer = SessionBuffer()
        question = '这里的“（被用户打断）”是什么意思？'
        heard = '“（被用户打断）”是你引用的文字。'
        buffer.add('session', 'space', question, heard, interrupted=True)
        self.assertEqual(buffer.messages('session', 'space', window=6), [
            {'role': 'user', 'content': question}, {'role': 'assistant', 'content': heard}])

    def test_studio_initialization_uses_the_expanded_context_budget(self):
        from studio.core.utils.cli.component import parse_args
        from studio.core.utils.runtime.component import configure
        agent = types.SimpleNamespace(use_space=lambda _: None)
        args = parse_args(['--backend', 'mlx', '--llm', 'deepseek', '--memory-llm', 'deepseek'])
        with patch('studio.core.utils.llm.initialize.credential', return_value=''), \
                patch('studio.core.utils.llm.initialize.configuration',
                      side_effect=lambda provider, role: {'provider': provider, 'config': {}}), \
                patch('studio.core.utils.turn_taking.backchannel.ENABLED'), \
                patch('studio.core.utils.turn_taking.backchannel.DEBUG'):
            configure(agent, args)
        text = 'synthetic detail ' * 40 + 'FINAL CONDITION'
        agent._SESSION_CONTEXT.add('session', 'space', text, 'synthetic reply')
        self.assertEqual(agent._SESSION_CONTEXT.messages('session', 'space', window=6)[0]['content'], text)

    def test_full_recent_turn_retains_final_condition_and_answer_conclusion(self):
        buffer = SessionBuffer()
        question = 'synthetic detail ' * 40 + 'FINAL CONDITION'
        answer = 'synthetic explanation ' * 40 + 'FINAL CONCLUSION'
        turn = buffer.add('session', 'space', question, answer)
        buffer.mark_complete(turn, True)
        self.assertEqual(buffer.messages('session', 'space', window=6), [
            {'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])

    def test_oversized_text_keeps_head_and_tail_and_uncommitted_state_is_bounded(self):
        buffer = SessionBuffer(text_limit=50, context_limit=120)
        for _ in range(30):
            buffer.add('session', 'space', 'HEAD' + 'x' * 300 + 'TAIL', 'ANSWER' * 20)
        turns = buffer.turns('session', 'space')
        self.assertLessEqual(sum(len(t.user_text) + len(t.assistant_text) for t in turns), 120)
        self.assertLessEqual(len(buffer._turn_context), len(turns))
        self.assertTrue(turns[-1].user_text.startswith('HEAD'))
        self.assertTrue(turns[-1].user_text.endswith('TAIL'))
        self.assertIn('…', turns[-1].user_text)
        buffer.clear_session('session')
        self.assertEqual(buffer._recent, {})
        self.assertEqual(buffer._turn_context, {})


if __name__ == '__main__':
    unittest.main()
