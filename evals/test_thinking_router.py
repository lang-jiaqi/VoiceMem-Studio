from __future__ import annotations

import unittest

from harness.reply_modes import (
    FAST,
    MEDIUM,
    SLOW,
    QwenThinkingRouter,
    ThinkingDecision,
    parse_level,
)


class ThinkingRouterTests(unittest.TestCase):
    def test_parses_only_supported_labels(self):
        self.assertEqual(parse_level("fast").level, FAST)
        self.assertEqual(parse_level("The label is MEDIUM.").level, MEDIUM)
        self.assertEqual(parse_level("slow\n").level, SLOW)
        self.assertEqual(parse_level("即时").level, FAST)
        self.assertEqual(parse_level("记忆").level, MEDIUM)
        self.assertEqual(parse_level("深思").level, SLOW)

    def test_invalid_output_uses_explicit_fallback(self):
        self.assertEqual(parse_level("unknown", MEDIUM).level, MEDIUM)

    def test_levels_map_to_deepseek_effort(self):
        self.assertEqual(ThinkingDecision(FAST).reasoning_effort, "none")
        self.assertEqual(ThinkingDecision(MEDIUM).reasoning_effort, "none")
        self.assertEqual(ThinkingDecision(SLOW).reasoning_effort, "high")

    def test_levels_map_to_reply_modes(self):
        self.assertEqual(ThinkingDecision(FAST).reply_mode, "direct")
        self.assertEqual(ThinkingDecision(MEDIUM).reply_mode, "memory")
        self.assertEqual(ThinkingDecision(SLOW).reply_mode, "memory_cot")

    def test_three_way_route_uses_memory_gate_only_as_invalid_output_fallback(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}

        def classify(text, answer, memory=False):
            router._predict = lambda *_: answer
            return router.classify(text, memory)

        self.assertEqual(classify("你好", "即时").level, FAST)
        self.assertEqual(classify("解释一下向量数据库", "即时").level, FAST)
        self.assertEqual(classify("我以前喜欢什么", "记忆").level, MEDIUM)
        self.assertEqual(classify("求这个函数的积分", "深思").level, SLOW)
        self.assertEqual(classify("我以前喜欢什么", "?", memory=True).level, MEDIUM)

    def test_integral_has_a_deterministic_slow_floor(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        router._predict = lambda *_: self.fail("the policy floor should skip inference")
        decision = router.classify("计算 x 平方的不定积分")
        self.assertEqual(decision.level, SLOW)
        self.assertEqual(decision.reply_mode, "memory_cot")

    def test_chinese_input_uses_a_chinese_policy_and_examples(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        seen = {}

        def predict(system, examples, prompt):
            seen.update(system=system, examples=examples, prompt=prompt)
            return "即时"

        router._predict = predict
        self.assertEqual(router.classify("介绍一下向量数据库").level, FAST)
        self.assertIn("中文语音助手", seen["system"])
        self.assertTrue(all(label in {"即时", "记忆", "深思"}
                            for _, label in seen["examples"]))


if __name__ == "__main__":
    unittest.main()
