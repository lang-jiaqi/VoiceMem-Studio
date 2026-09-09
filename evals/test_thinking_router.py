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

    def test_three_way_route_and_invalid_output_fallback(self):
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

    def test_router_owned_memory_rules_skip_model_inference(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        router._predict = lambda *_: self.fail("the memory rule should skip inference")
        self.assertEqual(router.classify("我上次说最喜欢什么", True).level, MEDIUM)
        self.assertEqual(router.classify("我明天有什么安排", True).level, MEDIUM)
        self.assertEqual(router.classify("请介绍一下你自己", True).level, FAST)

    def test_prefetch_hint_does_not_override_router_output(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        router._predict = lambda *_: "即时"
        decision = router.classify("介绍一下向量数据库", True)
        self.assertEqual(decision.level, FAST)

    def test_short_non_request_fragment_is_instant(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        router._predict = lambda *_: self.fail("a short fragment should skip inference")
        for text in ("数学", "x平方", "的积分", "sin x"):
            with self.subTest(text=text):
                self.assertEqual(router.classify(text, False).level, FAST)

    def test_integral_has_a_deterministic_slow_floor(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        router._predict = lambda *_: self.fail("the policy floor should skip inference")
        decision = router.classify("计算 x 平方的不定积分")
        self.assertEqual(decision.level, SLOW)
        self.assertEqual(decision.reply_mode, "memory_cot")

    def test_obvious_conversation_has_a_deterministic_instant_floor(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        router._predict = lambda *_: self.fail("the policy floor should skip inference")
        for text in ("你好", "请介绍一下你自己", "讲一个流浪猫的故事"):
            with self.subTest(text=text):
                self.assertEqual(router.classify(text).level, FAST)

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
        self.assertIn("当前用户：介绍一下向量数据库", seen["prompt"])
        self.assertIn("记忆预取提示：否", seen["prompt"])
        self.assertTrue(all(label in {"即时", "记忆", "深思"}
                            for _, label in seen["examples"]))

    def test_warmup_cannot_be_satisfied_by_a_policy_floor(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        calls = []
        router._predict = lambda *_: calls.append(True) or "即时"
        self.assertEqual(router.warmup().level, FAST)
        self.assertEqual(calls, [True])

    def test_recent_history_resolves_followups_and_is_part_of_cache_key(self):
        router = QwenThinkingRouter.__new__(QwenThinkingRouter)
        router._cache = {}
        prompts = []

        def predict(_system, _examples, prompt):
            prompts.append(prompt)
            return "记忆" if "周末安排" in prompt else "即时"

        router._predict = predict
        first = router.classify("为什么", False, [
            {"role": "user", "content": "天空为什么是蓝色的？"},
            {"role": "assistant", "content": "因为大气散射。"},
        ])
        second = router.classify("为什么", False, [
            {"role": "user", "content": "我上次的周末安排是什么？"},
            {"role": "assistant", "content": "还没有确认。"},
        ])
        self.assertEqual(first.level, FAST)
        self.assertEqual(second.level, MEDIUM)
        self.assertEqual(len(prompts), 2)
        self.assertIn("天空为什么是蓝色的", prompts[0])
        self.assertIn("周末安排", prompts[1])


if __name__ == "__main__":
    unittest.main()
