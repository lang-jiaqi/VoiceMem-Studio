from __future__ import annotations

import tempfile
import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.core.utils.reply_modes.initialize import (
    FAST,
    MEDIUM,
    SLOW,
    QwenThinkingRouter,
    ThinkingDecision,
    parse_level,
)


class ThinkingRouterTests(unittest.TestCase):
    def test_harness_examples_use_the_runtime_context_format(self):
        from studio.harness.reply_modes.policy import EXAMPLES
        router = QwenThinkingRouter(model='/unused')
        labels = set()
        for example, label in EXAMPLES:
            labels.add(label)
            recent, separator, current = example.partition('\n当前用户：')
            self.assertTrue(separator)
            recent = recent.removeprefix('最近对话：\n')
            history = []
            if recent != '无':
                for line in recent.splitlines():
                    role, content = line.split(': ', 1)
                    history.append({'role': {'用户':'user', '助手':'assistant'}[role],
                                    'content': content})
            self.assertEqual(router._context_prompt(current, history, False), example)
        self.assertEqual(labels, {'是', '否'})

    def test_policy_distinguishes_recall_from_reasoning_and_scopes_history(self):
        from studio.harness.reply_modes.policy import SYSTEM, EXAMPLES
        self.assertIn('不判断是否检索记忆', SYSTEM)
        self.assertIn('不延续上一轮的思考等级', SYSTEM)
        self.assertIn('询问已经得出的结果', SYSTEM)
        self.assertIn('疑问句形式的请求', SYSTEM)
        self.assertTrue(any('历史需求' in text and label == '是' for text, label in EXAMPLES))
        self.assertTrue(any('旅行日期' in text and label == '否' for text, label in EXAMPLES))

    def test_quality_sets_are_disjoint_and_report_false_and_missed_cot(self):
        from evals.router_quality import CASES, HOLDOUT_CASES, evaluate
        self.assertFalse({row[0] for row in CASES} & {row[0] for row in HOLDOUT_CASES})
        router = QwenThinkingRouter(model='/unused')
        router._predict = lambda *_: '否'
        result = evaluate(router, (
            ('合成普通问题', (), 'memory'),
            ('合成复杂问题', (), 'memory_cot')))
        self.assertEqual((result['false_cot'], result['missed_cot']), (0, 1))
        self.assertEqual((result['ordinary_cases'], result['deep_cases']), (1, 1))

    def test_missing_default_router_downloads_to_project_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "Qwen3-0.6B"
            router = QwenThinkingRouter.__new__(QwenThinkingRouter)
            router._local_model_dir = destination
            router._download_default = True
            router.model_name = str(destination)

            def download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "Qwen/Qwen3-0.6B")
                self.assertEqual(kwargs["local_dir"], str(destination))
                self.assertTrue(kwargs["tqdm_class"])
                destination.mkdir(parents=True)
                (destination / "config.json").write_text("{}", encoding="utf-8")
                (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
                (destination / "model.safetensors").write_bytes(b"test")

            with patch("huggingface_hub.snapshot_download", side_effect=download):
                selected = router._ensure_model_source()

        self.assertEqual(selected, str(destination))
        self.assertFalse(router._download_default)

    def test_configured_router_never_triggers_default_download(self):
        router = QwenThinkingRouter(model="/models/custom-router")
        with patch("huggingface_hub.snapshot_download") as download:
            self.assertEqual(router._ensure_model_source(), "/models/custom-router")
        download.assert_not_called()

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

    def test_only_depth_is_selected_by_the_model(self):
        router = QwenThinkingRouter(model='/unused')
        for index, (label, expected) in enumerate((('否', FAST), ('是', SLOW),
                                                 ('否。', FAST), ('不是深思', FAST),
                                                 ('记忆', FAST))):
            router._predict = lambda *_, answer=label: answer
            self.assertEqual(router.classify(f'合成问题{index}').level, expected)

    def test_memory_hint_cannot_change_depth_or_trigger_another_inference(self):
        from unittest.mock import Mock
        router = QwenThinkingRouter(model='/unused')
        router._predict = Mock(return_value='否')
        self.assertEqual(router.classify('我明天有什么安排？', True).level, FAST)
        self.assertEqual(router.classify('我明天有什么安排？', False).level, FAST)
        router._predict.assert_called_once()
        self.assertNotIn('预取', router._predict.call_args.args[2])

    def test_no_semantic_keyword_shortcuts_remain(self):
        from unittest.mock import Mock
        router = QwenThinkingRouter(model='/unused')
        router._predict = Mock(return_value='否')
        for text in ('你好', '我明天有什么安排？', '计算这个函数的积分', '嗯'):
            self.assertEqual(router.classify(text).level, FAST)
        self.assertEqual(router._predict.call_count, 4)

    def test_chinese_prompt_only_asks_for_reasoning_depth(self):
        router = QwenThinkingRouter(model='/unused')
        seen = {}
        def predict(system, examples, prompt):
            seen.update(system=system, examples=examples, prompt=prompt)
            return '否'
        router._predict = predict
        router.classify('介绍一下向量数据库')
        self.assertIn('中文语音助手', seen['system'])
        self.assertIn('不判断是否检索记忆', seen['system'])
        self.assertIn('当前用户：介绍一下向量数据库', seen['prompt'])
        self.assertTrue(all(label in {'是', '否'} for _, label in seen['examples']))

    def test_long_assistant_message_does_not_erase_user_context(self):
        router = QwenThinkingRouter(model='/unused')
        prompt = router._context_prompt('继续', [
            {'role': 'user', 'content': '请推导这个函数的积分。'},
            {'role': 'assistant', 'content': '说明' * 300},
        ], False)
        self.assertIn('请推导这个函数的积分。', prompt)
        self.assertLess(len(prompt), 400)

    def test_history_is_part_of_depth_cache(self):
        router = QwenThinkingRouter(model='/unused')
        prompts = []
        def predict(_system, _examples, prompt):
            prompts.append(prompt)
            return '是' if '推导' in prompt else '否'
        router._predict = predict
        simple = [{'role': 'user', 'content': '讲个故事。'}]
        deep = [{'role': 'user', 'content': '推导这个函数的积分。'}]
        self.assertEqual(router.classify('继续', history=simple).level, FAST)
        self.assertEqual(router.classify('继续', history=deep).level, SLOW)
        self.assertEqual(router.classify('继续', history=deep).level, SLOW)
        self.assertEqual(len(prompts), 2)

    def test_warmup_still_loads_the_inference_path(self):
        from unittest.mock import Mock
        router = QwenThinkingRouter(model='/unused')
        router._predict = Mock(return_value='否')
        self.assertEqual(router.warmup().level, FAST)
        router._predict.assert_called_once()


class ConcurrentDepthTests(unittest.IsolatedAsyncioTestCase):
    async def test_identical_requests_share_inference_and_one_cancelled_waiter_is_isolated(self):
        router = QwenThinkingRouter(model='/unused')
        started, release = threading.Event(), threading.Event()
        calls = []

        def predict(*args):
            calls.append(args)
            started.set()
            release.wait(2)
            return '是'

        router._predict = predict
        first = asyncio.create_task(router.classify_async('synthetic question'))
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            second = asyncio.create_task(router.classify_async('synthetic question', True))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertFalse(second.done())
        finally:
            release.set()
        self.assertEqual((await second).level, SLOW)
        self.assertEqual(len(calls), 1)
        self.assertEqual(router._classifications, {})
        self.assertEqual((await router.classify_async('synthetic question')).level, SLOW)
        self.assertEqual(len(calls), 1)

    async def test_different_contexts_are_not_coalesced(self):
        router = QwenThinkingRouter(model='/unused')
        router._predict = lambda _s, _e, prompt: '是' if 'deep context' in prompt else '否'
        results = await asyncio.gather(
            router.classify_async('continue', history=[{'role': 'user', 'content': 'deep context'}]),
            router.classify_async('continue', history=[{'role': 'user', 'content': 'ordinary context'}]))
        self.assertEqual([item.level for item in results], [SLOW, FAST])

    async def test_shared_failure_reaches_all_waiters_and_retry_can_succeed(self):
        router = QwenThinkingRouter(model='/unused')
        started, release = threading.Event(), threading.Event()
        calls = []

        def predict(*args):
            calls.append(args)
            started.set()
            release.wait(2)
            raise RuntimeError('synthetic failure')

        router._predict = predict
        first = asyncio.create_task(router.classify_async('synthetic question'))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            second = asyncio.create_task(router.classify_async('synthetic question'))
            await asyncio.sleep(0)
        finally:
            release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(item, RuntimeError) for item in results))
        self.assertEqual(len(calls), 1)
        self.assertEqual(router._classifications, {})
        router._predict = lambda *_: '否'
        self.assertEqual((await router.classify_async('synthetic question')).level, FAST)

    async def test_last_cancelled_waiter_reaps_pending_task(self):
        router = QwenThinkingRouter(model='/unused')
        started, release = threading.Event(), threading.Event()
        router._predict = lambda *_: (started.set(), release.wait(2), '否')[-1]
        waiter = asyncio.create_task(router.classify_async('synthetic question'))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertEqual(router._classifications, {})
        finally:
            release.set()


class MemoryRoutingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, Mock
        from studio.core.utils.routing.component import Routing
        from voicemem.stream import empty_result
        self.agent = Routing()
        self.agent._THINKING_ROUTER_ON = True
        self.agent._replay_id = lambda *_: ''
        self.fresh = empty_result()
        self.agent.vm = SimpleNamespace(
            classify=Mock(return_value=SimpleNamespace(slots=[], entities=[])),
            search=Mock(return_value=self.fresh))
        self.router = SimpleNamespace(classify_async=AsyncMock(return_value=ThinkingDecision(FAST, '普通')))
        self.enterContext(patch('studio.core.utils.routing.component.thinking_router', return_value=self.router))

    def pending(self, *, route='deep', prepared=False, stranger=False):
        from types import SimpleNamespace
        from studio.core.utils.contracts.component import Pending
        from voicemem.stream import empty_result
        result = SimpleNamespace(search_mode='full') if prepared else empty_result()
        return Pending('我想接着上回那本书的话题聊。', 'existing-context' if prepared else '',
                       result, route=route, stranger=stranger, replay='existing-replay')

    async def test_normal_depth_does_not_discard_prefetched_memory(self):
        pending = self.pending(prepared=True)
        result = pending.result
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertIs(pending.result, result)
        self.assertEqual(pending.memory_context, 'existing-context')
        self.assertEqual(pending.replay, 'existing-replay')
        self.agent.vm.search.assert_not_called()

    async def test_ordinary_followups_keep_gate_approved_memory(self):
        for text in ('我后天约了谁？', '所以最后的结果是多少？', '帮我想想午饭吃什么。'):
            with self.subTest(text=text):
                pending = self.pending(prepared=True)
                pending.text = text
                result = pending.result
                await self.agent.route_pending_thinking(pending)
                self.assertEqual(pending.reply_mode, 'memory')
                self.assertIs(pending.result, result)
                self.assertEqual(pending.memory_context, 'existing-context')
        self.agent.vm.search.assert_not_called()

    async def test_existing_gate_handles_history_without_studio_keyword_rules(self):
        from voicemem import gate
        for text in ('昨天聊到的那本书叫什么？', '我想接着上回那次旅行的话题聊。'):
            with self.subTest(text=text):
                pending = self.pending(prepared=True)
                pending.text = text
                pending.route = gate.route(text, semantic=False)
                self.assertEqual(pending.route, gate.DEEP)
                await self.agent.route_pending_thinking(pending)
                self.assertEqual(pending.reply_mode, 'memory')
                self.assertEqual(pending.memory_context, 'existing-context')
        self.agent.vm.search.assert_not_called()

    async def test_gate_memory_eligibility_completes_missing_retrieval(self):
        from voicemem import gate
        pending = self.pending()
        pending.text = '我明天有什么安排？'
        await self.agent.route_pending_thinking(pending)
        self.assertEqual((pending.reply_mode, pending.route), ('memory', gate.DEEP))
        self.agent.vm.classify.assert_called_once_with(pending.text)
        self.agent.vm.search.assert_called_once_with(pending.text, slots=[], entities=[], emotion='')
        self.assertIs(pending.result, self.fresh)

    async def test_gate_shallow_and_normal_depth_stay_direct(self):
        pending = self.pending(route='shallow', prepared=True)
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'direct')
        self.assertEqual(pending.memory_context, '')
        self.assertEqual(pending.replay, '')
        self.agent.vm.search.assert_not_called()

    async def test_personal_followup_searches_with_context_and_keeps_original_text(self):
        pending = self.pending(route='shallow')
        pending.text = '那她呢？'
        history = [
            {'role': 'user', 'content': '我妹妹安安的生日是哪天？'},
            {'role': 'assistant', 'content': '我先查到了哥哥的生日。'},
        ]
        await self.agent.route_pending_thinking(pending, history=history)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertEqual(pending.text, '那她呢？')
        self.assertEqual(pending.memory_query, '我妹妹安安的生日是哪天？ 那她呢？')
        self.agent.vm.classify.assert_called_once_with(pending.memory_query)
        self.agent.vm.search.assert_called_once_with(
            pending.memory_query, slots=[], entities=[], emotion='')
        self.router.classify_async.assert_awaited_once_with('那她呢？', history=history)

    async def test_short_schedule_followup_uses_previous_personal_request(self):
        pending = self.pending(route='shallow')
        pending.text = '那周日呢？'
        await self.agent.route_pending_thinking(pending, history=[
            {'role': 'user', 'content': '我上次说过的周末计划是什么？'},
            {'role': 'assistant', 'content': '周六有聚会。'},
        ])
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertIn('周末计划', pending.memory_query)
        self.agent.vm.search.assert_called_once()

    async def test_general_followup_and_backchannel_do_not_search(self):
        history = [{'role': 'user', 'content': '周末天气怎么样？'}]
        for text in ('那周日呢？', '那她呢？'):
            with self.subTest(text=text):
                pending = self.pending(route='shallow')
                pending.text = text
                await self.agent.route_pending_thinking(pending, history=history)
                self.assertEqual(pending.reply_mode, 'direct')
                self.assertEqual(pending.memory_query, '')
        pending = self.pending(route='backchannel')
        pending.text = '嗯嗯'
        await self.agent.route_pending_thinking(pending, history=[
            {'role': 'user', 'content': '我妹妹安安的生日是哪天？'}])
        self.assertEqual(pending.reply_mode, 'direct')
        self.agent.vm.search.assert_not_called()

    async def test_contextual_followup_replaces_ambiguous_prefetch(self):
        pending = self.pending(route='deep', prepared=True)
        pending.text = '那她呢？'
        old_result = pending.result
        await self.agent.route_pending_thinking(pending, history=[
            {'role': 'user', 'content': '我妹妹安安的生日是哪天？'}])
        self.assertIsNot(pending.result, old_result)
        self.agent.vm.search.assert_called_once_with(
            '我妹妹安安的生日是哪天？ 那她呢？', slots=[], entities=[], emotion='')

    async def test_stranger_followup_never_searches_owner_memory(self):
        pending = self.pending(route='shallow', stranger=True)
        pending.text = '那她呢？'
        await self.agent.route_pending_thinking(pending, history=[
            {'role': 'user', 'content': '我妹妹安安的生日是哪天？'}])
        self.assertEqual(pending.memory_query, '')
        self.assertEqual(pending.memory_context, '')
        self.agent.vm.search.assert_not_called()

    async def test_followup_uses_only_latest_user_turn(self):
        pending = self.pending(route='shallow')
        pending.text = '那她呢？'
        await self.agent.route_pending_thinking(pending, history=[
            {'role': 'user', 'content': '我妹妹安安的生日是哪天？'},
            {'role': 'assistant', 'content': '五月十日。'},
            {'role': 'user', 'content': '周末天气怎么样？'},
            {'role': 'assistant', 'content': '晴天。'},
        ])
        self.assertEqual(pending.reply_mode, 'direct')
        self.agent.vm.search.assert_not_called()

    async def test_disabled_depth_router_still_handles_personal_followup(self):
        self.agent._THINKING_ROUTER_ON = False
        pending = self.pending(route='shallow')
        pending.text = '那她呢？'
        await self.agent.route_pending_thinking(pending, history=[
            {'role': 'user', 'content': '我妹妹安安的生日是哪天？'}])
        self.assertEqual(pending.reply_mode, 'memory')
        self.agent.vm.search.assert_called_once()
        self.router.classify_async.assert_not_awaited()

    async def test_failed_contextual_search_clears_ambiguous_prefetch(self):
        self.agent.vm.search.side_effect = RuntimeError('synthetic retrieval failure')
        pending = self.pending(route='deep', prepared=True)
        pending.text = '那她呢？'
        await self.agent.route_pending_thinking(pending, history=[
            {'role': 'user', 'content': '我妹妹安安的生日是哪天？'}])
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertEqual(pending.memory_context, '')
        self.assertEqual(pending.replay, '')
        self.assertEqual(pending.result.hits, [])

    async def test_deep_reasoning_upgrades_shallow_to_memory_cot(self):
        self.router.classify_async.return_value = ThinkingDecision(SLOW, '深思')
        pending = self.pending(route='shallow')
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory_cot')
        self.agent.vm.search.assert_called_once()

    async def test_self_harness_can_prefer_deep_or_quick_without_losing_memory(self):
        pending = self.pending(route='shallow')
        pending.self_harness_profile = {
            "reply_modes": {"reasoning_depth": "deep"}}
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory_cot')

        self.router.classify_async.return_value = ThinkingDecision(SLOW, '深思')
        pending = self.pending(prepared=True)
        pending.self_harness_profile = {
            "reply_modes": {"reasoning_depth": "quick"}}
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertEqual(pending.memory_context, 'existing-context')

    async def test_classifier_failure_preserves_memory_and_privacy(self):
        self.router.classify_async.side_effect = RuntimeError('synthetic classifier failure')
        pending = self.pending(prepared=True)
        result = pending.result
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertIs(pending.result, result)
        self.assertEqual(pending.memory_context, 'existing-context')
        stranger = self.pending(prepared=True, stranger=True)
        await self.agent.route_pending_thinking(stranger)
        self.assertEqual(stranger.memory_context, '')
        self.assertEqual(stranger.result.hits, [])
        self.agent.vm.search.assert_not_called()

    async def test_stranger_cannot_use_owner_memory_even_in_deep_mode(self):
        self.router.classify_async.return_value = ThinkingDecision(SLOW, '深思')
        pending = self.pending(prepared=True, stranger=True)
        await self.agent.route_pending_thinking(
            pending, history=[{"role": "user", "content": "主人的私人信息"}])
        self.assertEqual(pending.memory_context, '')
        self.assertEqual(pending.replay, '')
        self.agent.vm.search.assert_not_called()
        self.assertEqual(self.router.classify_async.call_args.kwargs['history'], [])

    async def test_unavailable_search_does_not_silently_downgrade_memory_mode(self):
        self.agent.vm.search.side_effect = RuntimeError('synthetic retrieval failure')
        pending = self.pending()
        await self.agent.route_pending_thinking(pending)
        self.assertEqual(pending.reply_mode, 'memory')
        self.assertEqual(pending.memory_context, '')


if __name__ == "__main__":
    unittest.main()
