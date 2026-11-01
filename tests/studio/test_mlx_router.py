import asyncio
import os
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from studio.core.utils.reply_modes.mlx import MLXThinkingRouter
from studio.core.utils.reply_modes.initialize import thinking_router
from studio.core.utils.reply_modes.component import QwenThinkingRouter
from tests.helpers.gpu import cpu_gpu_loop


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.router = MLXThinkingRouter(model='/unused')
        self.prefills = []
        self.generations = []

        def template(messages, **options):
            rendered = ''.join(f"{m['role']}:{m['content']}|end|" for m in messages)
            return list(map(ord, rendered + ('assistant:' if options['add_generation_prompt'] else '')))

        self.tokenizer = types.SimpleNamespace(apply_chat_template=template,
            eos_token_ids={0}, decode=lambda tokens: ''.join(map(chr, tokens)))
        def model(tokens, cache):
            values = tokens[0].tolist()
            if ''.join(map(chr, values)).endswith('user:'):
                self.prefills.append(values)
            elif len(values) > 1:
                self.generations.append((list(cache[0].state), values))
            cache[0].state.extend(tokens[0].tolist())
            logits = np.zeros((1, 1, 40000))
            logits[0, 0, 0 if len(values) == 1 else ord('否')] = 1
            return logits
        self.router._load_mlx = Mock(return_value=(self.tokenizer, model))
        self.enterContext(patch.dict(sys.modules, {
            'mlx': types.ModuleType('mlx'),
            'mlx.core': types.SimpleNamespace(array=np.array, argmax=np.argmax, eval=lambda *_: None),
            'mlx_lm.models.cache': types.SimpleNamespace(make_prompt_cache=lambda _: [types.SimpleNamespace(state=[])]),
        }))

    def predict(self, text, system='policy'):
        return self.router._predict_on_gpu(system, [('example', '否')], text)

    def test_reuse_keeps_each_request_out_of_static_cache(self):
        self.predict('first private request')
        original = list(self.router._prefix_cache[0].state)
        self.predict('second private request')
        self.assertEqual(len(self.prefills), 1)
        self.assertEqual(self.generations[0][0], self.generations[1][0])
        self.assertEqual(self.router._prefix_cache[0].state, original)
        self.assertNotIn('private', ''.join(map(chr, original)))

    def test_cached_and_uncached_inputs_are_identical(self):
        self.predict('current request')
        cached, suffix = self.generations[-1]
        full = self.tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'policy'},
            {'role': 'user', 'content': 'example'},
            {'role': 'assistant', 'content': '否'},
            {'role': 'user', 'content': 'current request'}], add_generation_prompt=True)
        self.assertEqual(cached + suffix, full)

    def test_changed_policy_rebuilds_prefix(self):
        self.predict('one')
        self.predict('two', system='new policy')
        self.assertEqual(len(self.prefills), 2)
        self.assertNotEqual(self.generations[0][0], self.generations[1][0])

    def test_template_mismatch_uses_full_prompt(self):
        self.predict('one')
        original = self.tokenizer.apply_chat_template
        self.tokenizer.apply_chat_template = lambda messages, **kw: [123456] + original(messages, **kw)
        self.predict('two')
        self.assertEqual(self.generations[-1][0], [])
        self.assertEqual(self.generations[-1][1][0], 123456)

    def test_inference_is_submitted_to_shared_scheduler(self):
        with patch('voicemem.engine.gpu_loop') as loop:
            loop.return_value.run_steps.side_effect = lambda fn, **_: list(fn())[-1]
            self.assertEqual(self.router._predict('policy', [], 'text'), '否')
            loop.return_value.run_steps.assert_called_once()

    def test_model_exception_is_visible(self):
        self.router._load_mlx.side_effect = RuntimeError('test failure')
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            self.predict('text')


class FactoryTests(unittest.TestCase):
    def tearDown(self):
        thinking_router.cache_clear()

    def test_mlx_selected_only_for_mlx_studio(self):
        for backend, cls in [('mlx', MLXThinkingRouter), ('cuda', QwenThinkingRouter)]:
            thinking_router.cache_clear()
            with patch.dict(os.environ, {'STUDIO_BACKEND': backend, 'STUDIO_ROUTER_BACKEND': ''}):
                self.assertIs(type(thinking_router()), cls)

    def test_torch_override(self):
        with patch.dict(os.environ, {'STUDIO_BACKEND': 'mlx', 'STUDIO_ROUTER_BACKEND': 'torch'}):
            self.assertIs(type(thinking_router()), QwenThinkingRouter)

    def test_invalid_backend_rejected(self):
        with patch.dict(os.environ, {'STUDIO_ROUTER_BACKEND': 'invalid'}):
            with self.assertRaises(ValueError):
                thinking_router()


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_does_not_deliver_a_stale_result(self):
        router = MLXThinkingRouter(model='/unused')
        started, release = threading.Event(), threading.Event()
        def slow(*_):
            started.set()
            release.wait(3)
            yield '否'
        router._timed_prediction = slow
        with cpu_gpu_loop() as gpu, patch('voicemem.engine.gpu_loop', return_value=gpu):
            task = asyncio.create_task(router.classify_async('cancelled request'))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            router._timed_prediction = lambda *_: iter(['是'])
            self.assertEqual((await router.classify_async('next request')).level, 'slow')
            self.assertEqual(router._classifications, {})


if __name__ == '__main__':
    unittest.main()
