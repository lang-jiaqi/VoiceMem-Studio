"""Offline decision-head, feature, cancellation and independent-filler checks."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from studio.core.utils.reply_modes.component import FAST, SLOW
from studio.core.utils.reply_modes.decision import LocalDecisionRouter, create
from studio.core.utils.reply_modes.decision_features import audio_vector, combine, texts
from studio.core.utils.reply_modes import independent_filler
from finetune.decision_router.train import load_rows, fit, calibrate, probabilities


class FeaturesTests(unittest.TestCase):
    def test_current_request_and_history_have_separate_bounded_budgets(self):
        current, context = texts('head' + 'x' * 1000 + 'tail',
                                 [{'role': 'user', 'content': str(i) * 200} for i in range(8)])
        self.assertTrue(current.startswith('head'))
        self.assertTrue(current.endswith('tail'))
        self.assertEqual(len(context.splitlines()), 4)
        self.assertLess(len(context), 400)
        self.assertNotIn('0000', context)

    def test_missing_audio_is_distinct_from_observed_zero(self):
        self.assertFalse(audio_vector().any())
        state = audio_vector({'eot_probability': 0, 'span_s': 60})
        np.testing.assert_array_equal(state, [0, 0, 1, 1, 0, 1])
        self.assertEqual(combine([1, 0], [0, 1]).shape, (12,))
        for value in ({'eot_probability': 2}, {'span_s': -1}, {'silence_s': float('nan')},
                      {'silence_s': True}, {'unknown': .5}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audio_vector(value)


class RouterFixture:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.metadata = {'version': 1, 'prototype': False, 'encoder': 'e5',
                         'encoder_repo': 'intfloat/multilingual-e5-small',
                         'dimensions': 384, 'temperature': 1, 'threshold': .5}
        self.weights = np.zeros(3 * 384 + 6)
        self.weights[0] = 8
        self.save()
        self.environment = patch.dict(os.environ, {'VOICEMEM_LOCAL_EMBED_MODEL': 'e5'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def save(self):
        (self.path / 'decision.json').write_text(json.dumps(self.metadata))
        np.savez(self.path / 'head.npz', weights=self.weights, bias=np.asarray(0.0))

    def router(self):
        value = LocalDecisionRouter(self.path)
        def encode(values):
            result = np.zeros((len(values), 384), np.float32)
            result[:, 0] = [1 if item == 'deep fixture' else -1 for item in values]
            return result
        value._encode = encode
        return value


class CheckpointTests(RouterFixture, unittest.TestCase):
    def test_no_generative_weights_are_loaded_and_only_depth_is_returned(self):
        with patch('studio.core.utils.reply_modes.component.QwenThinkingRouter._load',
                   side_effect=AssertionError('Qwen must not load')):
            router = self.router()
            self.assertEqual(router.classify('deep fixture').level, SLOW)
            self.assertEqual(router.classify('ordinary fixture', True).level, FAST)
            self.assertEqual(router.classify('').level, FAST)

    def test_context_and_audio_changes_cannot_hit_an_old_decision(self):
        router = self.router()
        with patch.object(router, '_encode', wraps=router._encode) as encode:
            router.classify('fixture')
            router.classify('fixture')
            router.classify('fixture', history=[{'role': 'user', 'content': 'other context'}])
            router.classify('fixture', audio_state={'eot_probability': .9})
            self.assertEqual(encode.call_count, 3)

    def test_encoder_identity_and_calibration_are_verified(self):
        for key, value in (('version', 2), ('encoder_repo', 'different'),
                           ('dimensions', 768), ('temperature', 0), ('threshold', 2)):
            original = dict(self.metadata)
            self.metadata[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                LocalDecisionRouter(self.path)
            self.metadata = original
        self.save()
        with patch.dict(os.environ, {'VOICEMEM_LOCAL_EMBED_MODEL': 'bge-zh'}), self.assertRaises(ValueError):
            LocalDecisionRouter(self.path)

    def test_bad_weights_and_prototype_checkpoints_are_not_silently_accepted(self):
        for weights in (np.zeros(3), np.full(1158, np.nan)):
            np.savez(self.path / 'head.npz', weights=weights, bias=np.asarray(0.0))
            with self.assertRaises(ValueError):
                LocalDecisionRouter(self.path)
        self.metadata['prototype'] = True
        self.save()
        with self.assertRaisesRegex(ValueError, 'Prototype'):
            LocalDecisionRouter(self.path)
        self.assertIsInstance(create(model=str(self.path), options={'allow_prototype': True}), LocalDecisionRouter)
        with self.assertRaises(ValueError):
            create(model=str(self.path), options={'allow_prototype': 'false'})


class AsyncRouterTests(RouterFixture, unittest.IsolatedAsyncioTestCase):
    async def test_identical_waiters_share_work_and_one_cancel_does_not_kill_the_other(self):
        router = self.router()
        entered, release = threading.Event(), threading.Event()
        encode = router._encode
        calls = []
        def slow(values):
            calls.append(values)
            entered.set()
            release.wait(2)
            return encode(values)
        router._encode = slow
        first = asyncio.create_task(router.classify_async('deep fixture'))
        second = asyncio.create_task(router.classify_async('deep fixture'))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertEqual((await second).level, SLOW)
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)

    async def test_cancelled_last_waiter_cannot_cache_a_stale_native_result(self):
        router = self.router()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        encode = router._encode
        def slow(values):
            entered.set()
            release.wait(2)
            finished.set()
            return encode(values)
        router._encode = slow
        task = asyncio.create_task(router.classify_async('deep fixture'))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 1)
            await asyncio.sleep(.02)
        self.assertEqual(len(router._cache), 0)

    async def test_auxiliary_wording_delegates_without_loading_qwen(self):
        router = self.router()
        with patch.object(independent_filler, 'generate', new_callable=AsyncMock,
                          return_value='让我理一理。') as generate:
            text = await router.generate_short_text_async('system', 'prompt')
        self.assertEqual(text, '让我理一理。')
        generate.assert_awaited_once()


class TrainingTests(unittest.TestCase):
    def test_fit_and_disjoint_temperature_calibration_generalize_on_toy_features(self):
        features = np.asarray([[-3, 0], [-2, .1], [2, -.1], [3, 0]], float)
        labels = np.asarray([0, 0, 1, 1], float)
        weights, bias = fit(features, labels)
        calibration = np.asarray([[-1.5, 0], [1.5, 0]])
        temperature = calibrate(calibration @ weights + bias, np.asarray([0, 1]))
        scores = probabilities(np.asarray([[-1, 0], [1, 0]]) @ weights + bias, temperature)
        self.assertLess(scores[0], .5)
        self.assertGreater(scores[1], .5)

    def test_pilot_groups_do_not_cross_splits_and_observed_audio_is_not_fabricated(self):
        from finetune.decision_router.pilot import build
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            rows = build()
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertEqual(len(load_rows(path)), len(rows))
            self.assertFalse(any('audio_state' in row for row in rows))
            rows[0]['group'] = next(row['group'] for row in rows if row['split'] == 'train')
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'group crosses'):
                load_rows(path)


class LayaRouterTests(RouterFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        from finetune.decision_router.laya_train import QUESTION
        self.voice_metadata = {'version': 1, 'prototype': False, 'task': 'voice-dialogue-depth',
                               'question': QUESTION, 'report': {'observed_audio_rows': 0}}
        self.save_voice()
        (self.path / 'model.safetensors').write_bytes(b'fixture; never loaded')
        self.agent = SimpleNamespace(predict=Mock(return_value={
            'answers': {'depth': {'choice': 'deep', 'probabilities': {'ordinary': .1, 'deep': .9}}}}))
        self.module = SimpleNamespace(load=Mock(return_value=self.agent))
        self.enterContext(patch.dict('sys.modules', {'laya': self.module}))

    def save_voice(self):
        (self.path / 'voice_decision.json').write_text(json.dumps(self.voice_metadata))

    async def test_typed_head_keeps_training_state_and_does_not_enable_untrained_audio(self):
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter
        router = LayaDecisionRouter(self.path)
        self.assertFalse(router.accepts_audio_state)
        decision = await router.classify_async('synthetic request', audio_state={'eot_probability': .9})
        self.assertEqual(decision.level, SLOW)
        state = self.agent.predict.call_args.args[0]
        self.assertEqual(state['current_user'], 'synthetic request')
        self.assertNotIn('observed_audio', state)
        await router.classify_async('synthetic request')
        self.agent.predict.assert_called_once()

    async def test_observed_audio_and_context_reach_a_checkpoint_trained_for_them(self):
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter
        self.voice_metadata['report']['observed_audio_rows'] = 50
        self.save_voice()
        router = LayaDecisionRouter(self.path)
        audio = {'eot_probability': .7}
        await router.classify_async('synthetic request', history=[{'role': 'user', 'content': 'context'}], audio_state=audio)
        state = self.agent.predict.call_args.args[0]
        self.assertEqual(state['observed_audio'], audio)
        self.assertIn('context', state['recent_dialogue'])
        self.assertIsNot(state['observed_audio'], audio)

    async def test_long_audio_states_have_distinct_keys_and_field_order_is_canonical(self):
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter
        self.voice_metadata['report']['observed_audio_rows'] = 50
        self.save_voice()
        router = LayaDecisionRouter(self.path)
        await router.classify_async('fixture', audio_state={'span_s': 35, 'eot_probability': .7})
        await router.classify_async('fixture', audio_state={'eot_probability': .7, 'span_s': 35})
        self.assertEqual(self.agent.predict.call_count, 1)
        await router.classify_async('fixture', audio_state={'span_s': 60, 'eot_probability': .7})
        self.assertEqual(self.agent.predict.call_count, 2)

    async def test_invalid_probability_distributions_are_not_accepted(self):
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter
        self.agent.predict.return_value['answers']['depth']['probabilities']['deep'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            await LayaDecisionRouter(self.path).classify_async('synthetic request')

    async def test_onnx_runtime_can_load_without_duplicate_torch_weights(self):
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter
        graph = self.path / 'laya.onnx'
        graph.write_bytes(b'fixture graph; mocked loader')
        (self.path / 'model.safetensors').unlink()
        router = LayaDecisionRouter(self.path, backend='onnx')
        with patch('studio.core.utils.reply_modes.onnx_depth.OnnxDepthRuntime', return_value=self.agent) as load:
            self.assertEqual((await router.classify_async('synthetic request')).level, SLOW)
            load.assert_called_once_with(self.path.resolve(), str(graph), self.voice_metadata['question'])
        self.module.load.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'device=cpu'):
            LayaDecisionRouter(self.path, backend='onnx', device='mps')

    def test_ablation_metadata_does_not_enable_an_untrained_acoustic_branch(self):
        from studio.core.utils.reply_modes.laya import LayaDecisionRouter
        self.voice_metadata['report'].update(observed_audio_rows=100, trained_audio_state=False)
        self.save_voice()
        self.assertFalse(LayaDecisionRouter(self.path).accepts_audio_state)


class IndependentFillerTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_and_cancellation_close_auxiliary_stream(self):
        for cancel in (False, True):
            entered = asyncio.Event()
            closed = AsyncMock()
            exited = AsyncMock()
            class Stream:
                async def __aiter__(self):
                    entered.set()
                    await asyncio.Event().wait()
                    yield None
                close = closed
            class Client:
                def __init__(self, **kwargs):
                    self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.request))
                async def request(self, **kwargs):
                    return Stream()
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    await exited()
            with self.subTest(cancel=cancel), \
                 patch.object(independent_filler, 'configuration', return_value=('openai', 'aux', 'http://127.0.0.1', 'fixture')), \
                 patch('openai.AsyncOpenAI', Client):
                task = asyncio.create_task(independent_filler.generate('system', 'prompt', timeout_s=.1))
                await asyncio.wait_for(entered.wait(), 1)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    self.assertEqual(await task, '')
                closed.assert_awaited_once()
                exited.assert_awaited_once()

    def test_different_provider_or_endpoint_does_not_reuse_reply_credentials(self):
        env = {'VOICEMEM_STUDIO_PROVIDER': 'deepseek', 'VOICEMEM_STUDIO_API_KEY': 'reply-fixture',
               'VOICEMEM_STUDIO_MODEL': 'reply-model', 'STUDIO_FILLER_PROVIDER': 'openai'}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            independent_filler.configuration()
        env['OPENAI_API_KEY'] = 'openai-fixture'
        with patch.dict(os.environ, env, clear=True):
            cfg = independent_filler.configuration()
            self.assertEqual((cfg[1], cfg[3]), ('gpt-4o', 'openai-fixture'))
        env['STUDIO_FILLER_BASE_URL'] = 'https://different.invalid/v1'
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            independent_filler.configuration()

    def test_loopback_server_and_dedicated_credentials_are_supported(self):
        env = {'STUDIO_FILLER_PROVIDER': 'openai', 'STUDIO_FILLER_MODEL': 'auxiliary',
               'STUDIO_FILLER_BASE_URL': 'http://127.0.0.1:9999/v1', 'STUDIO_FILLER_API_KEY': ''}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(independent_filler.configuration(),
                             ('openai', 'auxiliary', env['STUDIO_FILLER_BASE_URL'], 'local'))
        env.update(STUDIO_FILLER_BASE_URL='https://example.invalid/v1', OPENAI_API_KEY='other')
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            independent_filler.configuration()

    async def test_non_thinking_filler_stream_closes_without_touching_reply_settings(self):
        calls = []
        stream = SimpleNamespace(close=AsyncMock())
        async def chunks():
            for text in ('让我', '认真整理一下。', '不会读到这里'):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        class Stream:
            def __aiter__(self):
                return chunks()
            close = stream.close
        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.request))
            async def request(self, **kwargs):
                calls.append(kwargs)
                return Stream()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                pass
        with patch.object(independent_filler, 'configuration', return_value=('deepseek', 'aux', 'https://example.invalid', 'fixture')), \
             patch('openai.AsyncOpenAI', Client):
            self.assertEqual(await independent_filler.generate('system', 'prompt'), '让我认真整理一下。')
        self.assertEqual(calls[0]['extra_body'], {'thinking': {'type': 'disabled'}})
        stream.close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
