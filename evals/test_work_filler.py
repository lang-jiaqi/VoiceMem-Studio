"""Offline work-filler policy, local generation and playback-race regressions."""
import asyncio
import queue
import sys
import threading
import time
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from studio.core.utils.contracts.component import ReplySink
from studio.core.utils.conversation.component import Conversation
from studio.core.utils.reply_modes.component import QwenThinkingRouter
from studio.core.utils.reply_modes.mlx import MLXThinkingRouter
from studio.core.utils.reply_modes.short_text import generate_short_text, torch_short_text
from studio.core.utils.self_harness.component import apply_turn_taking_profile
from studio.core.utils.turn_taking.component import HandoffKind, TurnTakingStateMachine
from studio.core.utils.turn_taking.filler import generate_local_filler


class WorkFillerPolicyTests(unittest.TestCase):
    def choose(self, machine, **kwargs):
        args = dict(main_audio_ready=False, reply_mode='memory_cot',
                    cached_ack_available=True, now=100.0)
        args.update(kwargs)
        return machine.decide_handoff(**args)

    def test_default_limits_and_validation(self):
        machine = TurnTakingStateMachine()
        self.assertEqual(machine.work_filler_probability, .3)
        self.assertEqual(machine.work_filler_cooldown_s, 20)
        for probability in (-1, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                TurnTakingStateMachine(work_filler_probability=probability)
        for cooldown in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                TurnTakingStateMachine(work_filler_cooldown_s=cooldown)

    def test_one_draw_per_confirmed_turn_including_retries(self):
        rng = Mock(random=Mock(side_effect=[.8, .1]))
        machine = TurnTakingStateMachine(filler_rng=rng)
        machine.commit_user_turn()
        self.assertIs(self.choose(machine).kind, HandoffKind.DIRECT)
        machine.begin_user_turn()
        self.assertIs(self.choose(machine).kind, HandoffKind.DIRECT)
        self.assertEqual(rng.random.call_count, 1)
        machine.commit_user_turn()
        self.assertIs(self.choose(machine).kind, HandoffKind.LLM_FILLER)
        self.assertIs(self.choose(machine).kind, HandoffKind.LLM_FILLER)
        self.assertEqual(rng.random.call_count, 2)

    def test_ready_text_and_ordinary_turns_do_not_draw(self):
        machine = TurnTakingStateMachine(filler_rng=Mock())
        for kwargs in ({'main_audio_ready':True}, {'spoken':False},
                       {'reply_mode':'direct'}, {'reply_mode':'memory'}):
            self.choose(machine, **kwargs)
        machine.filler_rng.random.assert_not_called()

    def test_probability_miss_does_not_fall_back_to_cached_ack(self):
        machine = TurnTakingStateMachine(initial_wait_s=10, work_filler_probability=0)
        self.assertEqual(self.choose(machine).reason, 'filler_probability')
        self.assertIs(self.choose(machine).kind, HandoffKind.DIRECT)

    def test_cooldown_covers_clip_and_twenty_seconds_then_new_turn(self):
        machine = TurnTakingStateMachine(work_filler_probability=1)
        machine.record_work_filler(4, now=100)
        machine.commit_user_turn()
        self.assertEqual(self.choose(machine, now=123).reason, 'filler_cooldown')
        # Waiting/retrying this same turn must not create a fresh opportunity.
        self.assertIs(self.choose(machine, now=125).kind, HandoffKind.DIRECT)
        machine.commit_user_turn()
        self.assertIs(self.choose(machine, now=125).kind, HandoffKind.LLM_FILLER)

    def test_cancelled_or_failed_attempt_does_not_start_cooldown(self):
        machine = TurnTakingStateMachine(work_filler_probability=1)
        self.choose(machine)
        machine.finish_reply()
        machine.commit_user_turn()
        self.assertIs(self.choose(machine, now=101).kind, HandoffKind.LLM_FILLER)

    def test_cooldown_is_session_local_and_does_not_change_short_ack(self):
        first = TurnTakingStateMachine(work_filler_probability=1, initial_wait_s=2)
        second = TurnTakingStateMachine(work_filler_probability=1)
        first.record_work_filler(4, now=100)
        self.assertIs(self.choose(second).kind, HandoffKind.LLM_FILLER)
        self.assertIs(self.choose(first, reply_mode='memory').kind, HandoffKind.CACHED_ACK)
        self.assertIsNot(first.filler_rng, first.backchannel.rng)

    def test_self_harness_turn_taking_overlay_is_session_local_and_resettable(self):
        machine = TurnTakingStateMachine(work_filler_probability=.3)
        apply_turn_taking_profile(machine, {"turn_taking": {
            "backchannel": "off", "work_filler": "silent"}})
        self.assertFalse(machine.backchannel.enabled)
        self.assertEqual(machine.work_filler_probability, 0)
        apply_turn_taking_profile(machine, {"turn_taking": {
            "backchannel": "more", "work_filler": "reassuring"}})
        self.assertTrue(machine.backchannel.enabled)
        self.assertEqual(machine.backchannel.frequency, "more")
        self.assertEqual(machine.work_filler_probability, .75)
        apply_turn_taking_profile(machine, {})
        self.assertEqual(machine.backchannel.frequency, "auto")
        self.assertEqual(machine.work_filler_probability, .3)


class LocalFillerTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_history_full_current_text_and_separate_prompt(self):
        router = types.SimpleNamespace(generate_short_text_async=AsyncMock(return_value='嗯，让我稍微理一下。'))
        text = '请结合这些限制给出比较。' * 30
        history = [{'role':'user','content':'old-excluded'},
                   {'role':'user','content':'保留的问题' * 50},
                   {'role':'assistant','content':'简短回复' * 50}]
        result = await generate_local_filler(text, history=history, router=router)
        self.assertEqual(result, '嗯，让我稍微理一下。')
        call = router.generate_short_text_async.call_args
        system, prompt = call.args
        self.assertIn('不认同自责', system)
        self.assertIn(text, prompt)
        self.assertNotIn('old-excluded', prompt)
        self.assertLess(len(prompt) - len(text), 300)
        self.assertEqual(call.kwargs, {'max_tokens':40, 'timeout_s':1.2})

    async def test_invalid_or_empty_generation_is_skipped(self):
        router = types.SimpleNamespace(generate_short_text_async=AsyncMock())
        for text in ('', '对', '<think>分析一下</think>', '解释\n多余内容', '很长' * 50,
                     '你能告诉我具体信息吗？', '分析时说“嗯，让我想想。'):
            router.generate_short_text_async.return_value = text
            self.assertEqual(await generate_local_filler('合成测试问题', router=router), '')

    async def test_control_prefix_removed_and_empty_input_never_calls_model(self):
        router = types.SimpleNamespace(generate_short_text_async=AsyncMock(return_value='温和|嗯，让我理一理。'))
        self.assertEqual(await generate_local_filler('合成测试问题', router=router), '嗯，让我理一理。')
        router.generate_short_text_async.reset_mock()
        self.assertEqual(await generate_local_filler('  ', router=router), '')
        router.generate_short_text_async.assert_not_awaited()

    async def test_cold_model_never_downloads_or_loads(self):
        router = QwenThinkingRouter(model='/unused')
        router._load = Mock(side_effect=AssertionError('must not load'))
        self.assertEqual(await router.generate_short_text_async('system', 'prompt'), '')
        router._load.assert_not_called()

    async def test_timeout_signals_worker_and_discards_late_text(self):
        finished = threading.Event()
        def work(system, prompt, tokens, cancelled, deadline):
            cancelled.wait(1)
            finished.set()
            return 'late output'
        router = types.SimpleNamespace(_model=object(), _short_text=work)
        self.assertEqual(await generate_short_text(router, 's', 'p', timeout_s=.05), '')
        self.assertTrue(await asyncio.to_thread(finished.wait, 1))

    async def test_cancellation_reaches_worker(self):
        entered, finished = threading.Event(), threading.Event()
        def work(system, prompt, tokens, cancelled, deadline):
            entered.set()
            cancelled.wait(1)
            finished.set()
            return ''
        router = types.SimpleNamespace(_model=object(), _short_text=work)
        task = asyncio.create_task(generate_short_text(router, 's', 'p'))
        self.assertTrue(await asyncio.to_thread(entered.wait, 1))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(await asyncio.to_thread(finished.wait, 1))

    async def test_generation_does_not_change_router_cache(self):
        router = QwenThinkingRouter(model='/unused')
        router._model = object()
        router._cache[('input','context')] = object()
        before = dict(router._cache)
        router._short_text = Mock(return_value='嗯，让我稍微想一想。')
        self.assertTrue(await router.generate_short_text_async('s', 'p'))
        self.assertEqual(router._cache, before)


class TorchShortTextTests(unittest.TestCase):
    def fixture(self, tokens):
        import torch
        from voicemem.utils.torch_lock import TORCH_LOCK
        calls = []
        token_ids = iter(tokens)
        cache = object()
        def forward(**kwargs):
            self.assertTrue(TORCH_LOCK._is_owned())
            calls.append(kwargs)
            logits = torch.full((1, 1, 10), -10000.0)
            logits[0, -1, next(token_ids)] = 0
            return types.SimpleNamespace(logits=logits, past_key_values=cache)
        model = Mock(side_effect=forward)
        model.generation_config = types.SimpleNamespace(eos_token_id=[9])
        tokenizer = Mock(return_value={'input_ids':torch.tensor([[1, 2]])})
        tokenizer.eos_token_id = 9
        tokenizer.apply_chat_template.return_value = 'rendered'
        def decode(ids, **_):
            self.assertFalse(TORCH_LOCK._is_owned())
            return ''.join({3:'嗯，', 4:'让我想想', 5:'。'}[i] for i in ids)
        tokenizer.decode.side_effect = decode
        router = types.SimpleNamespace(_tokenizer=tokenizer, _model=model, _device='cpu')
        return router, calls, cache

    def test_private_kv_and_per_step_lock_without_changing_global_rng(self):
        import torch
        router, calls, cache = self.fixture([3, 4, 5])
        rng = torch.random.get_rng_state().clone()
        result = torch_short_text(router, 's', 'p', 8, threading.Event(), time.monotonic()+2)
        self.assertEqual(result, '嗯，让我想想。')
        self.assertIsNone(calls[0]['past_key_values'])
        self.assertTrue(all(c['past_key_values'] is cache for c in calls[1:]))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(router._tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_token_budget_does_not_voice_a_truncated_sentence(self):
        router, _, _ = self.fixture([3, 4])
        self.assertEqual(torch_short_text(router, 's', 'p', 2, threading.Event(), time.monotonic()+2), '')

    def test_expired_request_never_enters_model(self):
        router = types.SimpleNamespace(_model=Mock())
        self.assertEqual(torch_short_text(router, 's', 'p', 8, threading.Event(), time.monotonic()-1), '')
        router._model.assert_not_called()

    def test_lock_wait_obeys_deadline_without_forward(self):
        from voicemem.utils.torch_lock import TORCH_LOCK
        router, calls, _ = self.fixture([3, 4, 5])
        results = []
        with TORCH_LOCK:
            worker = threading.Thread(target=lambda: results.append(torch_short_text(
                router, 's', 'p', 8, threading.Event(), time.monotonic()+.04)))
            worker.start()
            worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [''])
        self.assertEqual(calls, [])


class MLXShortTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_cache_and_gpu_owned_token_steps(self):
        import numpy as np
        made, forwards, workers = [], [], []
        cache = [types.SimpleNamespace(state=0)]
        def make_cache(_):
            made.append(threading.get_ident())
            return cache
        ids = iter([3, 4, 5])
        def forward(tokens, cache):
            forwards.append(threading.get_ident())
            logits = np.full((1, 1, 10), -10000.0)
            logits[0, -1, next(ids)] = 0
            return logits
        router = MLXThinkingRouter(model='/unused')
        router._model = Mock(side_effect=forward)
        router._tokenizer = types.SimpleNamespace(
            apply_chat_template=Mock(return_value=[1, 2]), eos_token_ids={9},
            decode=lambda ids: ''.join({3:'嗯，',4:'让我想想',5:'。'}[i] for i in ids))
        prefix = router._prefix_cache = object()
        mx = types.ModuleType('mlx.core')
        mx.array, mx.eval = np.array, lambda _:None
        mx.random = types.SimpleNamespace(key=lambda n:n, split=lambda k:(k+1,k+2),
            categorical=lambda logits,key: np.argmax(logits))
        root = types.ModuleType('mlx')
        root.core = mx
        models = types.ModuleType('mlx_lm.models.cache')
        models.make_prompt_cache = make_cache
        modules = {'mlx':root, 'mlx.core':mx, 'mlx_lm':types.ModuleType('mlx_lm'),
            'mlx_lm.models':types.ModuleType('mlx_lm.models'), 'mlx_lm.models.cache':models}
        jobs = []
        def submit(factory, **kwargs):
            self.assertEqual(kwargs, {'weight':1})
            stopped = threading.Event()
            job = types.SimpleNamespace(out=queue.Queue(), cancel=stopped.set)
            jobs.append(job)
            def run():
                gen = factory()
                try:
                    while not stopped.is_set():
                        try:
                            value = next(gen)
                        except StopIteration:
                            break
                        job.out.put(('ok',value))
                except Exception as exc:
                    job.out.put(('err',exc))
                finally:
                    gen.close()
                    job.out.put(None)
            worker = threading.Thread(target=run)
            workers.append(worker)
            worker.start()
            return job
        with patch.dict(sys.modules, modules), patch('voicemem.utils.gpu_loop.gpu_loop',
                return_value=types.SimpleNamespace(iter=submit)):
            result = await router.generate_short_text_async('s','p')
            for worker in workers:
                await asyncio.to_thread(worker.join, 1)
        self.assertEqual(result, '嗯，让我想想。')
        self.assertIs(router._prefix_cache, prefix)
        self.assertEqual(set(made+forwards), {workers[0].ident})
        self.assertNotEqual(workers[0].ident, threading.get_ident())

    async def test_queued_mlx_job_is_cancelled_at_deadline(self):
        router = MLXThinkingRouter(model='/unused')
        router._model = Mock(side_effect=AssertionError('queued work must not run'))
        cancelled = threading.Event()
        job = types.SimpleNamespace(out=queue.Queue(), cancel=cancelled.set)
        with patch('voicemem.utils.gpu_loop.gpu_loop',
                   return_value=types.SimpleNamespace(iter=lambda *a,**k:job)):
            self.assertEqual(await router.generate_short_text_async('s','p',timeout_s=.04), '')
            self.assertTrue(await asyncio.to_thread(cancelled.wait, 1))
        router._model.assert_not_called()


class FillerHandoffTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        async def send(message):
            self.events.append(message)
        agent = types.SimpleNamespace(BC_ECHO_WINDOW_S=4, MIC_RATE=24000,
            _SESSION_CONTEXT=types.SimpleNamespace(messages=lambda *a,**k:[]),
            HISTORY_TURNS=4, space_language=lambda _: 'zh', _speak_instruction=lambda _: '')
        self.session = Conversation(agent, types.SimpleNamespace(send_json=send))
        self.sink = ReplySink(send, AsyncMock())
        self.value = types.SimpleNamespace(text='合成测试问题', emotion='')
        self.decision = types.SimpleNamespace(kind=HandoffKind.LLM_FILLER)

    def release(self):
        return self.session.release_buffered_reply(self.sink, self.decision, None, self.value, object(), 'fixture')

    async def test_main_ready_cancels_unplayed_filler_without_cooldown(self):
        cancelled = asyncio.Event()
        async def work(*_):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        self.session.synthesize_work_filler = work
        task = asyncio.create_task(self.release())
        await asyncio.sleep(.01)
        await self.sink.send({'type':'answer_start'})
        await self.sink.send_audio(bytes(480))
        await asyncio.wait_for(task, 1)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(any(m['type']=='backchannel' for m in self.events))
        self.assertEqual(self.session.turn_taking._work_filler_until, 0)

    async def test_emitted_filler_waits_for_completion_and_records_cooldown(self):
        self.session.synthesize_work_filler = AsyncMock(return_value=('嗯，让我想想。', bytes(4800)))
        task = asyncio.create_task(self.release())
        for _ in range(10):
            await asyncio.sleep(.001)
            if self.events:
                break
        message = self.events[0]
        self.assertEqual(message['type'], 'backchannel')
        self.assertGreater(self.session.turn_taking._work_filler_until, time.monotonic()+19)
        await self.sink.send({'type':'answer_start'})
        await self.sink.send_audio(bytes(480))
        await asyncio.sleep(.01)
        self.assertFalse(task.done())
        self.assertEqual(len(self.events), 1)
        self.session.filler_done(message['filler_id'])
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.events[-1]['type'], 'answer_start')

    async def test_failed_or_empty_filler_releases_main_without_cooldown(self):
        for result in (('', b''), RuntimeError('synthetic failure')):
            self.session.synthesize_work_filler = AsyncMock()
            if isinstance(result, Exception):
                self.session.synthesize_work_filler.side_effect = result
            else:
                self.session.synthesize_work_filler.return_value = result
            await asyncio.wait_for(self.release(), 1)
            self.assertEqual(self.session.turn_taking._work_filler_until, 0)
            self.assertFalse(any(m['type']=='backchannel' for m in self.events))

    async def test_short_ack_does_not_consume_long_filler_cooldown(self):
        await self.session.emit_filler('嗯', bytes(480))
        self.assertEqual(self.session.turn_taking._work_filler_until, 0)

    async def test_failed_send_and_stale_completion_do_not_consume_cooldown(self):
        self.session.sock.send_json = AsyncMock(side_effect=ConnectionError)
        with self.assertRaises(ConnectionError):
            await self.session.emit_filler('嗯，让我想想。', bytes(480), filler_id='failed')
        self.session.filler_done('failed')
        self.assertEqual(self.session.turn_taking._work_filler_until, 0)

    async def test_delayed_playback_completion_extends_cooldown_only_once(self):
        event = self.session.filler_waiters['clip'] = asyncio.Event()
        self.session.turn_taking.record_work_filler(1, now=100)
        with patch('studio.core.utils.turn_taking.component.time.monotonic', return_value=105):
            self.session.filler_done('clip')
        self.assertTrue(event.is_set())
        self.assertEqual(self.session.turn_taking._work_filler_until, 125)
        with patch('studio.core.utils.turn_taking.component.time.monotonic', return_value=110):
            self.session.filler_done('clip')
        self.assertEqual(self.session.turn_taking._work_filler_until, 125)

    async def test_disconnect_cancels_unplayed_filler(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        async def work(*_):
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
        self.session.synthesize_work_filler = work
        task = asyncio.create_task(self.release())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        self.assertEqual(self.session.turn_taking._work_filler_until, 0)
        self.assertFalse(self.events)

    async def test_synthesis_uses_local_text_not_main_reply_api(self):
        async def tts(*_):
            yield bytes(480)
        memory = types.SimpleNamespace(reply_stream=Mock(side_effect=AssertionError('no API')),
            utils=types.SimpleNamespace(get=lambda _:types.SimpleNamespace(stream=tts)))
        with patch('studio.core.utils.conversation.component.generate_local_filler',
                   new=AsyncMock(return_value='嗯，让我想想。')) as local:
            text, pcm = await self.session.synthesize_work_filler(self.value, memory, 'fixture')
        self.assertEqual(text, '嗯，让我想想。')
        self.assertEqual(len(pcm), 480)
        local.assert_awaited_once()
        memory.reply_stream.assert_not_called()

    async def test_cancelled_synthesis_closes_tts_stream(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        async def tts(*_):
            try:
                yield bytes(480)
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
        memory = types.SimpleNamespace(utils=types.SimpleNamespace(get=lambda _:types.SimpleNamespace(stream=tts)))
        with patch('studio.core.utils.conversation.component.generate_local_filler',
                   new=AsyncMock(return_value='嗯，让我想想。')):
            task = asyncio.create_task(self.session.synthesize_work_filler(self.value, memory, 'fixture'))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(closed.is_set())


if __name__ == '__main__':
    unittest.main()
