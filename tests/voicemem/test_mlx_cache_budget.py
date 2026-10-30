"""Paging hysteresis and owner-thread changes to reclaimable MLX capacity."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests.helpers.gpu import cpu_gpu_loop
from voicemem.engine import GpuLoop
from voicemem.engine.scheduling.admission import ResourceAdmission


class PagingBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.resources = ResourceAdmission(clock=lambda: self.now)

    def sample(self, at, rate):
        self.now = at
        self.resources.paging(rate)
        return self.resources.cache_pressure()

    def test_spikes_and_small_rates_preserve_normal_budget(self):
        self.assertFalse(self.resources.cache_pressure())
        for at, rate in ((0, 100), (1, 0), (2, 8), (3, 8), (4, 8)):
            self.assertFalse(self.sample(at, rate))

    def test_sustained_pressure_and_quiet_recovery_have_hysteresis(self):
        for at in range(2):
            self.assertFalse(self.sample(at, 20))
        self.assertTrue(self.sample(2, 20))
        for at in range(3, 8):
            self.assertTrue(self.sample(at, 0))
        self.assertFalse(self.sample(8, 0))

    def test_fresh_pressure_interrupts_quiet_recovery(self):
        for at in range(3):
            self.sample(at, 20)
        for at in range(3, 7):
            self.assertTrue(self.sample(at, 0))
        self.assertTrue(self.sample(7, 20))
        for at in range(8, 13):
            self.assertTrue(self.sample(at, 0))
        self.assertFalse(self.sample(13, 0))

    def test_stale_feedback_expires_and_cannot_activate_new_samples(self):
        for at in range(3):
            self.sample(at, 20)
        self.now = 5
        self.assertFalse(self.resources.cache_pressure())
        self.assertFalse(self.sample(5, 20))
        self.assertFalse(self.sample(6, 20))
        self.assertTrue(self.sample(7, 20))


class MLXCacheBudgetTests(unittest.TestCase):
    def test_only_free_cache_budget_changes_and_recovers(self):
        gpu = GpuLoop()
        native = SimpleNamespace(set_cache_limit=Mock())
        with patch('voicemem.engine.scheduling.admission.resources') as resources:
            resources.cache_pressure.return_value = True
            gpu._update_cache_budget(native, 0)
            gpu._update_cache_budget(native, .5)
            gpu._update_cache_budget(native, 1)
            native.set_cache_limit.assert_called_once_with(128 * 2**20)
            self.assertEqual(gpu.snapshot()['cache_limit'], 512 * 2**20)
            self.assertEqual(gpu.snapshot()['effective_cache_limit'], 128 * 2**20)
            resources.cache_pressure.return_value = False
            gpu._update_cache_budget(native, 2)
            self.assertEqual(native.set_cache_limit.call_args.args, (512 * 2**20,))

    def test_explicit_small_or_disabled_cache_is_never_increased(self):
        for limit in (0, 64 * 2**20, 128 * 2**20):
            with self.subTest(limit=limit):
                native = SimpleNamespace(set_cache_limit=Mock())
                with patch('voicemem.engine.scheduling.admission.resources') as resources:
                    resources.cache_pressure.return_value = True
                    gpu = GpuLoop(cache_limit=limit)
                    gpu._update_cache_budget(native, 0)
                    self.assertEqual(gpu.snapshot()['effective_cache_limit'], limit)
                    native.set_cache_limit.assert_not_called()

    def test_budget_failure_does_not_drop_model_output_and_retries(self):
        gpu = GpuLoop()
        native = SimpleNamespace(set_cache_limit=Mock(side_effect=RuntimeError('synthetic')))
        with patch('voicemem.engine.scheduling.admission.resources') as resources:
            resources.cache_pressure.return_value = True
            with self.assertLogs('voicemem.engine.backends.mlx.scheduler', level='WARNING'):
                gpu._update_cache_budget(native, 0)
            self.assertEqual(gpu.snapshot()['cache_budget_error'], 'RuntimeError')
            self.assertEqual(gpu.snapshot()['effective_cache_limit'], 512 * 2**20)
            native.set_cache_limit.side_effect = None
            gpu._update_cache_budget(native, 1)
            self.assertIsNone(gpu.snapshot()['cache_budget_error'])
            self.assertEqual(gpu.snapshot()['effective_cache_limit'], 128 * 2**20)

    def test_native_budget_changes_precede_work_on_the_owner_thread(self):
        events = []
        with cpu_gpu_loop() as gpu, \
                patch('voicemem.engine.scheduling.admission.resources') as resources, \
                patch('mlx.core.set_cache_limit', side_effect=lambda size: events.append(
                    ('budget', size, threading.get_ident()))):
            resources.cache_pressure.return_value = True
            def work():
                events.append(('model', gpu.snapshot()['effective_cache_limit'], threading.get_ident()))
                return 42
            self.assertEqual(gpu.call(work), 42)
            self.assertEqual([(kind, size) for kind, size, _ in events],
                             [('budget', 512 * 2**20), ('budget', 128 * 2**20), ('model', 128 * 2**20)])
            self.assertTrue(all(owner == gpu._thread.ident for _, _, owner in events))

    def test_native_budget_failure_leaves_worker_usable(self):
        with cpu_gpu_loop() as gpu, \
                patch('voicemem.engine.scheduling.admission.resources') as resources, \
                patch('mlx.core.set_cache_limit', side_effect=[None, RuntimeError('synthetic')]):
            resources.cache_pressure.return_value = True
            with self.assertLogs('voicemem.engine.backends.mlx.scheduler', level='WARNING'):
                self.assertEqual(gpu.call(lambda: 42), 42)
            self.assertEqual(gpu.call(lambda: 43), 43)


if __name__ == '__main__':
    unittest.main()
