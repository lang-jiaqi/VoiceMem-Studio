"""Offline regressions for shared embedding initialization and Torch locking."""
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from voicemem.leftbrain import local_embedder
from voicemem.utils.torch_lock import TORCH_LOCK


class SharedModelTests(unittest.TestCase):
    def setUp(self):
        local_embedder.shared_model.cache_clear()
        self.env = patch.dict(os.environ, {"VOICEMEM_VERBOSE": "1"})
        self.env.start()

    def tearDown(self):
        local_embedder.shared_model.cache_clear()
        self.env.stop()

    def module(self, constructor):
        return patch.dict("sys.modules", {
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=constructor)})

    def test_omitted_positional_and_keyword_defaults_share_one_model(self):
        constructor = Mock(return_value=object())
        with self.module(constructor):
            first = local_embedder.shared_model("fixture")
            self.assertIs(local_embedder.shared_model("fixture", ()), first)
            self.assertIs(local_embedder.shared_model("fixture", tokenizer_kwargs=()), first)
            self.assertIs(local_embedder.shared_model(path="fixture"), first)
        constructor.assert_called_once()

    def test_construction_and_legacy_tokenizer_fallback_hold_torch_lock(self):
        calls = []
        def constructor(path, **kwargs):
            self.assertTrue(TORCH_LOCK._is_owned())
            calls.append(kwargs)
            if "processor_kwargs" in kwargs:
                raise TypeError("legacy fixture")
            return object()
        with self.module(constructor):
            first = local_embedder.shared_model("fixture")
            second = local_embedder.shared_model("fixture", (("padding_side", "left"),))
        self.assertIsNot(first, second)
        self.assertEqual(calls, [{}, {"processor_kwargs": {"padding_side": "left"}},
                                {"tokenizer_kwargs": {"padding_side": "left"}}])

    def test_simultaneous_cold_requests_construct_only_one_model(self):
        start = threading.Barrier(4)
        def constructor(path):
            time.sleep(.02)
            return object()
        create = Mock(side_effect=constructor)
        def load(index):
            start.wait(timeout=1)
            return (local_embedder.shared_model("fixture") if index % 2 else
                    local_embedder.shared_model("fixture", ()))
        with self.module(create), ThreadPoolExecutor(max_workers=4) as pool:
            models = list(pool.map(load, range(4)))
        self.assertTrue(all(model is models[0] for model in models))
        create.assert_called_once()

    def test_failed_load_is_not_cached_and_releases_torch_lock(self):
        constructor = Mock(side_effect=[RuntimeError("fixture failure"), object()])
        with self.module(constructor):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                local_embedder.shared_model("fixture")
            self.assertFalse(TORCH_LOCK._is_owned())
            model = local_embedder.shared_model("fixture")
            self.assertIs(local_embedder.shared_model("fixture", ()), model)
        self.assertEqual(constructor.call_count, 2)


if __name__ == "__main__":
    unittest.main()
