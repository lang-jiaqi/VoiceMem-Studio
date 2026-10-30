"""Studio selects verified decoding and consistent local repetition control."""
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from studio.core.utils.tts import initialize
from studio.core.utils.tts.cache import FastDepth


class SpeechFactoryTests(unittest.TestCase):
    def test_mlx_default_selects_verified_compilation_and_repetition_control(self):
        with patch.dict(os.environ, {'STUDIO_BACKEND': 'mlx'}, clear=True), \
                patch('studio.core.utils.tts.component.BreezeMLXTTS') as create:
            result = initialize._create.__wrapped__()
        self.assertIs(result, create.return_value)
        create.assert_called_once_with(model=str(initialize.MODEL),
            ref_audio=str(initialize.REFERENCE), cfg_scale=1.0, seed=42,
            chunk_frames=2, first_frames=2, max_tokens=750, depth_mode='compiled',
            repetition_penalty=1.1)

    def test_cuda_uses_its_existing_provider_and_device(self):
        with patch.dict(os.environ, {'STUDIO_BACKEND': 'cuda', 'STUDIO_TTS_DEVICE': 'cuda:1'},
                        clear=True), patch('studio.core.utils.tts.cuda.BreezeCUDATTS') as create:
            result = initialize._create.__wrapped__()
        self.assertIs(result, create.return_value)
        create.assert_called_once_with(ref_audio=str(initialize.REFERENCE), device='cuda:1')


class DepthWarmupTests(unittest.TestCase):
    def test_compilation_checks_real_activation_dtype_and_cfg_batch(self):
        for cfg, batch in ((1.0, 1), (1.5, 2)):
            with self.subTest(cfg=cfg):
                mx = ModuleType('mlx.core')
                mx.int32 = np.int32
                mx.array = np.array
                mx.zeros = np.zeros
                mx.concatenate = np.concatenate
                mx.eval = Mock()
                mx.random = SimpleNamespace(seed=Mock())
                hidden = np.full((1, 1, 8), .5, dtype=np.float32)
                backbone = Mock(return_value=hidden)
                backbone.make_cache.return_value = object()
                model = SimpleNamespace(backbone_model=backbone, num_codebooks=16,
                                        _depth_tokens=Mock(),
                                        _prompt_embeddings=Mock(return_value=hidden))
                depth = FastDepth(model, mode='compiled')
                profile = Mock()
                depth._function = Mock(return_value=profile)
                with patch.dict(sys.modules, {'mlx.core': mx, 'mlx': ModuleType('mlx')}):
                    depth.warmup(2, cfg, 'synthetic instruction')
                sample = profile.verify.call_args.args[1]
                self.assertEqual((sample.dtype, sample.shape), (np.dtype('float32'), (batch, 8)))
                np.testing.assert_array_equal(sample, np.full((batch, 8), .5))
                self.assertEqual(profile.verify.call_args.kwargs, {'seed': 0})
                self.assertEqual(model._depth_tokens.call_count, 2)
                prefill, decode = backbone.call_args_list
                self.assertIs(prefill.kwargs['cache'], decode.kwargs['cache'])
                self.assertIs(prefill.kwargs['input_embeddings'], hidden)
                mx.random.seed.assert_called_once_with(0)


if __name__ == '__main__':
    unittest.main()
