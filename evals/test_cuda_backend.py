"""Backend selection and local CUDA streaming contracts without model inference."""
import asyncio
from contextlib import aclosing
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from studio.core.utils.cli.component import parse_args
from studio.core.utils.environment.component import load_environment
from studio.core.utils.models.initialize import models
from studio.core.utils.startup.initialize import required_credentials
from studio.core.utils.tts.cuda import BreezeCUDATTS


class BackendConfigTests(unittest.TestCase):
    def test_plain_launch_selects_platform_backend_and_preserves_defaults(self):
        for system, backend, device in [('Linux', 'cuda', 'cuda:0'), ('Darwin', 'mlx', 'cpu')]:
            with self.subTest(system=system), patch.dict(os.environ, {}, clear=True), \
                    patch('studio.core.utils.cli.component.platform.system', return_value=system):
                args = parse_args([])
                self.assertEqual((args.backend, args.device, args.tts_device),
                                 (backend, device, device))
                self.assertEqual((args.mode, args.llm, args.space, args.lang, args.port),
                                 ('llm_tts', 'deepseek', 'studio-zh', 'zh', 8787))

    def test_explicit_space_still_overrides_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parse_args(['--space', 'demo-zh']).space, 'demo-zh')

    def test_shell_launchers_enable_verbose_and_forward_arguments(self):
        root = Path(__file__).resolve().parents[1]
        for backend in ('cuda', 'mlx'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run(
                    ['bash', str(root / f'scripts/run_studio_{backend}.sh'), '--check'],
                    cwd=cwd, env={**os.environ, 'STUDIO_PYTHON': '/bin/echo'},
                    text=True, capture_output=True, check=True)
                self.assertEqual(result.stdout.strip(),
                                 f'-m studio --backend {backend} --verbose --check')

    def test_module_and_legacy_launchers_use_the_same_entry_point(self):
        root = Path(__file__).resolve().parents[1]
        with patch('studio.core.core.main') as main, patch.object(sys, 'path', sys.path.copy()):
            runpy.run_module('studio', run_name='__main__')
            main.assert_called_once_with()
            main.reset_mock()
            runpy.run_path(str(root / 'web/run.py'), run_name='__main__')
            main.assert_called_once_with()

    def test_local_env_keeps_exported_credentials_and_handles_quotes(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, '.env').write_text('DEEPSEEK_API_KEY="file-key"\nSTUDIO_DEVICE=cuda:1\n')
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'exported-key'}, clear=True):
                load_environment(root)
                self.assertEqual(os.environ['DEEPSEEK_API_KEY'], 'exported-key')
                self.assertEqual(os.environ['STUDIO_DEVICE'], 'cuda:1')

    def test_cli_overrides_backend_and_devices(self):
        with patch.dict(os.environ, {'STUDIO_BACKEND': 'mlx', 'STUDIO_DEVICE': 'cpu'}, clear=True):
            args = parse_args(['--backend', 'cuda', '--device', 'cuda:1', '--tts-device', 'cuda:0'])
        self.assertEqual((args.backend, args.device, args.tts_device), ('cuda', 'cuda:1', 'cuda:0'))

    def test_credentials_follow_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            for provider, key in [('deepseek', 'DEEPSEEK_API_KEY'), ('openai', 'OPENAI_API_KEY'),
                                  ('qwen', 'DASHSCOPE_API_KEY')]:
                args = parse_args(['--backend', 'cuda', '--llm', provider])
                self.assertEqual(set(required_credentials(args)), {key})
            args = parse_args(['--mode', 'realtime'])
            self.assertEqual(set(required_credentials(args)), {'OPENAI_API_KEY'})

    def test_cuda_model_list_excludes_mlx_and_includes_codec(self):
        with patch.dict(os.environ, {}, clear=True):
            cuda = models(parse_args(['--backend', 'cuda']))
            mlx = models(parse_args(['--backend', 'mlx']))
        self.assertFalse(any('mlx' in item.repository.lower() for item in cuda))
        breeze = next(item for item in cuda if item.name == 'Breeze CUDA')
        self.assertIn('audio_tokenizer/*.safetensors', breeze.required)
        self.assertTrue(any('Breeze-TTS-2-mlx' in item.directory for item in mlx))


class CudaStreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = BreezeCUDATTS(instruction='test')

    async def asyncTearDown(self):
        await self.provider.aclose()

    async def test_voicemem_resolves_factory_to_shared_provider(self):
        from voicemem.orchestrator import Utils
        from studio.core.utils.tts.initialize import create, _create
        _create.cache_clear()
        try:
            with patch.dict(os.environ, {'STUDIO_BACKEND': 'cuda'}), \
                    patch('studio.core.utils.tts.cuda.BreezeCUDATTS', return_value=self.provider):
                first = Utils('multi_modal', None, 'unused', {'tts': create})
                second = Utils('multi_modal', None, 'unused', {'tts': create})
                self.assertIs(first.get('tts'), self.provider)
                self.assertIs(second.get('tts'), self.provider)
        finally:
            _create.cache_clear()

    async def test_ordered_pcm_and_exception_propagation(self):
        def chunks(text, instruction, cancelled):
            yield b'\x01\x00'
            yield b'\x02\x00'
        self.provider._segments = chunks
        self.assertEqual([pcm async for pcm in self.provider.stream('test')],
                         [b'\x01\x00', b'\x02\x00'])

        def fail(*args):
            yield b'\x00\x00'
            raise RuntimeError('synthesis failed')
        self.provider._segments = fail
        with self.assertRaisesRegex(RuntimeError, 'synthesis failed'):
            async for _ in self.provider.stream('test'):
                pass

    async def test_empty_text_never_loads_models(self):
        with patch.object(self.provider, '_segments') as generate:
            self.assertEqual([pcm async for pcm in self.provider.stream(' ')], [])
            generate.assert_not_called()

    async def test_early_close_releases_bounded_buffer_and_serial_worker(self):
        finished = threading.Event()
        calls = []

        def chunks(text, instruction, cancelled):
            calls.append(text)
            try:
                for _ in range(100):
                    if cancelled.is_set():
                        return
                    yield b'\x00\x00'
            finally:
                finished.set()

        self.provider._segments = chunks
        async with aclosing(self.provider.stream('first')) as stream:
            self.assertEqual(await anext(stream), b'\x00\x00')
        self.assertTrue(await asyncio.to_thread(finished.wait, 2))
        async with aclosing(self.provider.stream('second')) as stream:
            await asyncio.wait_for(anext(stream), 2)
        self.assertEqual(calls, ['first', 'second'])

    async def test_cancel_before_first_audio_wakes_consumer(self):
        started, finished = threading.Event(), threading.Event()

        def chunks(text, instruction, cancelled):
            started.set()
            try:
                cancelled.wait(2)
                if not cancelled.is_set():
                    yield b'\x00\x00'
            finally:
                finished.set()

        self.provider._segments = chunks

        async def consume():
            async with aclosing(self.provider.stream('first')) as stream:
                await anext(stream)

        task = asyncio.create_task(consume())
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(await asyncio.to_thread(finished.wait, 2))


if __name__ == '__main__':
    unittest.main()
