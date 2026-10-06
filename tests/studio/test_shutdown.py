"""Process-level termination checks without models, credentials or sockets."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]


SCRIPT = textwrap.dedent('''
    import atexit
    import multiprocessing
    import os
    from pathlib import Path
    import signal
    import sys
    import threading
    import types
    from unittest.mock import patch

    import uvicorn
    from studio.core.core import main

    mode = sys.argv[1]
    lock = multiprocessing.get_context('spawn').RLock()
    atexit.register(lambda: print('PYTHON_FINALIZED', flush=True))

    class Speech:
        async def aclose(self):
            print('TTS_CLOSED', flush=True)
            if mode == 'cleanup-error':
                raise RuntimeError('cleanup failed')

    class Agent:
        def __init__(self, args):
            self.vm = types.SimpleNamespace(utils={'tts': Speech()})

        def warmup(self):
            if mode == 'warmup-term':
                os.kill(os.getpid(), signal.SIGTERM)
            if mode == 'warmup-error':
                raise RuntimeError('warmup failed')

    events = []
    app = types.SimpleNamespace(router=types.SimpleNamespace(
        add_event_handler=lambda name, callback: events.append((name, callback))))
    args = types.SimpleNamespace(prepare_stage='', check=mode == 'check',
        no_file_log=True, mode='llm_tts', host='127.0.0.1', port=8787)

    def serve(*args, **kwargs):
        server = uvicorn.Server(uvicorn.Config(app))
        with server.capture_signals():
            os.kill(os.getpid(), signal.SIGINT if mode == 'interrupt' else signal.SIGTERM)
            print('SERVER_SHUTDOWN', flush=True)
        print('SERVER_RETURNED', flush=True)

    def module(name, **values):
        result = types.ModuleType(name)
        result.__dict__.update(values)
        return result

    modules = {
        'studio.core.utils.environment.component': module('environment',
            load_environment=lambda: None, prepare=lambda args: None),
        'studio.core.utils.startup.initialize': module('startup', inspect=lambda *args: None),
        'studio.core.utils.models.initialize': module('models', acquire_all=lambda *args: None),
        'studio.paths': module('paths', ROOT=Path('.')),
        'voicemem.utils.common.prompt_trace': module('trace', configure=lambda *args: None),
        'studio.core.voiceagent': module('voiceagent', VoiceAgent=Agent),
    }
    if mode == 'custom':
        def caller_handler(signum, frame):
            print('CALLER_HANDLER', flush=True)
        signal.signal(signal.SIGTERM, caller_handler)
    original = signal.getsignal(signal.SIGTERM)

    def run():
        try:
            main([])
        finally:
            print('HANDLER_RESTORED=' + str(signal.getsignal(signal.SIGTERM) == original), flush=True)

    with patch.dict(sys.modules, modules), patch.dict(os.environ, {'STUDIO_PUBLIC_DEMO': '0'}), \\
            patch('studio.core.core.parse_args', return_value=args), \\
            patch('studio.core.core.build_app', return_value=app), patch('uvicorn.run', side_effect=serve):
        if mode == 'thread':
            args.check = True
            worker = threading.Thread(target=run)
            worker.start()
            worker.join()
        else:
            run()
''')


@unittest.skipUnless(os.name == 'posix', 'Requires POSIX signal delivery')
class ShutdownTests(unittest.TestCase):
    def run_case(self, mode):
        result = subprocess.run([sys.executable, '-c', SCRIPT, mode], cwd=ROOT,
                                text=True, capture_output=True, timeout=20)
        self.assertIn('PYTHON_FINALIZED', result.stdout)
        self.assertIn('HANDLER_RESTORED=True', result.stdout)
        self.assertNotIn('leaked semaphore', result.stderr)
        return result

    def test_sigterm_finishes_server_and_python_cleanup(self):
        result = self.run_case('terminate')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SERVER_SHUTDOWN', result.stdout)
        self.assertIn('TTS_CLOSED', result.stdout)
        self.assertNotIn('SERVER_RETURNED', result.stdout)

    def test_termination_during_warmup_still_closes_tts(self):
        result = self.run_case('warmup-term')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('TTS_CLOSED', result.stdout)
        self.assertNotIn('SERVER_SHUTDOWN', result.stdout)

    def test_keyboard_interrupt_retains_its_exit_behavior(self):
        result = self.run_case('interrupt')
        self.assertEqual(result.returncode, -signal.SIGINT, result.stderr)
        self.assertIn('TTS_CLOSED', result.stdout)
        self.assertIn('KeyboardInterrupt', result.stderr)

    def test_caller_installed_handler_is_preserved(self):
        result = self.run_case('custom')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('CALLER_HANDLER'), 1)
        self.assertIn('SERVER_RETURNED', result.stdout)

    def test_check_returns_without_serving(self):
        result = self.run_case('check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('SERVER_SHUTDOWN', result.stdout)

    def test_thread_entry_does_not_install_signal_handlers(self):
        result = self.run_case('thread')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('Exception in thread', result.stderr)

    def test_startup_failure_keeps_failure_exit_code(self):
        result = self.run_case('warmup-error')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('warmup failed', result.stdout)
        self.assertIn('TTS_CLOSED', result.stdout)

    def test_cleanup_failure_is_not_suppressed(self):
        result = self.run_case('cleanup-error')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('cleanup failed', result.stderr)


if __name__ == '__main__':
    unittest.main()
