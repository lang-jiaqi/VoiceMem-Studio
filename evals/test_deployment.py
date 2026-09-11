"""Offline deployment contracts; never build images or start real services."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DeploymentTests(unittest.TestCase):
    def test_build_context_excludes_private_runtime_data(self):
        rules = (ROOT / '.dockerignore').read_text().splitlines()
        self.assertIn('**', rules)
        for rule in ('**/.env', '**/.env.*', '**/.venv*', 'prompt/logs',
                     'results', 'voicemem_memoryspace', 'studio/models', 'models'):
            self.assertIn(rule, rules)
        dockerfile = (ROOT / 'docker/Dockerfile.cuda').read_text()
        self.assertNotIn('COPY . .', dockerfile)
        self.assertNotIn('COPY .env', dockerfile)
        self.assertIn('USER studio', dockerfile)
        self.assertIn('python docker/smoke_check.py', dockerfile)
        self.assertIn('ln -s /home/studio/.cache/studio cache', dockerfile)

    def test_image_uses_pinned_breeze_source_and_shared_studio_entry(self):
        text = (ROOT / 'docker/Dockerfile.cuda').read_text()
        self.assertRegex(text, r'ARG BREEZE_REV=[0-9a-f]{40}\n')
        self.assertIn('torch==2.8.0', text)
        self.assertIn("'.[studio-cuda]'", text)
        self.assertIn('ENTRYPOINT ["python", "-m", "studio"]', text)
        self.assertIn('CMD ["--verbose"]', text)

    def test_cuda_dependency_pins_match_the_image_profile(self):
        import tomllib
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        cuda = config['project']['optional-dependencies']['studio-cuda']
        dockerfile = (ROOT / 'docker/Dockerfile.cuda').read_text()
        for requirement in ('torch==2.8.0', 'torchaudio==2.8.0', 'torchvision==0.23.0'):
            self.assertIn(requirement, cuda)
            self.assertIn(requirement, dockerfile)
        smoke = (ROOT / 'docker/smoke_check.py').read_text()
        self.assertIn("torch.version.cuda != '12.8'", smoke)
        self.assertIn("metadata.version(name)", smoke)
        self.assertNotIn('12.4', smoke)

    @unittest.skipUnless(shutil.which('docker'), 'Docker CLI is not installed')
    def test_compose_is_valid_and_preserves_gpu_and_data_boundaries(self):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('COMPOSE_', 'STUDIO_'))}
        env['STUDIO_ENV_FILE'] = str(ROOT / '.env.example')
        env.pop('TZ', None)
        result = subprocess.run(
            ['docker', 'compose', '--env-file', str(ROOT / '.env.example'),
             '-f', str(ROOT / 'compose.yaml'), 'config', '--format', 'json'],
            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        config = json.loads(result.stdout)
        service = config['services']['studio']
        self.assertEqual(service['image'], 'voicemem-studio:torch2.8-cu128')
        devices = service['deploy']['resources']['reservations']['devices']
        self.assertEqual(devices, [{'capabilities': ['gpu'], 'device_ids': ['0'], 'driver': 'nvidia'}])
        self.assertEqual(service['environment']['STUDIO_DEVICE'], 'cuda:0')
        self.assertEqual(service['environment']['STUDIO_TTS_DEVICE'], 'cuda:0')
        self.assertEqual(service['environment']['TZ'], 'Asia/Shanghai')
        self.assertEqual(service['ports'][0]['host_ip'], '127.0.0.1')
        self.assertEqual(service['ports'][0]['target'], 8787)
        self.assertEqual(service['environment']['STUDIO_DESKTOP_PET'], '0')
        self.assertEqual(sum(volume['type'] == 'volume' for volume in service['volumes']), 5)
        settings = [volume for volume in service['volumes'] if volume['type'] == 'bind']
        self.assertEqual(len(settings), 2)
        self.assertTrue(all(volume['read_only'] for volume in settings))
        self.assertEqual({volume['target'] for volume in settings}, {
            '/opt/voicemem-studio/studio/harness', '/opt/voicemem-studio/prompt/tts.json'})
        self.assertTrue(service['init'])
        self.assertNotIn('privileged', service)

    def test_mlx_setup_rejects_linux_before_modifying_environment(self):
        if os.uname().sysname == 'Darwin':
            self.skipTest('Linux rejection test')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['bash', str(ROOT / 'scripts/setup_studio_mlx.sh')],
                                    cwd=directory, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('原生 Apple Silicon', result.stderr)

    def test_native_mlx_setup_installs_upstream_pet_dependencies(self):
        script = (ROOT / 'scripts/setup_studio_mlx.sh').read_text()
        self.assertIn('npm ci --prefix "$studio_root/pet" --include=dev', script)

    def test_headless_mode_never_starts_a_desktop_process(self):
        from unittest.mock import patch
        from studio.web.pet_bridge import PetSupervisor
        with patch.dict(os.environ, {'STUDIO_DESKTOP_PET': '0'}), \
                patch('studio.web.pet_bridge.subprocess.Popen') as spawn:
            PetSupervisor().ensure_running('ws://127.0.0.1:8787/ws-pet')
        spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
