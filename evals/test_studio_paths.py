"""Offline checks for Studio code, resource, and runtime path boundaries."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from studio.paths import ROOT, STUDIO, MODELS, voice_directory, pet_directory


class StudioPathTests(unittest.TestCase):
    def test_runtime_assets_are_studio_owned(self):
        self.assertTrue((STUDIO / "resources/voice/noctelle_ref_short.wav").is_file())
        self.assertTrue((STUDIO / "resources/voice/backchannel").is_dir())
        self.assertTrue((STUDIO / "pet/assets/live2d/rattan/rattan.model3.json").is_file())
        self.assertFalse((ROOT / "voice").exists())
        self.assertFalse((ROOT / "pet").exists())

    def test_models_remain_outside_the_source_package(self):
        self.assertEqual(STUDIO.parent, ROOT)
        self.assertEqual(MODELS, Path(os.environ.get("STUDIO_MODELS_DIR") or
                                     STUDIO / "models").expanduser().resolve())

    def test_voice_sources_prefer_studio_but_accept_explicit_external_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studio = root / "studio"
            studio.mkdir()
            legacy = root / "voice"
            self.assertEqual(voice_directory(env={}, studio=studio, root=root), legacy.resolve())
            bundled = studio / "resources/voice"
            bundled.mkdir(parents=True)
            self.assertEqual(voice_directory(env={}, studio=studio, root=root), bundled.resolve())
            external = root / "reviewed-voice"
            self.assertEqual(voice_directory(env={"STUDIO_VOICE_DIR": str(external)},
                                             studio=studio, root=root), external.resolve())

    def test_pet_source_prefers_studio_and_preserves_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studio = root / "studio"
            studio.mkdir()
            self.assertEqual(pet_directory(env={}, studio=studio, root=root), (root / "pet").resolve())
            (studio / "pet").mkdir()
            self.assertEqual(pet_directory(env={}, studio=studio, root=root), (studio / "pet").resolve())

    def test_dotenv_overrides_are_loaded_before_resource_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models, voice = root / "custom-models", root / "reviewed-voice"
            (root / ".env").write_text(
                f"STUDIO_MODELS_DIR={models}\nSTUDIO_VOICE_DIR={voice}\n")
            env = os.environ.copy()
            env.pop("STUDIO_MODELS_DIR", None)
            env.pop("STUDIO_VOICE_DIR", None)
            script = (
                "import sys; "
                "from studio.core.utils.environment.component import load_environment; "
                "load_environment(sys.argv[1]); "
                "from studio.paths import MODELS, VOICE; "
                "print(MODELS); print(VOICE)"
            )
            result = subprocess.run(
                [sys.executable, "-c", script, str(root)], cwd=ROOT, env=env,
                text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(models.resolve()), str(voice.resolve())])


if __name__ == "__main__":
    unittest.main()
