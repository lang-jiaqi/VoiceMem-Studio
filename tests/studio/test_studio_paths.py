"""Offline checks for Studio code, resource, and runtime path boundaries."""
from pathlib import Path
import json
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


class StudioSpaceIsolationTests(unittest.TestCase):
    """Exercise exact-root compatibility without models or existing user data."""

    def agent(self, memory_root=""):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from studio.core.utils.spaces.component import Spaces
        agent = Spaces()
        agent.ARGS = SimpleNamespace(space="mine", lang="zh", llm="deepseek")
        agent.CONFIG = {"memory_root": str(memory_root), "reply": None}
        agent._SPACES = {}
        agent._LOCAL_LLM = None
        agent.ACTIVE_SPACE = ""
        agent._rt_persona = Mock(return_value="persona")
        return agent

    def test_explicit_root_uses_existing_metadata_and_counts_in_place(self):
        import json
        import sqlite3
        from unittest.mock import Mock, patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "existing-store"
            root.mkdir()
            metadata = root / "old-name.json"
            original = json.dumps({"space": {"language": "en"}, "existing": "preserve"})
            metadata.write_text(original)
            with sqlite3.connect(root / "old-name.sqlite") as db:
                db.execute("CREATE TABLE memories (id INTEGER)")
                db.execute("INSERT INTO memories VALUES (1)")
            agent = self.agent(root)
            memory = Mock()
            with patch("studio.core.utils.spaces.component.open_memory", return_value=memory) as opened:
                self.assertEqual(agent.use_space("mine"), "mine")
                self.assertIs(agent.vm, memory)
                self.assertEqual(agent.use_space("mine"), "mine")
                opened.assert_called_once()
                config = opened.call_args.args[0]
                self.assertEqual(config["memory_root"], str(root.resolve()))
                self.assertEqual(config["memory_language"], "en")
            listed = agent.list_spaces()
            self.assertEqual([(s["id"], s["count"], s["fixed"]) for s in listed], [("mine", 1, True)])
            self.assertEqual(metadata.read_text(), original)
            self.assertFalse((root / "mine").exists())
            self.assertFalse((root / "mine.json").exists())

    def test_fixed_root_rejects_other_spaces_before_side_effects(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "existing-store"
            root.mkdir()
            agent = self.agent(root)
            with patch("studio.core.utils.spaces.component.open_memory") as opened:
                for action in (agent.get_space, agent.use_space, agent.create_space):
                    with self.subTest(action=action.__name__), self.assertRaisesRegex(ValueError, "单个记忆库"):
                        action("another")
                with self.assertRaisesRegex(ValueError, "单个记忆库"):
                    agent.create_space("mine")
                opened.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(agent.ACTIVE_SPACE, "")

    def test_default_named_spaces_remain_separate_and_selectable(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent()
            with patch("studio.core.utils.spaces.component._ROOT", Path(directory)), \
                    patch("studio.core.utils.spaces.component.open_memory", side_effect=[object(), object()]) as opened:
                agent.use_space("mine")
                first = agent.vm
                agent.use_space("other")
                self.assertIsNot(agent.vm, first)
                self.assertEqual([call.args[0]["space"] for call in opened.call_args_list], ["mine", "other"])
                self.assertNotEqual(agent.space_dir("mine")[0], agent.space_dir("other")[0])
                self.assertEqual({item["id"] for item in agent.list_spaces()}, {"mine", "other"})
                self.assertTrue(all(not item["fixed"] for item in agent.list_spaces()))
                self.assertTrue(all(call.args[0]["memory_root"] == "" for call in opened.call_args_list))

    def test_http_fixed_space_contract_rejects_switch_and_creation(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from studio.web.transport import build_app
        with tempfile.TemporaryDirectory() as directory, patch("studio.web.transport.PetSupervisor"):
            agent = self.agent(directory)
            agent.ACTIVE_SPACE = "mine"
            app = build_app("llm_tts", session=lambda _: None, classify=lambda _: None,
                            spaces=(agent.list_spaces, agent.create_space, agent.use_space, lambda: agent.ACTIVE_SPACE))
            with TestClient(app) as client:
                state = client.get("/api/spaces").json()
                self.assertEqual(state["active"], "mine")
                self.assertTrue(state["spaces"][0]["fixed"])
                self.assertEqual(client.post("/api/spaces", json={"name": "other"}).status_code, 400)
                self.assertEqual(client.post("/api/spaces/other/use").status_code, 400)
                self.assertEqual(agent.ACTIVE_SPACE, "mine")

    def test_normal_studio_does_not_activate_demo_accounts(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        from studio.web.transport import build_app
        with patch("studio.web.transport.PetSupervisor"):
            app = build_app("llm_tts", session=lambda _: None, classify=lambda _: None)
            with TestClient(app) as client:
                for route in ("/auth/me", "/pet/avatar-controller.js"):
                    self.assertEqual(client.get(route).status_code, 404, route)
                home = client.get("/", headers={"User-Agent": "iPhone Mobile"}, follow_redirects=False)
                self.assertEqual(home.status_code, 200)

    def test_demo_rejects_unsupported_launches_before_model_work(self):
        from unittest.mock import patch
        from studio.core.core import main
        for flags in (("--host", "0.0.0.0"), ("--llm", "local"),
                      ("--mode", "realtime")):
            with self.subTest(flags=flags), \
                    patch.dict(os.environ, {"STUDIO_PUBLIC_DEMO": "1"}), \
                    patch("studio.core.utils.environment.component.load_environment"), \
                    patch("studio.core.utils.environment.component.prepare") as prepare:
                with self.assertRaises(SystemExit) as stopped:
                    main(["--check", *flags])
                self.assertEqual(stopped.exception.code, 1)
                prepare.assert_not_called()


class PackageResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = '''
import contextlib, io, json, tempfile, warnings
from pathlib import Path
from setuptools import Distribution
from setuptools.config.pyprojecttoml import apply_configuration
with warnings.catch_warnings(), tempfile.TemporaryDirectory() as temporary:
    warnings.simplefilter("ignore")
    with contextlib.redirect_stdout(io.StringIO()):
        distribution = Distribution()
        distribution.script_name = "setup.py"
        apply_configuration(distribution, str(Path("pyproject.toml").resolve()))
        distribution.command_options["egg_info"] = {"egg_base": ("test", temporary)}
        build = distribution.get_command_obj("build_py")
        build.ensure_finalized()
        files = sorted(str(Path(source) / file)
                       for package, source, target, names in build._get_data_files()
                       for file in names)
    print(json.dumps({"files": files, "packages": distribution.packages}))
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                text=True, capture_output=True, check=True)
        cls.metadata = json.loads(result.stdout)

    def test_sdk_runtime_prompts_and_sample_audio_are_included(self):
        included = set(self.metadata["files"])
        for path in (ROOT / "voicemem/leftbrain/data").glob("*.txt"):
            self.assertIn(str(path.relative_to(ROOT)), included)
        for name in ("speech", "input", "question", "cafe_song"):
            self.assertIn(f"voicemem/assets/{name}.wav", included)
        self.assertIn("studio/prompt/tts.json", included)
        self.assertIn("studio/prompt/llm_context.json", included)
        self.assertIn("studio/prompt/llm_system_zh.md", included)

    def test_existing_sdk_studio_helpers_remain_importable(self):
        packages = set(self.metadata["packages"])
        for package in ("voicemem", "studio.core.utils.tts",
                        "studio.core.utils.prompts"):
            self.assertIn(package, packages)
        self.assertFalse(any(name.startswith(("studio.models", "studio.apps",
                                             "studio.pet", "studio.resources"))
                             for name in packages))

    def test_wheel_excludes_partial_frontends_and_runtime_data(self):
        forbidden = ("studio/apps/", "studio/web/", "studio/pet/",
                     "studio/resources/", "studio/models/", "prompt/logs/",
                     "voicemem_memoryspace/", "results/")
        self.assertFalse([file for file in self.metadata["files"]
                          if file.startswith(forbidden)])


if __name__ == "__main__":
    unittest.main()
