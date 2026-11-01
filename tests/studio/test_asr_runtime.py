"""CPU session configuration, packaging and default provider isolation."""
from importlib import resources
from pathlib import Path
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from studio.core.utils.asr.component import XASRStreamingASR, cpu_provider
from voicemem.utils.audio.asr import OfflineASR, StreamingASR


class CpuSessionTests(unittest.TestCase):
    def test_packaged_options_stop_only_post_run_spinning(self):
        config = resources.files("studio.core.utils.asr").joinpath("cpu.config")
        self.assertEqual(config.read_text().splitlines(), [
            "SessionConfig.session.force_spinning_stop=1"])
        provider, path = cpu_provider().split(":", 1)
        self.assertEqual(provider, "cpu")
        self.assertTrue(Path(path).is_file())
        self.assertEqual(Path(path).read_text(), config.read_text())

    def test_missing_configuration_fails_before_entering_native_code(self):
        with patch("studio.core.utils.asr.component.Path.is_file", return_value=False), \
                patch("sherpa_onnx.OnlineRecognizer.from_transducer") as native:
            with self.assertRaisesRegex(FileNotFoundError, "CPU session configuration"):
                XASRStreamingASR("synthetic")
        native.assert_not_called()

    def test_streaming_weights_and_threads_keep_their_original_defaults(self):
        native = SimpleNamespace(create_stream=Mock())
        with patch("sherpa_onnx.OnlineRecognizer.from_transducer", return_value=native) as create, \
                patch.object(StreamingASR, "_pick", return_value="synthetic.onnx"):
            StreamingASR("synthetic")
            self.assertEqual(create.call_args.kwargs["provider"], "cpu")
            self.assertEqual(create.call_args.kwargs["num_threads"], 2)
            XASRStreamingASR("synthetic")
            options = create.call_args.kwargs
            self.assertEqual(options["provider"], cpu_provider())
            self.assertEqual(options["num_threads"], 1)
            self.assertEqual(options["decoding_method"], "greedy_search")
            self.assertEqual(options["model_type"], "zipformer2")

    def test_final_provider_is_explicit_and_retains_itn_and_language_detection(self):
        for provider in ("cpu", cpu_provider()):
            with self.subTest(provider=provider), \
                    patch("sherpa_onnx.OfflineRecognizer.from_sense_voice") as create:
                OfflineASR("synthetic", provider=provider)
                options = create.call_args.kwargs
                self.assertEqual(options["provider"], provider)
                self.assertEqual(options["num_threads"], 4)
                self.assertTrue(options["use_itn"])
                self.assertNotIn("language", options)

    def test_wheel_metadata_includes_options_and_requires_the_native_protocol(self):
        root = Path(__file__).resolve().parents[2]
        project = tomllib.loads((root / "pyproject.toml").read_text())
        patterns = project["tool"]["setuptools"]["package-data"]["studio.core.utils.asr"]
        self.assertIn("*.config", patterns)
        dependencies = project["project"]["dependencies"]
        self.assertIn("sherpa-onnx>=1.13.8", dependencies)
