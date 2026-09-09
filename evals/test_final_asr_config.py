"""Final-ASR model selection regressions without loading an inference runtime."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from voicemem.utils.defaults import default_utils


class FinalAsrConfigTests(unittest.TestCase):
    def test_int8_directory_is_the_zero_config_default(self):
        name = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "asr" / name
            model.mkdir(parents=True)
            (model / "tokens.txt").touch()
            (model / "model.int8.onnx").touch()
            fake = types.ModuleType("voicemem.utils.audio.asr")
            fake.OfflineASR = lambda path: path
            env = {"VOICEMEM_MODELS_DIR": directory, "VOICEMEM_FINAL_ASR": "1"}
            with patch.dict(os.environ, env, clear=False), \
                    patch.dict(sys.modules, {"voicemem.utils.audio.asr": fake}):
                os.environ.pop("VOICEMEM_FINAL_ASR_DIR", None)
                selected = default_utils(None, directory)["asr_final"]()
        self.assertEqual(Path(selected).name, name)

    def test_explicit_disable_skips_model_selection(self):
        with patch.dict(os.environ, {"VOICEMEM_FINAL_ASR": "0"}):
            self.assertIsNone(default_utils(None, "unused")["asr_final"]())


if __name__ == "__main__":
    unittest.main()
