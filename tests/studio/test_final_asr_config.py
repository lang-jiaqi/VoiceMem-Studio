"""Final-ASR model selection regressions without loading an inference runtime."""
from __future__ import annotations

import os
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voicemem.utils.defaults import default_utils
from voicemem.utils.audio.asr import OfflineASR


class FinalAsrConfigTests(unittest.TestCase):
    def test_studio_accepts_both_input_languages_and_cantonese(self):
        from studio.core.utils.asr import initialize
        with patch.object(initialize, "OfflineASR") as create:
            initialize.final()
        create.assert_called_once_with(
            str(initialize.MODELS / "asr/sensevoice-int8"),
            allowed_languages=("zh", "en", "yue"))

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


class FinalAsrLanguageTests(unittest.TestCase):
    def make_asr(self, text, language, *, allowed_languages=("zh", "en", "yue")):
        stream = types.SimpleNamespace(
            result=types.SimpleNamespace(text=text, lang=language),
            accept_waveform=Mock())
        recognizer = types.SimpleNamespace(create_stream=Mock(return_value=stream),
                                          decode_stream=Mock())
        create = Mock(return_value=recognizer)
        fake = types.SimpleNamespace(
            OfflineRecognizer=types.SimpleNamespace(from_sense_voice=create))
        with patch.dict(sys.modules, {"sherpa_onnx": fake}):
            asr = OfflineASR("unused-model", allowed_languages=allowed_languages)
        self.assertNotIn("language", create.call_args.kwargs)
        return asr, stream, recognizer

    def test_allowed_languages_preserve_complete_transcript_and_decode_once(self):
        for language, text in (("<|zh|>", "今天过得怎么样？"),
                               ("en", "How was your day?"),
                               ("<|yue|>", "今日過得點呀？"),
                               ("<|en|>", "We can discuss attention 机制.")):
            with self.subTest(language=language):
                asr, stream, recognizer = self.make_asr(text, language)
                samples = np.ones(160, dtype=np.float32)
                self.assertEqual(asr.transcribe(samples), text)
                recognizer.decode_stream.assert_called_once_with(stream)
                np.testing.assert_array_equal(stream.accept_waveform.call_args.args[1], samples)

    def test_japanese_korean_and_unknown_tags_return_no_refinement(self):
        for language in ("<|ja|>", "ko", "", None, "unexpected"):
            with self.subTest(language=language):
                asr, stream, recognizer = self.make_asr("misclassified speech", language)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(asr.transcribe(np.ones(160)), "")
                recognizer.decode_stream.assert_called_once_with(stream)
                self.assertIn("streaming transcript fallback", output.getvalue())
                self.assertNotIn("misclassified speech", output.getvalue())

    def test_reusable_recognizer_still_accepts_japanese_without_allowlist(self):
        asr, _, _ = self.make_asr("こんにちは", "<|ja|>", allowed_languages=None)
        self.assertEqual(asr.transcribe(np.ones(160)), "こんにちは")

    def test_empty_audio_does_not_decode_and_empty_output_does_not_warn(self):
        asr, _, recognizer = self.make_asr("", "<|ja|>")
        self.assertEqual(asr.transcribe([]), "")
        recognizer.create_stream.assert_not_called()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(asr.transcribe(np.ones(160)), "")
        self.assertEqual(output.getvalue(), "")

    def test_invalid_allowlist_fails_before_loading_model(self):
        for values in ((), ("auto",), ("invalid",), "en"):
            with self.subTest(values=values), patch.dict(sys.modules, {"sherpa_onnx": None}):
                with self.assertRaisesRegex(ValueError, "allowed_languages"):
                    OfflineASR("unused-model", allowed_languages=values)


if __name__ == "__main__":
    unittest.main()
