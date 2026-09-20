"""Regression checks for Studio's small VoiceMem integration boundary."""
import unittest
from unittest.mock import patch

from studio.core.voicemem import shared_embed_model


class SharedEmbeddingTests(unittest.TestCase):
    def test_perception_uses_the_memory_embedding_cache(self):
        from voicemem.leftbrain import local_embedder

        model = object()
        spec = object()
        with patch.object(local_embedder, "resolve", return_value=spec) as resolve, \
             patch.object(local_embedder, "resolve_path", return_value="cached-model") as path, \
             patch.object(local_embedder, "shared_model", return_value=model) as shared:
            self.assertIs(shared_embed_model(), model)

        resolve.assert_called_once_with()
        path.assert_called_once_with(spec)
        shared.assert_called_once_with("cached-model")

    def test_web_compatibility_alias_uses_the_same_bridge(self):
        from studio.web import transport

        self.assertIs(transport.shared_embed_model, shared_embed_model)

    def test_realtime_backchannel_uses_the_speech_provider_voice(self):
        from studio.core.utils.speech.component import Speech

        speech = Speech()
        speech.MODE = "realtime"
        speech._TTS_SHARED_VOICES = {"alloy"}
        with patch("studio.core.utils.tts.providers.TTS_VOICE", "alloy"), \
             patch("studio.core.utils.tts.providers.OpenAITTS") as create:
            self.assertIs(speech._backchannel_tts(), create.return_value)
        create.assert_called_once_with(voice="alloy")


if __name__ == "__main__":
    unittest.main()
