"""Compatibility import; implementation lives in studio.core.utils.tts.audio_timing."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.tts.audio_timing")
