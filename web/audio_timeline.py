"""Compatibility import; implementation lives in studio.core.utils.audio_timeline.component."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.audio_timeline.component")
