"""Compatibility import; implementation lives in studio.core.utils.tts.control."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.tts.control")
