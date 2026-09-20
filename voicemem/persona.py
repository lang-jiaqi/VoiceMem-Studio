"""Compatibility alias for the legacy Studio reply persona."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.prompts.legacy_persona")
