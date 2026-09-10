"""Compatibility import; implementation lives in studio.core.utils.logging_utils.component."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.logging_utils.component")
