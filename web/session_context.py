"""Compatibility import; implementation lives in studio.core.utils.session_context.component."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.session_context.component")
