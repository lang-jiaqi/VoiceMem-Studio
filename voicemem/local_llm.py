"""Compatibility import; implementation lives in studio.core.utils.llm.local."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.llm.local")
