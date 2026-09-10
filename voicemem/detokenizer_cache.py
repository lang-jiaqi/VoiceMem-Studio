"""Compatibility import; implementation lives in studio.core.utils.llm.detokenizer_cache."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.llm.detokenizer_cache")
