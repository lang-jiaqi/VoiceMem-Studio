"""Compatibility import; implementation lives in studio.web.transport."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.web.transport")
