"""Compatibility alias for Studio request tracing."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.logging_utils.prompt_trace")
