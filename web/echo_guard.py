"""Compatibility import; implementation lives in studio.core.utils.echo_guard.component."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.core.utils.echo_guard.component")
