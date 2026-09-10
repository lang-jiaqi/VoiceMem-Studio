"""Compatibility import; implementation lives in studio.harness.persona.policy."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.harness.persona.policy")
