"""Compatibility alias for the Studio-owned prompt configuration."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.prompt_config")
