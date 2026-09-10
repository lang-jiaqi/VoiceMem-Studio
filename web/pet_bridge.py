"""Compatibility import; implementation lives in studio.web.pet_bridge."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("studio.web.pet_bridge")
