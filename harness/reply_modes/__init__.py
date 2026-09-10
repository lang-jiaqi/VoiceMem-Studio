"""Compatibility exports for Studio reply routing."""
import importlib
import sys
sys.modules[f'{__name__}.thinking'] = importlib.import_module('studio.core.utils.reply_modes.component')
sys.modules[__name__] = importlib.import_module('studio.core.utils.reply_modes.initialize')
