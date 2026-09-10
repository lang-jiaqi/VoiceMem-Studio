"""Compatibility exports for the Studio turn-taking policy and execution."""
import importlib
import sys

for child, target in {
    'backchannel': 'studio.core.utils.turn_taking.backchannel',
    'filler': 'studio.core.utils.turn_taking.filler',
    'frequency': 'studio.harness.turn_taking.policy',
    'state_machine': 'studio.core.utils.turn_taking.component',
}.items():
    sys.modules[f'{__name__}.{child}'] = importlib.import_module(target)
sys.modules[__name__] = importlib.import_module('studio.core.utils.turn_taking.initialize')
