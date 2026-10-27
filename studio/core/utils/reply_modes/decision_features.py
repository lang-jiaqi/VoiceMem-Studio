"""Versioned features for an offline-trained voice-dialogue decision head."""
from __future__ import annotations

import math
import numpy as np

VERSION = 1
AUDIO_FIELDS = (('eot_probability', 1.0), ('silence_s', 2.0), ('span_s', 30.0))


def texts(text: str, history=None) -> tuple[str, str]:
    """Bound current text separately so history cannot truncate the request."""
    current = ' '.join(str(text).split())
    if len(current) > 256:
        current = current[:128] + ' … ' + current[-128:]
    rows = []
    for message in list(history or [])[-4:]:
        role = message.get('role')
        if role not in {'user', 'assistant'}:
            continue
        content = ' '.join(str(message.get('content') or '').split())[:80]
        if content:
            rows.append(f'{role}: {content}')
    return current, '\n'.join(rows)


def audio_vector(state=None) -> np.ndarray:
    """Encode observed signals and missingness without retaining any audio."""
    state = state or {}
    if not isinstance(state, dict) or set(state) - {key for key, _ in AUDIO_FIELDS}:
        raise ValueError('Invalid decision audio state')
    values, present = [], []
    for name, scale in AUDIO_FIELDS:
        item = state.get(name)
        if item is None:
            values.append(0.0)
            present.append(0.0)
            continue
        if isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item):
            raise ValueError(f'Invalid decision signal: {name}')
        if item < 0 or (name == 'eot_probability' and item > 1):
            raise ValueError(f'Invalid decision signal: {name}')
        values.append(min(float(item) / scale, 1.0))
        present.append(1.0)
    return np.asarray(values + present, dtype=np.float32)


def decision_state(text, history=None, audio_state=None):
    """Serialize bounded semantic context and only genuinely observed audio state."""
    current, recent = texts(text, history)
    audio_vector(audio_state)
    state = {'current_user': current, 'recent_dialogue': recent}
    observed = {name: audio_state[name] for name, _ in AUDIO_FIELDS
                if audio_state and audio_state.get(name) is not None}
    if observed:
        state['observed_audio'] = observed
    return state


def combine(current, context, audio_state=None) -> np.ndarray:
    """Join semantics, context interaction and optional acoustic observations."""
    current, context = np.asarray(current, np.float32), np.asarray(context, np.float32)
    if (current.ndim != 1 or context.shape != current.shape or not current.size
            or not np.isfinite(current).all() or not np.isfinite(context).all()):
        raise ValueError('Invalid decision embeddings')
    return np.concatenate((current, context, current * context, audio_vector(audio_state)))
