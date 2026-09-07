"""Reuse identical query encodings only within one retrieval operation.

Keyed by the actual model object and prefixed input. No cross-request memory
results, no stale DB hits, and no mixing of query/passage prefixes or models.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

_CACHE = ContextVar("query_embedding_cache", default=None)


@contextmanager
def query_embedding_scope():
    token = _CACHE.set({})
    try:
        yield
    finally:
        _CACHE.reset(token)


def encode_query(model, text):
    cache = _CACHE.get()
    key = (id(model), text)
    if cache is not None and key in cache:
        return cache[key][1].copy()
    vector = np.asarray(model.encode([text], normalize_embeddings=True)[0])
    if cache is not None:
        # Keep the model alive so id() cannot be reused within this scope.
        cache[key] = (model, vector.copy())
    return vector
