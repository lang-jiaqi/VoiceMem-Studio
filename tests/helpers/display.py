"""Isolated reply-display state without initializing perception models."""
import ast
import asyncio
import contextlib
import threading
import time
import types
from tests.helpers.studio import studio_tree, execute


def display_namespace():
    names = {"ReplySink", "_early_reply_compatible", "_send_reply_display",
             "voicemem_llm_tts", "_voicemem_llm_tts"}
    tree = studio_tree()
    code = ast.Module(body=[n for n in tree.body if getattr(n, "name", "") in names],
                      type_ignores=[])
    ns = dict(asyncio=asyncio, threading=threading, time=time, MEMORY_COT="memory_cot",
              aclosing=contextlib.aclosing,
              _REPLY_DISPLAY_LOCK=threading.Lock(), ACTIVE_SPACE="test", vm=object(),
              gate=types.SimpleNamespace(needs_memory=lambda _: True),
              note_hits=lambda _: None, audio_of=None, hit_cluster=None,
              utils=types.SimpleNamespace(hits_payload=lambda *a, **kw: {}),
              fill_tags=lambda *a, **kw: {"emotion": "平静"}, MIC_RATE=24000)
    execute(code.body, ns)
    return ns

