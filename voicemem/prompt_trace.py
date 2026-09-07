"""Opt-in local request journal. No credentials, no disk I/O on the speech loop."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import uuid

_context = ContextVar("prompt_trace", default=None)
_writer = None


class PromptWriter:
    def __init__(self, root):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.directory = Path(root) / f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.directory.mkdir(parents=True, mode=0o700)
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="prompt-journal", daemon=True)
        self.thread.start()

    def submit(self, event):
        # Freeze the exact request before callers can mutate message/history lists.
        data = json.dumps(event, ensure_ascii=False) + "\n"
        self.queue.put((event["trace_id"], data))

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                trace_id, data = item
                # trace_id is always internally generated, never a user path.
                path = self.directory / f"{trace_id}.jsonl"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as out:
                    out.write(data)
            except Exception as exc:
                print(f"[prompt] 保存失败：{type(exc).__name__}: {exc}", flush=True)
            finally:
                self.queue.task_done()

    def flush(self):
        self.queue.join()


def configure(root):
    global _writer
    if _writer is None:
        _writer = PromptWriter(root)
        atexit.register(_writer.flush)
    return _writer.directory


@contextmanager
def prompt_scope(**metadata):
    token = _context.set({"trace_id": uuid.uuid4().hex, **metadata})
    try:
        yield
    finally:
        _context.reset(token)


def record_request(kind, provider, request):
    """Call at provider boundary with an explicit allowlist of request fields.

    Never pass clients, headers or full provider configs (they can contain keys).
    Unscoped requests, e.g. backchannel synthesis, get their own journal file.
    """
    if _writer is None:
        return
    context = _context.get() or {"trace_id": uuid.uuid4().hex, "purpose": "unscoped"}
    try:
        _writer.submit({**context, "at": datetime.now().astimezone().isoformat(),
                        "kind": kind, "provider": provider, "request": request})
    except Exception as exc:
        print(f"[prompt] 记录失败：{type(exc).__name__}: {exc}", flush=True)
