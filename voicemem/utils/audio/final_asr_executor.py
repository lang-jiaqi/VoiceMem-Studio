"""Serial final-ASR execution with confirmed work ahead of queued snapshots."""
from concurrent.futures import Future, ThreadPoolExecutor
import heapq
import itertools
import threading


class FinalASRExecutor:
    """Prioritize pending confirmed decodes without interrupting native calls."""

    def __init__(self):
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-final")
        self._lock = threading.Lock()
        self._pending = []
        self._sequence = itertools.count()
        self._draining = False
        self._closed = False

    def submit(self, fn, *, speculative=False):
        """Return a cancellable future; only one recognizer call runs at a time."""
        future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("Final ASR executor is shut down")
            heapq.heappush(self._pending, (
                int(speculative), next(self._sequence), future, fn))
            if not self._draining:
                self._draining = True
                self._worker.submit(self._drain)
        return future

    def _drain(self):
        while True:
            with self._lock:
                if not self._pending:
                    self._draining = False
                    return
                _, _, future, fn = heapq.heappop(self._pending)
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def shutdown(self, wait=True):
        """Finish submitted native work before releasing the backing executor."""
        with self._lock:
            self._closed = True
        self._worker.shutdown(wait=wait)
