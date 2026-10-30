"""One process-owned MLX thread with fair execution and bounded async output.

Factories, generator steps and cleanup stay on this thread. Native calls are
cooperatively cancelled between steps; model state belongs to each provider.
"""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import dataclass, field
import queue
import threading
import time

from ...scheduling.context import inference_scope


_DONE = None


class _AsyncOutput:
    """Deliver worker output only on its owning event loop."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue()

    def put(self, item):
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:
            if not self.loop.is_closed():
                raise


class Job:
    """One stream; cancellation is observed between native generator steps."""

    def __init__(self, solo=False, *, out=None, max_pending=0, label="mlx"):
        self.out = queue.Queue() if out is None else out
        self._cancelled = False
        self.solo = solo
        self.context = copy_context()
        self.label = label
        self.submitted = time.perf_counter()
        self.max_pending = max_pending
        self._pending = 0
        self._mutex = threading.Lock()
        self._wake = lambda: None

    def cancel(self):
        """Wake the worker to skip queued work or close a paused generator."""
        self._cancelled = True
        self._wake()

    @property
    def cancelled(self):
        return self._cancelled

    @property
    def ready(self):
        with self._mutex:
            return not self.max_pending or self._pending < self.max_pending

    def produce(self, item):
        with self._mutex:
            self._pending += 1
        self.out.put(("ok", item))

    def acknowledge(self):
        with self._mutex:
            self._pending -= 1
        self._wake()


@dataclass
class _Entry:
    factory: object
    job: Job
    weight: int
    runtime: float = 0.0
    generator: object = None
    last_step: float = field(default_factory=time.monotonic)
    cost: float = 0.0


class GpuLoop:
    """Own MLX execution, fair compute shares and asynchronous output admission."""

    def __init__(self, *, cache_limit=536870912):
        if cache_limit < 0:
            raise ValueError("MLX cache limit must not be negative")
        self.cache_limit = cache_limit
        self._effective_cache_limit = cache_limit
        self._cache_budget_sampled = float('-inf')
        self._cache_budget_error = None
        self._jobs = queue.Queue()
        self._thread = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.active_count = 0
        self._state = {"runnable": 0, "backpressured": 0}
        self._step_started = None
        self._step_model = None
        self._memory = {}

    def snapshot(self):
        """Read scheduler pressure without invoking MLX or retaining outputs."""
        started = self._step_started
        return {**self._state, **self._memory, "active": self.active_count,
                "model": self._step_model,
                "queued": self._jobs.qsize(), "cache_limit": self.cache_limit,
                "effective_cache_limit": self._effective_cache_limit,
                "cache_budget_error": self._cache_budget_error,
                "native_step_ms": (time.perf_counter() - started) * 1000 if started else 0}

    def _update_cache_budget(self, mx, now):
        """Adjust free-cache capacity between native steps on this owner thread."""
        if now - self._cache_budget_sampled < 1:
            return
        self._cache_budget_sampled = now
        from ...scheduling.admission import resources
        limit = min(self.cache_limit, 134217728) if resources.cache_pressure() else self.cache_limit
        if limit == self._effective_cache_limit:
            self._cache_budget_error = None
            return
        try:
            # MLX reclaims excess free buffers on the next allocation. Active
            # weights, compiled functions and provider-owned KV stay intact.
            mx.set_cache_limit(limit)
        except Exception as exc:
            reason = type(exc).__name__
            if reason != self._cache_budget_error:
                import logging
                logging.getLogger(__name__).warning('[mlx-cache] budget_unavailable=%s', reason)
            self._cache_budget_error = reason
        else:
            self._cache_budget_error = None
            self._effective_cache_limit = limit
            print(f'[mlx-cache] free_cache_limit_mib={limit / 1048576:.0f}', flush=True)

    def _ensure(self):
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="gpu-loop", daemon=True)
                self._thread.start()

    def on_thread(self):
        """Report thread ownership so nested initialization can execute directly."""
        return threading.current_thread() is self._thread

    def iter(self, make_gen, weight=1, solo_first=False, *, label="mlx"):
        """Return a synchronous Job with the existing ok/err/None queue protocol.

        Factories execute lazily on the MLX thread. Weight is a share of measured
        compute time; solo_first gives the first yield precedence over other jobs.
        Synchronous consumers retain the existing unbounded queue contract.
        """
        job = Job(solo=solo_first, label=label)
        self._submit(make_gen, job, weight)
        return job

    def _submit(self, make_gen, job, weight):
        job._wake = self._wake.set
        self._jobs.put((make_gen, job, max(1, weight)))
        self._wake.set()
        self._ensure()

    async def aiter(self, make_gen, weight=1, solo_first=False, *, max_pending=8,
                    service=None, label="mlx"):
        """Yield bounded output without pool threads; errors reach the consumer.

        At most max_pending items may be produced ahead of consumer advancement.
        Cancellation closes the native generator at its next safe boundary.
        """
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        out = _AsyncOutput()
        with inference_scope(service) if service is not None else nullcontext():
            job = Job(solo=solo_first, out=out, max_pending=max_pending, label=label)
        self._submit(make_gen, job, weight)
        try:
            while (item := await out.queue.get()) is not _DONE:
                kind, value = item
                if kind == "err":
                    raise value
                try:
                    yield value
                finally:
                    job.acknowledge()
        finally:
            job.cancel()

    def call(self, fn):
        """Run one function on the MLX thread and return or raise its result."""
        if self.on_thread():
            return fn()

        def once():
            yield fn()
        job = self.iter(once)
        tag, payload = job.out.get()
        job.out.get()
        if tag == "err":
            raise payload
        return payload

    def run_steps(self, make_gen, *, cancelled=None, label="mlx"):
        """Drain cooperative steps from a native caller and return the last value.

        Queued cancellation skips the factory; active cancellation waits for
        owner-thread cleanup and returns None. Exceptions reach the caller.
        Calls already on the owner thread execute inline without rescheduling.
        """
        if cancelled is not None and cancelled.is_set():
            return None
        if self.on_thread():
            generator = iter(make_gen())
            result = None
            try:
                while cancelled is None or not cancelled.is_set():
                    try:
                        result = next(generator)
                    except StopIteration:
                        return result
                return None
            finally:
                close = getattr(generator, "close", None)
                if close is not None:
                    close()

        def guarded():
            if cancelled is not None and cancelled.is_set():
                return
            generator = iter(make_gen())
            try:
                while cancelled is None or not cancelled.is_set():
                    try:
                        value = next(generator)
                    except StopIteration:
                        return
                    yield value
            finally:
                close = getattr(generator, "close", None)
                if close is not None:
                    close()

        job = self.iter(guarded, label=label)
        result = error = None
        try:
            while True:
                if cancelled is not None and cancelled.is_set():
                    job.cancel()
                try:
                    item = job.out.get(timeout=.05)
                except queue.Empty:
                    continue
                if item is _DONE:
                    if error is not None:
                        raise error
                    if job.cancelled or (cancelled is not None and cancelled.is_set()):
                        return None
                    return result
                kind, value = item
                if kind == "err":
                    if error is None:
                        error = value
                else:
                    result = value
        finally:
            job.cancel()

    def _run(self):
        try:
            import mlx.core as mx
            mx.set_default_device(mx.gpu)
            mx.set_cache_limit(self.cache_limit)
            memory_sampled = 0.0
        except Exception as exc:
            # Initialization failures must reach every queued caller rather
            # than leave futures waiting on an exited worker.
            while True:
                _, job, _ = self._jobs.get()
                if not job.cancelled:
                    job.out.put(("err", exc))
                job.out.put(_DONE)
        active = []
        virtual_time = 0.0
        while True:
            self._wake.clear()
            try:
                while True:
                    self._begin(self._jobs.get_nowait(), active)
            except queue.Empty:
                pass
            for entry in tuple(active):
                if entry.job.cancelled:
                    entry.job.context.run(self._cancel, entry.generator, entry.job)
                    active.remove(entry)
            self.active_count = len(active)
            ready = [e for e in active if e.job.ready]
            self._state = {"runnable": len(ready), "backpressured": len(active) - len(ready)}
            for entry in ready:
                entry.runtime = max(entry.runtime, virtual_time)
            if not ready:
                self._wake.wait()
                continue
            solo = [e for e in ready if e.job.solo]
            now = time.monotonic()
            aged = [e for e in ready if now - e.last_step >= .5]
            entry = (solo[0] if solo else
                     min(aged, key=lambda e: e.last_step) if aged else
                     min(ready, key=lambda e: e.runtime))
            virtual_time = max(virtual_time, min(e.runtime for e in ready))
            self._update_cache_budget(mx, now)
            self._step_started = time.perf_counter()
            self._step_model = entry.job.label
            try:
                # Admission precedes factory execution too: a newly submitted
                # model load or prefill cannot bypass the chosen compute share.
                value = entry.job.context.run(self._step, entry)
            except StopIteration:
                if entry.job.cancelled:
                    entry.job.context.run(self._cancel, entry.generator, entry.job)
                else:
                    entry.job.out.put(_DONE)
                active.remove(entry)
            except Exception as exc:
                entry.job.out.put(("err", exc))
                entry.job.context.run(self._cancel, entry.generator, entry.job)
                active.remove(entry)
            else:
                entry.job.produce(value)
                entry.job.solo = False
                # Charge execution, excluding device admission and consumer waits.
                entry.runtime += max(entry.cost, .000001) / entry.weight
                entry.last_step = time.monotonic()
            finally:
                self._step_started = None
                self._step_model = None
                now = time.monotonic()
                if now-memory_sampled >= 1:
                    memory_sampled = now
                    # Allocator inspection stays on the process MLX thread.
                    if hasattr(mx, "get_active_memory") and hasattr(mx, "get_cache_memory"):
                        try:
                            self._memory = {"allocated_bytes": mx.get_active_memory(),
                                            "cached_bytes": mx.get_cache_memory()}
                        except Exception:
                            self._memory = {"sampling_unavailable": True}
            self.active_count = len(active)

    @staticmethod
    def _step(entry):
        if entry.job.cancelled:
            raise StopIteration
        started = time.perf_counter()
        first = entry.generator is None
        failed = False
        try:
            if entry.generator is None:
                entry.generator = iter(entry.factory())
            return next(entry.generator)
        except StopIteration:
            raise
        except BaseException:
            failed = True
            raise
        finally:
            entry.cost = time.perf_counter() - started
            from ...telemetry.metrics import record
            record(entry.job.label, "mlx", entry.cost,
                   queue_s=started-entry.job.submitted if first else 0, error=failed)

    @staticmethod
    def _begin(pending, active):
        factory, job, weight = pending
        if job.cancelled:
            job.out.put(_DONE)
            return
        floor = min((e.runtime for e in active), default=0.0)
        active.append(_Entry(factory, job, weight, floor))

    @staticmethod
    def _cancel(gen, job):
        """Close on the MLX thread, retaining cleanup failures for the consumer."""
        try:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
        except Exception as exc:
            job.out.put(("err", exc))
        finally:
            job.out.put(_DONE)


_LOOP = None
_LOOP_LOCK = threading.Lock()


def gpu_loop():
    """Return the lazy process singleton without importing MLX on the caller."""
    global _LOOP
    if _LOOP is None:
        with _LOOP_LOCK:
            if _LOOP is None:
                _LOOP = GpuLoop()
    return _LOOP
