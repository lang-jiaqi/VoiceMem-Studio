"""Studio routing implementation."""
import asyncio
import re
import time
from studio.core.utils.reply_modes.initialize import DIRECT, thinking_router
from studio.core.utils.reply_modes.component import FAST, MEDIUM, SLOW, ThinkingDecision
from voicemem import gate
from voicemem.memory_api import build_memory_context
from studio.core.utils.contracts.component import Pending


_ZH_FOLLOWUP = re.compile(r"^(?:那|那么|她|他|它|他们|她们|这个|那个|这些|那些|还有)")
_EN_FOLLOWUP = re.compile(r"^(?:what about|how about|and (?:her|him|them|that|it))\b", re.I)


def contextual_memory_query(text: str, history) -> str:
    """Return a bounded search query for a short follow-up to personal context."""
    current = (text or "").strip()
    compact = re.sub(r"\s+", "", current)
    english_followup = bool(_EN_FOLLOWUP.match(current))
    too_long = ((len(current.split()) > 12 or len(current) > 120)
                if english_followup else len(compact) > 18)
    if (not history or not compact or too_long
            or gate.is_backchannel(current)):
        return ""
    if not (current.endswith(("?", "？", "呢", "吗", "么"))
            or _EN_FOLLOWUP.match(current)):
        return ""
    if not (_ZH_FOLLOWUP.match(current) or _EN_FOLLOWUP.match(current)):
        return ""
    previous = next((item.get("content", "").strip() for item in reversed(history)
                     if item.get("role") == "user"), "")
    if not previous or gate.route(previous, semantic=False) != gate.DEEP:
        return ""
    return f"{previous[:160]} {current}"


class Routing:
    async def _ensure_pending_memory(self, pending: Pending, memory_vm,
                                     query: str = "") -> None:
        """Populate memory after a final route upgrades a speculative shallow turn."""
        query = query or pending.text
        if (query == pending.text and pending.route == gate.DEEP
                and getattr(pending.result, "search_mode", "gated") != "gated"):
            return

        def search():
            from voicemem.leftbrain.query_embedding import query_embedding_scope
            from voicemem.lang import language_scope
            with query_embedding_scope(), language_scope(
                    getattr(pending, "language", "") or getattr(memory_vm, "memory_language", "en")):
                classification = memory_vm.classify(query)
                return memory_vm.search(
                    query,
                    slots=classification.slots,
                    entities=classification.entities,
                    emotion=pending.emotion,
                )

        pending.result = await asyncio.to_thread(search)
        pending.memory_context = build_memory_context(pending.result)
        pending.route = gate.DEEP
        pending.replay = self._replay_id(pending.text, pending.result)

    async def route_pending_thinking(self, pending: Pending, memory_vm=None,
                                     history=None) -> Pending:
        """Combine VoiceMem memory eligibility with local reasoning depth off-loop."""
        started = time.monotonic()
        if getattr(pending, 'timing', None) is not None:
            pending.timing = {**pending.timing, 'route_started_at': started}
        from studio.core.utils.self_harness.component import reasoning_preference
        if pending.stranger:
            history = []
        depth_preference = reasoning_preference(
            getattr(pending, "self_harness_profile", None))
        query = (contextual_memory_query(pending.text, history)
                 if not pending.stranger and pending.route != gate.BACKCHANNEL else "")
        pending.memory_query = query
        if not self._THINKING_ROUTER_ON and depth_preference == "auto":
            if query:
                from studio.core.utils.reply_modes.initialize import MEMORY
                pending.reply_mode = MEMORY
                try:
                    await self._ensure_pending_memory(pending, memory_vm or self.vm, query)
                except Exception as exc:
                    from voicemem.stream import empty_result
                    pending.route = gate.DEEP
                    pending.result = empty_result()
                    pending.memory_context = ""
                    pending.replay = ""
                    print(f"[route] memory retrieval failed: {type(exc).__name__}: {exc}",
                          flush=True)
            if getattr(pending, 'timing', None) is not None:
                routed_at = time.monotonic()
                pending.timing.update(route_finished_at=routed_at,
                                      route_ms=(routed_at-started)*1000)
            return pending
        memory_vm = memory_vm or self.vm
        memory_required = gate.needs_memory(pending.route) or bool(query)
        memory_ms = 0.0

        async def retrieve():
            nonlocal memory_ms
            memory_started = time.monotonic()
            try:
                await self._ensure_pending_memory(pending, memory_vm, query)
            except Exception as exc:
                # The selected route still reaches the provider with an explicit
                # no-memory directive rather than silently degrading to instant.
                pending.route = gate.DEEP
                from voicemem.stream import empty_result
                pending.result = empty_result()
                pending.memory_context = ""
                pending.replay = ""
                print(f"[route] memory retrieval failed: {type(exc).__name__}: {exc}",
                      flush=True)
            finally:
                memory_ms = (time.monotonic() - memory_started) * 1000

        # Memory already required by Gate does not depend on reasoning depth.
        # Both tasks belong to this turn; cancellation must reap the retrieval.
        memory_task = (asyncio.create_task(retrieve())
                       if memory_required and not pending.stranger else None)
        try:
            if depth_preference == "deep":
                decision = ThinkingDecision(SLOW, 'self-harness:deep')
            elif depth_preference == "quick":
                decision = ThinkingDecision(FAST, 'self-harness:quick')
            elif self._THINKING_ROUTER_ON:
                try:
                    router = thinking_router()
                    options = {'history': history}
                    if getattr(router, 'accepts_audio_state', False) is True:
                        observed = getattr(pending, 'audio_state', None)
                        options['audio_state'] = dict(observed) if observed is not None else None
                    decision = await router.classify_async(pending.text, **options)
                except Exception as exc:
                    print(f"[thinking] 深思判断不可用（{type(exc).__name__}），保留 VoiceMem 记忆资格", flush=True)
                    decision = ThinkingDecision(FAST, 'unavailable')
            else:
                decision = ThinkingDecision(FAST, 'router-disabled')
            decision = ThinkingDecision(
                SLOW if decision.level == SLOW else (MEDIUM if memory_required else FAST), decision.raw)
            classified = time.monotonic()
            pending.reply_mode = decision.reply_mode
            if pending.stranger or decision.reply_mode == DIRECT:
                # Speaker privacy outranks routing, including reasoning upgrades.
                from voicemem.stream import empty_result
                pending.result = empty_result()
                pending.memory_context = ""
                pending.replay = ""
                pending.route = gate.SHALLOW
            elif memory_task is not None:
                await memory_task
            else:
                await retrieve()
        finally:
            if memory_task is not None:
                if not memory_task.done():
                    memory_task.cancel()
                await asyncio.gather(memory_task, return_exceptions=True)
        routed_at = time.monotonic()
        if getattr(pending, 'timing', None) is not None:
            pending.timing.update(route_finished_at=routed_at,
                                  route_ms=(routed_at-started)*1000)
        print(f"[route] {decision.display_name} → {decision.reply_mode}"
              f" / reasoning={decision.reasoning_effort} "
              f"(model={(classified - started) * 1000:.0f}ms"
              f" memory={memory_ms:.0f}ms"
              f" total={(time.monotonic() - started) * 1000:.0f}ms)", flush=True)
        return pending
