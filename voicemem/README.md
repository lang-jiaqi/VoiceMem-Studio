# VoiceMem package maintenance

## Structure

This package follows the implementation boundaries of
[upstream VoiceMem](https://github.com/xzf-thu/VoiceMem/tree/main/voicemem).
Existing modules contain their implementations directly. New functionality belongs
in the subsystem that owns its behavior; do not add a parallel directory tree or
module redirects to make the root appear smaller.

| Responsibility | Implementation owner |
| --- | --- |
| Public facade, configuration and lazy exports | `core.py`, `config.py`, `__init__.py` |
| Convenience memory API | `memory_api.py` |
| Cross-component search/ingest, capabilities and result contract | `orchestrator.py` |
| Streaming ASR, confirmation, speculation and turn snapshots | `stream.py` |
| Memory eligibility and interruption routes | `gate.py` |
| Per-operation language and Space fallback | `lang.py` |
| Process-wide model role configuration | `llm_config.py` |
| Reply provider interfaces and remote adapters | `reply.py` |
| Shared speech providers and PCM timing contracts | `tts.py`, `audio_timing.py` |
| Opt-in startup probes | `startup_check.py` |
| Factual query classification, graph candidates and ranking | `leftbrain/brain.py` |
| Fact persistence, backend IDs and Space mirror | `leftbrain/memory_repository.py` |
| Slot tags, summaries and tag backfill | `leftbrain/memory_repository_v2.py` |
| Vector backend | `leftbrain/mem0_backend_store.py` |
| Cognitive schema and slot extension | `leftbrain/cognitive_graph/store.py`, `store_v2.py` |
| Shared embedding models | `leftbrain/local_embedder.py` |
| Embedding contracts, hit types and lexical scoring helpers | `leftbrain/local_memory_store.py` |
| Affective search, ingestion and traits | `rightbrain/` |
| Audio perception and speaker permissions | `utils/audio/` |
| Shared request journal | `utils/common/prompt_trace.py` |
| Accelerator scheduling and serialization | `utils/gpu_loop.py`, `utils/torch_lock.py` |

The root contains thirteen implementation modules: the twelve upstream module
names plus `gate.py`. Studio persona, prompt configuration, local reply generation,
Breeze implementation, tone controls and decoder caches live in `studio/` and are
imported from there directly. The shared journal has one implementation under
`utils/common/`, used directly by both packages.

## Existing layers

`memory_repository_v2.py` extends the base repository's writes with slot metadata.
It does not replace the default query pipeline, which remains in `LeftBrain`.
Its graph-store type is selected during base initialization so base schema
migration runs once. Both repository files contain real implementations.

`local_memory_store.py` contains shared contracts and helpers, not another vector
database. `local_e5_embedder.py` preserves the E5-specific constructor;
configurable embeddings use `local_embedder.py` and its shared cache.
`cognitive_graph/store_v2.py` extends the existing schema with tags and summaries.
These layers retain their original algorithms and data contracts.

`Orchestrator` retains its component cache, Space identity, locks and background
work. `stream.py` retains the ASR worker, final-ASR executor, speculation and lazy
perception state. Language ContextVars, model overrides, TTS instances and the
request journal each have one owner. Plain `import voicemem`, resolving `VoiceMem`
and importing `reply.py` do not initialize models or import Studio. Selecting an
optional Studio speech adapter or using the existing default persona can load
Studio explicitly.

The Python wheel includes SDK prompts, sample audio and the Studio Python helpers
used by optional speech providers and the default persona. The complete Studio
application runs from a repository checkout. Its UI, avatars and model weights
are not part of the wheel.

## Historical opt-in APIs

`memory_repository_v2.py::search_slot_based` retains its direct-call query and
summary behavior. The default `LeftBrain` pipeline does not call it. Older
`utils/fusion/` and emotion layer/graph/store APIs remain available to explicit
callers.

`memory_repository.py::search_with_graph` and `search_combined` are deprecated,
unsupported legacy entry points. Their names and parameter lists remain available,
but direct calls immediately raise `NotImplementedError` with guidance to use
`VoiceMem.search()`. They perform no query, model call or background work.
The optional fusion adapter uses its existing plain search when a graph provider
explicitly raises `NotImplementedError`; working custom graph providers keep their
graph results. Studio continues to use the existing `Orchestrator`/`LeftBrain` pipeline.

Stored legacy response-experience types and public backfill/maintenance methods
remain available. A method's absence from the current query path does not justify
removing persisted data or changing its returned schema.

## Change rules

- Keep original implementation locations and use direct imports.
- Preserve constructors, injection, memory IDs, tables, metadata, prompts,
  thresholds, retrieval order and cancellation during structural maintenance.
- Keep Studio interaction policy and UI in `studio/`.
- Keep operation language, Space defaults and UI locale separate.
- Capture context before scheduling background work and retain existing locks.
- Remove private code only after checking direct and dynamic references.
- Comments explain durable contracts, ownership and failure behavior.

Focused checks:

```bash
.venv/bin/python -m unittest tests.voicemem.test_memory_structure \
  tests.voicemem.test_stream_lifecycle tests.voicemem.test_asr_finish \
  tests.studio.test_bilingual_pipeline tests.voicemem.test_query_embedding \
  tests.voicemem.test_shared_embedding tests.studio.test_prompt_logging
```

These use synthetic fixtures and temporary stores. They do not establish live
ASR quality, model performance or end-to-end perceived latency.
