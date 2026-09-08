# VoiceMem Repository Instructions

This file applies to the entire repository. Follow explicit user instructions
first. Use this document for operating rules; use `ARCHITECTURE.md` for system
design and ownership boundaries.

## Required context

Before changing a public contract, data flow, state owner, concurrency model,
provider interface, persistence behavior, or more than one top-level subsystem:

1. Read `ARCHITECTURE.md`.
2. Identify the layer that owns the behavior.
3. List the direct and asynchronous consumers of the changed contract.
4. Update `ARCHITECTURE.md` in the same patch if the design changes.

For a localized fix, read the surrounding implementation and its direct caller
and tests. Do not load unrelated files merely to gather context.

## Repository map

- `voicemem/core.py`: public `VoiceMem` facade.
- `voicemem/orchestrator.py`: cross-component search and ingest workflows.
- `voicemem/stream.py`: streaming ASR/VAD state and speculative retrieval.
- `voicemem/leftbrain/`: factual memory and retrieval.
- `voicemem/rightbrain/`: affective and behavioral memory.
- `voicemem/utils/`: replaceable capabilities and shared infrastructure.
- `voicemem/config.py`: declarative provider configuration.
- `voicemem/reply.py`, `voicemem/tts.py`: reply and speech output providers.
- `web/`: demo application, WebSocket transport, session state, and playback.
- `examples/`, `evaluation/`, `finetune/`: examples, benchmarks, and training.

## Standard workflow

### 1. Inspect

- Start with `git status --short --branch`.
- Treat existing modifications and untracked files as user-owned.
- Reproduce or locate evidence before proposing a fix.
- A request to diagnose, explain, review, or report does not authorize code
  changes.

### 2. Scope

- Define the requested outcome and the smallest owning layer.
- Separate root cause from adjacent cleanup opportunities.
- Do not bundle unrelated refactors, formatting, dependency changes, or comment
  cleanup into a functional patch.
- Ask before making a choice that materially changes product behavior, public
  APIs, persistence, deployment, cost, or external systems.

### 3. Implement

- Make the smallest coherent change that solves the problem.
- Preserve backward compatibility and existing injection points unless the task
  explicitly permits a breaking change.
- Prefer the standard library and existing dependencies.
- Keep application policy out of the memory core and vendor payloads behind
  adapters.
- Preserve cancellation, ordering, state ownership, and exception visibility
  when moving work into background tasks.

### 4. Verify

- Run focused checks while iterating and broader checks for shared contracts.
- Test both synchronous and asynchronous failure paths when relevant.
- Review the final diff for accidental files, secrets, generated data, and scope
  expansion.
- State exactly which live, GPU, network, paid-API, benchmark, or training paths
  were not exercised.

### 5. Handoff

- Lead with the result, then summarize behavior and verification.
- Do not claim a path was tested when it was only inspected or syntax-checked.
- Do not commit, push, rewrite history, publish, or modify an external service
  unless the user explicitly requests that action.
- When asked for Git commands, provide exact commands with an explicit file
  list. Do not use `git add .` in a dirty worktree.

## Authorization and side effects

Use one general rule: do not expand the task's scope or create unnecessary side
effects.

Obtain explicit approval before actions that are destructive, difficult to
reverse, costly, affect external systems or people, or are not a necessary part
of the requested implementation. This includes changes to persistent data,
remote repositories, deployed services, credentials, large dependency/model
downloads, long-running compute, and training or full benchmark runs.

Read-only inspection and narrowly scoped local verification are allowed when
they directly support the task. Prefer reversible operations and exact targets.

## Architecture rules

- The dependency direction is from applications toward `voicemem`; the core
  package must not depend on browser or transport code.
- Keep model and service providers replaceable. Shared memory behavior consumes
  normalized contracts, not vendor-specific payloads.
- Core memory modes and Web reply modes are separate axes. A shared behavior
  change must be checked in every affected mode.
- Do not block the asyncio/WebSocket event loop with synchronous inference,
  network calls, model loading, or slow storage work.
- Capture turn, session, user, language, and Memory Space ownership before
  scheduling background work.
- Generated, sent, buffered, rendered, and heard output are distinct states.
  Interruption context uses only the heard prefix.
- Keep `import voicemem` lightweight; preserve lazy loading of heavy components.

See `ARCHITECTURE.md` for the full layer map, flows, contracts, and state
lifetime.

## Source code language and comments

### Language

- Write new or substantially revised production comments and docstrings in
  English.
- Do not mix Chinese and English prose inside one comment block.
- Existing Chinese comments are legacy text. Do not add to them. Translate only
  comments directly affected by the current code change and only when the
  technical meaning can be preserved.
- User-facing UI text, localized documentation, prompts, memory-language
  examples, and test fixtures may use the language required by their behavior.

### What comments should contain

Add a comment only when names, types, and code structure cannot communicate a
durable constraint. Appropriate subjects are:

- public contracts, side effects, and failure modes;
- ownership and lifecycle boundaries;
- concurrency, cancellation, ordering, and protocol invariants;
- the reason for a non-obvious algorithmic decision.

Public APIs should use concise docstrings that describe inputs, outputs, side
effects, and raised errors. Keep local comments close to the smallest code
region they govern. Move cross-module explanations to `ARCHITECTURE.md` or a
focused document under `docs/`.

### What comments must not contain

Do not put any of the following in source comments or docstrings:

- credentials, tokens, private endpoints, internal hostnames, or authorization
  details;
- personal names, user conversations, memory contents, local usernames,
  machine-specific paths, or other identifying information;
- debugging transcripts, incident narratives, failed-attempt history, or notes
  about how an agent produced the code;
- comparisons with unrelated products or frameworks;
- unverifiable performance claims or measurements without a reproducible setup;
- dead code, commented-out implementations, or ownerless future-work notes;
- prose that merely restates the next line of code.

Use `TODO(<issue-or-owner>): action` only for an actionable and owned follow-up.
Do not use TODO comments as a substitute for completing the requested behavior.

Comment-only cleanup should be a separate, reviewable task organized by
subsystem. It must not change runtime behavior.

## Code and documentation style

- Follow the surrounding structure and naming. Python uses four-space
  indentation and targets Python 3.10 or newer.
- Use type hints at public and cross-module boundaries. Use dataclasses when
  they clarify exchanged state.
- Avoid broad exception handling unless the boundary must degrade gracefully;
  keep exceptions visible through a result or actionable log.
- Keep logs concise and structured. Never log secrets or authorization headers.
  Gate high-frequency diagnostics behind an existing debug setting.
- Keep Chinese and English README sections aligned when user-visible behavior
  changes.
- Public documentation may describe only behavior visible in the repository or
  information explicitly approved for publication. Exclude private roadmaps,
  internal operations, incident details, and unreleased plans.
- Do not add generated artifacts or machine-specific setup instructions to
  source control.

## Environment and data safety

- Use Python 3.10 or newer and prefer an existing working environment.
- Do not replace an environment solely because an isolated agent sandbox cannot
  execute an interpreter outside the repository. Inspect the user's logs and
  active interpreter first.
- Treat configured memory roots, `voicemem_memoryspace/`, databases, recordings,
  `results/`, model directories, caches, virtual environments, and local backups
  as user data.
- Tests and reproductions must use temporary or dedicated memory roots. Never
  write fixtures into the user's default Memory Space.
- Read credentials from environment variables or ignored local configuration.
  Never write real values into code, docs, examples, tests, or committed logs.

## Commands and verification

Common setup and run commands:

```bash
python -m pip install -e .
python -m pip install -e ".[slm]"
bash scripts/download_models.sh
python web/run.py --mode realtime --host 0.0.0.0 --port 8787
python web/run.py --mode llm_tts --host 0.0.0.0 --port 8787
```

These are references, not instructions to run every command. Select only what
the task requires.

Focused checks:

```bash
python -m py_compile path/to/changed_file.py
python -m unittest tests.test_offline_engine tests.test_session_context
node --check web/pcm-player-worklet.js
git diff --check
```

New files under `tests/` are intentionally ignored. Keep local regression tests
there; do not force-add them unless the user explicitly asks to publish tests.
Existing tracked tests may be updated when their covered behavior changes.

For audio or streaming changes, verify the applicable start, partial update,
pause, resume, underflow, drain, interruption, cancellation, disconnect, and
stale-event paths. Prefer synthetic data and fake providers for unit tests.

## Git hygiene

- Preserve user changes during pull, rebase, and conflict resolution.
- Never use destructive reset or checkout commands as a shortcut.
- Stage only reviewed files and inspect the staged diff before committing.
- Never stage environments, model weights, caches, logs, memory databases,
  recordings, generated results, or local experiment directories.
- Before a requested commit or push, verify author name, author email, target
  branch, and remote URL.

## Final review checklist

- The change belongs to the correct architecture layer.
- Public and provider contracts remain compatible or are explicitly migrated.
- No event-loop blocking, orphaned task, stale state, or cross-session ownership
  issue was introduced.
- Both affected modes and error paths were considered.
- Comments follow the English and information-safety rules.
- No private information, credentials, user data, or machine-specific details
  entered the diff.
- Tests and documentation match the behavior, and untested paths are disclosed.
