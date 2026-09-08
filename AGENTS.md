# VoiceMem Studio Repository Instructions

This file applies to the entire repository. Follow explicit user instructions
first. Use this document for operating rules and `ARCHITECTURE.md` for system
design, state ownership, and pipeline boundaries.

## Required context

Before changing a public contract, latency-critical flow, state owner,
concurrency model, prompt protocol, provider interface, persistence behavior,
or more than one top-level subsystem:

1. Read `ARCHITECTURE.md`.
2. Identify the layer that owns the behavior.
3. Trace direct, background, and browser-side consumers.
4. Update `ARCHITECTURE.md` in the same patch if the design changes.

For a localized fix, read the implementation, its caller, and focused
regressions. Avoid loading unrelated files or sensitive runtime traces.

## Repository map

- `voicemem/`: memory framework plus Studio providers, routing, prompt support,
  and local inference utilities.
- `voicemem/leftbrain/`, `voicemem/rightbrain/`: factual and affective memory.
- `voicemem/stream.py`: ASR/VAD/EOT state, turn routing, and speculative search.
- `voicemem/gate.py`: `backchannel`, `shallow`, and `deep` turn routes.
- `voicemem/local_llm.py`, `voicemem/tts.py`, `voicemem/breeze_tts.py`: local
  and remote reply/speech providers.
- `voicemem/utils/gpu_loop.py`, `voicemem/utils/torch_lock.py`: process-level
  accelerator scheduling.
- `web/run.py`: Studio composition root and both reply modes.
- `web/harness.py`, `harness/`: dialogue policy, pause handling, tone tags, and
  spoken backchannels.
- `web/mic-capture-worklet.js`, `web/echo_guard.py`: capture-time echo defense.
- `web/audio_timeline.py`, `web/pcm-player-worklet.js`: output timing and PCM
  playback.
- `prompt/`: editable prompt and TTS configuration; `prompt/logs/` contains
  sensitive runtime traces.
- `evals/`: Studio latency and behavioral regressions.
- `tests/`: core regressions; new local tests are ignored by default.
- `evaluation/`, `finetune/`: benchmark and training workflows.
- `voice/`: reviewed speech assets and local voice-reference material.

## Standard workflow

### Inspect

- Start with `git status --short --branch`.
- Treat existing changes and untracked files as user-owned.
- Reproduce or locate concrete evidence before proposing a fix.
- A request to diagnose, explain, review, or report does not authorize code
  changes.

### Scope

- Define the requested outcome and smallest owning layer.
- Separate the root cause from adjacent cleanup opportunities.
- Do not combine behavior changes with unrelated refactors, formatting,
  dependency updates, prompt rewrites, or comment cleanup.
- Ask before making a choice that materially changes product behavior, public
  APIs, stored data, deployment, cost, or external systems.

### Implement

- Make the smallest coherent change that solves the requested problem.
- Preserve existing provider and injection contracts unless a breaking change
  is explicitly accepted.
- Prefer the standard library and existing dependencies.
- Keep memory behavior in `voicemem`, Studio interaction policy in the harness,
  and browser/transport behavior in `web`.
- Preserve ordering, cancellation, exception visibility, and state ownership
  when introducing concurrency.

### Verify

- Use focused regressions while iterating and broaden checks for shared
  contracts.
- Test partial success, cancellation, disconnect, stale event, and fallback
  paths when relevant.
- Review the final diff for secrets, runtime data, generated files, and scope
  expansion.
- Distinguish static checks, deterministic regressions, and live audio/model
  tests in the handoff.

### Handoff

- Lead with the result and summarize verification and remaining limitations.
- Do not claim a path was tested when it was only inspected or syntax-checked.
- Do not commit, push, rewrite history, publish, or change an external service
  unless the user explicitly requests that action.
- When asked for Git commands, provide exact commands with an explicit file
  list. Do not use `git add .` in a dirty worktree.

## Authorization and side effects

Use one general rule: do not expand the task or create unnecessary side
effects.

Obtain explicit approval before an action that is destructive, difficult to
reverse, costly, affects external systems or people, or is not necessary for
the requested implementation. Prefer read-only checks, reversible operations,
and exact targets.

## Studio architecture rules

- `VoiceMem` memory semantics are independent from reply, TTS, UI, and
  transport providers. Vendor-specific payloads stay at adapter boundaries.
- Core memory modes and Studio reply modes are separate axes. Shared behavior
  changes must be checked in every affected path.
- The asyncio/WebSocket loop must not perform synchronous model inference,
  blocking I/O, or slow persistence.
- All MLX generation uses the process-level `GpuLoop`. Do not submit local MLX
  LLM or TTS work from an independent thread or stream.
- Torch/MPS work that shares a model or device follows the existing lock and
  executor boundaries. Do not hold a cross-thread lock while waiting for work
  that needs the same lock.
- ASR streaming, final-ASR refinement, EOT, Turn Gate routing, and early reply
  generation are distinct stages. Preserve their cancellation and stale-result
  guards.
- Early reply output remains private to `ReplySink` until the final turn
  confirms that the speculative input is reusable.
- Generated, sent, buffered, rendered, and heard output are distinct states.
  Interrupted context uses only the heard prefix.
- Capture session, language, turn, output ID, and Memory Space ownership before
  scheduling background work.
- Keep `import voicemem` lightweight and preserve lazy loading.

## Prompt and dialogue policy

- `web/harness.py` owns the Studio Web system prompt, dialogue controls,
  unfinished-utterance policy, and Web context directives.
- `harness/backchannel.py` owns whether and how Studio emits spoken
  backchannels. `harness/speak_tag.py` owns the tone-label protocol.
- `prompt/llm_*.md` and `prompt/llm_context.json` are package/default prompt
  inputs; they do not replace the Web system prompt.
- `prompt/tts.json` owns shared TTS tone instructions and backchannel synthesis
  styles.
- Prompt/config changes require parser validation and the corresponding prompt
  regressions. Do not silently fall back on malformed configuration.
- Prompt changes are behavior changes. Keep them separate from unrelated Python
  refactors and report them explicitly.

## Source code language and comments

### Language

- Write new or substantially revised production comments and docstrings in
  English.
- Do not mix Chinese and English prose in one comment block.
- Existing Chinese comments are legacy text. Do not add to them. Translate only
  comments directly affected by the current change and only when meaning is
  preserved.
- UI copy, localized prompts, language fixtures, and public bilingual
  documentation may use the language required by their behavior.

### Appropriate comments

Use comments only for durable information not clear from names, types, or code
structure:

- public contracts, side effects, and failure modes;
- state ownership and lifecycle;
- concurrency, cancellation, ordering, cache, and protocol invariants;
- the reason for a non-obvious algorithmic choice.

Public APIs should have concise docstrings covering inputs, outputs, side
effects, and raised errors. Keep local comments near the smallest region they
govern. Move cross-module explanations to `ARCHITECTURE.md` or a focused file in
`docs/`.

### Prohibited comment content

Do not put these in source comments or docstrings:

- credentials, tokens, private endpoints, internal hostnames, or authorization
  details;
- personal names, user conversations, memory contents, local usernames,
  machine paths, device identifiers, or other identifying information;
- prompt-trace excerpts, incident narratives, debugging transcripts,
  failed-attempt history, or notes about how an agent produced the code;
- comparisons with unrelated products or frameworks;
- measurements without a reproducible fixture, environment, metric, and
  documented purpose;
- dead code, commented-out implementations, or ownerless future-work notes;
- prose that merely restates the next line of code.

Use `TODO(<issue-or-owner>): action` only for an owned and actionable follow-up.
Comment-only cleanup is a separate task and must not alter runtime behavior.

## Sensitive and generated data

- Treat Memory Spaces, databases, recordings, `results/`, caches, model
  directories, environments, and local backups as user data.
- Treat `prompt/logs/` as sensitive: entries may contain full prompts,
  conversation history, and retrieved memory. Do not inspect, quote, modify, or
  stage them unless the task explicitly requires that trace.
- Treat voice-reference and recorded-audio files as identity-bearing media. Do
  not replace, copy, publish, or derive new assets without authorization.
- Public documentation may include only repository-visible behavior or
  information explicitly approved for publication. Exclude private roadmaps,
  internal operations, incident details, and unreleased plans.
- Read credentials from environment variables or ignored local configuration.
  Never write real values into source, docs, prompts, examples, tests, or logs
  intended for Git.
- Tests and reproductions use temporary or dedicated Memory Spaces, never the
  user's active space.

## Environment and commands

The package metadata supports Python 3.10 or newer, but use Python 3.12 for
Studio development and its current eval suite. Studio's local MLX path also
requires a compatible native Apple Silicon environment; a sandboxed Linux or
non-Metal run is not equivalent. Inspect the active interpreter and installed
versions before changing an environment.

Common entry points:

```bash
python -m pip install -e .
python web/run.py --mode llm_tts --llm deepseek --space demo-zh --lang zh --confirm_ms 200
python web/run.py --mode llm_tts --llm local --space demo-zh --lang zh --confirm_ms 200
python web/run.py --mode realtime --space demo-zh --lang zh
```

These are references, not commands to run for every task.

## Verification

Select checks that cover the changed ownership boundary.

```bash
python -m py_compile path/to/changed_file.py
python -m unittest tests.test_offline_engine tests.test_session_context
python -m unittest evals.test_dialogue_harness evals.test_prompt_config
python -m unittest evals.test_prompt_logging evals.test_prewarm_scheduling
node evals/test_mic_capture.cjs
node evals/test_transcript_ui.cjs
git diff --check
```

Additional `evals/` scripts may require models, credentials, platform-specific
hardware, or benchmark fixtures. Inspect them before running. A latency script
is not a unit test, and synthetic timing does not establish live perceived
latency.

New files under `tests/` are ignored by default. Keep local regressions there;
do not force-add them unless the user explicitly asks to publish tests.

For streaming or audio changes, cover the applicable capture, partial ASR,
final ASR, EOT, gate route, early-speculation commit/reject, backchannel, echo,
pause/resume, underflow, drain, interruption, disconnect, and stale-output
paths.

## Git hygiene

- Preserve existing work during pull, rebase, and conflict resolution.
- Never use destructive reset or checkout as a shortcut.
- Stage only reviewed files and inspect the staged diff before committing.
- Never stage runtime traces, user memory, recordings, environments, caches,
  model weights, generated results, or local experiments.
- Before a requested commit or push, verify author name, author email, target
  branch, and remote URL.

## Final review checklist

- The change belongs to the correct Studio layer.
- Memory, dialogue policy, prompt, provider, and browser responsibilities remain
  separated.
- Shared behavior is consistent across affected reply/provider paths.
- No event-loop blocking, GPU-stream violation, lock inversion, orphaned task,
  stale result, or cross-session ownership issue was introduced.
- Comments follow the English and information-safety rules.
- Sensitive traces, user data, credentials, local paths, and generated files are
  absent from the diff.
- Focused regressions and documentation match the behavior; untested live paths
  are disclosed.
