# Local depth routing

The recommended router is Qwen3-0.6B on MLX or Torch. It selects ordinary versus
deep reasoning; VoiceMem continues to own memory eligibility. The resulting
`ThinkingDecision` composes these signals into the existing reply modes. Routing
cannot end a voice turn or release private speculative output.

Startup prepares default weights under `studio/models/reply-router/Qwen3-0.6B`.
Mac uses the process MLX scheduler; CUDA uses the selected Torch device and
existing device locks. `STUDIO_ROUTER_BACKEND=mlx` or `torch` can explicitly
select the implementation. Explicit model paths remain caller-owned.

The router receives current text and bounded conversation history. Session
Harness prompt edits apply through the captured prompt scope and are included
in classification cache keys. Installed adapters retain the classification,
short-text generation and warmup contracts documented in
[component bindings](../component_config/README.md).

## Waiting speech

Default long waiting speech uses four neutral phrases per language. Startup
prepares all eight PCM16 clips using the selected speech provider and default
speech rate. Stable voice identities persist clips under ignored
`cache/work-filler`; provider identity, language, exact text and TTS instructions
are part of the cache key. Providers without a stable identity keep memory-only
banks. No classifier, chat API, disk I/O or TTS inference runs on a warmed hit.
Each conversation avoids its last three emitted phrases.

Changed voice preferences use the prepared text and synthesize with the captured
instructions rather than replay stale audio. Explicit `filler_system` or
`filler_input` edits keep dynamic generation through Qwen's local short-text
generator. Default cached fillers make no such generation requests.

The 1.5-second default eligibility threshold, cooldown, readiness race, silent
preference, full clip playback and client completion acknowledgement remain in
the conversation handoff owner. Preparing phrases does not authorize playing
them on ordinary turns or while the user is speaking.
