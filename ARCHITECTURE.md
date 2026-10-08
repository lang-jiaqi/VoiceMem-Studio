# VoiceMem Studio Architecture

This document describes the current Studio architecture, ownership boundaries,
runtime flows, and extension points. It is a design reference, not a development
log or benchmark report.

## 1. Product scope

VoiceMem Studio is a conversational voice application built around the
`voicemem` memory framework. The repository contains both:

- the reusable memory package; and
- a latency-oriented voice experience with browser capture, turn-taking,
  configurable prompts, local or remote reply models, speech synthesis,
  interruption, and observability.

The memory system does not depend on a particular reply model, speech provider,
or transport. Studio composes those concerns at the application boundary.

```mermaid
flowchart LR
    Browser --> Capture[Capture and echo defense]
    Capture --> Turn[Streaming turn pipeline]
    Turn --> Memory[VoiceMem memory plane]
    Turn --> Dialogue[Dialogue policy]
    Memory --> Reply[Reply pipeline]
    Dialogue --> Reply
    Reply --> Speech[Speech pipeline]
    Speech --> Playback[Playback and interruption]
    Playback --> Browser
    Playback --> Context[Session and memory finalization]
```

## 2. Architectural planes

### Client plane

Owned by `studio/apps/ui/`, `studio/web/transport.py`, and the AudioWorklets:

- microphone permission and browser audio graph;
- browser acoustic echo cancellation;
- residual playback-reference filtering in `mic-capture-worklet.js`;
- WebSocket text, PCM, and playback-checkpoint messages;
- PCM buffering and rendering in `pcm-player-worklet.js`;
- subtitles, Memory Space controls, and memory visualization.

The client reports rendered source-sample progress. Network receipt does not
mean that audio was heard.

### Conversation plane

Owned by `studio/core/voiceagent.py`, its `utils/` modules, and `studio/harness/`:

- turn lifecycle and session wiring;
- unfinished-utterance handling;
- spoken backchannel policy and cached clips;
- tone-label parsing and TTS instruction selection;
- allowlisted, conversation-scoped Self Harness overlays selected by the reply model;
- early reply buffering and commitment;
- interruption and output cancellation;
- short-term Session Context.

This plane may use memory results but does not define factual or affective
storage semantics.

The four policy files in `studio/harness/{persona,speaking_style,reply_modes,
turn_taking}/policy.py` are immutable runtime defaults. Each exposes its
pre-Self-Harness prompt as a `DEFAULT_*` value, and execution remains in the
corresponding `studio/core/utils/` component. Studio uses one prompt per purpose
without language variants. A shared persona rule permits natural Chinese/English
conversation, mixed terminology and explicit user language requests. No per-turn
system message forces a reply language. `--lang` supplies an initial UI/default
language; stored Space language is the fallback for ambiguous first inputs and
the opening, not a restriction on speech or stored facts.

`studio/harness/self_harness/policy.py` is the fifth, coordinating policy. It
declares a typed overlay schema spanning persona interaction style, speech rate,
tone, reply length, reasoning depth, spoken backchannel frequency, and work
filler behavior. The same reply call emits a private `<self_harness>` JSON
prefix before its tone tag. The reply pipeline removes that prefix, rejects
unknown domains, fields, values, and changes larger than two fields, and stores
accepted changes on the current WebSocket `Conversation`. It never rewrites the
four default prompts, Python source, provider settings, or arbitrary runtime
parameters.

Self Harness changes are small session overlays. The model may emit them only
for explicit user preferences; transient content cannot change the profile.
Current profile context marks fields changed during the preceding two replies.
A conflicting request in that window is emitted as a candidate but staged by
the runtime instead of being activated. The model asks for confirmation; only
the same candidate repeated on the next explicit request is applied. This
hysteresis is enforced for the stored profile and the current reply's TTS, while
relative speech-rate requests move only one step. The runtime also enforces the
schema and per-turn change bound independently of the prompt.
Speech rate is appended to request-scoped TTS instructions, reasoning preference
is composed with Gate memory eligibility, and turn-taking preferences mutate
only the session state machine. Speculative replies stage their entire update
and apply it only when that reply is committed, so a discarded EOT prediction
cannot alter later turns. Realtime speech-to-speech prompts do not use this text
control protocol.

The owning browser can read and change the same profile through
`self_harness_get`, `self_harness_update`, and `self_harness_state` WebSocket
messages. The server exposes only enum values and display labels, never private
prompt or TTS instructions. Ordered enum fields are shown as discrete slider
stops; unordered enums use ordinary choices. An explicit settings choice applies
immediately, while later conflicting model changes still use the confirmation
window. The settings page also supports a bounded, session-only Persona prompt
through `harness_prompt_update`. It is appended as a user instruction and does
not rewrite the immutable Persona source prompt.

The local Qwen3-0.6B classifier selects ordinary or deep reasoning after ASR.
Studio combines that depth with VoiceMem memory eligibility into the existing
`fast`, `medium`, or `slow` reply modes. Missing default weights are downloaded into the
`studio/models/reply-router/Qwen3-0.6B` directory with visible progress.
Explicit model paths remain caller-owned.

`studio/core/core.py` exposes service setup and the chronological controls:
listen, merge continuation, stop warming, wait on unfinished speech, route,
interrupt, commit early output or start a reply, and cancel on disconnect.
`VoiceAgent` composes application utilities. `Conversation` owns per-WebSocket
state and tasks explicitly; utility methods implement each control without
moving model inference onto the event loop. GPU and Torch schedulers remain
process-scoped. Speculative generation captures its memory instance and space
before scheduling; cancellation also reaps pending route work.

The App's first local launch chooses the VoiceMem memory API, prepares the
memory-stage models, then chooses the Studio Agent reply API. It passes separate
`--memory-llm` and `--llm` values into one Studio backend. The direct Python CLI
retains its simpler compatibility behavior: an interactive launch without
`--llm` asks once for DeepSeek, Qwen, OpenAI, or an MLX-local reply model;
remote selections are also used for VoiceMem memory work, while local replies
default memory work to DeepSeek. Explicit `--memory-llm` and `--llm` select
the roles separately. `--check`, explicit provider flags, realtime mode, and
non-interactive commands do not prompt. Non-interactive startup defaults to
DeepSeek reply, Breeze TTS, and the `studio-zh` Memory Space.
An explicit `--space` selects another existing or new space; stored language and
memory data are preserved when the default selection changes.
Without `--memory_root`, each selectable Space owns its existing named directory
under the repository memory parent. An explicit `--memory_root` preserves the
VoiceMem contract of one exact database directory, not a parent for new Spaces.
Studio binds that directory to the startup `--space`, reads its existing metadata
and counts, and rejects creating or selecting another Space. The browser exposes
this fixed selection and hides new-Space controls. Existing memory is neither
moved nor rewritten to introduce child directories.
Qwen is selectable with `--llm qwen`: `qwen3.6-flash` uses the international
DashScope OpenAI-compatible endpoint with streamed content and request-scoped
thinking. Credentials come from `DASHSCOPE_API_KEY` or ignored `.env.qwen`;
ordinary replies disable thinking, while deep reasoning preserves the router's
selection. Memory workers and display rewriting use the selected memory model.
Startup reports missing credentials, dependencies, versions, assets, policies,
and weights before opening memory.
Repository `.env` files supply credentials and deployment options without overriding
exported environment variables. Both platforms use `python -m studio` after
activating their separately installed environment; `python web/run.py` remains
the compatibility entry point. `studio/scripts/run_cuda.sh` and
`studio/scripts/run_mlx.sh` select the corresponding interpreter and backend;
the root `scripts/run_studio_*.sh` paths are compatibility wrappers. They
enable verbose terminal logging, and forward arguments to that same entry point.
The backend defaults to CUDA on Linux and MLX on macOS, with CUDA devices
defaulting to `cuda:0`. Environment variables and explicit CLI flags remain
optional overrides for the backend and ASR/router or TTS device. Complete legacy weights are linked
into `studio/models/`; missing artifacts download there with resumable caching.
The DeepSeek provider is also passed into VoiceMem's internal extraction,
annotation, and cleanup workers, so their model and endpoint cannot fall back to
an OpenAI model name.
The MLX extra pins Transformers 5.16.1 with Hugging Face Hub 1.x to satisfy
MLX Audio 0.5.1. The separate CUDA extra pins Torch 2.8.0, Transformers 4.57.3,
Hub 0.x and qwen-tts 0.1.1 for the native Breeze streaming runtime.
The memory package accepts Transformers 4.52.3 through 5.x;
its recognition and reply contracts remain unchanged.
Selected model warmup failures prevent serving a silently degraded pipeline.
`--check` performs inspection only. Model directories contain weights, while
model classes and factories live in `core/utils/<component>/`.

### Deployment boundary

The Python wheel distributes the VoiceMem SDK, sample audio and runtime prompts.
Studio Python helpers remain included for existing SDK speech-provider and default
persona imports. The complete Studio application runs from a repository checkout;
its UI, avatar assets and model weights are not wheel resources. Package-data
patterns deliberately exclude partial Studio frontends and reviewed Studio voices.

Native Apple Silicon setup installs the Studio extra with
`studio/constraints-macos-py312.txt`, a Python 3.12 profile pinning direct and
transitive dependencies, then runs `pip check`. It does not apply these Mac
constraints to the independent CUDA profile. Local runtime logs under
`prompt/logs/` remain ignored.

Linux/NVIDIA deployment uses the root compatibility `compose.yaml` and
`studio/deploy/Dockerfile.cuda` to run
the same Studio entry point and in-process Breeze provider. The image pins the
independent Breeze source revision and CUDA dependency profile. The default
image tag is `voicemem-studio:torch2.8-cu128`, with Torch/TorchAudio 2.8.0,
TorchVision 0.23.0 and CUDA 12.8 validated during the build. The host retains
ownership of its NVIDIA driver. Model weights are acquired at runtime.
Only GPU 0 is exposed to the container. The process
runs as a non-root user with an init process; memory, results, request logs,
model weights and compiler/download caches live in separate persistent volumes.
The harness directory and TTS JSON are read-only configuration mounts, loaded on restart.
Credentials are runtime configuration, never image build inputs. Build-context
filters exclude private traces, memory, recordings, checkpoints and host environments.
The default published port is loopback-only; exposing the application requires
deployment-level HTTPS and access control. HTTP health becomes available only
after the existing model warmup completes.
Native command-line startup also binds loopback by default. The CUDA container
explicitly binds `0.0.0.0` inside its isolated network namespace so Compose can
publish the service on the host's loopback address.

Apple Silicon retains native MLX deployment through the existing launch script;
`studio/scripts/setup_mlx.sh` prepares the matching environment without replacing
an existing environment or credential file. No Metal-in-Linux-container path or
host inference proxy is added. Neither deployment changes memory semantics,
provider contracts, speech segmentation or model scheduling.
`npm start` is the canonical source entry and runs local Python setup after the
VoiceMem API choice, only when the environment marker is absent or the dependency
definition changed. `studio/scripts/start_app.sh` owns the Mac App shortcut;
`scripts/start_studio_app.sh` and `scripts/setup_studio_mlx.sh` are compatibility wrappers.
The lower-level backend launchers remain
available independently.
The native setup also installs the locked desktop-pet dependencies. Headless
containers disable spawning Electron through `STUDIO_DESKTOP_PET=0` while
retaining the upstream pet-observer WebSocket and events. Linux, including WSL,
never auto-spawns the desktop pet even when the environment flag is enabled.
Native macOS launches keep the optional automatic pet lifecycle by default.

`studio/core/voicemem.py` is the integration boundary for creating VoiceMem and
opening its native stream; Studio perception obtains VoiceMem's process-shared
embedding model through the same bridge rather than importing the Web transport.
The current deployment is one process: memory is
initialized before Studio starts accepting connections. There is no memory RPC
server or second startup command. Streaming ASR, final ASR, VAD, EOT, and their
worker/epoch guards remain in VoiceMem because the memory package also uses them.
`studio/core/utils/asr/` initializes shared recognizers from Studio weight paths.
`studio/paths.py` owns Python resource lookup: reviewed voice assets are in
`studio/resources/voice/` and pet assets are in `studio/pet/`; legacy repository
siblings remain a compatibility fallback. The repository root still owns runtime Memory Spaces,
results, and legacy model reuse, while weights default to `studio/models/`.
The startup loader reads `.env` before importing `studio.paths`, so optional
model and voice directory overrides are effective on a fresh process.
Studio-owned prompts use `studio.prompt_config` as their path authority;
Studio startup inspects that effective prompt directory rather than assuming
the repository root. The desktop pet preparation follows the same Studio-first,
legacy-second source rule. The App-managed Python backend requires the package
entry point and root package metadata, while Docker startup separately requires
Compose; local MLX/WSL startup is not coupled to a Studio Compose file.

`studio/web/` owns browser assets, HTTP/WebSocket transport, and the desktop
pet bridge. Startup rejects the unsupported `STUDIO_PUBLIC_DEMO` flag before
loading dependencies, models or Memory Spaces.

In Studio `llm_tts`, a stranger turn excludes retrieved memory and recent
session messages from the reply request. The speaker gate's identity decision
stays in server state and diagnostics; it is not included in reply prompts.
The gate also skips or discards speculative replies before playback. Background
ingestion and speaker tracking keep their lifecycle so a later matching voice
can clear the gate.

`studio/apps/` owns the Windows/macOS Electron desktop client and pet. Linux is
a backend deployment target, not a desktop release target. Windows runs capture,
playback and the pet natively, while local CUDA inference belongs in WSL2 or its
Docker backend. macOS uses native MLX or a remote service. Both clients share
the same Web UI and observer contracts. The backend root serves the shared
frontend from `studio/apps/ui/`, with assets under `/ui/`. Its first page reuses the original mode-selection homepage, with labeled
image cards linking to the technical or digital-human visual style. A centered
settings shell lives inside each style, with an icon sidebar for appearance,
text, conversation preferences, and components. The conversation panel mirrors
Self Harness snapshots and uses a Persona text area, a fixed-choice tone control,
and discrete sliders only for ordered fields. Backchannel frequency uses a
four-node curve editor whose nodes are the expected acknowledgement quotas for
the 0–3s, 3–6s, 6–10s, and 10s+ speech phases. Fractional values are sampled to
integer per-turn targets while preserving their expectation; the shared
first-six-second cap and the refractory cooldown still apply. Preset Self Harness
values provide the initial curve, while an explicit user curve is scoped to the
current conversation and is reflected in subsequent snapshots. The component
settings tab presents the fixed
input, memory, reply, speech, and pet components as a draggable canvas. Only
normalized card positions are stored in browser local storage; components cannot
be removed. `/ui` responses use `Cache-Control: no-store`, and the entry pages
version their settings assets so an Electron reload cannot retain an older
settings implementation after an update. `/api/components` supplies non-secret provider, model, endpoint, and
readiness labels. For an App-owned backend, a narrowly scoped isolated preload
allows the main-frame settings page to update the memory or reply model service
and request a supervised backend restart. The main process validates endpoints,
keeps keys out of command arguments and backend responses, and encrypts persisted
keys with Electron `safeStorage`. A new configuration is persisted only after its
backend becomes ready; failure restarts the previous configuration. Existing or
remote services remain read-only because their lifecycle belongs to the server.
Before replacing an App-owned backend, Electron waits for the old child to exit
so the replacement does not race it for port 8787. Failed replacements also
exit before the previous configuration restarts. The managed server bounds
Uvicorn's graceful shutdown while the old UI still has an open WebSocket.
Electron force-stops only its own child if it remains after that grace period;
if the child still does not exit, the old configuration stays selected and no
replacement process starts.
`/legacy` retains the previous Studio renderer;
`/classic` retains the older demo. The desktop opens this same root and keeps its
connection configuration window hidden on a successful startup; connection errors
reveal configuration. Desktop clients require the updated backend assets.

`studio-client.js` owns one page-local WebSocket and AudioContext, reuses the
existing capture and PCM AudioWorklets, and sends rendered-sample checkpoints,
actual playback RMS levels, pause/resume, and filler completion on the existing
protocol. The RMS event feeds the App-owned Live2D pet and is not treated as a
playback checkpoint. The client retains a bounded, memory-only PCM cache for
completed replies so the reply speaker control replays the audio that the
backend actually generated. When the user starts the microphone, the client
sends `conversation_start` after capture is ready. The server generates at most
one short opening per WebSocket session through the normal reply and TTS stream;
text sends and settings connections do not trigger it. The opening may draw on
low-sensitivity facts from the current Space, and falls back to an ordinary
greeting. The model selects a tone for the opening from those facts using a
detached auto-tone snapshot. This tone neither updates Self Harness nor
participates in the cross-turn tone smoothing used by later replies. Only its
heard assistant text enters the session context; it creates no user transcript
or long-term memory ingest.

Interrupted replies retain only the source samples
reported as rendered by the playback worklet. Replays do not emit playback
checkpoints and the cache is discarded on page refresh. UI replies and
perception come from backend events. Confirmed input IDs deduplicate transcripts
and merge continuations; interruption uses the backend's heard prefix. Streaming
reply deltas update the active text node without rebuilding prior chat turns.
Changing style, conversation, or leaving the page closes its connection. UI
language changes retain the connection and do not reopen or rewrite memory. Chat
lists are page-local and reset on refresh; opening a previous list item starts a
new backend context for subsequent input. Both styles let each new chat choose a
Memory Space. Chats retain that choice and share the Space's persistent memory;
switching chats selects the corresponding backend Space before further turns.
The technical and digital-human pages render assistant Markdown
through one DOM-only preview, including headings, emphasis, lists, quotes, links,
tables, code and delimited LaTeX math. Math is recognized before Markdown escapes
or emphasis so `\[...\]`, `\(...\)`, `$$...$$` and `$...$` retain their commands
and subscripts. Fences labelled `math`, `latex` or `tex` also use math layout;
ordinary code, escaped dollars and currency remain literal. A vendored KaTeX
0.18.10 runtime, stylesheet, fonts and license ship with the UI; users do not
need a browser CDN or an additional npm install for formulas. KaTeX receives untrusted input with resource/HTML commands disabled, fresh macros
per expression, bounded expansion, bounded size and an 8192-character input
limit. Incomplete streaming math waits for closure; malformed final input falls
back to text without failing the reply. Up to 128 rendered expressions are
cached in the page to avoid repeated parsing of unchanged formulas on each
streaming frame. Formula base size inherits the reply font; superscripts and
fractions retain their mathematical proportions, and wide display formulas
scroll inside the reply. Root containers retain their page-specific typography; nested
paragraphs, headings, code and tables inherit the existing font size and line
height. Desktop reading-size rules target the reply root, not nested Markdown
paragraphs, so the common stylesheet cannot enlarge only part of an answer.
Browser text-size adjustment is fixed at 100 percent to prevent independent
mobile block inflation; user zoom and content-size settings remain available.
Heading weight distinguishes sections without enlarging subtitle text.
Raw HTML remains text, images display their alt text, and only HTTP, HTTPS or
mailto links are clickable. Streaming paints coalesce per animation frame;
completion/reset cancels stale paints. User transcripts remain plain text, and
copy/history retain source Markdown. Interrupted previews render the heard
prefix without requiring closing formatting markers.
The brain backdrop remains an illustration, while graph nodes are rebuilt from
the current Space's `/api/memories` snapshot. Node placement uses the visible
hemisphere bounds and keeps space between memory nodes and domain labels on
each resize. Hovering or selecting a node shows
a compact card beside it, following the Web brain's interaction pattern. The page checks for asynchronous
ingest changes after a turn and redraws only when memory content changes. Its
background canvas is redrawn on resize or view changes rather than every frame.

The technical page's liquid orb owns its WebGPU device and animation loop.
Transient adapter, device or rendering failures show the existing static preview
and allow at most three delayed recovery attempts per foreground visit. Recovery
retains the current voice state and motion parameters. Unsupported WebGPU and
shader compilation errors do not retry. Hidden, offscreen and settings views
pause frames and pending retries. Page departure releases the device and uniform
buffer, invalidates asynchronous initialization callbacks, and a restored page
reinitializes on demand. Healthy rendering keeps its existing shader, resolution
and animation cadence.

The panels show real per-turn recall results without demo records or rule replies. A local
configuration page owns connection IPC. The Studio renderer has only the model-service
bridge described above and no Node integration. Both renderers use context isolation and
sandboxing. Microphone requests are limited to main-frame audio from the selected
origin and require user approval; remote connections require HTTPS, while HTTP
is accepted only on loopback. Settings and browser state live in the desktop
application-data directory, not in the backend's memory or credential files.

The source desktop entry (`npm start`) first verifies its locked Electron, PixiJS,
and Pixi Live2D files. Missing packages trigger `npm ci --include=dev`; after
that, a missing Electron binary runs Electron's installer explicitly, since the
package does not declare an npm install script. A complete
installation performs no package-manager or network work. The managed install
explicitly enables lifecycle scripts so a user-level npm `ignore-scripts`
setting cannot leave Electron without its platform binary. It then owns an optional local backend lifecycle.
The separate `npm run start:remote` entry performs the same locked client-resource
preparation but removes managed-backend state before Electron starts. It opens
the connection configuration with an empty, required address, without first
waiting on the default loopback service and without checking local Python, WSL, accelerators,
credentials, or inference models. This is the source entry for Intel Macs,
clients without a compatible local GPU, and already running local or remote services.
Without encrypted App model configuration, local startup first selects the
VoiceMem memory provider and reads its key through masked terminal input, then
prepares the missing or stale Apple Silicon Python environment. A short-lived
preparation process checks and acquires the shared memory,
perception, and transcription models. Keys are inherited only by preparation,
Electron, and the backend process, never command arguments. After preparation,
the user selects the Studio Agent's dialogue-model API; a different provider
requires its own masked key, while the same provider reuses the memory key.
The last menu option clears managed state and opens only the connection page;
the address is required before the style selector. This is not a second reply
service: VoiceMem's selected API is used for memory work, while Studio owns
the complete agent (memory integration, ASR, dialogue, TTS, and UI). The `start:remote` shortcut
avoids even the memory preparation. VoiceMem initialization remains inside the
eventual Studio backend, not a second service. Later managed launches reuse
the App configuration without another terminal prompt. After the first
backend reaches readiness, Electron persists them only through OS-backed
`safeStorage`; the connection settings file remains secret-free.
Electron starts `.venv/bin/python` with the MLX backend on macOS, or invokes
`.venv-cuda/bin/python` through `wsl.exe` with the CUDA backend on Windows. Both
paths prefer loopback port 8787 and select an available loopback port if it is
occupied. They disable the backend-owned pet, acquire the
remaining reply and speech models, run strict warmups, wait for the shared Web
page for up to 30 minutes by default, and only then create and show the style selector.
The managed backend echoes a per-launch instance ID on the Web root response;
the App accepts readiness only from that instance and reports an occupied port
if another service wins the selected port before startup completes. Existing-service
connections do not require this instance ID.
During managed model-service changes, the App retains ownership of the old
backend until it exits and prevents connection actions from cancelling startup
or restart. App exit signals either the running or stopping owned backend.
The timeout can be increased for unusually slow first downloads through the
desktop launch environment. Existing-service connections retain their shorter
readiness timeout. Managed startup never
creates the local connection/status window; preparation progress stays in the
terminal, and startup failures use a native error dialog. The app stops this
owned process on exit. Missing Python, WSL, driver, or dependency prerequisites produce a startup
error; model artifacts are checked and downloaded with resumable provider
caches. The desktop entry does not install or modify system components.

Studio's main-thread entry converts the default SIGTERM action into a normal
Python exit. Uvicorn retains signal handling during serving and completes its
graceful shutdown before replaying the signal; TTS cleanup and Python resource
finalizers can then run. The same handler covers interruption during preparation
or warmup, preserves caller-installed handlers, and restores the original handler
when the entry returns. App restart keeps its existing timeout and forced-stop
fallback.

On Windows, an opt-in configuration setting can also start an existing local NVIDIA
Compose service through a local Docker named pipe. Docker Desktop must already
be running with its WSL2 backend; the app never starts or installs the Docker
engine or WSL itself. The Unix-socket path remains available to isolated Linux
development tests, not as a Linux desktop release. Startup uses the user-approved project and optional local override,
never builds or pulls an image or recreates an existing container, reads the published port and waits for the Web
page to become ready. Connection attempts own cancellable CLI and readiness
work; stale attempts cannot replace a newer window. Closing the app never stops
the shared container. Installed desktop packages can still connect to an already
running local or remote service; the source-managed backend requires the repository
and its prepared Python environment. There is no Metal-in-Docker path. Desktop packages contain the shell, Electron
runtime and pet display resources, not inference weights, Python, recordings,
credentials or memory data.

The desktop app also owns one optional transparent pet window in the same
Electron application. Packaging selects the `studio/pet/` renderer, animation,
observer, rattan/white-vine Cubism model, expressions and Pixi runtime without forking them or
bundling another Electron. The generated HTML connection CSP is adapted for the
desktop package, and the pet session permits the Cubism Core source plus the
selected observer. Resource preparation backs up recognized Canvas and older
Live2D build output before migration, excludes backups from the package, and
rejects unknown files. Its isolated preload exposes window controls,
not Docker or settings APIs; IPC validates the pet's exact main frame and document.
The pet session permits bundled resources, the Cubism Core source and the selected
`/ws-pet` endpoint and denies device permissions. Its call button sends a trusted,
main-frame-only IPC request to toggle the existing Studio page's voice control with
a user gesture; the App renderer retains microphone permissions, capture, playback,
and conversation ownership. Desktop voice capture may continue while the App window
is hidden. Opening settings pauses microphone capture but retains the current
WebSocket so session preferences can be read and changed; page navigation and
explicit stop still end it. Service changes close
the old observer before opening the new one; closing the Studio window closes the
pet. Position and a 40–150% expanded-window scale live in desktop app data; older
position-only files default to 100%. The shared renderer uses a portrait call
background, a single voice toggle and a collapse button, without numbered action
controls. Dragging any of its four corners resizes around the opposite corner, and
dragging the character or background moves the window. Window controllers constrain bounds to
the display work area and resize without restarting the avatar pose. The dot
keeps its fixed size, and manual collapse suppresses observer-triggered expansion
until the user explicitly reopens it. Opening or focusing Studio leaves pet
placement under user control. The standalone controller provides the same controls
with its own saved position and scale. Native users can disable the backend's
automatic pet with the existing `STUDIO_DESKTOP_PET=0` to avoid duplicate windows;
containers already do this. The standalone launch and Web playback contracts remain unchanged.

Original `web/run.py` remains a thin compatibility entry point. Studio-specific
implementations have one owner under `studio/` and callers import them directly.
The old top-level `harness/` forwarding package is removed: Studio policies
are imported through `studio.harness` only. Studio-specific offline speech/TTS
development scripts live in `studio/tools/`; they are never part of startup.
The root `web/` compatibility launcher stays separate from VoiceMem's own Web
demo and is not a directory to copy wholesale during migration.

### Memory plane

Owned by the reusable `voicemem` package, following upstream module boundaries:

- `core.py`, `config.py`, `__init__.py`: public facade, configuration and lazy exports;
- `orchestrator.py`: cross-component search/ingest, capabilities and `SearchResult`;
- `stream.py`: ASR workers, turn snapshots, confirmation and speculative retrieval;
- `gate.py`: memory eligibility and interruption routes;
- `lang.py`, `llm_config.py`: language context and process-wide model roles;
- `reply.py`, `tts.py`, `audio_timing.py`: shared reply/speech adapters and contracts;
- `startup_check.py`: optional capability probes;
- `leftbrain/`, `rightbrain/`: factual and affective memory;
- `memory_api.py`: prompt-ready memory helpers.

These files contain their implementations directly. There is no module redirection
layer or parallel runtime/input/output package tree. Search, ingest and streaming
retain their original state, locks, executors and cancellation boundaries.

`leftbrain/memory_repository.py` owns backend writes, the Space mirror and
maintenance APIs. `memory_repository_v2.py` extends those writes with slot metadata
and summaries; it is an implementation, not a second independent memory engine.
`LeftBrain` continues to own classification, candidate expansion and ranking.
The extension selects its graph-store type before construction so base schema
initialization and migration run once. Existing tables and query algorithms remain
unchanged. See `voicemem/README.md` for the owner inventory and historical opt-in APIs.

`leftbrain/mem0_backend_store.py` caches one Mem0 client per resolved local
Qdrant directory. Before sharing it, the adapter serializes native Qdrant SDK
operations with one client-owned reentrant lock, also used when Mem0's entity
store reuses that client. Searches cannot observe vector and payload arrays
halfway through a write. Embedding, extraction, BM25 encoding, Mem0 history,
and result reranking stay outside this boundary. Each native operation releases
the lock; a multi-fact background ingestion does not hold it across the batch.
Different directories have independent locks. A contending query can wait for
an in-progress native operation; this does not guarantee zero additional latency
or make embedded storage safe for access from multiple processes.

The repository's historical `search_with_graph()` and `search_combined()` are
deprecated and unsupported. Direct calls immediately raise `NotImplementedError`
and direct callers to `VoiceMem.search()`; they do not query stores or start work.
The optional fusion adapter falls back to its existing plain search only when a
graph provider explicitly raises `NotImplementedError`. Custom graph providers
remain supported. Studio's cognitive-graph, ranking and dual-brain search flow
does not call these legacy entry points and remains unchanged.

### Inference and scheduling plane

Owned by provider modules and shared schedulers:

- `studio/core/utils/llm/local.py`: local MLX reply generation and prefix/KV caching;
- `voicemem/tts.py`, `studio/core/utils/tts/component.py`, `cache.py`: speech providers and local
  synthesis support;
- `utils/gpu_loop.py`: the single process-level MLX execution thread;
- `utils/torch_lock.py`: serialization for shared Torch/MPS work;
- the streaming ASR worker and dedicated final-ASR executor in `voicemem/stream.py`.

Provider adapters translate configuration and provider events into shared
reply, text, PCM, and timing contracts.

### Configuration and observability plane

Owned by:

- `studio/harness/`: Studio Web persona, context directives, and dialogue
  controls;
- `studio/prompt/llm_*.md` and `studio/prompt/llm_context.json`: package/default LLM prompt
  inputs;
- `studio/prompt/tts.json`: TTS tone and backchannel synthesis configuration;
- `studio/prompt_config.py`: validated Studio prompt loading and caching;
- `voicemem/utils/common/prompt_trace.py`: shared asynchronous request tracing;
- `studio/core/utils/logging_utils/component.py`: runtime log routing;
- `tests/`: deterministic regressions and shared synthetic fixtures;
- `evals/`: manually run latency and model-quality evaluations.

Prompt traces may contain complete conversation and memory context. They are
runtime data even though their schema is part of the observability design.

## 3. Dependency direction

```text
browser / application
        -> conversation and provider adapters
        -> VoiceMem public API
        -> orchestrator
        -> left brain / right brain / audio capabilities
        -> stores and model adapters
```

Allowed cross-cutting infrastructure includes normalized contracts,
configuration, locks, schedulers and logging. The memory pipeline does not import
Studio. Studio-specific persona, prompt loading, local reply, Breeze and tone
controls are imported directly from their owners under `studio/`. Shared speech
adapters and timing contracts live in `voicemem/tts.py` and `audio_timing.py`.
Their optional Studio providers retain the existing injection and provider cache.
Plain `import voicemem` and importing the API reply module do not load Studio or
models. Using the existing default reply persona loads Studio lazily.
VoiceMem and Studio import `voicemem.utils.common.prompt_trace` directly, preserving
one writer and one task-local trace context.

`studio/core/voicemem.py::open_memory` composes the existing ASR, reply and speech
implementations. Each component's `initialize.py` owns its defaults and construction;
the bridge passes the resulting reply callable and ASR/TTS factories into VoiceMem.
Replacement implementations use the same contracts:

| Component | Existing boundary | State ownership |
| --- | --- | --- |
| Streaming ASR | `feed`, `flush`, `reset`, `text`; `new_stream` for independent decoders | Decoder state per connection; reusable model weights |
| Final ASR | `transcribe(audio)` | Refinement adapter; VoiceStream owns scheduling and stale-result guards |
| Reply LLM | Async callable accepting text, memory context and history, yielding text deltas | Request generation; Conversation owns dialogue state |
| TTS | `stream(text, instruction)`, yielding PCM or timed audio chunks | Provider synthesis; Conversation owns playback and interruption |

No additional plugin registry or abstract interface is required. Provider credentials,
model acquisition and warmup belong to the selected implementation's initialization,
not memory retrieval or turn policy. The built-in startup manifest describes the
default implementations; replacing one also requires its matching readiness checks.

## 4. Mode and provider model

Three independent choices shape a Studio run.

### Memory mode

| Mode | Memory capability profile |
| --- | --- |
| `left_brain_single` | Factual memory without affective/audio perception |
| `text_mode` | Text memory with affective memory logic |
| `multi_modal` | Text memory plus ASR, voiceprint, and audio perception |

### Reply mode

| Mode | Output path |
| --- | --- |
| `llm_tts` | Reply provider emits text; TTS provider emits PCM |
| `realtime` | Realtime provider emits transcript and PCM directly |

### Provider selection

Studio can select remote or local reply and TTS implementations. Provider
selection changes generation and scheduling details, not Session Context,
memory semantics, output identity, or heard-text finalization.

The local MLX profile has additional scheduling constraints described in
Section 10.

The reusable package targets Python 3.10 or newer. The current Studio local
profile and eval workflow are standardized on Python 3.12.

## 5. Core contracts

| Contract | Owner | Meaning |
| --- | --- | --- |
| `VoiceMem` | `voicemem/core.py` | Public memory facade |
| `SearchResult` | `voicemem/orchestrator.py` | Combined left/right retrieval |
| `Turn` | `voicemem/stream.py` | Confirmed user text and prepared memory |
| `StreamState` | `voicemem/stream.py` | Partial ASR/VAD/EOT state |
| Gate route | `voicemem/gate.py` | `backchannel`, `shallow`, or `deep` |
| `Pending` | `studio/core/utils/contracts/component.py` | Application-ready confirmed turn |
| `ReplySink` | `studio/core/utils/contracts/component.py` | Hidden speculative output timeline |
| `TimedAudioChunk` | `voicemem/audio_timing.py` | Optional PCM text alignment |
| `AudioTimeline` | `studio/core/utils/audio_timeline/component.py` | One output's text/media clock |
| `SessionTurn` | `studio/core/utils/session_context/component.py` | Unpersisted dialogue context |

Shared code consumes these meanings rather than provider-native objects.

Input display messages use an opaque `input_turn_id`, distinct from assistant
`output_id` and persistent history IDs. Capture retains it across partials and
the accepted input, then rotates it for the next turn. An explicit continuation
merge includes `replace_input_turn_id` so the UI updates the original user bubble
instead of guessing from elapsed time. Untagged events retain legacy display
behavior. These identifiers carry no memory-write or interruption authority.

## 6. Input and turn flow

```mermaid
flowchart LR
    Mic[Browser microphone] --> AEC[Browser AEC]
    AEC --> Residual[Residual echo guard]
    Residual --> WS[WebSocket PCM]
    WS --> Stream[VoiceStream]
    Stream --> ASR[Streaming ASR worker]
    Stream --> VAD[VAD]
    Stream --> EOT[EOT scoring]
    Stream --> Gate[Turn Gate]
    ASR --> Final[Final ASR refinement]
    VAD --> Confirm[Turn confirmation]
    EOT --> Confirm
    Final --> Confirm
    Gate --> Confirm
```

### Capture

Browser AEC is the first echo-control stage. The microphone worklet receives a
playback reference and suppresses only highly correlated residual blocks. The
server-side text guard provides a second defense against assistant or
backchannel echo reaching ASR.

Capture batches target short, regular PCM frames. Capture and playback share a
sample-clock relationship so latency and echo logic can use explicit media
positions rather than wall-clock guesses.

The browser owns one live connection attempt and its microphone resources.
Startup is registered before opening the WebSocket; another start-button click
cancels that attempt instead of creating a second capture. Every asynchronous
startup continuation checks its owner, and late permission results have their
tracks stopped. End/disconnect/page exit invalidates ownership before removing
node callbacks, disconnecting the capture and playback-reference edges, stopping
tracks and closing the owned socket. PCM callbacks use the socket and sample
clock captured by their own attempt, never a later connection. AudioWorklet
module loading may be shared for one audio context, but capture nodes are not.
The ScriptProcessor fallback follows the same cleanup and ownership rules.
Old socket events and cancelled startup-memory responses cannot affect the new
session. This is a per-page guard, not a cross-client or server-wide session limit.

### ASR and final refinement

Studio defaults to FunASR `paraformer-zh-streaming`, which supports Chinese and
English independently of UI or Space language. Acquisition and strict warmup
use the same local FunASR artifact manifest. Each connection owns a decoder
cache, input buffer and cumulative text. Studio caches model prototypes by
backend/device and clones their stream state; accounts and Spaces reuse weights
instead of loading another FunASR model. Cold construction and inference follow
`TORCH_LOCK`. Stateless final-ASR weights are reused as well.
`--asr-device` / `STUDIO_ASR_DEVICE` selects streaming ASR independently.
CUDA's `--device` supplies its fallback; Mac keeps automatic ASR selection
independently of the Torch router device. Startup reports the requested ASR device and
the factory reports its actual device, separately from MLX routing and TTS.
Explicit English adapter calls retain the reusable Zipformer path. FunASR
buffers 600 ms chunks and appends 50 ms of zero-valued right context at flush.
Resumed input starts a fresh decoder cache while retaining the utterance text;
reset clears that text and cache for the next utterance.

Streaming ASR runs in a dedicated serial worker so chunk inference does not
block WebSocket input. At turn end, full-audio ASR refinement may run in a
separate final-ASR executor. Epoch checks prevent obsolete worker results from
overwriting a newer turn.
The recognizer remains serial. Confirmed decodes precede queued speculative
snapshots, and a stream cancels its unstarted snapshots at finalization.
Already-running native calls finish normally; different audio snapshots are
never treated as interchangeable.
`VoiceStream.prepare_audio` constructs connection-owned ASR and VAD adapters
off the event loop. VAD recurrent state and segment queues are private to a
connection; consumed segments are discarded because capture only needs speech
activity. Built-in recognizers expose `new_stream` for independent decoder
state. Custom injected objects without that method keep their existing object
semantics. Capture explicitly closes its stream and background perception tasks
on completion, error, cancellation or generator closure. Both chained and
Realtime session consumers close their capture iterator explicitly. Worker shutdown rejects
new input, invalidates queued epochs, resolves pending flushes and enqueues a stop
command. An active native call is allowed to return before the daemon exits;
connection cleanup waits only for the existing bounded ASR deadline, never
forcibly interrupts native inference or closes shared weights.

Studio's final SenseVoice recognizer keeps automatic language detection, accepting
Mandarin, English and Cantonese result tags. Nonempty results tagged as another
language or with an unknown tag return no refinement, so the existing stream
flush, frozen speculative snapshot or latest partial transcript supplies the
fallback. This prevents an out-of-scope language result from replacing usable
streaming text without another decode, translation or extra model. The reusable
VoiceMem recognizer remains unrestricted unless an allowlist is explicitly
supplied. The separate acoustic-emotion transcriber also uses automatic input
language detection rather than forcing Mandarin.

### Conversation language ownership

Capture estimates the main input language from confirmed prose, using recent
language in the same WebSocket and Space for ambiguous short inputs. Code,
formula syntax and isolated identifiers do not determine that fallback. The
captured `Pending.language` travels with routing, speculative generation,
continuations, fillers and deferred ingestion; an early output with a different
language estimate cannot be adopted. Partial recognition may select an eligible
backchannel bank but does not commit the session's language fallback.

Reply text follows the shared persona and user context naturally. TTS selects
instructions from the actual generated segment, so an explicit user request to
answer in another language works without a forced output-language control field.
Tone labels stay canonical; English tone and speech-rate instructions are
localized at the speech boundary. Backchannel caches are keyed by language and
voice. Missing matching clips stay silent rather than playing the other language
or synthesizing a bank during a live pause. Tone smoothing belongs to the owning
conversation, not the account-wide agent. Speculative tone is staged on the
output timeline and only committed with an accepted, heard reply prefix.

VoiceMem retains each instance's stored default. Studio enables
`follow_input_language`, and captured ingestion language scopes generated affect,
trait and attribution descriptions. Context variables isolate concurrent tasks;
native threads and executor jobs explicitly capture their parent's context.
Constructing or opening another Space does not change a process override.
Factual extraction preserves input language, and multilingual E5 retrieval uses
the same index for either language; short cross-language queries can still rank
imperfectly. Canonical slot/emotion keys stay fixed.
Existing persisted memories and Space metadata are not translated or migrated.
The compatibility `/api/lang` endpoint changes UI language and reports
`reply_lang: "auto"`.

ASR finalization gives full-audio refinement an 80 ms preference window, then
accepts the first non-empty result from refinement or streaming flush. Passing
the preference window does not cancel refinement while streaming is unfinished.
The two decoders share a one-second finalization deadline, including queue time;
if neither produces usable text, the last available transcript is retained and
the streaming worker advances its epoch to discard stale work. Timeout fallbacks
are logged even when detailed ASR timing is disabled. Standalone offline probes
and speculative snapshots use the same one-second wait limit. Cancelling a wait
does not forcibly terminate native inference, but late results cannot update
turn text. These limits bound ASR waits, not VAD turn detection, memory retrieval,
reply generation or playback.

The complete captured audio remains available for archive and final decoding
even when obsolete streaming chunks are skipped.

### VAD, EOT, and pause policy

VAD provides acoustic speech/silence state. EOT provides semantic completeness
evidence. `PauseGate` applies Studio dialogue policy for unfinished phrases and
spoken backchannels. The timeout remains the fallback when semantic evidence is
not decisive.

These stages answer different questions and keep separate state:

- VAD: is speech acoustically active?
- EOT: does the utterance sound semantically complete?
- Pause policy: should Studio wait because the user may continue?
- final ASR: what is the best final transcript?

## 7. Turn Gate and memory retrieval

`voicemem/gate.py` assigns a confirmed utterance to one of three routes:

| Route | Interrupt active output | Inject factual memory |
| --- | --- | --- |
| `backchannel` | No | No |
| `shallow` | Yes | No |
| `deep` | Yes | Yes |

The gate combines a closed backchannel vocabulary, high-precision lexical
rules, and an embedding fallback. Its complete-utterance result owns baseline
memory eligibility, including in Studio `llm_tts`. The depth classifier cannot
veto a Gate-approved memory turn. Deep reasoning can additionally request memory
to preserve the existing `memory_cot` mode. Speaker privacy still takes priority.
For short anaphoric follow-ups, Studio can also carry forward the latest user
turn in the same session and Memory Space when its lexical Gate route requested
memory. It classifies and searches with the bounded combined query while keeping
the current utterance as the reply input. This uses no additional model judge;
ordinary turns and backchannels keep their existing route.

For deep turns, speculative classify/search starts while speech is still in
progress. Query embedding work is shared within a search scope, and device
access follows the existing Torch lock boundary. A shallow or backchannel route
returns a normalized empty result rather than `None`.

```mermaid
flowchart LR
    Text --> Route[Turn Gate]
    Route -->|backchannel| Continue[Keep listening]
    Route -->|shallow| Empty[Normalized empty memory]
    Route -->|deep| Search[Classify and search]
    Search --> Left[LeftBrain rank]
    Search --> Right[RightBrain retrieve]
    Left --> Result[SearchResult]
    Right --> Result
```

## 8. Early reply generation

EOT can provide enough confidence to start reply work before final turn
confirmation. This is a latency optimization, not a change to turn semantics.

`ReplySink` initially buffers reply JSON events and PCM in one ordered private
timeline. Speculative replies remain private until final ASR and turn confirmation
show that the speculative input still covers the final utterance. Accepted user
transcripts are published separately by the conversation loop after input filters
and continuation merging, before routing or reply handoff. Cancelling a reply
cannot discard that user's display record. This publication never writes Session
Context or memory; their existing reply-finalization paths retain ownership.

```text
high EOT score
  -> freeze an immutable audio snapshot
  -> run final ASR for that snapshot in the background
  -> start speculative reply and TTS from the refined snapshot text
  -> buffer output in ReplySink
  -> finalize transcript and route
  -> compatible: commit buffered timeline and continue live
  -> incompatible: cancel provider work and discard the timeline
```

Cancellation removes stale reply, TTS, display, and GPU work before a new
response becomes authoritative.

For conversation-managed chained speech, `Reply` only generates output;
`Conversation` owns playback waiting and exactly-once context finalization.
Speculative generation may finish while still private, but cannot start a
playback-completion timeout, save history or enqueue ingest. Rejection only
cancels/discards it. Acceptance binds finalization to the confirmed `Pending`,
including its final text, audio and routing/privacy metadata, while retaining
the existing generated reply and reuse decision. Normal replies and delayed
continuation follow-ups use the same finalizer. Cancellation before a handoff
task's first execution still reaps any adopted generator and finalizes the
confirmed user input once. Legacy direct `Reply` callers retain their existing
standalone finalization path.

Studio uses EOT both to start speculative reply work and, after acoustic
silence plus pause-policy approval, to end the Studio user turn. `VoiceStream` owns the
immutable audio snapshot and final-ASR refinement; `studio/core/utils/capture/component.py` owns the policy
that starts LLM/TTS generation from the refined snapshot text. Streaming ASR
continues to update the browser while that work runs. Resumed speech before turn
commit cancels the buffered work before any transcript or audio is sent. Once EOT
commits the turn, the final ASR transcript becomes authoritative without requiring
an exact match to the earlier streaming hypothesis. Minor ASR repairs and spoken
fillers keep the fast path; material continuation after the frozen EOT text
cancels the stale reply and starts one reply from the complete turn. Conversation-
managed reply jobs do not emit user transcripts, including EOT snapshots. The
legacy `ReplySink` transcript replacement remains available for direct callers.
The UI finalizes user records by input ID and ignores later partials or duplicate
finals for those records; a real new input remains free to update live text. Generated speech
stays buffered until turn confirmation becomes the commit point for playback.

Clearly unfinished voice turns have a separate continuation path. The Studio session
plays an eligible cached acknowledgement, keeps the turn interruptible, and
merges resumed speech back into the unfinished text. If silence reaches the
2.5-second follow-up deadline measured from the last voiced frame, it sends a continuation-specific instruction to
the reply model so the assistant gently asks the user to finish their thought.

Pause protection and delayed follow-up use separate predicates. Brief planning
prefaces may keep the existing continuation window, but only explicitly
incomplete clauses authorize an acknowledgement plus delayed follow-up. Ordinary
negations, isolated unknown characters, complete word suffixes, and question
prefaces do not authorize that follow-up. Resumed speech cancels the timer and
merges the continuation; disconnect cancels pending work. Deferred tasks capture
their memory instance and space, and discard output if that ownership changes.
Trailing conjunctions remain incomplete even without an ASR punctuation boundary.
Follow-up wording acknowledges substantive preceding content when present; bare
openings receive a brief invitation to continue without invented explanations.
Speculative reply work is cancelled before entering continuation waiting.

The Web pause gate does not add a second minimum to the configured turn
confirmation: a complete voice turn remains eligible at the application's EOT
or `confirm_s` boundary. Only a lexically unfinished clause, question preface,
short subject/time lead-in, or complement-taking tail gets a 1.2 second
continuation window. Resumed speech clears that window, so the completed
utterance again uses the normal confirmation latency. Until confirmation,
partial ASR remains transient UI state rather than a chat bubble. It may seed
buffered speculative work, but that work is cancelled if speech resumes and
cannot become an accepted reply before confirmation.

After confirmation, Studio combines the Gate and local reasoning classifier:
ordinary reasoning with Gate-approved memory is `memory` (mem), ordinary reasoning
without it is `direct` (instant), and deep reasoning is `memory_cot` (mem+cot).
Only direct turns or privacy overrides discard speculative memory. A normal or
unavailable depth classification cannot erase Gate-approved results. Memory
modes reuse eligible speculative results or complete retrieval before generation.
A reply mode or contextual memory query change between an early snapshot and
final ASR invalidates buffered early output. Contextual follow-ups also check
that both paths retrieved the same memory before releasing early output.

One session-scoped `TurnTakingStateMachine` then chooses the handoff. Ready
audio is released directly. An ordinary predicted wait may use a cached
acknowledgement, while `memory_cot` may request an LLM-generated work filler
when main audio is not ready after an actual wait threshold. The wait is measured
from the last voiced frame, including confirmation and routing; generation and
playback of fillers begin only after the turn is confirmed. Main reply work runs
concurrently during that wait. Ready audio, audio-free completion or an error
releases the buffered reply without generating a filler. The harness configures
the wait threshold and session cooldown separately from probabilistic in-speech
acknowledgements. Self Harness `silent` disables work fillers; `reassuring` uses
a shorter wait threshold, while `auto` restores the default. A cooldown-blocked
turn remains ineligible across handoff retries even if the cooldown later expires.
A sent work filler reserves its clip duration plus cooldown;
an acknowledged playback completion extends that deadline when playback started late.
Skipped work fillers do not fall back to a cached acknowledgement. Existing
readiness races still cancel unplayed fillers if the main reply wins. Work-filler
eligibility has no random draw; readiness, preferences, cooldown and successful
synthesis still determine whether a filler is spoken.
First audio observations update the session estimate used by later
decisions. Main reply work runs into a `ReplySink` while either filler plays.
For every emitted end-of-turn filler, the browser reports actual playback
completion before that sink releases `answer_start` or main PCM. Generation
remains concurrent, but spoken filler and main audio never overlap or hard-cut
each other.

Work-filler wording comes from the already-loaded Qwen3-0.6B weights, not the
main reply API. It uses a separate non-thinking short-generation prompt, the
complete current input and at most two bounded history messages, without
changing depth classification or retrieving additional memory. Cold weights,
over-budget output, invalid text, failure or a queue-inclusive generation timeout
skip the optional bridge. Token budget and cancellation limit background work;
native inference already in progress cannot be forcibly interrupted. The
legacy reply-stream filler helper remains available to direct callers.

## 9. Reply, prompt, and speech flow

### Reply context

The reply input combines:

```text
Studio system prompt
+ current user input
+ recent session history
+ route-eligible persistent memory
+ contextual directives
```

Reply mode remains one public per-turn signal, composed from two separate inputs:
VoiceMem's existing Gate for memory eligibility, and Qwen3-0.6B for reasoning depth.
The model receives current text plus up to four context messages sharing the
existing 320-character history budget. It answers only whether deep reasoning
is needed and does not receive memory-prefetch hints. Studio-specific topic,
date and greeting regex shortcuts are not used. Each decision uses one off-loop
classification request, not a second model judge or a keyword override.
The editable depth policy treats recall, ordinary explanation, simple formula
application and requests for an already-derived result as ordinary reasoning.
Explicit current deep-thinking requests and genuinely complex derivation,
diagnosis or multi-constraint planning remain deep. History resolves references;
an earlier difficult topic or a verbose assistant answer does not carry a sticky
depth into the next turn. Few-shot examples use the same bounded-history/current-
input framing as runtime requests. An ordinary depth decision still retains
Gate-approved memory.
Inference executes on the shared `GpuLoop` for the MLX backend, using the same local Qwen weights
quantized to 8 bits at load time. Its immutable KV prefix includes only the
system prompt and examples; every request receives a separate cache copy after
verifying the token prefix. CUDA and explicit `STUDIO_ROUTER_BACKEND=torch`
retain the Torch path under the process Torch lock. Depth decisions are cached
by text and bounded
context, independent of the Gate; changing memory eligibility recomposes the
final route without rerunning depth.
Identical concurrent async requests also share their in-flight depth calculation
within an event loop. Only text and bounded history form the key, not memory or
turn objects. One cancelled waiter does not cancel another; the last cancelled
waiter releases the task, and native inference already running may finish safely.
Invalid model output uses ordinary reasoning; model exceptions preserve Gate
memory eligibility. The legacy class name and
hint argument remain for caller compatibility, but that hint no longer
participates in depth classification.
Provider-neutral request options carry the required reasoning across async reply
iteration: `direct` and `memory` use non-thinking generation, while `memory_cot`
uses high effort. Reasoning content remains private and is never spoken.
The DeepSeek/Qwen SSE adapter requires nonempty answer content at normal EOF;
reasoning-only completion is a failure, including `finish_reason=length`.
Before any nonblank answer content is emitted, it may retry once with a fresh
connection and the same input, reasoning effort and token budget. Partial answers
are never replayed, and cancellation does not retry. Stream diagnostics record
completion reason, duration and character counts without logging generated text.
The first-event timeout includes reasoning events; it does not impose a separate
deadline on reasoning duration or reduce the requested reasoning effort.

Auxiliary conversation-title generation follows the configured Studio reply
provider and model with its own short prompt. In particular, DeepSeek reply
mode uses DeepSeek credentials and never sends an OpenAI model name to that
endpoint.

`build_reply_context` is the shared context builder used by actual generation
and local-model prewarming. Keeping one builder preserves local prefix-cache
compatibility.
All Studio reply adapters opt into `context_as_system`: backend context follows
the immutable persona in the leading system message, history follows that
message, and the final user message contains only the actual input. The static
persona remains a reusable prefix; changing backend context precedes history.
OpenAI, DeepSeek, Qwen and local MLX replies share the message composer; local
prewarming uses the same ordering. Direct VoiceMem provider callers retain the
default context-in-user format and existing custom callable signatures.

The Studio persona treats relevant facts retrieved for the current turn as
available evidence even when absent from recent session history. Earlier
assistant denials of recall are not user facts and cannot override that evidence;
the reply should briefly correct its earlier mistake. Explicit user corrections
take precedence over older memory. Unrelated results, missing details, and
unresolved factual conflicts still require uncertainty rather than invented
answers. This is a reply policy within the existing generation call; it adds no
retrieval, verification request, or rewrite of stored history or memory.
`persona.memory_reply_context` places a short handling rule adjacent to each
nonempty retrieved-memory block in both text/TTS and realtime replies. The rule
distinguishes factual evidence from internal affective notes and applies to
Chinese and English facts without forcing the answer language. Empty-memory and
stranger paths receive no retrieved-memory block or handling rule.

### Prompt ownership

The Studio Web system prompt and dialogue context live in `studio/harness/`.
Legacy reply/default prompt files live in `studio/prompt/llm_*.md` and
`studio/prompt/llm_context.json`. TTS tone configuration is loaded from
`studio/prompt/tts.json`. The installed Studio package owns these files,
its parser and legacy reply persona. Callers import `studio.prompt_config` and
`studio.core.utils.prompts.legacy_persona` directly. `STUDIO_PROMPT_DIR` can replace the complete
configuration directory; the older `VOICEMEM_PROMPT_DIR` remains a fallback.
The provider-neutral request journal lives in `voicemem.utils.common.prompt_trace`;
both packages import this single state owner directly.
Request logs remain external runtime data under the ignored root `prompt/logs/`.

Prompt configuration is parsed and cached by `prompt_config.py`; malformed or
incomplete configuration fails validation rather than changing behavior
silently. Editing a default policy requires restart because prompt files are not
read in the speech loop. Self Harness profile changes are in-memory overlays and
take effect without editing or reloading those defaults.

### Self Harness, tone and TTS

The reply model prefixes tagged replies with a private Self Harness header and
then a tone tag. The speaking policy specifies the tag after the private header
so the two prompt rules agree on ordering.
`studio/core/utils/self_harness/component.py` validates and
removes the header. `studio/core/utils/tts/control.py` removes the tone tag,
smooths abrupt tone transitions when no fixed tone is selected, and converts it
into a TTS instruction. Neither control prefix is spoken or stored as assistant text.
Leading control headers are stripped even when the Self Harness overlay is
disabled; updates are applied only when that overlay is enabled. A reply-local
`PrivateControlFilter` also strips misplaced or repeated reserved control blocks
after tone parsing, before browser deltas, speech projection and source history.
It never applies updates from body text, retains only possible split delimiters,
and discards unfinished private blocks at EOF. Cancellation and provider failure
do not flush its pending delimiter. Only a validated leading header may update
conversation preferences.
Control delimiters tolerate case and surrounding whitespace variants in both
the leading parser and body filter. A known pipe/bracket tone immediately after
a removed body control block is also stripped, without changing the active tone
or preferences. Ordinary prose and code outside those boundaries are preserved.
The reply pipeline buffers a possible leading tone tag across deltas. If the
provider ends normally before that buffer becomes a recognized tag or reaches
the streaming fallback length, nonempty buffered text is delivered once as
plain reply text to subtitles, the audio timeline and TTS. Recognized tag-only
output and whitespace remain silent. Cancellation and provider failure discard
the pending prefix instead of flushing it; speculative output still follows
the existing `ReplySink` commitment and heard-prefix rules.

The TTS layer accepts plain 24 kHz mono PCM16 bytes and optional
`TimedAudioChunk` alignment metadata. Segment concurrency is selected by the
provider; local GPU providers can require serialized segments.

The SDK `voicemem.tts.speak_stream()` helper serializes synthesis while consuming
reply deltas in a separate task. Input, callback and synthesis errors wake the
audio consumer and propagate instead of leaving it waiting for output. Cancellation
or explicit generator closure cancels and awaits both tasks. Callers that break
iteration early must close the generator, for example with `contextlib.aclosing`.
Studio uses its own speech pipeline rather than this helper.

Studio consumes reply deltas in an owned task alongside segmentation, synthesis
and audio delivery. A failed synthesis segment, including a stream that returns
no nonempty audio, fails the reply instead of silently skipping that segment.
Stage failure cancels and awaits the remaining pipeline tasks, marks the timeline
interrupted and reports a generic error scoped to its output ID. Failed replies
never send `answer_done`; exception details are not client messages. Normal
streaming does not add a model call, retry or wait before first audio.
The request-options and reply-capture wrappers close their underlying streams
from the producing task, including cancellation while delivering a text delta.

Studio's default speech factory shares one Breeze instance across Spaces, selecting
MLX or CUDA from the existing backend profile. VoiceMem receives that factory through
its capability-injection contract; memory algorithms do not select speech vendors.
Reply and filler generation accept a conversation-owned speech provider through the
same `stream(text, instruction)` contract. Optional `preconnect` and `aclose` hooks
remain at the server lifecycle boundary. New adapters can use these existing
contracts without adding vendor-specific branches to routing or dialogue policy.

`studio/core/utils/tts/segmentation.py` owns sentence-first text boundaries and
the pending text buffer. A reply-local segmenter consumes plain text after tone
parsing, independently of LLM iteration, so a stalled token stream cannot prevent
a deadline flush. Sentence endings release promptly; comma boundaries are a
fallback for long phrases. Bounded first/rest waits and maximum lengths prevent
indefinite buffering. Confirmed active playback headroom permits a longer bounded
wait for subsequent phrases, without changing client playback or filler gates.
After private control and tone parsing, Studio sends the original Markdown to
the browser and keeps it as the reply history. A reply-local `MarkdownSpeech`
parser emits separate speech text plus source code-point boundaries before
sentence segmentation. It removes common formatting and link targets. Fenced
code becomes a brief viewing cue as soon as its bounded language header is
available; its body is skipped while the model continues streaming. Fences
labelled `math`, `latex` or `tex` use formula cues. Delimited math (`$...$`,
`$$...$$`, `\(...\)` and `\[...\]`) uses at most 256 source characters of
lookahead: short numeric expressions become spoken operators, short coordinate
tuples become comma-separated values without delimiters, and complex
expressions become viewing cues. Recognizable complex syntax can release a cue
before the closing delimiter arrives. Inline code retains short identifiers and
simple expressions; long, multiline or symbol-dense snippets use an inline
reference. Numeric expressions are parsed, never evaluated. Ordinary unmarked
arithmetic remains unchanged, and currency text is preserved when distinguishable
from math. Pipe-prefixed tables with a separator row become comma-separated
spoken cells. Cue wording rotates from a random reply-local starting position
in the generated prose language, falling back to the captured input language,
without shared cross-session cue counters. Language detection reads the same
accumulated reply lazily when a structured cue, arithmetic expression or TTS
segment needs it; ordinary deltas do not repeatedly scan the growing reply.
Repeated
block cues require 64 new prose characters and are capped at three per type per
reply; adjacent blocks do not each add another announcement. Short inline
references remain available where omitting them would break a spoken sentence.
The speaking harness asks the reply model to retain complete display material
and supply useful oral explanations around it, without duplicating viewing
cues. The projection itself does not infer or summarize a formula's meaning and
does not issue a second LLM request. Ambiguous syntax uses bounded lookahead;
ordinary prose does not wait for the entire reply or an emphasis closing marker.
Only normal EOF flushes pending syntax.
Segment and provider timing offsets refer to the actual synthesized text;
`AudioTimeline` maps heard prefixes back to the original Markdown before browser,
context or memory consumers receive them. Formatting-only skipped ranges advance
source boundaries without adding speech duration. A viewing cue maps to the
source material it references, rather than pretending the code was literally
read aloud; following unheard prose stays outside that span. Buffered short
inline content does not advance source boundaries before its speech is emitted.
Realtime/direct timeline callers retain their original plain-text behavior. Only unsent text may be
regrouped. Cancellation reaps the segmenter together with synthesis and
delivery, and discarded text is never flushed into a replacement reply. This
policy is shared by CUDA and MLX; their inference and PCM chunk settings are unchanged.

### Spoken backchannels

Pause acknowledgements and delayed continuation are separate decisions. Pauses
with at least four alphanumeric characters can receive a probabilistic cached
acknowledgement, including expressive statements; ordinary questions are
excluded unless they invite acknowledgement. Only explicit dangling clauses
schedule a delayed continuation prompt. Chinese ASR spacing and comma variants
are normalized for that clause check. Complete turns do not receive a cached
acknowledgement merely because reply generation is pending.
Explicit incomplete two-character openings (such as 今天 or 因为) bypass the
four-character acknowledgement minimum. Introductions such as 我现在是这么想的
also retain the continuation window. Completed answers and greetings do not.
The PCM comfort-noise level stays constant during speech and silence.
Incomplete clauses may emit a quiet continuer during an eligible in-speech
pause; delayed continuation remains a separate confirmed-turn decision. Each
utterance samples acknowledgement quotas independently for four elapsed-time
phases. The first three seconds select two clips with 80 percent probability or
one clip otherwise. Three to six seconds select one clip with 80 percent
probability or none otherwise. Six to ten seconds select one, two, or three
clips with 30 percent probability each, and none with 10 percent probability.
From ten seconds onward, the remaining utterance selects two clips with 80
percent probability or none otherwise.
Quotas are upper bounds when the user supplies fewer eligible pauses. Available
audio and enabled backchannels remain prerequisites. Buffered EOT speculation
does not suppress these in-speech acknowledgements. After a confirmed barge-in,
the user can receive them again once the main reply stops; the utterance's
original echo reference is retained. Active replies, unconfirmed barge-ins,
and detected echo still suppress acknowledgement playback.
Self-introduction segments use a shared TTS arc for both DeepSeek and Qwen:
bright and proud initially, then explicitly sad and slower at the limitation
clause. Qwen 3.6 additionally receives a model-specific prompt and general voice
instruction. Instructions are attached to each segment, not spoken as text.

The persona uses an optimistic, proud fictional superintelligence identity with
the limitation of being unable to physically accompany the user. Identity
questions use a fixed introduction in the persona prompt. Knowledge and memory
claims remain grounded; this characterization does not grant additional tools.

`studio/core/utils/turn_taking/backchannel.py` decides whether to emit a short acknowledgement during
a user pause and selects a token appropriate to language and context. Audio is
served from reviewed or prepared clips because generation on the live pause
window is too late. Backchannels share playback and echo-reference plumbing but
do not become normal assistant replies.

## 10. Local inference scheduling

### MLX

Local MLX reply, reasoning-depth routing, and TTS work share one process-level
`GpuLoop`. The loop owns the GPU execution thread and advances active generators in weighted turns.
Startup completes a short TTS utterance after other model warmups, exercising
continuous codec generation and cleanup before accepting conversations.

Some speech jobs receive temporary first-chunk priority; afterward they rejoin
weighted scheduling. Cancellation closes the generator and removes it from the
active set. Creating an independent MLX thread or stream bypasses this safety
and scheduling model.
Auxiliary Qwen work fillers run as non-exclusive token-stepped jobs on this
same scheduler. Each has a private KV cache and leaves the classifier's static
prefix cache untouched.

### CUDA

`studio/core/utils/tts/cuda.py` loads the configured Breeze streaming checkout
and checkpoint locally. Its public `stream(text, instruction)` contract matches
MLX: sample-aligned 24 kHz PCM16. No separate HTTP service or listening port is
required. One shared provider serializes model loading, generation and codec
cleanup on a dedicated worker. A bounded output queue limits buffered PCM;
cancellation is observed between acoustic frames and queued work checks its
cancellation flag before entering inference. Application shutdown closes the
worker. Initial and subsequent acoustic batches preserve the MLX chunk settings.
CUDA depth decoding uses the existing compiled CUDA-graph path by default, with
profile warmup completed before serving. The master `fast_all` override remains
unset so it cannot disable that stage. Completed requests report delivery RTF;
`evals/breeze_cuda_latency.py` measures warmed delivery and same-GPU ASR contention.

ASR/router and TTS devices are explicit configuration. Shared GPU use still
competes for resources; separate devices can be selected without changing the
conversation pipeline. CUDA startup validates the selected devices, code checkout,
and CUDA checkpoint including its bundled codec, and never acquires MLX weights.
DeepSeek-only deployments require only DeepSeek credentials.

### Torch/MPS

Torch-backed embedding and related MPS operations use the process-level lock in
`utils/torch_lock.py`. Lock scope covers device inference, not unrelated search
coordination or waits on work that may need the same lock.
Shared embedding lookup and construction also acquire this lock before checking
the cache, preventing concurrent cold loads from initializing MPS alongside ASR
or another encoder. Equivalent omitted, positional and keyword tokenizer defaults
resolve to one cache entry; memory, Gate, slot classification and Studio perception
reuse the same weights. Distinct tokenizer settings retain distinct entries.
The optional local work-filler decoder also reuses this lock, releasing it
between forward passes so a whole sentence does not monopolize ASR/router
access. Each request owns its KV cache and checks cancellation and deadline
before each step and while acquiring the lock.

### Hot-path priority

Studio tracks whether reply or speech output is on the user-visible hot path.
Background perception and ingest may wait for an idle window, subject to a
bounded fallback so memory work cannot starve indefinitely.

Acoustic-emotion tasks belong to the conversation or realtime session and capture
their output ID, Memory Space and VoiceMem instance. Ownership is checked before
and after inference. A new output cancels old tasks, and session closure cancels
and awaits them. Native inference already running in a thread may finish, but its
stale result cannot update the client. Accepted tag updates carry the output ID;
reply audio never awaits this background result.

## 11. Output, playback, and interruption

```mermaid
flowchart LR
    Provider --> Timeline[Output ID and AudioTimeline]
    Timeline --> PCM[WebSocket PCM]
    PCM --> Worklet[PCM player worklet]
    Worklet --> Progress[Rendered source samples]
    Progress --> Cutoff[Heard-text cutoff]
    Cutoff --> UI[Visible history]
    Cutoff --> Session[Session Context]
    Cutoff --> Memory[Memory attribution]
```

The canonical Web media format is 24 kHz mono PCM16. Each assistant output has
an output ID. Late audio, subtitle, checkpoint, and cancellation events resolve
against that ID.

The local Breeze adapter emits an initial acoustic batch sized to satisfy the
browser's existing admission buffer in one delivery. This avoids a redundant
one-frame codec call and the subsequent wait for a second server chunk without
raising the browser prebuffer or delaying audible playback.

Turn fillers use the browser's independent backchannel path so they do not
become main-output timeline content. Once a backchannel starts, interruption,
reset, and a new `answer_start` never truncate it. Main PCM generation and
transport may continue in parallel, but browser playback remains paused until
the active backchannel finishes; this preserves a seamless handoff without
adding the clip duration to model or TTS work. In-speech backchannels remain
independent and do not mutate the main reply state.

Generated, sent, buffered, rendered, and heard output are distinct states.
Browser-rendered source samples determine the interruption cutoff. Text mapping
uses provider alignment when available, completed-segment duration otherwise,
and calibrated speech rate as the fallback.

Studio timelines explicitly separate `generated_samples` (including private
`ReplySink` PCM) from `sent_samples` recorded after a successful transport send.
Only playback checkpoints establish rendered progress, bounded by delivery;
without a report the confirmed progress is zero. A playback-completion timeout
does not promote buffered or delivered audio to heard audio. Legacy timeline
callers can retain append-as-emission and elapsed-time fallback behavior, but
Studio's chained and realtime sessions use explicit delivery tracking.
An interruption freezes both sample cutoff and mapped text before cancelling
workers, so later clock ticks, alignments, audio or checkpoints cannot change
the UI/history prefix. Ordinary finalization also freezes its final checkpoint.
Realtime keeps its existing single `close_turn` owner and shares these playback
rules. These bookkeeping changes add no model calls or pre-audio waits.

Interruption separates reversible detection from cancellation:

1. Candidate speech pauses playback while preserving the PCM queue.
2. ASR growth, explicit control text, backchannel routing, and echo rejection
   confirm or reject the candidate.
3. Rejection resumes the same output.
4. Confirmation clears browser playback, cancels reply/provider work, and
   finalizes only the heard assistant prefix.

Generated, sent, buffered, rendered, and heard output are different states. The
unheard generated tail is not conversation history.

On a supported native desktop platform, the HTTP application can start its supervised desktop pet during server startup,
using the configured listening port, and stops that process during server shutdown.
Linux/WSL skips process startup while retaining all pet broadcast routes.
Opening the browser page is not required to launch the pet.
The optional desktop pet observes the existing pet WebSocket without starting
another conversation. Its renderer follows output-identified playback checkpoints
for lifecycle and actual playback RMS for Live2D mouth movement. Pause,
interruption, drain and disconnect close the mouth. Backchannel, ordinary reply
and sadness events select white-vine expressions and programmatic gestures. While
speaking, the behavior controller starts a gesture promptly and schedules spaced
random torso gestures. The model's physics drives both arms from torso rotation;
brief arm accents are added after physics rather than replacing that output. Pointer
gaze temporarily overrides natural wandering and fades when the cursor leaves.
The pet starts
in the `lie` resting state. Conversation startup and detected user voice select
the `sit` interaction state; a local silence timer returns it to rest. Both states
share one Live2D model and scene. VAD transitions come from the
existing capture state or realtime provider, not raw microphone packet arrival.
Conversation closure clears active voice state; resting suppresses random gestures.

## 12. Session and persistent memory

Session Context contains turns not yet represented by persistent memory. It is
isolated by WebSocket session and Memory Space.

```text
current user input
+ unpersisted Session Context
+ retrieved persistent memory
-> reply provider
```

After a normal or interrupted reply, ingest runs outside the response path.
The completion callback removes the session turn only when durable memory was
created. Non-persistent dialogue remains within a bounded rolling context until
the session ends.
Interruption remains a `SessionTurn` state flag. Provider dialogue messages
contain the heard assistant content without appending runtime annotations to
the assistant's utterance.

Only confirmed turns enter this path. Chained conversations use one guarded
finalizer for normal completion, interruption, errors, follow-ups and disconnect;
unaccepted EOT snapshots never enqueue memory work. The finalizer uses the
captured final user input and frozen heard assistant prefix, not the speculative
input or full generated tail. Missing playback reports can therefore omit heard
words from context rather than inventing unconfirmed playback. Messages preserve
up to 4,000 characters each; oversized messages keep their beginning and ending
with an explicit omission marker. Recent history retains at most six turns and
12,000 content characters. Uncommitted context uses the same total character
budget, evicting its oldest turns and completion lookup entries when full.

Background ingest captures the target `VoiceMem` instance and Memory Space when
scheduled, along with the confirmed input language. A later UI space change or
another account's language cannot redirect or relabel an existing write.
An agent tracks unfinished native writes separately from its asyncio task set.
Returning from `ingest(async_facts=True)` does not complete a native write; its
durable completion callback or a failed ingest releases that pending write. Browser completion messages contain status and IDs, never backend error
details.

## 13. State ownership

| State | Owner | Lifetime |
| --- | --- | --- |
| Active model-role and provider configuration | Backend process | Process |
| Managed desktop model-service configuration | Electron main process / encrypted app data | App installation |
| Prompt templates and parsed prompt cache | Harness / prompt config | Process |
| MLX scheduler | `GpuLoop` | Process |
| Torch device serialization | `TORCH_LOCK` | Process |
| `VoiceMem` capability cache | `VoiceMem` instance | Instance |
| Factual and affective memory | Memory Space stores | Persistent |
| Streaming ASR/VAD/EOT/gate state | `VoiceStream` | Input turn/session |
| Turn-taking phase, latency estimate, and backchannel policy | `TurnTakingStateMachine` | WebSocket session |
| Self Harness profile, Persona supplement, and confirmation state | `SelfHarnessState` | WebSocket session |
| Reply router model | `studio/core/utils/reply_modes` | Process |
| Reply mode | Confirmed `Pending` turn | Turn |
| Recent input-language fallback | WebSocket session + Memory Space | Session |
| Captured input language | Confirmed `Pending` and ingest context | Turn/task |
| Early output buffer | `ReplySink` | Speculative assistant output |
| Short-term dialogue | `SessionBuffer` | WebSocket session + Memory Space |
| Text/media alignment | `AudioTimeline` | Assistant output ID |
| PCM queue and echo reference | Browser worklets | Assistant output ID/session |
| Background ingest | Captured turn and `VoiceMem` | Until completion |
| Background acoustic emotion | Conversation or realtime session, captured output and Space | Until completion or cancellation |

Turn-specific state is explicit. Process globals are reserved for configuration,
shared model caches, and schedulers whose process-wide behavior is intentional.
`SessionBuffer.clear_session` removes both uncommitted context and bounded recent
turns for every Memory Space owned by the closing WebSocket session.

## 14. Observability and evaluation

Studio has three distinct verification categories:

- deterministic Python regressions in `tests/voicemem/` and `tests/studio/`;
- browser/worklet and desktop simulations in `tests/frontend/`, with visual
  checks in `tests/manual/`;
- latency, quality, and benchmark scripts in `evals/` and
  `evaluation/`.

`tests/helpers/` owns synthetic capture, reply, conversation and display fixtures;
test cases do not import other test modules to reuse their state. Python discovery
uses `-t .` so `tests/voicemem` and `tests/studio` cannot shadow runtime packages.
`tests/README.md` documents the Python, Node and manual entry points. Tests are
development source and are not imported by application startup or bundled in
the Python wheel or Electron application.

`prompt_trace.py` records allowlisted provider requests asynchronously so disk
I/O does not block speech. `prompt/logs/` entries can include system prompts,
history, and retrieved memory; they are sensitive runtime traces.

Synthetic regressions verify state transitions and protocol behavior. Live
perceived latency, voice quality, Metal stability, microphone behavior, and
network-provider performance require the corresponding native environment.

## 15. Extension map

| Change | Primary owner |
| --- | --- |
| Public memory API | Thin facade plus owning memory subsystem |
| Fact extraction or retrieval | `voicemem/leftbrain/` |
| Affective or behavioral memory | `voicemem/rightbrain/` |
| Cross-brain search or ingest | `voicemem/orchestrator.py` |
| ASR, VAD, EOT, speaker, scene, emotion | `voicemem/utils/audio/` and config |
| Turn routing | `voicemem/gate.py` and streaming regressions |
| Studio persona or pause policy | `studio/harness/` |
| Spoken backchannel behavior | `studio/core/utils/turn_taking/` and Web playback |
| Tone-label protocol | `studio/core/utils/tts/control.py`, prompts, and TTS wiring |
| Reply provider | `voicemem/reply.py` or `studio/core/utils/llm/local.py`, then config |
| Three-way reply routing | `studio/core/utils/reply_modes/` and Web composition root |
| TTS provider | `voicemem/tts.py` or provider module, then config |
| GPU scheduling | `voicemem/utils/gpu_loop.py` |
| Studio prompt parsing | `studio/prompt_config.py` and `studio/prompt/` schema |
| Prompt tracing | `voicemem/utils/common/prompt_trace.py` |
| Early generation | `ReplySink`, EOT callback, and cancellation path |
| Capture echo control | Browser mic worklet and server echo guard |
| Playback timing and heard prefix | Audio timeline and both reply modes |
| Browser UI and visualization | `studio/web/voicemem.html` |
| Transport | Application boundary; memory contracts remain stable |
| Persistent schema | Owning store plus explicit migration and rollback |

A cross-layer feature begins with a normalized contract. Each implementation
then remains in its owning layer.

## 16. Architecture update rule

Update this document when a change modifies:

- a layer's responsibility or dependency direction;
- a public or provider-neutral contract;
- input, reply, playback, interruption, or persistence flow;
- state ownership, isolation, cache, or lifetime;
- GPU, thread, lock, or cancellation semantics;
- prompt ownership or the relationship between reply modes.

Tuning values, local machine observations, incident history, and temporary
experiments belong in focused evaluation artifacts, not this overview.

## Small-model emotion processing

Audio emotion uses SenseVoiceSmall on CPU. Memory detectors and Studio reuse one
process-level transcriber; initialization and inference follow the Torch lock.
The classifier returns localized acoustic labels. It does not load a multimodal
language model or generate causal explanations from audio. The separate existing
emotion2vec background classifier remains available in Studio.

The optional fusion interface accepts a provider-neutral `TurnAttributor` via
`attributor=`. `SmallEmotionAttributor` preserves the result schema with observed
affect, supplied V/A and transcript retrieval terms; semantic evidence and causal
graph deltas remain empty. Existing persisted records are not rewritten.
Retired multimodal adapter modules and their exports have been removed.
