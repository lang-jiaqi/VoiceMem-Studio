# Component bindings

Studio keeps its current pipeline as the recommended default. An App-owned
backend can select a small set of API protocols under **Settings → Components**.
Changes stay in drafts until **Apply and restart**. The desktop validates the
whole batch, stops its old child before starting the replacement, and persists
the new configuration only after startup succeeds. Failed startup restores the
previous configuration. Existing/Docker/remote backends remain server-managed.
Restart the desktop App after updating its preload code.

## Supported selections

| Role | Recommended implementation | Optional built-in API |
| --- | --- | --- |
| `asr` | X-ASR-zh-en 480ms (CPU, INT8, one thread) | Deepgram Listen v1 WebSocket |
| `asr_final` | SenseVoice | Installed adapter |
| `vad` | Silero | Installed adapter |
| `eot` | Smart Turn | Installed adapter |
| `router` | Qwen3 0.6B | Installed adapter |
| `speech` | Breeze MLX/CUDA | OpenAI-compatible streaming speech |

See [reply routing](../reply_modes/README.md) for default model selection and
waiting speech. Installed adapters retain the router interface listed below.

The recommended ASR downloads a pinned official ONNX revision on first launch,
then converts it locally and caches INT8 weights before warmup. This startup
step does not run in the live audio worker. Set `VOICEMEM_ASR=funasr` to retain
Paraformer or `VOICEMEM_ASR=sherpa` for the previous bilingual Zipformer. Neither
override affects Deepgram or an installed ASR adapter.

Memory and reply keep separate DeepSeek, Qwen/DashScope, and OpenAI-compatible
Chat service settings. MLX-local replies also accept an explicit model path or
compatible model ID. Compatibility depends on the selected model's protocol;
an arbitrary HTTP endpoint is not sufficient.

Deepgram consumes mono PCM16 at 16 kHz and returns cumulative interim/final
text. Studio uses the [Listen v1 streaming protocol](https://developers.deepgram.com/reference/speech-to-text/listen-streaming)
with provider endpointing disabled. Harness still controls VAD, end-of-turn,
speculation, and final-ASR refinement. The configured service root ends in
`/v1`; the adapter appends `/listen`. Speech APIs must implement streaming
`audio/speech` PCM16 output at 24 kHz and support the requested model/voice and
tone instructions. Startup synthesizes a short disposable check and, when
backchannels are enabled, primes two voice-specific clips. This can incur a
small provider charge. Reviewed default Breeze recordings are unchanged.

## Headless configuration

Set `STUDIO_COMPONENTS` to a JSON object; absent roles select `recommended`.
For example, the following selects the built-in optional API protocols:

```dotenv
STUDIO_COMPONENTS='{"asr":{"provider":"deepgram","model":"nova-3","baseUrl":"https://api.deepgram.com/v1","language":"zh"},"speech":{"provider":"openai","model":"gpt-4o-mini-tts","baseUrl":"https://api.openai.com/v1","voice":"alloy"}}'
STUDIO_ASR_API_KEY=
STUDIO_SPEECH_API_KEY=
```

Supply real credentials only through ignored environment configuration or the
desktop's API Key field. They are not accepted inside `STUDIO_COMPONENTS` or
adapter options. The desktop encrypts persisted keys with OS-backed
`safeStorage` and returns only non-secret configuration to the UI. Empty keys
retain a saved key only for the same provider, endpoint, and adapter factory.
Changing the destination requires a new key, except for loopback services or
adapters that do not require authentication. Remote endpoints require HTTPS;
local loopback endpoints may use HTTP. URLs cannot embed credentials or query
parameters. Restore the recommended selection to remove an optional binding.

`STUDIO_VALIDATE_SERVICES` is a transient desktop-owned list of changed remote
memory/reply roles. It triggers bounded synthetic streaming probes on startup;
ordinary starts do not add these LLM requests. API checks never include user
conversations or retrieved memory, and never ingest their synthetic input.

## Installed adapters

Advanced settings accept `package.module:create`, a callable already installed
in the backend's Python environment. This imports trusted local Python code;
it does not install packages, execute uploaded scripts, or adapt arbitrary model
protocols automatically. A factory is synchronous and receives:

```python
def create(*, model: str, base_url: str, api_key: str, options: dict):
    return Adapter(model=model, base_url=base_url, api_key=api_key, **options)
```

The returned adapter must satisfy its existing Harness boundary:

| Role | Required interface and result |
| --- | --- |
| `asr` | Sync `feed(float32_mono_16k) -> str`, `flush() -> str`, `reset()`, `new_stream()`; text is cumulative |
| `asr_final` | Sync `transcribe(float32_mono_16k) -> str` on the final-ASR executor |
| `vad` | Fast sync `is_speech(float32_mono_16k) -> bool`, `new_stream()` |
| `eot` | Sync `score(float32_mono_16k) -> float` and numeric `threshold`, both in `[0, 1]` |
| `router` | Async `classify_async(text, memory_prefetch_hint=False, history=None) -> ThinkingDecision`, async `generate_short_text_async(system, prompt, *, max_tokens=40, timeout_s=1.2) -> str`, sync `warmup()` |
| `speech` | Async generator `stream(text, instruction=None)`, yielding complete mono PCM16 bytes at 24 kHz or `TimedAudioChunk` at that rate |

`new_stream()` must return independent mutable decoder state; weights may be
shared. Streaming ASR may expose sync `close()` to release a transport on reset
or disconnect. `ThinkingDecision` comes from
`studio.core.utils.reply_modes.component` and retains `fast`, `medium`, and
`slow`; the router also supplies optional spoken filler generation. Studio
uses prepared waiting speech independently of the classifier; explicit custom
filler prompts retain the adapter's short-text generation interface.
Speech generators must close network responses/native work in `finally` and propagate cancellation.
Backchannel cache identity includes factory, model, endpoint and options; adapters
may additionally expose `cache_voice_id` to distinguish internal voice versions.

Initialization owns model readiness. Local MLX work must use the shared
`voicemem.engine.GpuLoop`; Torch work must retain existing device locks and
executor boundaries. VAD runs per frame on the input loop, so a network call or
slow synchronous inference does not satisfy that boundary. Other synchronous
input methods retain their existing worker/executor placement. Async router and
speech methods must offload blocking model work themselves. API clients retained
by custom adapters must be safe across startup and server loop lifetimes.
