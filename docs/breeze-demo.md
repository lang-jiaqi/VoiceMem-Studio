# Breeze MLX demo

`llm_tts` now defaults to `breeze_mlx`. An explicit `TTS_BACKEND` still wins.
The library default and the remote `breeze` provider are unchanged.
The demo uses `voice/noctelle_ref_short.wav` and its `.txt` transcript to hold the
reference voice across sentences, with a gentle female delivery instruction.
This is a configured reference, not a new voice selected through listening tests.
Reference recordings are local-only and not committed. On another machine, supply
your own authorized reference with `VOICEMEM_BREEZE_REF_AUDIO` and
`VOICEMEM_BREEZE_REF_TEXT` (or a matching `.txt` sidecar).

## Current dialogue setup

Use `--mode llm_tts --llm deepseek --space demo-zh --lang zh --confirm_ms 200`.
Set `DEEPSEEK_API_KEY` in the terminal environment; do not put it in a file committed
to Git. Breeze remains local (`TTS_BACKEND=breeze_mlx`).

The terminal defaults to latency and the effective TTS emotional instruction,
plus warnings/errors. `--verbose` restores full diagnostic output. Full logs
still go to `results/logs/`. Actual LLM messages and each Breeze segment's text,
instruction and reference transcript are saved under `prompt/logs/`; see
[`prompt/README.md`](../prompt/README.md). Prompt files are private local data,
not part of the Git repository. Editable runtime templates live directly in
`prompt/`: `llm_system_*.md`, `llm_tone_rule_*.md`, `tts.json`, and
`llm_context.json`. Edit these files and restart; no Python prompt edits are needed.

Use native Apple Silicon Python 3.12 with `mlx-audio==0.5.1`, `mlx==0.32.2`.
Both versions are already installed in the Python below. Weights are downloaded
to `models/tts/Breeze-TTS-2-mlx-4bit` (research/noncommercial model license).

```bash
cd /Users/langjiaqi/voicemem_opensource
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 tools/test_breeze_tts.py
afplay results/breeze_noctelle.wav
```

The smoke test warms one utterance, writes 24 kHz mono PCM16 WAV, and reports
first PCM latency and RTF (under 1 means faster than playback).

```bash
cd /Users/langjiaqi/voicemem_opensource
TTS_BACKEND=breeze_mlx VOICEMEM_FINAL_ASR=0 VOICEMEM_EOT_ENDS_TURN=1 \
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 web/run.py \
  --mode llm_tts --llm local --space demo-zh --lang zh \
  --confirm_ms 200 --backchannel --port 8787
```

Open http://localhost:8787 after startup completes and refresh the page to load
the updated microphone worklet. Startup prepares 10 Chinese acknowledgement
tokens in two styles using the same reference voice (including 嗯嗯、啊、啊啊、对).
This first startup takes longer; later runs use the on-disk cache. Live sessions
only read cached clips and use the existing Backchannel.offer trigger.
`--no-backchannel` disables this preparation and playback.
To use a cloud reply model, change `--llm local` to `--llm openai` and supply
the normal API credentials. TTS remains local.

The supplied standalone benchmark is preserved in `tools/breezetts2_mac_fast.py`.
The adapter uses its FrameKVCache/FastDepth implementation, serialized on the
shared GPU worker, and clears codec state on completion, errors and cancellation.
It also caches immutable reference prompt embeddings, avoiding reference audio
and reference transcript encoding on every sentence. Target text still goes
through the normal text encoder and backbone.
`VOICEMEM_BREEZE_REF_AUDIO` can select another reference; put an accurate transcript
in the matching `.txt` file or set `VOICEMEM_BREEZE_REF_TEXT`.

Verification: adapter regression tests cover registry, reference validation,
PCM clipping, codec cleanup and error propagation. The Codex execution sandbox
cannot access Metal, so actual synthesis, voice quality and live demo latency
must be checked in the user's native terminal with the commands above.

## Echo and latency update

### Final ASR no longer waits behind streaming backlog

Keep `VOICEMEM_FINAL_ASR=1` to enable this path (it is the library default).
After VAD confirms the turn, complete-audio offline decoding and streaming
flush start together. A non-empty offline result can finish the turn without
waiting for obsolete streaming chunks. Queued old audio is discarded only
after successful offline decoding; an in-flight chunk is allowed to finish,
but its result cannot overwrite the next turn. Empty/failed/disabled offline
decoding still waits for the complete streaming fallback.

The full captured audio is retained for decoding/archive. If the streaming
tail was skipped, early-reply coverage is checked against the full offline
text, not a stale partial. No VAD silence threshold or echo guard is weakened.

With `VOICEMEM_ASR_DEBUG=1`, look for
`完整复核（绕过流式积压）` and the breakdown
`判停/VAD后等待 · ASR并行收尾 · 回合确认`.
These are real runtime timings; unit tests do not establish live latency.
Restart the existing demo command to load this change (keep DeepSeek + Breeze).

- Browser AEC stays on. The mic AudioWorklet also receives the actual mixed
  playback bus (reply, backchannel and replay). It suppresses residual blocks
  only at correlation >=0.88 over a 350ms delay search; independent speech is
  retained. This is a residual echo guard, not a replacement for browser AEC.
- Capture batches are 20ms instead of 2048 samples / 24kHz (~85ms). The fallback
  uses 512 samples. Playback and microphone share a sample clock.
- Text echo matching removes punctuation and handles a single ASR insertion in
  a sufficiently long phrase. Confirmed echo cannot increment interruption
  evidence or create a new reply; evidence expires after playback plus 2s.
- Short first clauses such as “在呢，” can launch TTS. The playback jitter buffer
  remains 160ms: the old trace already showed underflow, so shrinking it without
  measuring synthesis throughput would risk more gaps.
- `[lat] 闭嘴→首帧` uses the last server-received VAD speech frame. A separate
  browser playback-start checkpoint includes buffering and acknowledgement
  transit. These are software estimates, not a measured microphone-to-DAC test.
  Text-only turns report N/A for speech-end timing instead of a false 0ms.

Run deterministic regressions with:

```bash
python3 -m unittest discover -s tests -p 'test_demo_audio.py' -v
python3 -m unittest discover -s tests -p 'test_breeze_tts.py' -v
node evals/test_mic_capture.cjs
```
