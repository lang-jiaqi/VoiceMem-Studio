# Breeze Studio

Use native Apple Silicon macOS and Python 3.12:

```bash
python -m pip install -e '.[studio]'
python web/run.py --mode llm_tts --space demo-zh --lang zh --confirm_ms 200 --verbose
```

The default reply provider is DeepSeek. Supply provider credentials through the
launching process environment. Add `--check` to inspect requirements without
opening memory or downloading weights. See [Studio](../studio/README.md).

`studio/core/utils/tts/initialize.py` selects the reviewed reference, Breeze
weights, seed, first acoustic batch, and cached depth decoder. All MLX work uses
the shared GPU scheduler. Reference recordings remain in `voice/`; missing
reviewed backchannel clips produce silent waiting, never a generated replacement.

Breeze requires `mlx==0.32.2` and `mlx-audio==0.5.1`. Startup verifies those
versions, downloads missing weights into `studio/models/`, and runs first-chunk
warmup before accepting browser connections. First and regular chunks retain
two acoustic frames and the browser retains its existing admission buffer.

`--llm local` uses the local reply adapter and its cancellable prefix cache.
Final ASR refinement, EOT, speculative output, heard-prefix interruption, and
filler playback completion remain separate stages with their existing guards.

Native microphone, Metal, voice quality, and end-to-end latency checks require
the full model environment; deterministic regressions alone do not establish
perceived latency or voice equivalence.

Studio uses `transformers==5.16.1` and `huggingface-hub>=1.5,<2` with
`mlx-audio==0.5.1`. Install the complete `studio` extra together; do not use
`--no-deps` to bypass dependency constraints.
