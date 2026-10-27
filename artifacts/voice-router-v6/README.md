# Voice router v6 — experimental snapshot

This directory versions the task-specific routing model with its runtime source.
The complete encoder and decision head contain 102,274,064 parameters. Inputs are
ASR text, bounded recent dialogue and observed EOT/silence/voiced-span state.
The router selects ordinary/deep reasoning; VoiceMem owns memory eligibility,
and the separate auxiliary generator owns work-filler wording.

## Files and Git LFS

- `eager/`: original safetensors checkpoint, configuration and tokenizer for Torch.
- `onnx/`: calibrated FP32 ONNX deployment, configuration and tokenizer for CPU.
- `profile.cpu.json`: CPU ONNX component selection, suitable for Apple Silicon testing.
- `profile.cuda.json`: Torch CUDA component selection; adjust the visible device as needed.
- `SOURCE.json`, `LICENSE`, `NOTICE.txt`: pinned base-model provenance and license.
- `EVALUATION.json`: scoped results and known limitations.
- `SHA256SUMS`: checksums for this curated snapshot.
- `COMMIT_PATHS.txt`: exact repository-relative paths for the code/model snapshot commit.

Both weight files are stored through Git LFS. After checking out the branch:

```bash
git lfs install --local
git lfs pull
```

To retrieve only the CPU weight when other LFS content has not been downloaded:

```bash
git lfs pull --include="artifacts/voice-router-v6/onnx/model.onnx"
```

Keep each complete model directory together. `base` and `init_checkpoint` in the
metadata record training provenance; inference uses the local files in the
chosen directory. The original base weights are not needed for inference.

## Select the adapter

In Components → Router → Adapter, select:

- Factory: `studio.core.utils.reply_modes.encoder_depth:create`
- Model: absolute path to this snapshot's `onnx` directory for CPU, or `eager` for Torch.
- CPU options: `{"backend":"onnx","device":"cpu","allow_prototype":true}`
- CUDA options: `{"backend":"eager","device":"cuda:0","allow_prototype":true}`
- Router API Key and Base URL: empty.

When starting the Python backend from the repository root, the supplied profiles
also work through the existing startup environment configuration:

```bash
export STUDIO_COMPONENTS="$(cat artifacts/voice-router-v6/profile.cpu.json)"
export STUDIO_FILLER_PROVIDER=deepseek
python -m studio
```

Use the project's existing environment and credentials for the reply/filler
provider. `STUDIO_FILLER_*` can instead select an existing local compatible
service. Component selection takes effect at backend startup. The snapshot
does not change the repository's recommended-router default.

## Scope and limitations

Authored test: 103/106. Frozen behavior regression: 20/20. Additional authored
edge regression: 27/30. Fresh designed speech: 48/48 ASR/noise views from 12
source contexts. Legacy held-out synthetic ASR: 712/716.

Some requests to hide the reasoning process can still be mistaken for disabling
deep reasoning. Completed-result recall and requests to supply a complex
derivation remain imperfect. These correlated authored/ASR views are not an
overall natural-dialogue accuracy estimate. Native M4 deployment has not been
measured; use the model as an experimental routing option.

Training tools and the complete input/adapter boundaries are documented in
`finetune/decision_router/README.md` and `ARCHITECTURE.md` at the repository root.
