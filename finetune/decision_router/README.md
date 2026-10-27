# Voice-dialogue decision training

This is an experimental replacement for Studio's reasoning-depth classifier.
It uses ASR text, bounded recent dialogue, and optional **observed** acoustic
state. It does not replace ASR, VAD, turn confirmation, memory eligibility or TTS.
It does not train or contain TypeSafe Jev weights.

## Upstream training recipe

The main experiment uses [Laya](https://github.com/NandhaKishorM/laya), its
multilingual checkpoint, and its actual `laya.train` implementation. The wrapper
does not reimplement or rename its objectives:

- `soft-ce`: cross-entropy against the target probability distribution.
- `rlcd`: upstream's Gaussian-logit-sampling policy-gradient term, rewarded by
  proper scoring rules, plus the same soft cross-entropy term.
- Choice options are shuffled during training with targets remapped to match.
- Encoder and typed decision head are trained with separate learning rates.
- Temperature calibration uses a separate set of complete semantic/dialogue
  groups. The wrapper bypasses upstream's random item-level split so different
  paraphrases of one seed cannot leak across train/calibration/test sets.
- Compare both objectives on the same data and frozen holdout, rather than
  assuming that the reinforcement-learning term improves this particular task.

[AnyJev](https://github.com/nokia-applied-research/AnyJev) is another reference:
its training-free readout corrects label/option-order bias; its Tacit checkpoints
use self-distillation from reasoning into a one-forward decision. Those are
different training/inference approaches. The present wrapper reproduces Laya's
objectives; it does not claim to reproduce proprietary Jev training or Tacit.

## Data

One JSONL row:

```json
{"id":"case-001","group":"dialogue-001","split":"train","text":"请把剩余的证明完成。","history":[{"role":"user","content":"证明这个递推式收敛。"},{"role":"assistant","content":"还没处理参数等于零的情况。"}],"needs_deep":true,"source":"reviewed-domain-data"}
```

Required splits are `train`, `calibration`, and `test`, each with both labels.
All snapshots, ASR variants, voices and paraphrases belonging to one source
utterance/dialogue must share its `group` and split. Keep speakers and sessions
disjoint when evaluating real recordings. `source` must describe provenance;
training scripts do not obtain consent, invent labels or upload recordings.

Optional `audio_state` fields are `eot_probability` in `[0,1]`, nonnegative
`silence_s`, and nonnegative `span_s` (the captured first-to-last voiced span,
including pauses). Missing values stay missing. This is acoustic-state fusion,
not a model that reads raw audio. A Laya checkpoint trained without observed
audio refuses to enable that branch at inference. Real audio features require
real paired training and validation, not invented values on text rows.

The pilot generator creates synthetic text only:

```bash
python -m finetune.decision_router.pilot --out results/decision-router/pilot.jsonl
```

The generated topics and wording have substantial shared template structure.
Its internal test score is a pipeline check, not a real conversation benchmark.
Use `evals.router_quality.HOLDOUT_CASES` as a separate wording check, followed by
independent natural-dialogue and native Mac evaluation.

## Laya training

Use a separate training environment/check-out for the optional upstream Laya
dependency. The Studio default environment and running backend need not change.
Record the upstream commit and public checkpoint revision used for each run.
The tested source revision is `b9fe351ce81e2d974bca2bf219c1e4030c0a05ea`,
with `convaiinnovations/laya-multilingual` checkpoint revision
`1720e3e3357cfe1e281542e223f8273b0890ca34`. With Studio's existing dependencies,
install that reviewed source checkout without replacing its Torch stack:

```bash
python -m pip install --no-deps --no-build-isolation /path/to/laya
```

The repository version need not have a matching PyPI release. Use the pinned
checkout for reproducible training and serving rather than assuming it does.

```bash
python -m finetune.decision_router.laya_train \
  --laya-source /path/to/laya \
  --source-revision UPSTREAM_COMMIT \
  --model /path/to/laya-multilingual \
  --model-revision CHECKPOINT_REVISION \
  --data results/decision-router/pilot.jsonl \
  --out results/decision-router/laya-soft-ce \
  --loss soft-ce --epochs 2 --device cuda:0
```

Run again from the same base with `--loss rlcd` and a separate output directory.
Outputs contain normal Laya checkpoint files and `voice_decision.json`, including
the trained question, source revisions, data digest and independent metrics.
All exported checkpoints remain marked as prototypes.

Compare against the existing Qwen classifier without any reply API calls:

```bash
python -m evals.decision_router --backend laya \
  --laya-source /path/to/laya \
  --checkpoint results/decision-router/laya-soft-ce \
  --compare-qwen /path/to/Qwen3-0.6B \
  --device cpu --out results/decision-router/comparison.json
```

The eval explicitly clears result caches. Measure cold loading separately from
warm inference. Linux CPU/CUDA results do not establish Apple Silicon latency,
memory pressure or competition with ASR/TTS.

## Complete observed-speech training

### Contrastive dialogue revision

`dialogue_data.py` authors paired explicit requests and negative/quoted requests,
implicit tasks, ordinary requests and context-dependent continuations. A whole
wording family, both labels, all contexts and all audio variants belong to one
partition. It checks semantic-input duplication and requires contextual examples
of every category in validation, calibration and test. These remain authored
synthetic examples, not independent natural conversation measurements.

```bash
python -m finetune.decision_router.dialogue_data \
  --out results/decision-router/dialogue-v2/authored.jsonl \
  --speech-out results/decision-router/dialogue-v2/speech-source.jsonl

python -m finetune.decision_router.audio_data \
  --data results/decision-router/dialogue-v2/speech-source.jsonl \
  --out results/decision-router/dialogue-v2/speech

python -m finetune.decision_router.assemble_v2 \
  --authored results/decision-router/dialogue-v2/authored.jsonl \
  --legacy results/decision-router/previous-joint.jsonl \
  --speech results/decision-router/dialogue-v2/speech/paired.jsonl \
  --out results/decision-router/dialogue-v2/joint.jsonl
```

Review observed transcripts for lost negation or unusable input before assembly;
`--exclude` accepts a JSON list of quarantined observation IDs. An original
utterance's intended label is not sufficient evidence when ASR changes its
meaning. `assemble_v2.py` preserves the authored partitions, discards prior
consensus weak labels and reuses only supervised legacy **training** rows.
Legacy validation/calibration/test data remains frozen for regression.

Train from the pinned multilingual base using `laya_train.py`. Validation
reports semantic categories, group accuracy, history coverage and label errors;
checkpoint selection gives categories equal weight. Probability calibration
remains separate. The trainer optionally accepts `--encoder-lr` and `--head-lr`.
`evals.dialogue_decision` measures all semantic rows, their audio-state ablations,
the frozen wording check and an optional private regression file. Regression
inputs and original dialogue logs belong in ignored results, never public test
fixtures or training data. A repaired regression is not an independent accuracy
estimate, and cannot alone qualify a replacement.

`intent_data.py` adds contrasts for execution versus terminology, negation scope,
speaker intent and changes of instruction. It keeps a previously inspected test
partition regression-only, and creates fresh held-out wording families. Its
fresh test focuses on explicit intent; also run the earlier six-category,
observed-speech and frozen regressions to cover ordinary and implicit requests.
`--label-smoothing` in `laya_train.py` softens training targets only. Validation,
calibration and tests retain hard labels. Compare a low-learning-rate full
fine-tune with a frozen-encoder control on validation before choosing a model;
keep independent tests out of selection.

```bash
python -m evals.dialogue_decision --checkpoint /path/to/candidate \
  --laya-source /path/to/laya --data results/decision-router/dialogue-v2/joint.jsonl \
  --out results/decision-router/dialogue-v2/qualification.json
```

`audio_data.py` synthesizes designed speech with no reference recording, runs
the actual streaming/final recognizers, and measures VAD/EOT state on clean and
noisy waveforms. It preserves observed transcripts without filling failed
recognition with reference text. Sample-keyed augmentation seeds are stable
across partitions/resumption; `--reuse-wav` reprocesses cached waves.

`public_prepare.py` reads a fixed-revision Apache-2.0 VoiceAssistant-400K shard,
retains conversation history, and prepares actual ASR/acoustic inputs. Local
Qwen/Laya consensus supplies explicitly weak training-only labels; disagreements
are reserved. Weak labels never become evaluation answers.

`assemble.py` detects duplicate/conflicting cases and makes four group-disjoint
partitions. With `validation` present, `laya_train.py` evaluates each epoch,
retains the best validation checkpoint, then fits calibration separately.
`--ignore-audio` supports a semantic-only control without enabling an untrained
acoustic-state branch. ASR/EOT models remain fixed; the route encoder and
decision head are fine-tuned.

```bash
python -m finetune.decision_router.assemble --data text.jsonl paired.jsonl public.jsonl \
  --out results/decision-router/joint.jsonl

python -m finetune.decision_router.laya_train \
  --laya-source /path/to/laya --source-revision UPSTREAM_COMMIT \
  --model /path/to/laya-multilingual --model-revision CHECKPOINT_REVISION \
  --data results/decision-router/joint.jsonl \
  --out results/decision-router/voice-router --loss soft-ce --epochs 4 --device cuda:0

python -m evals.voice_decision --checkpoint results/decision-router/voice-router \
  --laya-source /path/to/laya --data results/decision-router/joint.jsonl \
  --compare-qwen /path/to/Qwen3-0.6B --out results/decision-router/voice-test.json
```

Evaluation reports rows/groups, separate ASR views, false deep triggers and deep
recall. Noisy/ASR views are correlated; their count is not a count of independent
speakers or natural conversations. `--audio-mode early` checks missing-span
input as seen during private early preparation.

`export.py` uses the inspected upstream exporter and verifies dynamic graph
behavior. It prepares `mixed`, `int8`, and `encoder-int8` CPU bundles with
independently refitted calibration. Validation preference alone does not release
a bundle: preserve frozen behavior checks and observed-speech qualification.
Retain higher precision when compression regresses.

```bash
python -m finetune.decision_router.export \
  --checkpoint results/decision-router/voice-router --laya-source /path/to/laya \
  --data results/decision-router/joint.jsonl --out results/decision-router/cpu-bundles
```

`--resume` reuses completed evaluations after checking checkpoint metadata. The
ONNX adapter uses two CPU threads and disables idle spinning. Select it with
`{"backend":"onnx","device":"cpu","allow_prototype":true}`. Only graph,
tokenizer and config are required; duplicate PyTorch weights are unnecessary.
Install the pinned runtime wheel using `pip install --no-deps --no-index /path/to/laya.whl`
to preserve the existing Torch stack.

Checkpoint files must be readable by the actual service user. The supplied CUDA
image runs as UID/GID `10001`; root-trained safetensors default to owner-only
access. Grant the selected checkpoint to that service group when deploying
across users. Native Mac extraction uses the current account.

## Opt-in Studio adapter and independent fillers

### Task-specific encoder

`encoder_train.py` provides a separate binary classifier with shared BERT
weights applied independently to current text and recent history. Its head
combines both representations, their interaction and measured acoustic state.
The isolated router encoder does not change the memory embedding model or any
Space vectors. One supported open base is
[hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base), pinned
by model revision and source checksums in the training record.

```bash
python -m finetune.decision_router.encoder_train --model /path/to/bert-base \
  --model-revision FIXED_REVISION --data /path/to/frozen-decisions.jsonl \
  --out results/decision-router/encoder --epochs 4

python -m finetune.decision_router.encoder_export \
  --checkpoint results/decision-router/encoder --out results/decision-router/encoder-onnx
```

`--init-checkpoint` continues from a previous isolated router. Validation selects
the saved epoch, while calibration and tests retain hard labels. The exported
FP32 graph preserves calibration and checks multiple batch sizes and long/short
sequences. Use factory `studio.core.utils.reply_modes.encoder_depth:create`, a
complete checkpoint directory, and either `{"device":"cuda:0","allow_prototype":true}`
or `{"backend":"onnx","device":"cpu","allow_prototype":true}`. The CPU bundle
needs only `model.onnx`, `depth_encoder.json`, `encoder/` and `tokenizer/`.

`hard_examples.py` explicitly promotes inspected authored cases into training
and creates fresh validation/calibration families. Their old evaluation scores
must no longer be reported as independent. `scope_state_data.py` adds reasoning
versus output-detail contrasts and dialogue completion states. Private real
conversation regressions remain excluded. `audio_data.py --select-file` takes a
JSON list of source IDs for new held-out speech; `evals.voice_decision` also
accepts evaluation-only audio manifests. These workflows use designed speech
without a reference recording and never substitute reference text for ASR.

Install the reviewed Laya runtime in the backend environment, then use the
existing **Components → Router → Adapter** selection, or equivalent environment:

```dotenv
STUDIO_COMPONENTS='{"router":{"provider":"adapter","factory":"studio.core.utils.reply_modes.laya:create","model":"/path/to/trained-checkpoint","options":{"device":"cpu","allow_prototype":true}}}'
STUDIO_FILLER_PROVIDER=deepseek
STUDIO_FILLER_MODEL=deepseek-v4-flash
```

`allow_prototype` is an explicit experiment switch, not a deployment-quality
claim. The default Qwen router is unchanged. Selecting the adapter skips Qwen
model acquisition and never loads Qwen for filler wording. The optional filler
instead uses a separately owned non-thinking chat request. It can target a
separate open model served locally:

```dotenv
STUDIO_FILLER_PROVIDER=openai
STUDIO_FILLER_MODEL=your-local-auxiliary-model
STUDIO_FILLER_BASE_URL=http://127.0.0.1:8080/v1
STUDIO_FILLER_API_KEY=
```

Without overrides, the auxiliary generator follows configured visible-reply
provider settings (DeepSeek when none are configured). It never changes the
main reply model, prompt or generation parameters. A different provider or
endpoint does not inherit another service's role-specific credential. Startup
checks configuration without sending a billable probe. Independent text work
uses `STUDIO_FILLER_TIMEOUT_S` (three seconds by default) and starts privately
during the confirmed handoff wait. Main readiness cancels unused work; speech
keeps the original eligibility/threshold and complete-playback gates. The local
Qwen text budget is unchanged.

## Shared-encoder baseline

`train.py` also trains a very small NumPy head on the process-shared frozen
memory encoder. Its factory is `studio.core.utils.reply_modes.decision:create`.
This is a speed/quality baseline rather than Laya/RLCD training. Changing the
memory encoder or stored Space vectors is not part of this experiment.

```bash
python -m finetune.decision_router.train --data results/decision-router/pilot.jsonl \
  --out results/decision-router/linear-head
```

Checkpoint identity and prototype validation prevent accidental silent default
replacement. Keep generated data, caches and model checkpoints outside Git.
