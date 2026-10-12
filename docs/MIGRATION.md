# Corpus migration and runtime-only deployment

LipTrace separates the source model, packaged recognizer, exported runtime,
recorded CTC decisions and reference-label assessment. A successful export or a
matching transcript alone does not establish numerical fidelity or recognition
accuracy. The current production adapter is a fixed-size LipNet-style character
CTC model; two checkpoints do not constitute two architecture adapters.

## Install by role

For training, checkpoint registration, export and corpus comparison:

```bash
git clone https://github.com/ntthienphuc/liptrace.git
cd liptrace
python -m pip install ".[training,onnx]"
```

When working from this checkout, use `".[training,onnx]"`. An ONNX-only machine
requires `liptrace[runtime]`; it needs no PyTorch, pandas, tqdm or ONNX exporter.
The deployment contains `model.onnx` and `manifest.json`, with a fixed BCTHW
input and BTC output. It preserves source hashes but cannot verify an absent
checkpoint's contents. A trusted manifest digest remains an external input.

```bash
liptrace bundle --checkpoint best.pt --out source_bundle
liptrace export --bundle source_bundle --runtime-only --out deployment
liptrace inspect --bundle deployment --expect-manifest-sha256 TRUSTED_DIGEST
liptrace trace --bundle deployment --runtime onnx --video mouth.mp4 --out trace_onnx
```

## Declare a case

Paths resolve against the case file. Videos must already be mouth cropped.
`source_group` is a declared recording identifier, not verified speaker identity.

```json
{
  "schema": "liptrace-migration-case-v1",
  "bundle": "source_bundle",
  "deployment": "deployment",
  "samples": [
    {
      "sample_id": "clip-001",
      "source_group": "source-recording-001",
      "video": "mouth/clip-001.mp4",
      "expected_video_sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_LOWERCASE_SHA256",
      "reference": "xin chao",
      "reference_status": "reviewed"
    }
  ]
}
```

Omit `reference` for unlabeled inputs; the status then defaults to `absent`.
Labeled statuses are `reviewed`, `historical` or `unreviewed`; these are user
declarations. The toolkit never promotes a label to reviewed. Normalization
does not remove Vietnamese accents. Label compatibility diagnostics record
unseen characters and minimum CTC steps; they do not silently alter the label
or exclude the video from fidelity assessment.

```bash
liptrace --threads 2 migrate --case case.json --out migration_run \
  --warmup 1 --repetitions 3 --atol 0.00001 --rtol 0.0001
```

The destination must be new. Strict-fidelity failure returns exit code 2 after
saving the complete report. Missing videos, wrong hashes and execution failures
remain in all-planned comparison denominators. Successful labeled predictions
have separately reported coverage; failed predictions are not silently scored
as successful predictions or erased from comparison counts.

Outputs include `receipt.json`, `progress.json`, `report.json`,
`predictions.csv` and per-sample stage arrays/results. Receipts bind software
source files, case, model, graph, profile, video bytes and numerical tolerances.
Local paths, frames, transcripts and logits may require separate permissions.

## Compare an original implementation

An optional `original_adapter` names a trusted local `file.py:factory`.
Supply `checkpoint` and explicitly pass `--allow-original-adapter`. The
factory receives `(checkpoint, profile)` and returns a callable accepting
`video_path`. Its result must contain float32 `tensor` (BCTHW), float32
`logits` (BTC) and a string `transcript`. Original tensor/logits/path/output
are compared with the packaged trace. Original decoded frames and sample
indices are not independently captured by this interface.

This executes local Python code; it is not an executable model-profile field.
The case receipt pins adapter source and checkpoint hashes. Hashes establish
consistency with those identities, not the trustworthiness of the code.
`examples/legacy_reference_adapter.py` imports a user-supplied legacy trainer
through `LIPTRACE_LEGACY_TRAINER`; the trainer itself is not redistributed.

## Numerical, path and task fidelity

`liptrace compare` defaults to strict stages. `--policy task-output` reports
numeric, path and transcript agreement separately and permits a changed CTC
path only when profile/source identity, internal trace consistency and decoded
output/metrics agree. The report always retains strict fidelity and the first
divergence. Corpus runs retain both outcomes and use strict fidelity for their
overall exit status. Tolerances must be fixed before evaluating a real case.

## Timing and metrics

Loaded-backend measurements separate decode, packing, forward and CTC decode.
Warmup observations are excluded; raw measured repetitions and pooled mean,
p50 and p95 remain in the report. Serial clips/s is derived from the mean
pipeline duration. Loading and trace-write measurements are recorded separately.
These host CPU timings exclude mouth extraction, camera, network, UI and Jetson.
They do not establish concurrent server capacity or live recognition latency.

CER/WER include deterministic S/D/I edit counts. Both the mean of sample ratios
and corpus-weighted edit/reference-unit ratios are reported. Empty reference
sample ratios retain the legacy 0/1 convention; corpus denominators use the
recorded sum with a minimum of one. Label statuses and failure coverage remain
separate. Reference review and legitimate data access are research obligations.

## Public reproduction without human data

```bash
python examples/migration_demo.py --out generated_migration --repetitions 3
```

This generates three clips and an untrained, seeded tiny checkpoint. It
exercises packing/padding/sampling, PyTorch, runtime-only export and corpus
comparison. It downloads no human videos or pretrained weights and supports
engineering reproduction only.
