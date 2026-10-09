# Contracts and API

## Supported adapter

`LipNetBackbone`: three 3D convolution blocks with spatial pooling, two
bidirectional GRUs, dropout and character logits `[B,T,C]`. Input is
float32 `[B,1,T,H,W]`. There is no temporal downsampling, so the audit's effective
CTC length equals `max_frames`. Blank is the last index, after the ordered charset.
Image size is divisible by eight. Export fixes batch size at one and fixes T/H/W.

All actual decoded frames are read (up to 4,096 frames and 256 MiB), sampled with
rounded uniform indices for long clips, or padded by repeating the last frame
for short clips. OpenCV BGR-to-gray, `INTER_AREA` resize and float32
`uint8/255*2-1` complete the profile. A decoded-prefix codec failure cannot always
be distinguished from normal end-of-file. Variable frame geometry is rejected.

The recovered trainer's model structure is retained. LipTrace changes the older
metadata-count-driven reader to actual-frame decoding; this is not a claim of
bit-identical historical preprocessing for malformed containers. Train and replay
use the same new reader and packing implementation.

Training accepts `--device auto|cpu|cuda` and `--threads N`; auto chooses CUDA
when available. The generated demo explicitly trains on CPU, and the public
replay/ONNX path is CPU-only in v0.1.0.

## Bundles

`model.pt` and `manifest.json` are required; exported bundles also have
`model.onnx`. Checkpoint registration removes optimizer state and private training
paths, retains model state/charset/blank/geometry, and creates a closed profile.
Unsupported normalization, decoder, architecture or preprocessing is rejected.
Per-file SHA-256 verifies bytes; checkpoint metadata must also equal the profile.

From v0.1.1, bundle creation inherits `args.normalization` when the checkpoint
declares it and preserves that field in the sanitized checkpoint. A conflicting
explicit override fails before writing an output directory. Checkpoints without
that historical metadata use the explicit choice or `legacy-lower-v1` fallback;
the tool cannot reconstruct their original label policy from weights alone.
Existing v0.1.0 bundles omitted this field and remain readable. For those bundles,
use an externally trusted manifest digest: their normalization declaration cannot
be cross-checked against the sanitized checkpoint. Rebuilding from the original
training checkpoint supplies the stronger binding.

```python
from liptrace.bundle import create_bundle, read_bundle, export_onnx
identity = create_bundle('best.pt', 'bundle')
read_bundle('bundle', expected_manifest_sha256=identity['manifest_sha256'])
export_onnx('bundle', 'exported')
```

Use a trusted manifest hash outside the bundle when identity matters. Replacing
both artifacts and their manifest can bypass a purely local checksum comparison.
`torch.load(..., weights_only=True)` narrows deserialization, but artifacts remain
resource-bearing inputs, not authenticated or sandboxed files.

## Dataset audit

```python
from liptrace.dataset import audit_dataset
report = audit_dataset(['train.csv', 'val.csv', 'test.csv'], max_frames=32)
assert report['valid'], report['issues']
```

CSV-relative paths work independently of the current directory. Empty labels,
missing groups, missing/failed media, cross-role group/path/hash overlap, conflicting
labels on identical media, unseen validation/test characters, invalid supplied
timestamps and infeasible CTC labels are errors. All issues are collected.
Within-role duplicates and many-to-one normalization collisions are warnings.
Duplicate CSV headers and rows with extra or missing fields are errors. A row
whose column count is ambiguous is reported and omitted from parsed records;
its split is invalid regardless of the remaining rows.

Minimum CTC length is the number of characters plus the number of adjacent equal
characters: `book` needs five output steps. Do not remove repeated characters to
make a label fit. Training derives characters only from training labels, selects
the checkpoint using validation mean CER, and evaluates test predictions afterward.
Audit inspecting held-out labels for contract violations is disclosed; those labels
do not extend the training charset. A failed audit stops before gradient updates.
Audit records unverified source groups, not biometric identity or verified speakers.

## Replay

```python
from liptrace.replay import trace_video, compare_traces
trace_video('bundle', 'mouth.mp4', 'trace_a', reference='xin chao')
trace_video('exported', 'mouth.mp4', 'trace_b', runtime='onnx', reference='xin chao')
report = compare_traces('trace_a', 'trace_b')
```

Trace arrays use compressed NPZ loaded with `allow_pickle=False`; JSON stores
profile/source identities, environment, transcript and optional metrics. Replay
checks profile and source-video identity, then decoded BGR frames, sample indices,
packed tensor, logits, CTC path, transcript and metrics. It reports all comparisons
and the first divergent stage. Tiny numeric differences can change an argmax, so
equal transcripts and paths are checked separately from float tolerance.

Version 0.1.1 also validates array shape, dtype and finite values before comparison,
then independently derives sample indices/tensors from the saved frames, CTC paths
from logits, transcripts from paths and metrics from the declared reference. The
`trace_consistency` result is required for a passing comparison. When matching
traces contain the same stale derived value, `first_divergence` names e.g.
`left_trace.ctc_path`. When traces differ, the first cross-trace difference remains
the primary diagnosis and both consistency reports are retained. This check uses
the current OpenCV packing implementation; it does not prove that saved logits
were produced by the named model or authenticate edited JSON. The reference text
is an input declaration, not independently established ground truth.

Forward timings are one CPU call without warmup, excluding loading, decoding,
preprocessing, disk writes and UI. They are execution diagnostics, not a latency
benchmark. Replay arrays retain input frames: do not publish human traces casually.
