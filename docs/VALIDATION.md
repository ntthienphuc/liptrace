# Software validation

## Corpus migration and runtime-only deployment — 2026-10-12

The v0.2.0 implementation passed **90 local tests** on Windows/Python 3.11.9.
A local natural-input case compares the independently imported original trainer,
LipTrace package and ONNX Runtime CPU for two recovered checkpoints. Both
original/package and package/ONNX comparisons passed **17/17** per checkpoint
at `atol=1e-5, rtol=1e-4`, with no failed or omitted inputs. This is **34 model–clip
executions on 17 unique clips**, not 34 independent clips or two architecture
adapters. Numeric tolerance, CTC path, transcript and trace consistency passed.

Sixteen pre-existing 96×96 mouth clips represent distinct source-video IDs; a
recovered 7-frame clip additionally exercises repeat-last padding. Source-video
IDs are not verified speaker identities. The 16 labels remain unreviewed, all
contain characters unseen by these charsets, and 12 need more than 32 CTC steps.
Their diagnostic CER/WER is retained locally but is not used as an accuracy
benchmark. No historical test accuracy is re-established.

| Model | Runtime | Mean forward (ms) | Mean pipeline (ms) | p50 pipeline (ms) | p95 pipeline (ms) | Serial clips/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| unigram | torch CPU | 230.34 | 258.39 | 254.38 | 302.88 | 3.87 |
| unigram | onnx CPU | 234.14 | 263.19 | 256.01 | 333.53 | 3.80 |
| mixed | torch CPU | 225.64 | 251.18 | 245.21 | 305.55 | 3.98 |
| mixed | onnx CPU | 213.12 | 237.40 | 234.43 | 278.23 | 4.21 |

These measurements use an Intel Core i7-10750H host, two CPU threads, one warmup
and three measured repetitions per clip/backend (51 measured observations per
row). Percentiles pool recorded repetitions. Models are reused after loading;
pipeline includes video decode, packing, forward and CTC decode. Camera, mouth
extraction, model loading, network, UI and trace persistence are excluded. Runs
are sequential and unrandomized; small timing differences do not establish a
backend performance advantage. No Jetson result is reported.

The runtime-only route is tested in an isolated environment without PyTorch,
ONNX exporter, pandas or tqdm. Deployment metadata records that absent source
checkpoint contents cannot be verified there. New tests cover same-profile
wrong-model rejection, missing/hash-mismatched rows, weighted metrics and changed
CTC paths with preserved output. Public reproduction uses generated data only:
`python examples/migration_demo.py --out migration_demo --repetitions 3`.

Human videos, pretrained checkpoints and raw traces are not redistributed.
[Sanitized identities and measurements](validation_20261012.json) retain source
hashes and counts; exact local recipes/receipts live with the paper evidence.
The public CI and final wheel status must be checked against the new release
commit, independently of these local results.

## Maintenance validation — 2026-10-09

The v0.1.1 candidate passed **57 tests** in Windows/Python 3.11.9 after installing
the built wheel. Thirteen new regression cases cover inherited normalization,
ambiguous CSV input, invalid trace geometry and matching but internally stale
replay records. The existing fault benchmark still detects 14/14 constructed
faults and accepts its one healthy fixture. These are controlled cases, not
estimates of field error prevalence or natural-corpus recognition accuracy.

An installed-wheel demo run from outside the checkout passed one synthetic CPU
training epoch, repeated PyTorch replay, and ONNX replay. All nine cross-trace
checks and all five within-trace consistency checks passed on each side. Maximum
logit error was 1.1920928955078125e-7 at atol=1e-5 and rtol=1e-4. `pip check`,
distribution-content checks and `twine check` passed. Dated machine-readable
evidence is in [validation_20261009.json](validation_20261009.json).

The source and wheel contain identical `LICENSE` and `Licence.txt` copies. Support
is through GitHub Issues. The initial v0.1.0 CI run was confirmed green during this
audit; v0.1.1 CI must be checked against its actual commit/tag before publication.

## Initial v0.1.0 evidence

Initial local check: Windows x64, Python 3.9.13, PyTorch 2.8.0+cpu, NumPy 2.0.2,
OpenCV 4.14.0. A separate clean Python 3.11 environment and fresh GitHub runners
are used to verify installability; current run status is visible in Actions.

The separate clean Windows Python 3.11.9 environment passed `pip check`, all
44 tests, the 14-fault benchmark, and the full ONNX demo invoked outside the
checkout. It used NumPy 2.4.6, PyTorch 2.8.0+cpu and OpenCV 4.14.0. Exact installed
versions are listed in `requirements-tested-win311.txt` (a Windows observation,
not a portable lockfile). Both local demos observed the same maximum logit error.

| Experiment | Initial observed result | Scope |
| --- | --- | --- |
| Automated tests including ONNX | 44 passed | Contract, data and replay regression tests |
| Deliberately injected fault benchmark | 14/14 rejected with expected diagnosis | Finite constructed fault set; not a population estimate |
| Healthy fixture in that benchmark | 1/1 accepted | One controlled healthy split set |
| Synthetic training demo | One epoch, 4 train + 4 validation + 4 test clips | Runnable trainer and validation-selected bundle; no useful accuracy claim |
| Repeated PyTorch replay | All nine comparison stages passed | One generated clip and tiny model |
| PyTorch vs ONNX Runtime CPU replay | All nine stages passed; logits max absolute error 1.1920928955078125e-7 | One generated clip/model at atol=1e-5, rtol=1e-4 |
| Original local checkpoint compatibility | 2/2 load and finite forward | Synthetic zero tensor, no historical test-set evaluation |

The deliberately injected fault experiment is in `scripts/fault_benchmark.py`.
The independent expected outcomes are recorded before checking the software's
response. Unit tests additionally check first-divergence localization for five
array stages and a case where logits pass tolerance but the CTC path changes.

The historical compatible checkpoints produced `[1,32,18]` and `[1,32,19]`
logits, with 11,340,018 and 11,340,275 model parameters respectively. Their hashes
and the sanitized software receipts are in `docs/validation_summary.json`.
Checkpoint files and human clips remain outside the public source and release.

Cross-platform CI runs core checks on Ubuntu/Python 3.10 and 3.12 and
Windows/Python 3.11, and optional ONNX checks on Ubuntu/Python 3.11. Core jobs
intentionally omit ONNX dependencies, so the ONNX test skips there. The ONNX job
and local full-extra checks execute it. See uploaded job artifacts for each
environment's exact reports rather than inferring identical floating-point error
from the initial Windows observation.

The source/wheel audit rejects checkpoints, real video, replay arrays, environment
folders and recognizable credential markers. This audit does not prove every
possible private string is absent; the release is curated from source-only files.

No natural-corpus recognition benchmark, Jetson/TensorRT runtime, live-camera
test, controlled baseline performance comparison or independent reuse study was
conducted for this software release. An ONNX graph checking successfully is not
itself a runtime-parity result; the replay comparison supplies that result.
