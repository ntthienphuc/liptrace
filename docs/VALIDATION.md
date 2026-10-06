# Software validation

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
