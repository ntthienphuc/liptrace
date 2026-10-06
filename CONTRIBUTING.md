# Contributing

Install CPU PyTorch as in the README, then `python -m pip install ".[dev,onnx]"`.
Run `python -m pytest -q` and `liptrace demo --out demo_new --onnx`.
Use a new output directory for every demo.

Report software bugs with the package version, runtime versions, command, input
contract and a small redistributable fixture. Do not upload private human videos,
secrets, local paths containing personal details or checkpoints without permission.

Changes to charset, normalization, sampling, decoder or metric semantics require
a versioned profile change and tests for compatibility. A new model adapter needs
shape/length tests, export/replay evidence and attribution for any reused code.
Keep algorithmic claims separate from software validation results. Contributions
must be yours or include compatible licensing and retained notices.
