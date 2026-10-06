# Reproduce software validation

Use the commands in the README to create a fresh environment. Then:

```bash
python -m pip install ".[dev,onnx]"
python -m pip check
python -m pytest -q
liptrace demo --out validation_run --onnx
python scripts/fault_benchmark.py --out fault_report.json
python -m build
python -m twine check dist/*
python scripts/check_dist.py --dist dist
```

Windows PowerShell accepts the same commands; replace the final wildcard with
explicit artifact names if the shell does not expand them for a tool.

Outputs to retain: `fault_report.json`, `validation_run/receipt.json`,
`validation_run/onnx_comparison.json`, `validation_run/run/dataset_audit.json`,
`validation_run/run/summary_metrics.json` and the runtime environment.
Dataset audit and training outputs can contain local paths and labels. Review
before sharing; the checked-in validation summary omits those private paths.

`compare` requires equal source-video and semantic-profile identities. Decoded
frames, sample indices and CTC path require exact equality; float tensors/logits
use `atol=1e-5,rtol=1e-4`; transcripts and optional metrics require equality.
Different tests or tolerances must be reported explicitly.

Release assets are a wheel, source distribution and `SHA256SUMS.txt`. Verify
downloaded files against that checksum file, then install the wheel in a clean
environment and run `liptrace demo`. A checksum file distributed with the assets
detects accidental corruption; obtain its identity from a trusted release source.
CI repeats the checks from a fresh checkout and uploads validation reports.

These checks do not retrain/evaluate a natural Vietnamese corpus, verify every
checkpoint from the historical paper, or establish Jetson deployment performance.
