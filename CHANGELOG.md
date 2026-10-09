# Changelog

## 0.1.1 — 2026-10-09

- Inherit normalization from a checkpoint's training metadata; reject a conflicting override and preserve the declared mode in sanitized checkpoints. Legacy bundles remain readable with their documented trust limitation.
- Validate replay array geometry, dtype and finite values against the profile.
- Independently check recorded sampling, tensors, CTC paths, transcripts and metrics; two identically inconsistent traces no longer pass merely because they match each other.
- Reject duplicate CSV headers and rows whose width differs from the header, collecting explicit audit errors.
- Add 13 focused regression cases and a dated journal-readiness evidence boundary.

## 0.1.0 — 2026-10-06

- Public Python package and CLI built from the maintainer's Vietnamese LipNet research implementation.
- Training-only charset, validation checkpoint selection and final test reporting.
- Sanitized model bundles bind weights, ordered charset, blank index and closed preprocessing profiles.
- Dataset audit collects exact-byte duplicates, source-group overlap, missing/invalid media, timestamp errors, unseen characters, normalization collisions and repeated-label CTC infeasibility.
- Trace decoded frames, sampled indices, input tensors, logits, CTC paths, transcripts and optional CER/WER.
- Fixed-geometry ONNX export and CPU ONNX Runtime replay; discrete parity required in addition to numerical tolerance.
- Synthetic train/export/replay demo, fault regression tests, CI, MIT license and versioned citation.
