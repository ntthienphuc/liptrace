# Software metadata for manuscript preparation

These values describe the v0.1.1 maintenance release. Confirm the final public
tag and archive identifiers when submitting a manuscript; this table does not
claim SoftwareX acceptance or a software DOI.

| Code | Field | Value |
| --- | --- | --- |
| C1 | Current code version | 0.1.1 |
| C2 | Permanent link to code/repository used for this code version | [GitHub tag v0.1.1](https://github.com/ntthienphuc/liptrace/tree/v0.1.1); freeze an archival identifier for the submitted version if available |
| C3 | Legal code license | MIT; identical root `LICENSE` and `Licence.txt`; third-party exclusions documented separately |
| C4 | Code versioning system used | Git |
| C5 | Software code languages, tools and services used | Python; PyTorch; NumPy; OpenCV; pandas; tqdm; optional ONNX/ONNX Runtime; setuptools; GitHub Actions |
| C6 | Compilation requirements, operating environments and dependencies | Python >=3.9; declared dependency bounds in `pyproject.toml`; observed Windows and CI combinations in [VALIDATION.md](VALIDATION.md). CPU demo requires no redistributed human video or pretrained weights |
| C7 | Link to developer documentation/manual | [Versioned README](https://github.com/ntthienphuc/liptrace/blob/v0.1.1/README.md), [contracts/API](CONTRACTS.md), [reproduction](../REPRODUCE.md) |
| C8 | Support email for questions | No public support email is declared; software support is available through [GitHub Issues](https://github.com/ntthienphuc/liptrace/issues). The corresponding author must supply the manuscript contact email |

Reproducible capsule: not deposited; source, build assets and reproduction
instructions are provided in the versioned repository/release.

Support: [GitHub Issues](https://github.com/ntthienphuc/liptrace/issues).
Maintainer: Nguyễn Trần Thiên Phúc. The manuscript correspondence email is pending
author confirmation; no email address is inferred from Git identity.

The current journal template should determine the final field order/names. A
repository tag is a version identifier, not an independent preservation service.
Model files, human recordings and trace frames require their own redistribution
permission and are outside the source license grant.
