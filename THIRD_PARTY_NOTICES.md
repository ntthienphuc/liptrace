# Source, dependencies and data

LipTrace's public source is licensed under MIT by Nguyễn Trần Thiên Phúc, who
supplied the original research implementation and authorized its code release.
The public package extends the recovered PyTorch trainer and model; it does not
vendor the repositories compared in `docs/RELATED_WORK.md`.

LipNet-style architecture and CTC are prior work and must be cited, even though
using an architecture is distinct from copying an implementation. The original
LipNet paper is [Assael et al., 2016](https://arxiv.org/abs/1611.01599). The
maintainer's checkpoint-compatible model is not a newly invented architecture.

Direct dependencies (not incorporated into LipTrace's source license):

| Distribution | Upstream license | Primary license source |
| --- | --- | --- |
| PyTorch | BSD-3-Clause, with additional third-party notices | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| NumPy | BSD-3-Clause | [NumPy LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pandas | BSD-3-Clause | [pandas LICENSE](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| OpenCV Python headless | Packaging MIT; OpenCV Apache-2.0 in current 4.x; bundled-component notices | [opencv-python LICENSE](https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt), [third-party notices](https://github.com/opencv/opencv-python/blob/4.x/LICENSE-3RD-PARTY.txt) |
| tqdm | MIT and MPL-2.0 portions | [tqdm LICENCE](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| ONNX (optional) | Apache-2.0 | [ONNX LICENSE](https://github.com/onnx/onnx/blob/main/LICENSE) |
| ONNX Runtime (optional) | MIT, with third-party notices | [ONNX Runtime LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |

Build/test tooling has its own licenses. Installed binary distributions can
bundle codecs, numerical libraries and other components with additional terms.
Check the notices shipped with the versions you distribute. This repository's
wheel contains only LipTrace's Python source and notices, not dependency binaries.

No research checkpoint, face/mouth video, transcript collection or mined label
bank is licensed or redistributed here. The demo generates geometric synthetic
frames locally. Users must have permission to process and distribute their own
inputs, outputs, reference transcripts, checkpoints and replay arrays.
