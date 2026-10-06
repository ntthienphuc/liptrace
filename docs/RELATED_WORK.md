# Positioning against existing software

Checked against primary repositories on 2026-10-06. The table describes what
their documentation and inspected components establish; it is not a claim that
their entire codebases lack other features.

| Prior software | Established functionality | LipTrace's narrower software focus |
| --- | --- | --- |
| [LipNet-PyTorch](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/tree/40209e09c49553c00c25c7d41faa3706aea3c625) | LipNet-style PyTorch training/testing, GRID data splits and a demo | Bind this project's compatible checkpoint to ordered characters and preprocessing; collect auditable CTC/data faults |
| [Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/182b62837773ab01052d4ac21ef1d2203ea7d267) | Reusable AVSR/VSR training, evaluation, preparation, model zoo and inference tutorials | A smaller character-CTC workflow whose semantic decisions are recorded alongside tensors |
| [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/98adec82349b3ae22aa3f753d733b9a1a84d497d/tools/Polygraphy) | Backend execution, saved results and numerical output comparison | Replay decoded video, sampling, input tensors, logits, discrete CTC path, transcript and optional metrics as an ordered chain |

Numerical parity, manifests, duplicate detection and unified workflows each have
prior art. LipTrace's contribution is their concrete integration for a working
CTC mouth-clip recognizer, with deliberate fault tests that demonstrate the failure
modes it handles. This is an engineering/research-software contribution. It does
not establish a new recognition method, the first Vietnamese lip-reading tool,
superiority over these projects, or suitability for every CTC architecture.

## Sources to cite in a software paper

1. Assael, Shillingford, Whiteson and de Freitas. *LipNet: End-to-End Sentence-level Lipreading*. 2016. [arXiv:1611.01599](https://arxiv.org/abs/1611.01599).
2. Graves, Fernández, Gomez and Schmidhuber. *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML 2006. [Author-hosted paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).
3. Ma et al. *Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels*. ICASSP 2023. [Official implementation and citation](https://github.com/mpc001/auto_avsr).
4. NVIDIA. *Polygraphy*. [Official software and examples](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy).
5. The maintainer's Vietnamese LipNet conference project. [DOI:10.1109/CICN70047.2026.11594266](https://doi.org/10.1109/CICN70047.2026.11594266).

Use verified bibliographic exports when drafting the final reference list. The
current toolkit is not a published SoftwareX article and has no assigned software
DOI. A release DOI can be added after an actual archival deposit.
