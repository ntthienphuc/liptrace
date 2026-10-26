# Context-aware interval transcription

Research companion for **Context-Aware Interval Transcription for Visual Speech Recognition**.
Public research release, 26 October 2026. The files are published in the `research/context_aware_interval_2026-10-24` subtree of [ntthienphuc/liptrace](https://github.com/ntthienphuc/liptrace/tree/main/research/context_aware_interval_2026-10-24). The existing VLR v2 Zenodo record is used for dataset citation; no separate software DOI has been created.

The task is to transcribe a requested time interval while allowing a bounded amount of real
neighboring video. The recognizer is frozen. The comparison separates:

| Arm | Recognition input | Decoder input | Returned words |
|---|---|---|---|
| A: crop-first | Target I | States encoded from I | Native prediction |
| B: contextual direct | Context-expanded J | Target states encoded within J | Native prediction |
| C: prediction-aligned projection | J | All states of J | Predicted words assigned to I |
| U_word / U_token | J | Same decoded sequence as C | Uniform temporal allocation |
| F: full-clip lookup | Entire clip | Entire clip | Predicted words assigned to I |

B and C observe identical pixels. Projection aligns its own predicted tokens using visual CTC
scores. Evaluation transcripts and audio-derived word times are not supplied to recognition.
The tested allocation uses half-open midpoint ownership; it cannot correct lexical errors in
the contextual prediction. Sliding and full-transcript results use a different endpoint from
requested-interval WER.

## Start with numerical reproduction

From this directory, using Python 3.9 or newer:

```sh
python reproduce.py
python -m pip install -r requirements-core.txt
python -m unittest discover -s code -p "test_*.py"
```

`reproduce.py` uses only Python's standard library. It recomputes the 84 principal aggregate
rows from 307,650 numerical observations covering 14,650 clips, all eight primary paired
comparisons and 40 matched-decoder/context cells. Bootstrap resampling uses 10,000 draws per
cell and the original source order and seed. Typical local run time is under two minutes.
The resulting `Reproduced_Results.json` records the checks actually completed.

The unit tests exercise CTC prefix probabilities against exhaustive path enumeration,
token grouping, target ownership, identical-input controls, deferred reference access,
sliding coverage and stitched logits. Their synthetic fixtures test software contracts;
they are not empirical recognition results.

## Contents

- `code/`: the small inference core, scoring, source-bootstrap analysis and tests.
- `data/`: twelve paper CSVs, source counts and compressed per-clip numerical observations.
- `examples/`: 27 actual prepared VLR frames with frame indices, hashes and attribution.
- `engine/`: the preserved model adapters and necessary upstream source files. Historical
  paths are retained inside this directory so code identities remain traceable.
- `config/Required_Assets.json`: exact checkpoint/tokenizer identities and required paths.
- `licenses/`: third-party notices and the scope of proposed author licensing.

There are no credentials, raw LRS2 recordings, complete third-party video collections,
or model weights in this lightweight package. The image examples are temporal samples for
illustration; they must not be treated as complete videos for recognition.

## Run recognition with authorized assets

Install a separate runtime environment with `requirements-runtime.txt`. Obtain the original
media and checkpoints using `DATA_AND_MODELS.md`. Place assets under `engine/` at the relative
paths in `config/Required_Assets.json`, retaining their registered SHA-256 values. Source files
already present under `engine/` must not be replaced by a different upstream version.

Prepared inputs are uint8, 25 fps, mouth regions of size 96 by 96: `[T,96,96]` for VLR/English,
or `[T,3,96,96]` RGB for the Spanish/native ViCocktail path. Each adapter retains the native
center crop and normalization. The `infer.py` entry point does not perform face tracking,
raw-video cropping, resampling, transcript alignment or model training.

```sh
python infer.py --model VLR_T0 --pixels /path/to/prepared_clip.npy \
  --target-frames 50 --context-frames 16 --position center \
  --decoder ctc40 --device cpu --output predictions.json
```

Run a fresh process for each model family. The public entry point returns A/B/C predictions
without reference transcripts. The preserved engine also implements the original hybrid,
allocation and sliding controls; its calibrated settings are summarized below. A clean-copy
CPU replay has a separate receipt, with the exact tested model and extent of validation.
Numerical reproduction is not a claim that all four neural models were rerun from scratch.

## Frozen experimental settings

The principal request is at most 50 frames, with at most 16 neighboring frames per side.
The principal placement is SHA256(original clip ID) modulo three: start, center, end.
The 1/2/3-second duration experiment uses centered targets; these are different request schedules.
Pure CTC has beam width 40, token pruning 64 and no external language model. VLR selects greedy
for A/B/C and hybrid CTC weight 0.5 for F. Other corpora select hybrid weight 0.1 for A/C/F and
0.5 for B, with zero length bonus. Matched greedy and pure-CTC comparisons hold decoding fixed.
The original job JSON files are necessary for exact cohort regeneration; access and remaining
publication prerequisites are documented in `DATA_AND_MODELS.md`.

`WER = 100 * sum(S+D+I) / sum(reference units)`. Counts are pooled within each corpus.
Vietnamese units are whitespace-delimited orthographic units, not necessarily linguistic words.
No cross-language mean WER is reported. Principal intervals are family-adjusted;
context-stratified intervals are post-hoc pointwise summaries of frozen predictions.

## License and citation

See `licenses/README.md` before redistribution. The current author choice is MIT for
original core code and CC BY 4.0 for derived numerical tables; replace the author
placeholder before publication. Third-party code retains its own license, and CC BY-NC
components are not covered by the MIT core-code license.
The VLR image attribution is in `examples/Frame_Provenance.json`.
Use the paper's bibliography for the original datasets, Auto-AVSR and vsr-low. The final
paper citation and author metadata will be inserted before submission. The existing VLR v2 Zenodo record remains the dataset citation; the software release is identified by the GitHub commit above.





