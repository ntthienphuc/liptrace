# Finalization status for the current manuscript revision

This folder is a separate revision of the earlier Pattern Recognition working draft.
The earlier draft remains unchanged. The revision keeps the existing scientific scope:
context-aware interval transcription with crop-first, contextual-direct and
prediction-aligned readout controls.

## Completed in this revision

- Added two publication figures built from real prepared VLR inputs and frozen cohort
  statistics. The VLR frame figure records source, frame and payload provenance and cites
  the versioned VLR record with CC BY 4.0 attribution.
- Updated data and code availability statements to distinguish derived numerical results,
  restricted corpora, local checkpoints and third-party licenses.
- Built `Manuscript/Research_Paper_OpenScience_2026-10-24.pdf` from the revised LaTeX source.
  The build has 53 pages and the warning scan found no LaTeX warning, overfull/underfull
  box, or undefined-reference match. Selected pages were rendered and visually inspected.
- Reproduced 84 aggregate rows from 307,650 numerical observations covering 14,650 clips;
  41/41 release tests pass. A fresh clean extraction of the archive reproduced the same
  numerical result and passed the same tests.
- Replayed one real VLR clip from a clean runtime copy. A/B/C greedy predictions,
  token counts and interval geometry matched the frozen prediction record. This is a
  bounded replay, not a claim of four-model end-to-end reruns.

## Evidence deliberately excluded

The recent prefix-recovery, independent-readout, and partial/incomplete recognition
branches do not provide a completed, independent improvement to the paper's primary
question. They remain in their original phase folders and are not silently folded into
the manuscript. The `upper_teacher_count/17` CPU retry completed 512 updates and passed an independent
1,280-row replay. It is exposed-development control evidence, without a held-out result,
and does not change the manuscript claims or primary tables.

## Release boundary

`Context_Aware_Interval_OpenScience_Release.zip` is a compact, reviewable release. It
contains source code, derived numerical tables, tests, manifests, the revised manuscript,
and license/access notes. It excludes raw recordings, credentials and checkpoint weights.
The selected release licenses are MIT for original code and CC BY 4.0 for derived numerical
tables. The release is published in the `research/context_aware_interval_2026-10-24`
subtree of `ntthienphuc/liptrace` at commit `5f72387`; author metadata and the copyright
placeholder still need replacement before submission, and the existing VLR Zenodo record
remains the dataset citation.

## Scientific decision

No new server run is justified by the completed evidence at this checkpoint. The central
paper claim is already supported by the frozen four-corpus analysis; the incremental
projection gain is modest and decoder-dependent, while contextual direct explains most of
the improvement. A future method run should be opened only if it answers a named limitation
with a locked comparator and held-out confirmation.
