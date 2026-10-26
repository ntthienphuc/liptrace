# Paper and open-science handoff — 26 October 2026

Workspace: `E:\Project\Lip`

This handoff describes the separate, reproducible manuscript/release revision. The earlier
`PatternRecognition_Submission_Draft_2026-10-22` is preserved and has not been overwritten.

## Stable deliverables

| Artifact | Current evidence |
|---|---|
| `Manuscript/Research_Paper_OpenScience_2026-10-24.pdf` | 53 pages; SHA-256 `49532465dd16bba35e115555183a70592b56396f872a952e19851eb19a088452` |
| `Context_Aware_Interval_OpenScience_Release.zip` | CRC pass; authoritative entry count and SHA-256 are in `Verification/OpenScience_Release_Receipt.json` |
| `Verification/Paper_Revision_Receipt.json` | PDF build, warning scan and visual-inspection receipt |
| `Verification/OpenScience_Release_Receipt.json` | External archive identity and complete payload inventory |
| `FINAL_STATUS_2026-10-26.md` | Scope, evidence boundary and publication prerequisites |
| root `result.txt` | Append-only project status and decisions |

## What is verified

- The revised PDF contains authentic VLR mouth-region examples and corpus-geometry figures.
  The figures have provenance entries and were rendered and inspected on representative
  pages including the first page, data-figure pages, the main results area, availability
  material and the final page.
- The lightweight release reproduces the completed numerical evidence for 14,650 clips and
  307,650 observation records, including 84 principal aggregate rows, 8 primary comparisons,
  40 matched/context cells and 10,000 bootstrap draws per cell.
- A clean extraction of the current archive passes the numerical reproduction script and the
  complete 41/41 release unit-test suite.
- A proper archive scan found no raw recordings, model-weight files, credentials or
  credential-like paths. The package contains source-level code, permitted derived tables,
  settings, provenance and reconstruction notes.
- The independent CPU retry for `upper_teacher_count/17` completed and was audited, but it
  is exposed-development evidence only: it has no held-out evaluation and is excluded from
  the paper's primary claims.
- No Python training or evaluation job is running at the time of this handoff.

## Scientific scope

The manuscript studies context-aware interval transcription for visual speech recognition.
It separates the requested text interval from the permitted visual observation interval.
The main completed evidence compares crop-first recognition, contextual direct decoding and
prediction-aligned projection across four corpora and three languages, with matched-decoder,
context-intervention, duration, whole-transcript and workload controls.

The paper's interpretation is intentionally bounded. Contextual encoding explains most of
the gain; projection adds a smaller, decoder-dependent readout effect. Full-clip recognition
remains the stronger endpoint when a complete transcript is wanted. Prefix-recovery,
independent-readout, partial/incomplete and exploratory training branches remain in their
original phase folders and are not presented as completed efficacy evidence.

## Public release and dataset provenance

The compact public source/data release is in the research subtree of
[ntthienphuc/liptrace](https://github.com/ntthienphuc/liptrace/tree/main/research/context_aware_interval_2026-10-24).
The release-content commit is
`5f72387a68e447da7b67766087cf7c1537048008`; the pre-finalization public verification
commit was `be28b6eb48f203cf6ebd92be710054bbf0dbe091`.

The dataset citation uses the existing
[VLR v2 Zenodo record](https://zenodo.org/records/19705466). No raw VLR media, LRS2 media,
MuAViC/mTEDx media, ViCocktail recordings, restricted metadata or local checkpoints are
redistributed. LRS2 remains subject to the Oxford--BBC agreement; other corpus terms remain
those of their original releases. The release's MIT and CC BY 4.0 notices apply only to
the original study code and derived numerical tables, respectively.

## Publication prerequisites

1. Replace the author, affiliation and correspondence placeholders in the LaTeX manuscript.
2. Replace the author/copyright placeholders in the selected MIT and CC BY 4.0 notices.
3. Rebuild the PDF and refresh the package checksums after those metadata edits.
4. Confirm the final citation metadata and any optional software DOI before submission.
5. Keep the corpus-specific access and rights statements unchanged unless the relevant
   upstream terms are independently rechecked.

The local `AUTHOR_METADATA_GUIDE.md` gives the LaTeX metadata examples. The original
Overleaf draft was inspected without editing. The user subsequently authorized updating
a separate Overleaf copy; that deployment is documented in `Overleaf_Free_2026-10-26/`.
The user will fill author identities and copyright names; these are not invented here.

The archive contains `verification/Content_Manifest.json` and `verify_release.py`.
The archive hash receipt lives beside the ZIP, not inside it: embedding that receipt would
make it stale or self-referential. Numerical reproduction and integrity checks are separate.
