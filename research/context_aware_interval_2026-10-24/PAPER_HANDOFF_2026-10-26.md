# Paper and open-science handoff — 26 October 2026

Workspace: `E:\Project\Lip`

This handoff describes the separate paper/release revision. The earlier
`PatternRecognition_Submission_Draft_2026-10-22` is preserved and must not be
overwritten.

## Stable deliverables

| Artifact | Current evidence |
|---|---|
| `Manuscript/Research_Paper_OpenScience_2026-10-24.pdf` | 53 pages; SHA-256 `92f40f8df5864ee597506424e9f8eb2b4b1587f0952ef63c4d3605d3556da22b` |
| `Context_Aware_Interval_OpenScience_Release.zip` | CRC pass; current SHA-256 is recorded in `Verification/OpenScience_Release_Receipt.json` |
| `Verification/Paper_Revision_Receipt.json` | Build and visual-inspection receipt |
| `Verification/OpenScience_Release_Receipt.json` | Archive content and hash receipt |
| `FINAL_STATUS_2026-10-26.md` | Scope, evidence boundary and publication prerequisites |
| root `result.txt` | Append-only project status and decisions |

## What is verified

- The revised PDF includes real VLR mouth-region examples and cohort-geometry figures.
  The figures were rendered and inspected on pages 12–13; pages 1, 34 and 53 were
  also rendered for layout checks.
- Numerical reproduction passes for 14,650 clips and 307,650 observations, including
  84 principal aggregate rows, 8 primary comparisons and 40 matched/context cells.
- The release unit suite passes 41/41 tests. A clean extraction of the final archive
  reproduces the same result and passes the same tests.
- The archive contains no checkpoint weights, raw video/audio or credential-like paths.
- A clean runtime CPU replay matches A/B/C predictions, token counts and interval
  geometry for one real VLR clip. This remains a bounded replay, not a full rerun of
  every model/corpus.

## Scientific scope

The manuscript's claim remains context-aware interval transcription. It separates the
video interval used for recognition from the text interval returned to a request. The
main evidence is the four-corpus comparison of crop-first, contextual direct and
prediction-aligned projection, with matched-decoder, context-intervention, duration,
whole-transcript and workload controls. Projection's incremental gain is modest and
decoder-dependent; contextual direct accounts for most of the improvement. Full-clip
recognition remains the stronger endpoint for complete transcripts.

Recent prefix-recovery, independent-readout and partial/incomplete branches are kept in
their original phase folders and are excluded from the manuscript because they do not
provide a completed independent improvement to the primary question.

## Live work that must not be mistaken for paper evidence

`Research_Anchored_CPU_Recognition_Retry_2026-10-26` has a separate retry process for
`upper_teacher_count/17`. At the last handoff observation it was still running at step
276 with no `Complete.json`. Do not restart or stop it based on this file. If it reaches
a complete receipt, inspect `Complete.json`, `Evaluation.json`, process/log hashes and
the independent verifier before considering any paper update.

## Remaining publication prerequisites

1. Replace author, affiliation and correspondence placeholders in the manuscript.
2. Replace the author placeholder in the selected MIT and CC BY 4.0 notices and record
   the final author/copyright metadata.
3. Supply a permanent repository/archive destination and final citation metadata.
4. Keep the LRS2 agreement, MuAViC CC BY-NC terms, ViCocktail access uncertainty and
   local checkpoint conditions explicit in the final data/code availability statement.
5. Do not call the package publicly released until the destination and rights are fixed.


