# GitHub release and existing Zenodo dataset handoff

The reviewable package is published in the `research/context_aware_interval_2026-10-24`
subtree of the public GitHub repository [`ntthienphuc/liptrace`](https://github.com/ntthienphuc/liptrace/tree/f4537c446da12e65d985d708da7198de02aa92b0/research/context_aware_interval_2026-10-24).
The dataset citation uses the existing VLR v2 Zenodo record; this handoff does not create
a duplicate dataset record or upload VLR recordings. No separate software DOI has been
created yet.

## Public layout

The subtree contains the reviewable code, configuration, derived tables, examples,
provenance receipts, manuscript source and compiled PDF. The lightweight release excludes
raw recordings, restricted LRS2 material, complete third-party video collections,
credentials and model weights. Those files remain outside the public release unless their
owners' terms explicitly allow redistribution.

The original study code is covered by the MIT notice in `licenses/LICENSE-MIT.txt`.
Derived numerical tables are covered by the CC BY 4.0 notice in
`licenses/LICENSE-DATA-CC-BY-4.0.txt`. These notices do not relicense source recordings,
restricted metadata, checkpoints or third-party code; their own notices remain under
`licenses/`.

## Reproduction and provenance

The published commit is `f4537c446da12e65d985d708da7198de02aa92b0`. From the research
subtree, run `python reproduce.py` followed by
`python -m unittest discover -s code -p "test_*.py"`. The clean extraction check completed
with 41/41 tests passing and the recorded reproduction reproduced 14,650 clips,
307,650 numerical observations, 84 principal aggregate rows and 8 primary comparisons.
The included verification receipts record the source inventory, hashes, real-frame replay,
data export and paper build.

## Zenodo and dataset citation

Use the existing [VLR v2 Zenodo record](https://zenodo.org/records/19705466) for the VLR
dataset citation and its current access/license metadata. Do not upload VLR media, LRS2
recordings, MuAViC/mTEDx material, ViCocktail recordings or local checkpoints from this
package. LRS2, MuAViC/mTEDx, ViCocktail and local checkpoints retain their source access
conditions.

If the authors later want a software DOI, create a new versioned archive from an
explicitly tagged GitHub commit and record that DOI in the manuscript. That future action
is separate from the existing VLR dataset record and is not claimed here.

## Current verified identity

- Published GitHub commit: `f4537c446da12e65d985d708da7198de02aa92b0`
- PDF SHA-256: `3b222b53f7fb326057a43603519fba4c98344b3c12d8bea2e7207f26e3b0d7a9`
- Archive checksum: see `verification/OpenScience_Release_Receipt.json`
- Numerical scope: 14,650 clips and 307,650 observations
- Release tests: 41/41 pass from a clean extraction
- Paper figures: real VLR prepared frames and cohort geometry with provenance

The manuscript still contains author and correspondence placeholders that must be
replaced before submission. The public release is a code-and-derived-results handoff;
it does not claim that the manuscript has been submitted or that a separate software DOI
already exists.
