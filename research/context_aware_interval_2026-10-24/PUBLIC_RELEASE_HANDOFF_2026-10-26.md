# GitHub release and existing Zenodo dataset handoff

The reviewable package is ready for a GitHub source release. The dataset citation uses
the existing VLR v2 Zenodo record; this handoff does not create a duplicate dataset record
or upload VLR recordings. No external repository or new software DOI has been created by
this handoff.

## Proposed public layout

Use the contents of `Context_Aware_Interval_OpenScience_Release.zip` as the release
root. Keep the following visible at the repository root:

- `README.md`, `DATA_AND_MODELS.md`, `FINAL_STATUS_2026-10-26.md` and this handoff;
- `LICENSE-MIT.txt` for original code and `LICENSE-DATA-CC-BY-4.0.txt` for derived
  numerical tables;
- `code/`, `config/`, `data/`, `examples/`, `engine/`, `licenses/`, `Manuscript/`
  and `verification/`.

The lightweight archive excludes raw recordings, restricted LRS2 material, complete
third-party video collections, credentials and model weights. Do not add those files
to the public repository unless their owners' terms explicitly allow it.

## GitHub preparation

1. Create a repository with the final author-approved name and add the archive contents.
2. Replace `[Author names to be inserted]`, the manuscript author placeholders and the
   correspondence address before the first public commit.
3. Keep the MIT notice for original code and the CC BY 4.0 notice limited to derived
   tables. Preserve every third-party notice under `licenses/`.
4. Run `python reproduce.py` and `python -m unittest discover -s code -p "test_*.py"`
   from the repository root after the upload.
5. Tag the exact commit used for the Zenodo archive and record the commit hash in the
   paper and Zenodo metadata.

## Zenodo and dataset citation

Use the existing [VLR v2 Zenodo record](https://zenodo.org/records/19705466) for the
VLR dataset citation and its current access/license metadata. Do not upload VLR media,
LRS2 recordings, MuAViC/mTEDx material, ViCocktail recordings or local checkpoints from
this package. State explicitly that LRS2, MuAViC/mTEDx, ViCocktail and local checkpoints
retain their source access conditions.

If the authors later want a separate software DOI, create a new versioned archive from
the tagged GitHub commit and record that DOI in the manuscript. That future action is
separate from the existing VLR dataset record and is not claimed here.

## Current verified identity

- PDF SHA-256: `92f40f8df5864ee597506424e9f8eb2b4b1587f0952ef63c4d3605d3556da22b`
- Archive checksum: see `verification/OpenScience_Release_Receipt.json`
- Numerical scope: 14,650 clips and 307,650 observations
- Release tests: 41/41 pass from a clean extraction
- Paper figures: real VLR prepared frames and cohort geometry with provenance

The handoff is a preparation artifact. It is not evidence that GitHub or Zenodo
publication has already occurred.
