# Data and model access

This companion separates research outputs produced by the study from third-party recordings,
labels and pretrained weights. Public access to a download is not a grant to redistribute it.
The manuscript uses frozen corpus/model pairs; a newer release is not automatically equivalent.

## Corpora

| Corpus | Original source | Distribution in this companion |
|---|---|---|
| VLR v2 | [Zenodo 19705466](https://zenodo.org/records/19705466), DOI 10.5281/zenodo.19705466 | Numerical results and 27 prepared frames with attribution. DOI metadata reports CC BY 4.0. Full media are obtained from the original record. |
| LRS2 | [Oxford--BBC dataset](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) | Derived numerical counts. Original videos/metadata require the BBC data-sharing agreement; no password, transcript or original file list is distributed here. |
| Spanish MuAViC/mTEDx | [Official MuAViC](https://github.com/facebookresearch/muavic) | Derived numerical counts and original access instructions. The repository reports CC BY-NC 4.0; TED/TEDx source terms also apply. |
| ViCocktail | [Author dataset](https://huggingface.co/datasets/nguyenvulebinh/ViCocktail), [author repository](https://github.com/nguyenvulebinh/viCocktail) | Derived numerical counts. The dataset card inspected on 24 October does not provide a clear dataset license field, so this package does not redistribute its recordings. |

The VLR license was checked through the DOI's [DataCite metadata](https://api.datacite.org/dois/10.5281/zenodo.19705466).
The accompanying local verification records the returned rights metadata. Frame provenance includes
the original clip identifier, source recording, frame index and preparation/payload hashes.
The figure retains the prepared crop, overlays and blur and uses no generative modification.

## Numerical observations

`data/Principal_Per_Clip.csv.gz` contains one row per frozen clip/cell/arm/decoder observation.
It stores edit counts and frame geometry. It excludes reference text, hypotheses and media.
The clip key is SHA256 of UTF-8 `model_id:original_clip_id`. Recording-source aliases retain
the original sorted order to reproduce the exact bootstrap draws. These aliases are not
verified speaker identities and are not a claim of anonymous-person data.

`Primary_Source_Counts.json` reproduces the selected-system comparisons.
`Source_Counts.json` reproduces matched greedy/pure-CTC and context-stratum comparisons.
The CSVs document principal, duration, intervention, whole-transcript, timing and query-cost
outcomes. Only the principal comparisons and matched/context intervals are fully recomputed by
the included numerical command; other CSVs are preserved exports and are identified as such.
No references are reconstructed from numerical counts.

## Models

| Model | Source / status |
|---|---|
| EN_B1 | [Auto-AVSR model zoo](https://github.com/mpc001/auto_avsr#model-zoo), `vsr_trlrs2lrs3vox2avsp_base.pth` |
| ES_NATIVE | [vsr-low pinned revision](https://github.com/JeongHun0716/vsr-low/tree/0114f5f1248a55507c551ed88a69bb978ec9f126), `es_mted_vox_avs_wer_45.7.pth` |
| VLR_T0 | Local Auto-AVSR continuation, seed 17, selected at step 3,000. Not a dataset-author checkpoint. Public weight deposit still required. |
| VC_NATIVE | Local ViCocktail Auto-AVSR, epoch 13 / step 55,986. Not the ViCocktail authors' AV-HuBERT model. Public weight deposit still required. |

All checkpoint, tokenizer and units hashes are specified in `config/Required_Assets.json` and
`data/Model_Registry.csv`. The VLR loader also needs its registered initialization checkpoint;
both local Vietnamese model histories and their redistribution terms must be reviewed before
depositing weights. Optimizer states and training caches are not needed for inference release.
The Spanish model filename's WER is an upstream filename, not this paper's result.

The preserved Spanish adapter verifies its upstream source archive as well as imported files.
Its download URL and exact expected archive hash are recorded among required assets. Upstream
archive bytes may change despite an unchanged commit; a different hash requires investigation,
not silently deleting the check.

## What remains before a complete submission-ready release

1. Replace the author list and final paper citation; the code-and-derived-results subtree is already published at `ntthienphuc/liptrace`, commit `c86f104`.
2. Replace the author placeholder in the MIT and CC BY 4.0 notices and preserve the final provenance.
3. Deposit the two local Vietnamese checkpoints and their tokenizer/initialization dependencies,
   or explicitly state their access conditions. A hash alone is not public model availability.
4. Finish portable acquisition/preparation recipes and exact cohort reconstruction. Restricted
   corpus labels must be obtained by each recipient under the original agreements.
5. Verify end-to-end reconstruction for all four model/corpus pairs with those accessible assets.

The current candidate supports complete principal numerical reproduction and a bounded real
VLR replay. It does not claim that these remaining full-inference release steps have passed.

## Journal data policy

[Elsevier research-data guidance](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines)
provides several journal-specific options, including availability statements when data cannot
be shared. The exact current [Pattern Recognition author guide](https://www.sciencedirect.com/journal/pattern-recognition/publish/guide-for-authors)
returned HTTP 403 in this check. Its current journal-specific wording therefore still needs
confirmation in the author/submission interface. No blanket claim is made that every original
third-party dataset can or must be relicensed publicly.

