"""Explicit sample-mean and corpus-weighted edit metrics, without training deps."""
from .contracts import normalize


def edit_counts(reference, hypothesis):
    """Minimum edits, ties resolved substitution, deletion, then insertion.

    O(len(hypothesis)) memory. Counts are deterministic, not a unique alignment.
    """
    previous = [(i, 0, 0, i) for i in range(len(hypothesis) + 1)]
    for i, token in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, other in enumerate(hypothesis, 1):
            if token == other:
                current.append(previous[j - 1])
                continue
            distance, substitutions, deletions, insertions = previous[j - 1]
            substitute = (distance + 1, substitutions + 1, deletions, insertions)
            distance, substitutions, deletions, insertions = previous[j]
            delete = (distance + 1, substitutions, deletions + 1, insertions)
            distance, substitutions, deletions, insertions = current[j - 1]
            insert = (distance + 1, substitutions, deletions, insertions + 1)
            current.append(min((substitute, delete, insert), key=lambda item: item[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {"edits": distance, "substitutions": substitutions,
            "deletions": deletions, "insertions": insertions,
            "reference_units": len(reference)}


def sample_metrics(reference, hypothesis, normalization="legacy-lower-v1"):
    ref = normalize(reference, normalization)
    hyp = normalize(hypothesis, normalization)
    chars = edit_counts(ref, hyp)
    words = edit_counts(ref.split(), hyp.split())
    return {"reference": ref, "hypothesis": hyp,
            "characters": chars, "words": words,
            "cer": chars["edits"] / chars["reference_units"] if ref else float(bool(hyp)),
            "wer": words["edits"] / words["reference_units"] if ref.split() else float(bool(hyp.split())),
            "exact_match": ref == hyp}


def corpus_metrics(pairs, normalization="legacy-lower-v1"):
    """Successful labeled pairs only; caller must also publish failure coverage."""
    records = [sample_metrics(row["reference"], row["hypothesis"], normalization) for row in pairs]
    totals = {}
    for kind in ("characters", "words"):
        totals[kind] = {key: sum(row[kind][key] for row in records) for key in
                        ("edits", "substitutions", "deletions", "insertions", "reference_units")}
    n = len(records)
    return {"schema": "liptrace-corpus-metrics-v1", "successful_labeled_samples": n,
            "normalization": normalization,
            "sample_mean": {key: sum(row[key] for row in records) / n if n else None
                            for key in ("cer", "wer", "exact_match")},
            "corpus_weighted": {
                "cer": totals["characters"]["edits"] / max(1, totals["characters"]["reference_units"])
                if n else None,
                "wer": totals["words"]["edits"] / max(1, totals["words"]["reference_units"])
                if n else None},
            "counts": totals,
            "empty_reference_samples": sum(not row["reference"] for row in records),
            "denominator_policy": "legacy sample empty-reference 0/1; corpus max(1, summed reference units)",
            "scope": "conditional on successful labeled pairs; not failure-adjusted accuracy"}


def reference_diagnostics(reference, profile):
    normalized = normalize(reference, profile["normalization"])
    required = len(normalized) + sum(a == b for a, b in zip(normalized, normalized[1:]))
    unseen = sorted(set(normalized) - set(profile["charset"]))
    return {"normalized_character_count": len(normalized), "min_ctc_steps": required,
            "available_ctc_steps": profile["model"]["max_frames"], "unseen_characters": unseen,
            "ctc_feasible": required <= profile["model"]["max_frames"],
            "representable_by_charset": not unseen,
            "scope": "label compatibility only; does not verify transcript, crop or recognition"}
