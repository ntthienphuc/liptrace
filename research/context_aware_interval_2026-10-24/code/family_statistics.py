"""Paired source-cluster percentile intervals for the two frozen E1 contrasts."""

import math
import random


def quantile(ordered, probability):
    if not ordered or not 0 <= probability <= 1:
        raise ValueError("Nonempty sorted samples and a probability are required")
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    weight = coordinate - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_sources(sources, draws=10000, seed=20261020, family_size=8):
    if not sources or draws < 1 or family_size < 1:
        raise ValueError("Positive sample, draw and family sizes are required")
    names = [row["source_id"] for row in sources]
    if names != sorted(set(names)):
        raise ValueError("Unique source IDs must be sorted for deterministic resampling")
    for row in sources:
        for field in ("A_errors", "B_errors", "C_errors", "units", "clips"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError("Source counts must be nonnegative integers")
        if not row["source_id"] or row["clips"] < 1:
            raise ValueError("Every source needs an ID and at least one paired clip")
    total_units = sum(row["units"] for row in sources)
    if total_units == 0:
        raise ValueError("Observed aggregate reference denominator is zero")
    total_clips = sum(row["clips"] for row in sources)
    total_errors = {arm: sum(row[arm + "_errors"] for row in sources) for arm in ("A", "B", "C")}
    columns = [(row["C_errors"] - row["A_errors"], row["C_errors"] - row["B_errors"], row["units"])
               for row in sources]
    generator = random.Random(seed)
    samples = {"C_minus_A": [], "C_minus_B": []}
    undefined = 0
    for unused in range(draws):
        difference_a = 0
        difference_b = 0
        units = 0
        for unused_source in range(len(sources)):
            selected = columns[generator.randrange(len(columns))]
            difference_a += selected[0]
            difference_b += selected[1]
            units += selected[2]
        if units:
            samples["C_minus_A"].append(100 * difference_a / units)
            samples["C_minus_B"].append(100 * difference_b / units)
        else:
            undefined += 1
    adjusted_tail = 0.05 / (2 * family_size)
    contrasts = []
    for baseline in ("A", "B"):
        name = "C_minus_" + baseline
        ordered = sorted(samples[name])
        intervals = None
        adjusted = None
        if ordered:
            intervals = [quantile(ordered, probability) for probability in (0.025, 0.975)]
            adjusted = [quantile(ordered, probability) for probability in (adjusted_tail, 1 - adjusted_tail)]
        difference = total_errors["C"] - total_errors[baseline]
        leave_out = []
        per_source = []
        for row in sources:
            delta = row["C_errors"] - row[baseline + "_errors"]
            denominator = total_units - row["units"]
            estimate = 100 * (difference - delta) / denominator if denominator else None
            if estimate is not None:
                leave_out.append(estimate)
            per_source.append({
                "source_id": row["source_id"], "clips": row["clips"], "units": row["units"],
                "difference_errors": delta, "leave_one_source_out_delta_pp": estimate,
            })
        contrasts.append({
            "contrast": name, "estimate_pp": 100 * difference / total_units,
            "difference_errors": difference, "reference_units": total_units,
            "ci95_marginal": intervals, "ci_family_bonferroni": adjusted,
            "family_size": family_size, "individual_confidence": 1 - 0.05 / family_size,
            "adjusted_tail_probability": adjusted_tail,
            "draws": draws, "valid_draws": len(ordered), "undefined_draws": undefined, "seed": seed,
            "status": "complete" if ordered and not undefined else "undefined_bootstrap_denominator",
            "loso_range_pp": [min(leave_out), max(leave_out)] if leave_out else None,
            "loso_negative": sum(value < 0 for value in leave_out),
            "loso_positive": sum(value > 0 for value in leave_out),
            "source_errors_negative": sum(row["difference_errors"] < 0 for row in per_source),
            "source_errors_positive": sum(row["difference_errors"] > 0 for row in per_source),
            "source_errors_zero": sum(row["difference_errors"] == 0 for row in per_source),
            "per_source": per_source,
        })
    return {
        "method": "paired_source_cluster_percentile_bootstrap",
        "source_clusters": len(sources), "paired_clips": total_clips, "reference_units": total_units,
        "max_source_unit_share": max(row["units"] for row in sources) / total_units,
        "max_source_clip_share": max(row["clips"] for row in sources) / total_clips,
        "wer_percent": {arm: 100 * errors / total_units for arm, errors in total_errors.items()},
        "sources": sources, "contrasts": contrasts,
    }
