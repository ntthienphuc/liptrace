"""Reproduce published principal counts and uncertainty using only this release."""

import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from family_statistics import summarize_sources


def table(name):
    with (HERE / "data" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=1e-11, abs_tol=1e-10), (actual, expected)


def main():
    totals = defaultdict(Counter)
    records = 0
    clips = set()
    with gzip.open(HERE / "data/Principal_Per_Clip.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in ("model", "cell", "arm", "decoder"))
            group = totals[key]
            for field in ("units", "substitutions", "deletions", "insertions", "errors"):
                group[field] += int(row[field])
            assert int(row["errors"]) == sum(int(row[field]) for field in
                                              ("substitutions", "deletions", "insertions"))
            group["clips"] += 1
            clips.add((row["model"], row["clip_key"]))
            records += 1
    assert len(clips) == 14650
    checked_totals = 0
    for row in table("Main_Results.csv"):
        if row["stratum"] != "all":
            continue
        counts = totals[tuple(row[field] for field in ("model", "cell", "arm", "decoder"))]
        for field in ("units", "substitutions", "deletions", "insertions", "errors", "clips"):
            assert counts[field] == int(row[field]), (row["model"], field)
        close(100 * counts["errors"] / counts["units"], row["WER_percent"])
        checked_totals += 1
    primary_rows = table("Main_Comparisons.csv")
    primary = json.loads((HERE / "data/Primary_Source_Counts.json").read_text())
    primary_checks = 0
    for item in primary:
        statistics = summarize_sources(item["sources"])
        for contrast in statistics["contrasts"]:
            row = next(row for row in primary_rows if row["model"] == item["model"]
                       and row["contrast"] == contrast["contrast"])
            close(contrast["estimate_pp"], row["estimate_pp"])
            for index, suffix in enumerate(("low", "high")):
                close(contrast["ci_family_bonferroni"][index], row["adjusted_ci_" + suffix + "_pp"])
                close(contrast["ci95_marginal"][index], row["marginal_ci95_" + suffix + "_pp"])
            primary_checks += 1
    matched_rows = table("Matched_Decoder_Intervals.csv") + table("Context_Availability.csv")
    groups = json.loads((HERE / "data/Source_Counts.json").read_text())
    matched_checks = 0
    for item in groups:
        statistics = summarize_sources(item["sources"])
        row = next(row for row in matched_rows if row["model"] == item["model"]
                   and row["decoder"] == item["decoder"] and row["context_stratum"] == item["stratum"])
        assert statistics["paired_clips"] == int(row["clips"])
        for arm in ("A", "B", "C"):
            close(statistics["wer_percent"][arm], row[arm + "_wer"])
        for contrast in statistics["contrasts"]:
            name = contrast["contrast"]
            close(contrast["estimate_pp"], row[name + "_pp"])
            close(contrast["ci95_marginal"][0], row[name + "_ci95_low"])
            close(contrast["ci95_marginal"][1], row[name + "_ci95_high"])
        matched_checks += 1
    report = {"status": "passed", "clips": len(clips), "numerical_records": records,
              "principal_aggregate_rows": checked_totals, "primary_comparisons": primary_checks,
              "matched_and_context_cells": matched_checks, "bootstrap_draws_per_cell": 10000,
              "scope": "Numerical reproduction from released counts, not fresh neural inference"}
    output = HERE / "Reproduced_Results.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
