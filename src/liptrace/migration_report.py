"""All-row migration coverage and explicitly separated label-status metrics."""
import csv
import numpy as np
from pathlib import Path
from .metrics import corpus_metrics


def summarize(rows, expected, normalization):
    total = len(rows)
    comparisons = {}
    for name in expected:
        available = [row["comparisons"][name] for row in rows if name in row["comparisons"]]
        strict_key = "strict_fidelity" if name == "torch_onnx" else "pass"
        strict = sum(bool(item[strict_key]) for item in available)
        task = sum(bool(item.get("task_output_preserved", item[strict_key])) for item in available)
        comparisons[name] = {"planned": total, "evaluated": len(available),
                             "missing": total - len(available), "strict_pass": strict,
                             "task_output_pass": task, "strict_pass_over_all_planned": strict / total,
                             "task_output_pass_over_all_planned": task / total}
    metrics = {}
    for backend in ("original", "torch", "onnx"):
        metrics[backend] = {}
        for state in ("reviewed", "historical", "unreviewed"):
            planned = [row for row in rows if row["reference_status"] == state]
            pairs = [{"reference": row["reference"], "hypothesis": row["executions"][backend]["transcript"]}
                     for row in planned if backend in row["executions"]]
            metrics[backend][state] = {"planned_labeled_samples": len(planned),
                                       "successful_labeled_samples": len(pairs),
                                       "coverage": len(pairs) / len(planned) if planned else None,
                                       "assessment": corpus_metrics(pairs, normalization),
                                       "scope": "review status is a case declaration, not verified by software"}
    errors = sum(bool(row["errors"]) for row in rows)
    timing = {}
    for backend in ("torch", "onnx"):
        entries = [row["benchmarks"][backend] for row in rows if backend in row["benchmarks"]]
        if not entries:
            continue
        observations = {stage: [value for entry in entries for value in entry["observations_ms"][stage]]
                        for stage in ("decode", "pack", "forward", "ctc_decode", "pipeline")}
        timing[backend] = {"successful_samples": len(entries),
                           "measured_repetitions": len(observations["pipeline"]),
                           "serial_clips_per_second": 1000 / float(np.mean(observations["pipeline"])),
                           "stages": {stage: {"mean_ms": float(np.mean(values)),
                                               "p50_ms": float(np.percentile(values, 50)),
                                               "p95_ms": float(np.percentile(values, 95))}
                                      for stage, values in observations.items()},
                           "environment": entries[0]["environment"],
                           "providers": entries[0]["providers"], "threads": entries[0]["threads"],
                           "scope": entries[0]["scope"],
                           "aggregation": "pooled repetitions, conditional on completed measurements"}
    strict_all = all(item["strict_pass"] == total for item in comparisons.values())
    return {"planned_samples": total, "rows_with_errors": errors,
            "source_group_values": len({row["source_group"] for row in rows}),
            "comparisons": comparisons, "metrics_by_label_status": metrics,
            "host_timing": timing,
            "pass": bool(total and expected and strict_all and errors == 0),
            "coverage_policy": "every planned row remains in comparison denominators; "
                               "accuracy ratios are conditional and separated by label status"}


def write_predictions(root, rows):
    fields = ["sample_id", "source_group", "reference_status", "reference",
              "original_prediction", "torch_prediction", "onnx_prediction",
              "original_torch_pass", "torch_onnx_strict", "torch_onnx_task", "errors"]
    with (Path(root) / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            original = row["comparisons"].get("original_torch", {})
            exported = row["comparisons"].get("torch_onnx", {})
            writer.writerow({"sample_id": row["sample_id"], "source_group": row["source_group"],
                             "reference_status": row["reference_status"], "reference": row.get("reference"),
                             **{f"{name}_prediction": row["executions"].get(name, {}).get("transcript", "")
                                for name in ("original", "torch", "onnx")},
                             "original_torch_pass": original.get("pass", ""),
                             "torch_onnx_strict": exported.get("strict_fidelity", ""),
                             "torch_onnx_task": exported.get("task_output_preserved", ""),
                             "errors": "; ".join(item["stage"] + ": " + item["message"]
                                                for item in row["errors"])})
