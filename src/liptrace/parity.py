"""Separate strict implementation fidelity from preserved decoded output."""
from .replay import compare_traces


def compare_task_traces(left, right, atol=1e-5, rtol=1e-4):
    strict = compare_traces(left, right, atol, rtol)
    checks = {row["stage"]: row["pass"] for row in strict["checks"]}
    consistency = all(row["pass"] for row in strict["trace_consistency"].values())
    identity = checks["profile"] and checks["source_video"]
    discrete = checks["ctc_path"] and checks["transcript"]
    output = bool(identity and consistency and checks["transcript"] and checks["metrics"])
    return {"schema": "liptrace-task-parity-v1",
            "strict_fidelity": strict["pass"],
            "identity_agreement": bool(identity), "internal_consistency": bool(consistency),
            "numeric_agreement": bool(checks["tensor"] and checks["logits"]),
            "path_agreement": bool(checks["ctc_path"]),
            "transcript_agreement": bool(checks["transcript"]),
            "task_output_preserved": output,
            "path_changed_output_preserved": bool(output and not discrete),
            "first_divergence": strict["first_divergence"], "strict_report": strict,
            "scope": "recorded output fidelity; neither ground-truth correctness nor deployment reliability"}
