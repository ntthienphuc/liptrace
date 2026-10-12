"""Compare a trusted original implementation with the packaged trace."""
import numpy as np
from .replay import read_trace, decode_path, trace_consistency


def compare_original(original, packaged, profile, atol=1e-5, rtol=1e-4):
    document, arrays = read_trace(packaged)
    checks = [{"stage": "profile", "pass": document["profile"] == profile},
              {"stage": "packaged_trace_consistency",
               "pass": trace_consistency(document, arrays)["pass"]}]
    for key in ("tensor", "logits"):
        value = original.get(key)
        expected = arrays[key]
        valid = isinstance(value, np.ndarray) and value.shape == expected.shape and value.dtype == expected.dtype
        finite = bool(valid and np.isfinite(value).all())
        passed = bool(finite and np.allclose(value, expected, atol=atol, rtol=rtol))
        checks.append({"stage": key, "pass": passed, "shape_dtype_valid": valid,
                       "max_abs_error": float(np.max(np.abs(value.astype(np.float64) - expected)))
                       if finite else None})
    valid_logits = checks[3]["shape_dtype_valid"] and np.isfinite(original["logits"]).all()
    path = original["logits"].argmax(-1)[0] if valid_logits else None
    transcript = decode_path(path, profile["charset"], profile["blank_index"]) if valid_logits else None
    checks.append({"stage": "ctc_path",
                   "pass": bool(valid_logits and np.array_equal(path, arrays["ctc_path"]))})
    consistent = isinstance(original.get("transcript"), str) and original["transcript"] == transcript
    checks.append({"stage": "original_ctc_consistency", "pass": bool(consistent)})
    checks.append({"stage": "transcript",
                   "pass": bool(consistent and original["transcript"] == document["transcript"])})
    failed = next((row["stage"] for row in checks if not row["pass"]), None)
    lookup = {row["stage"]: row["pass"] for row in checks}
    task = all(lookup[key] for key in ("profile", "packaged_trace_consistency",
                                      "original_ctc_consistency", "transcript"))
    return {"schema": "liptrace-original-parity-v1", "pass": failed is None,
            "task_output_preserved": bool(task),
            "numeric_agreement": bool(lookup["tensor"] and lookup["logits"]),
            "first_divergence": failed, "checks": checks,
            "scope": "original tensor/logits/greedy output; original decoded frames and sampling indices "
                     "are not independently captured; trusted reference source pinned in case receipt"}
