from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pavir.artifacts import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "freeze_pavir_config",
    ROOT / "tools" / "freeze_pavir_config.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _inputs(root: Path) -> tuple[dict[str, Path], Path]:
    routes = root / "routes.json"
    routes.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "pavir-routes-v1",
                "corpus": "vicocktail",
                "edges": [
                    {
                        "edge_id": "e1",
                        "target_text": "a",
                        "confusion_text": "b",
                        "target_token_ids": [1],
                        "confusion_token_ids": [2],
                        "route": "E",
                        "tone_only": False,
                        "support": 20,
                        "source_count": 10,
                    }
                ],
            }
        )
    )
    occurrences = root / "occurrences.jsonl"
    occurrences.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "pavir-occurrences-v1",
                "corpus": "vicocktail",
                "inputs": {"routes_sha256": sha256_file(routes)},
            }
        )
        + canonical_json_bytes(
            {
                "sample_id": "vicocktail:train:clip",
                "events": [
                    {"edge_id": "e1", "target_start": 0, "target_end": 1}
                ],
            }
        )
    )
    prototypes = root / "prototypes.npz"
    np.savez(
        prototypes,
        keys=np.asarray(["e1:target", "e1:confusion"]),
        vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
    )
    canonical = root / "canonical.jsonl"
    canonical.write_bytes(
        canonical_json_bytes({"schema_version": "canonical-id-map-v1"})
        + canonical_json_bytes(
            {
                "split": "train",
                "lookup_key": "clip",
                "sample_id": "vicocktail:train:clip",
                "source_id": "source",
            }
        )
    )
    forbidden = root / "forbidden.json"
    forbidden.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "pavir-forbidden-ids-v1",
                "sample_ids": ["vicocktail:test:held"],
            }
        )
    )
    sources = {
        "routes": routes,
        "occurrences": occurrences,
        "prototypes": prototypes,
        "canonical_ids": canonical,
        "forbidden_ids": forbidden,
    }
    calibration = root / "calibration.json"
    calibration.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "pavir-gradient-calibration-v1",
                "optimizer_steps": 0,
                "batches": 64,
                "seed": 42,
                "max_frames": 6000,
                "minimum_active_batches": 8,
                "targets": {
                    "visual_native_median": 0.05,
                    "context_native_median": 0.025,
                },
                "observed_unscaled": {
                    "visual_native_ratio_median": 0.5,
                    "context_native_ratio_median": 0.25,
                },
                "locked_lambdas": {
                    "lambda_visual": 0.1,
                    "lambda_context": 0.1,
                    "achieved_visual_median_ratio": 0.05,
                    "achieved_context_median_ratio": 0.025,
                    "achieved_total_median_budget": 0.075,
                },
                "config_contract": {
                    "corpus": "vicocktail",
                    "seed": 42,
                    "block_index": 6,
                    "visual_margin": 0.15,
                    "context_margin": 0.10,
                    "context_batch_fraction": 0.50,
                    "artifacts": {
                        name: sha256_file(path)
                        for name, path in sources.items()
                    },
                },
            }
        )
    )
    return sources, calibration


def _argv(
    sources: dict[str, Path],
    calibration: Path,
    output: Path,
) -> list[str]:
    return [
        "freeze_pavir_config.py",
        "--corpus",
        "vicocktail",
        "--routes",
        str(sources["routes"]),
        "--occurrences",
        str(sources["occurrences"]),
        "--prototypes",
        str(sources["prototypes"]),
        "--canonical-ids",
        str(sources["canonical_ids"]),
        "--forbidden-ids",
        str(sources["forbidden_ids"]),
        "--calibration",
        str(calibration),
        "--total-updates",
        "20000",
        "--output-dir",
        str(output),
    ]


def test_freeze_binds_calibration_to_exact_artifacts(tmp_path, monkeypatch):
    sources, calibration = _inputs(tmp_path)
    output = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", _argv(sources, calibration, output))
    MODULE.main()
    manifest = json.loads(
        (output / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["ramp_updates"] == 2000
    assert manifest["calibration"]["sha256"] == sha256_file(calibration)

    payload = json.loads(calibration.read_text(encoding="utf-8"))
    payload["config_contract"]["artifacts"]["routes"] = "0" * 64
    calibration.write_bytes(canonical_json_bytes(payload))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(sources, calibration, tmp_path / "rejected"),
    )
    with pytest.raises(ValueError, match="different routes"):
        MODULE.main()
