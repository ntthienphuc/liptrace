from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pavir.artifacts import canonical_json_bytes, sha256_file
from pavir.campaign import (
    ARM_NAMES,
    authorize_gate,
    build_campaign_bundle,
    build_route_shuffled_payload,
    count_padded_batches,
    resolve_evaluation,
    route_counts,
    validate_gate_marker,
    validate_registry,
)


def _routes():
    return {
        "schema_version": "pavir-routes-v1",
        "corpus": "vicocktail",
        "split": "vc-calib",
        "route_counts": {"E": 2, "D": 2, "A": 1},
        "edges": [
            {
                "edge_id": f"edge-{index}",
                "target_text": f"t{index}",
                "confusion_text": f"c{index}",
                "target_token_ids": [index + 1],
                "confusion_token_ids": [index + 11],
                "route": route,
                "tone_only": route == "D",
                "support": support,
                "source_count": 10,
            }
            for index, (route, support) in enumerate(
                (("E", 20), ("D", 21), ("E", 80), ("D", 33), ("A", 90))
            )
        ],
    }


def test_route_shuffle_is_seeded_nonidentity_and_preserves_precommit_contract():
    routes = _routes()
    keys = {
        f"edge-{index}:{endpoint}"
        for index in range(4)
        for endpoint in ("target", "confusion")
    }
    first = build_route_shuffled_payload(routes, prototype_keys=keys, seed=42)
    second = build_route_shuffled_payload(routes, prototype_keys=keys, seed=42)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert route_counts(first) == route_counts(routes)
    before = {row["edge_id"]: row["route"] for row in routes["edges"]}
    after = {row["edge_id"]: row["route"] for row in first["edges"]}
    assert before != after
    assert after["edge-4"] == "A"
    for original, shuffled in zip(routes["edges"], first["edges"]):
        assert {
            key: value for key, value in original.items() if key != "route"
        } == {key: value for key, value in shuffled.items() if key != "route"}
    audit = first["route_shuffle_control"]
    assert audit["existing_prototypes_only"] is True
    assert audit["support_totals_are_diagnostic_not_a_selection_constraint"] is True


def test_route_shuffle_refuses_posthoc_prototype_extraction():
    with pytest.raises(ValueError, match="post-hoc prototype"):
        build_route_shuffled_payload(_routes(), prototype_keys=(), seed=42)


def test_padded_batch_counter_matches_locked_workload_examples():
    assert count_padded_batches([6000] * 4000) == 4000
    assert count_padded_batches([100, 100, 100], max_frames=200, num_buckets=2) == 2


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_bytes(canonical_json_bytes(payload))
    return path


def _campaign_fixture(root: Path) -> Path:
    frozen = root / "frozen"
    frozen.mkdir()
    routes = _routes()
    routes["edges"] = routes["edges"][:2]
    routes["route_counts"] = {"E": 1, "D": 1, "A": 0}
    routes_path = _write(frozen / "routes.json", routes)
    occurrences = frozen / "occurrences.jsonl"
    occurrences.write_text(
        json.dumps(
            {
                "schema_version": "pavir-occurrences-v1",
                "corpus": "vicocktail",
                "inputs": {"routes_sha256": sha256_file(routes_path)},
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "vicocktail:train:clip0000",
                "events": [{"edge_id": "edge-0", "target_start": 0, "target_end": 1}],
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "vicocktail:train:clip0001",
                "events": [{"edge_id": "edge-1", "target_start": 0, "target_end": 1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prototypes = frozen / "prototypes.npz"
    np.savez(
        prototypes,
        keys=np.asarray(
            [
                "edge-0:target",
                "edge-0:confusion",
                "edge-1:target",
                "edge-1:confusion",
            ]
        ),
        vectors=np.asarray(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float16
        ),
    )
    canonical = frozen / "canonical_ids.jsonl"
    with canonical.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version": "canonical-id-map-v1"}) + "\n")
        for index in range(4000):
            stream.write(
                json.dumps(
                    {
                        "split": "train",
                        "lookup_key": f"clip{index:04d}",
                        "sample_id": f"vicocktail:train:clip{index:04d}",
                        "source_id": f"source-{index}",
                    }
                )
                + "\n"
            )
    forbidden = _write(
        frozen / "forbidden.json",
        {"sample_ids": ["vicocktail:val:promote"]},
    )
    artifacts = {
        name: {"path": path.name, "sha256": sha256_file(path)}
        for name, path in (
            ("routes", routes_path),
            ("occurrences", occurrences),
            ("prototypes", prototypes),
            ("canonical_ids", canonical),
            ("forbidden_ids", forbidden),
        )
    }
    config = _write(
        frozen / "locked.json",
        {
            "schema_version": "pavir-vsr-config-v1",
            "corpus": "vicocktail",
            "seed": 42,
            "block_index": 6,
            "visual_margin": 0.15,
            "context_margin": 0.10,
            "lambda_visual": 0.2,
            "lambda_context": 0.1,
            "ramp_updates": 2000,
            "context_batch_fraction": 0.5,
            "artifacts": artifacts,
        },
    )
    _write(
        frozen / "BUNDLE_MANIFEST.json",
        {
            "schema_version": "pavir-frozen-bundle-v1",
            "config": {"path": config.name, "sha256": sha256_file(config)},
            "calibration_sha256": "a" * 64,
            "total_updates": 20000,
            "ramp_updates": 2000,
            "edge_count": 2,
            "occurrence_sample_count": 2,
            "prototype_count": 4,
        },
    )
    data_root = root / "data"
    data_root.mkdir()
    train = root / "train.csv"
    train.write_text(
        "".join(
            f"vicocktail,clip{index:04d},6000,1,/cache/{index}.npy\n"
            for index in range(4000)
        ),
        encoding="utf-8",
    )
    calib_label = _write(root / "calib.csv", b"vicocktail,c,1,1,/c.npy\n")
    promote_label = _write(root / "promote.csv", b"vicocktail,p,1,1,/p.npy\n")
    calib_manifest = _write(root / "calib.jsonl", b'{"sample_id":"c"}\n')
    promote_manifest = _write(root / "promote.jsonl", b'{"sample_id":"p"}\n')
    checkpoint = _write(root / "w0.ckpt", b"fixture")
    tokenizer = _write(root / "tokenizer.model", b"model")
    units = _write(root / "units.txt", b"<unk> 1\n")
    precommit = _write(
        root / "PRECOMMIT.json",
        {
            "schema_version": "pavir-campaign-precommit-v1",
            "training": {
                "seed": 42,
                "optimizer": "AdamW",
                "learning_rate": 1e-5,
                "weight_decay": 0.03,
                "precision": "bf16-mixed",
                "max_padded_frames": 6000,
                "epochs": 5,
                "warmup_epochs": 1,
                "early_stopping": False,
                "checkpoint_selection": "fixed final",
                "initialization": "weights-only corpus-specific W0",
                "optimizer_scheduler_reset": True,
            },
            "short_screen": {
                "updates": 2000,
                "restart_each_arm_from_w0": True,
                "arms": list(ARM_NAMES),
                "route_shuffle": {
                    "population": "evidence-sufficient frozen E/D edges only",
                    "preserve_exact_E_and_D_edge_counts": True,
                    "preserve_prototypes_and_gradient_budget": True,
                    "seed": 42,
                    "reject_identity_permutation": True,
                },
            },
        },
    )
    data_registry = _write(
        root / "DATA_REGISTRY.json",
        {
            "schema_version": "pavir-data-registry-v1",
            "corpora": {
                "vicocktail": {
                    "w0_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
                    "training": {
                        "auto_avsr_labels_path": str(train),
                        "auto_avsr_labels_sha256": sha256_file(train),
                    },
                    "firewall": {
                        "vc_calib": {
                            "manifest_path": str(calib_manifest),
                            "manifest_sha256": sha256_file(calib_manifest),
                            "auto_avsr_labels_sha256": sha256_file(calib_label),
                        },
                        "vc_promote": {
                            "manifest_path": str(promote_manifest),
                            "manifest_sha256": sha256_file(promote_manifest),
                            "auto_avsr_labels_sha256": sha256_file(promote_label),
                        },
                    },
                    "canonical_ids": {"path": str(canonical), "sha256": sha256_file(canonical)},
                    "forbidden_promote_test_ids": {
                        "path": str(forbidden),
                        "sha256": sha256_file(forbidden),
                    },
                }
            },
            "tokenizer": {
                "sentencepiece_model": {"sha256": sha256_file(tokenizer)},
                "auto_avsr_units": {"sha256": sha256_file(units)},
            },
        },
    )
    implementation = _write(
        root / "IMPLEMENTATION_LOCK.json",
        {
            "schema_version": "pavir-implementation-lock-v1",
            "status": "LOCKED_BEFORE_W0_CONFUSION_GRAPH_RESULTS",
        },
    )
    source = _write(root / "source.py", {"fixture": True})
    relative = source.relative_to(root).as_posix()
    aggregate = hashlib.sha256(
        (
            relative
            + "\0"
            + sha256_file(source)
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    code_snapshot = _write(
        root / "CODE_SNAPSHOT.json",
        {
            "schema_version": "pavir-code-snapshot-v1",
            "project_root": str(root),
            "implementation_lock": {
                "sha256": sha256_file(implementation),
            },
            "aggregate_sha256": aggregate,
            "file_count": 1,
            "files": [{"path": relative, "sha256": sha256_file(source)}],
        },
    )
    output = root / "arms"
    build_campaign_bundle(
        source_bundle_dir=frozen,
        output_dir=output,
        precommit=precommit,
        data_registry=data_registry,
        implementation_lock=implementation,
        code_snapshot=code_snapshot,
        w0_checkpoint=checkpoint,
        data_root=data_root,
        train_label=train,
        vc_calib_label=calib_label,
        vc_calib_manifest=calib_manifest,
        vc_promote_label=promote_label,
        vc_promote_manifest=promote_manifest,
        tokenizer_model=tokenizer,
        tokenizer_units=units,
    )
    return output / "ARM_REGISTRY.json"


def test_campaign_bundle_and_gate_marker_are_hash_bound(tmp_path):
    registry_path = _campaign_fixture(tmp_path)
    registry = validate_registry(registry_path)
    assert registry["arm_order"] == list(ARM_NAMES)
    assert set(registry["arms"]) == set(ARM_NAMES)
    assert registry["scientific_lock"]["screen_updates"] == 2000
    assert registry["scientific_lock"]["full_updates"] == 20000
    assert registry["evaluation_firewall"]["clean_test_path_recorded"] is False
    assert len(
        (registry_path.parent / "preflight/overfit32.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 32

    with pytest.raises(PermissionError, match="promotion"):
        resolve_evaluation(
            registry_path,
            arm="full",
            split="clean-test",
            gate_marker=None,
            clean_label=tmp_path / "does-not-exist.csv",
            clean_manifest=tmp_path / "does-not-exist.jsonl",
        )

    summary = _write(
        tmp_path / "screen_summary.json",
        {
            "schema_version": "pavir-analysis-v1",
            "split": "vc-calib",
            "evaluation_status": "complete",
            "campaign": {
                "campaign_identifier": sha256_file(registry_path),
                "campaign_registry": {
                    "path": str(registry_path.resolve()),
                    "sha256": sha256_file(registry_path),
                },
                "campaign_content_id": registry["campaign_content_id"],
                "protocol_family_id": registry["protocol_family_id"],
                "corpus": "vicocktail",
            },
            "gates": {"short_screen": {"status": "pass", "checks": {}}},
        },
    )
    marker_path = tmp_path / "SCREEN_PASS.json"
    authorize_gate(
        registry_path=registry_path,
        summary_path=summary,
        gate="short_screen",
        output_path=marker_path,
    )
    validate_gate_marker(
        registry_path=registry_path,
        marker_path=marker_path,
        required_gate="short_screen",
    )
