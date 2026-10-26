import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.e2e_asr_pavir import PAViRE2E
from pavir.artifacts import canonical_json_bytes, sha256_file


def test_legacy_loader_target_shape_is_normalized_without_value_drift():
    canonical = torch.tensor([[1, 2, -1], [3, -1, -1]])
    legacy = canonical.unsqueeze(1)
    observed = PAViRE2E._normalize_loader_targets(legacy)
    assert observed.shape == canonical.shape
    assert torch.equal(observed, canonical)
    assert torch.equal(PAViRE2E._normalize_loader_targets(canonical), canonical)
    with pytest.raises(ValueError, match="targets must be"):
        PAViRE2E._normalize_loader_targets(canonical.unsqueeze(-1))


def _lambda_zero_config(root: Path) -> Path:
    routes = {
        "schema_version": "pavir-routes-v1",
        "corpus": "vicocktail",
        "edges": [
            {
                "edge_id": "inactive-e",
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
    route_path = root / "routes.json"
    route_path.write_bytes(canonical_json_bytes(routes))
    occurrence_path = root / "occurrences.jsonl"
    occurrence_path.write_text(
        json.dumps(
            {
                "schema_version": "pavir-occurrences-v1",
                "corpus": "vicocktail",
                "inputs": {"routes_sha256": sha256_file(route_path)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prototype_path = root / "prototypes.npz"
    np.savez(
        prototype_path,
        keys=np.array(["inactive-e:target", "inactive-e:confusion"]),
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
    )
    canonical_path = root / "canonical.jsonl"
    canonical_path.write_text(
        json.dumps({"schema_version": "canonical-id-map-v1"}) + "\n"
        + json.dumps(
            {
                "split": "train",
                "lookup_key": "unused",
                "sample_id": "vicocktail:train:unused",
                "source_id": "unused",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        name: {"path": path.name, "sha256": sha256_file(path)}
        for name, path in (
            ("routes", route_path),
            ("occurrences", occurrence_path),
            ("prototypes", prototype_path),
            ("canonical_ids", canonical_path),
        )
    }
    config = {
        "schema_version": "pavir-vsr-config-v1",
        "corpus": "vicocktail",
        "seed": 42,
        "block_index": 6,
        "visual_margin": 0.15,
        "context_margin": 0.10,
        "lambda_visual": 0.0,
        "lambda_context": 0.0,
        "ramp_updates": 10,
        "context_batch_fraction": 0.5,
        "artifacts": artifacts,
    }
    path = root / "config.json"
    path.write_bytes(canonical_json_bytes(config))
    return path


@pytest.mark.integration
def test_checkpoint_state_and_lambda_zero_native_equivalence(tmp_path):
    checkpoint_path = os.environ.get("PAVIR_W0")
    if not checkpoint_path:
        pytest.skip("set PAVIR_W0 to run the full checkpoint/model contract")
    if not torch.cuda.is_available():
        pytest.skip("full model contract requires CUDA")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    odim = int(state["ctc.ctc_lo.weight"].size(0))
    config = _lambda_zero_config(tmp_path)
    native = E2E(odim, "video", ctc_weight=0.1)
    pavir = PAViRE2E(
        odim,
        "video",
        ctc_weight=0.1,
        pavir_config=str(config),
    )
    native.load_state_dict(state, strict=True)
    pavir.load_state_dict(state, strict=True)
    assert tuple(native.state_dict()) == tuple(pavir.state_dict())
    assert sum(p.numel() for p in native.parameters()) == sum(
        p.numel() for p in pavir.parameters()
    )

    device = torch.device("cuda")
    native.to(device).train()
    pavir.to(device).train()
    video = torch.randn(1, 12, 1, 96, 96, device=device)
    lengths = torch.tensor([12])
    target = torch.tensor([[1, 2, 3]], device=device)

    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
    native_output = native(video, lengths, target)
    native_cpu_rng = torch.random.get_rng_state().clone()
    native_cuda_rng = torch.cuda.get_rng_state().clone()

    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
    pavir_output = pavir(
        video,
        lengths,
        target,
        sample_ids=["vicocktail:train:unused"],
        update=0,
        apply_pavir=True,
    )
    assert torch.equal(native_output[0], pavir_output["loss"])
    assert torch.equal(native_output[1], pavir_output["loss_ctc"])
    assert torch.equal(native_output[2], pavir_output["loss_att"])
    assert native_output[3] == pavir_output["decoder_acc"]
    assert torch.equal(native_cpu_rng, torch.random.get_rng_state())
    assert torch.equal(native_cuda_rng, torch.cuda.get_rng_state())

    # A PAViR training state remains a native inference state.
    reloaded_native = E2E(odim, "video", ctc_weight=0.1)
    reloaded_native.load_state_dict(pavir.cpu().state_dict(), strict=True)
