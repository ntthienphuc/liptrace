from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "verify_pavir_spatial_stability",
    ROOT / "tools" / "verify_pavir_spatial_stability.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_substitution_stability_key_includes_reference_position():
    first = MODULE.substitution_keys("tôi đi học đi", "tôi đã học đi")
    second = MODULE.substitution_keys("tôi đi học đi", "tôi đi học đã")
    assert first == {(1, "đi", "đã")}
    assert second == {(3, "đi", "đã")}
    assert first.isdisjoint(second)
