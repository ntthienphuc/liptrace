from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from pavir.route_confirm_v3 import (
    EDGE_SEED_DOMAIN,
    PRODUCTION_BOOTSTRAP_ITERATIONS,
    PRODUCTION_PERMUTATIONS,
    PROTOCOL_PRECOMMIT_SHA256,
    BootstrapResult,
    EdgeEvidence,
    EdgeSupport,
    PartitionSupport,
    RouteDecision,
    SourceSplit,
    SupportThresholds,
    audit_edge_support,
    audit_route_mass,
    benjamini_hochberg,
    benjamini_yekutieli,
    complete_non_tone_bh,
    decide_route,
    deterministic_source_partition,
    deterministic_source_split,
    edge_seed,
    evaluate_edge,
    finalize_route_family,
    fit_weighted_probe,
    mann_whitney_auc,
    one_sample_auc_tost,
    source_bootstrap_ci,
    source_endpoint_balanced_weights,
    source_macro_auc,
    within_source_permutation_test,
)


def test_production_monte_carlo_defaults_are_locked() -> None:
    assert PRODUCTION_PERMUTATIONS == 199_999
    assert PRODUCTION_BOOTSTRAP_ITERATIONS == 10_000
    assert (
        PROTOCOL_PRECOMMIT_SHA256
        == "63d9a2513bd66f35833db92329a98c5672620a56fd0087c48ae322fc5b85359d"
    )
    precommit = (
        Path(__file__).resolve().parents[4]
        / "pavir_campaign"
        / "route_confirm_v3"
        / "PROTOCOL_PRECOMMIT.json"
    )
    assert hashlib.sha256(precommit.read_bytes()).hexdigest() == (
        PROTOCOL_PRECOMMIT_SHA256
    )


def test_global_source_split_is_exact_endpoint_blind_and_order_invariant() -> None:
    sources = [f"source-{index:04d}" for index in range(2_714)]
    first = deterministic_source_split(sources)
    second = deterministic_source_split(list(reversed(sources)) + sources[:17])

    assert first == second
    assert len(first.fit_sources) == 1_899
    assert len(first.confirm_sources) == 815
    assert set(first.fit_sources).isdisjoint(first.confirm_sources)
    assert set(first.fit_sources) | set(first.confirm_sources) == set(sources)
    record = first.to_record()
    assert record["protocol_precommit_sha256"] == PROTOCOL_PRECOMMIT_SHA256
    assert record["fit_count"] == 1_899


def test_source_partition_rows_bind_digest_rank_and_fail_closed_inventory() -> None:
    with pytest.raises(ValueError, match="inventory mismatch"):
        deterministic_source_partition(["a", "b"])
    rows = deterministic_source_partition(
        ["b", "a", "c"],
        expected_source_count=None,
    )
    assert len(rows) == 3
    assert [row.rank for row in rows] == [0, 1, 2]
    assert [row.partition for row in rows].count("route_fit") == 2
    for row in rows:
        assert row.source_digest_sha256 == hashlib.sha256(
            b"pavir-route-confirm-v3\0" + row.source_id.encode("utf-8")
        ).hexdigest()


def test_edge_seed_uses_documented_unsigned_big_endian_conversion() -> None:
    edge_id = "xin→xinh"
    expected = int.from_bytes(
        hashlib.sha256(EDGE_SEED_DOMAIN + edge_id.encode("utf-8")).digest()[:4],
        byteorder="big",
        signed=False,
    )
    assert edge_seed(edge_id) == expected
    assert 0 <= expected < 2**32


def test_source_endpoint_weights_equalize_every_cell_and_have_mean_one() -> None:
    endpoints = np.asarray(
        ["target", "confusion", "confusion", "confusion", "target", "target", "confusion"]
    )
    sources = np.asarray(["s1", "s1", "s1", "s1", "s2", "s2", "s2"])
    weights = source_endpoint_balanced_weights(endpoints, sources)

    assert weights.dtype == np.float32
    assert float(np.mean(weights)) == pytest.approx(1.0)
    cell_sums = [
        float(np.sum(weights[(sources == source) & (endpoints == endpoint)]))
        for source in ("s1", "s2")
        for endpoint in ("target", "confusion")
    ]
    assert cell_sums == pytest.approx([cell_sums[0]] * 4)


def test_support_excludes_and_reports_one_class_sources_before_floors() -> None:
    split = SourceSplit(
        fit_sources=("f1", "f2", "f_one"),
        confirm_sources=("c1", "c2", "c_one"),
    )
    endpoints, sources = [], []
    for source in ("f1", "f2", "c1", "c2"):
        endpoints.extend(("target", "confusion"))
        sources.extend((source, source))
    endpoints.extend(("target", "confusion"))
    sources.extend(("f_one", "c_one"))
    vectors = np.arange(len(sources) * 2, dtype=np.float32).reshape(-1, 2)

    audit = audit_edge_support(
        vectors,
        endpoints,
        sources,
        split,
        thresholds=SupportThresholds(
            fit_occurrences_per_endpoint=2,
            confirm_occurrences_per_endpoint=2,
            fit_paired_sources=2,
            confirm_paired_sources=2,
            maximum_source_endpoint_fraction=0.5,
        ),
    )

    assert audit.passed
    assert audit.fit.one_class_sources == ("f_one",)
    assert audit.confirm.one_class_sources == ("c_one",)
    assert audit.fit.included_endpoint_occurrences == (2, 2)
    assert audit.confirm.included_endpoint_occurrences == (2, 2)
    assert audit.fit.maximum_source_endpoint_fractions == pytest.approx((0.5, 0.5))
    assert len(audit.fit.included_indices) == 4
    assert len(audit.confirm.included_indices) == 4


def test_support_rejects_endpoint_concentration_over_twenty_percent() -> None:
    split = SourceSplit(
        fit_sources=("f1", "f2"),
        confirm_sources=("c1", "c2"),
    )
    endpoints = []
    sources = []
    for source, repeats in (("f1", 4), ("f2", 1), ("c1", 1), ("c2", 1)):
        for endpoint in ("target", "confusion"):
            endpoints.extend([endpoint] * repeats)
            sources.extend([source] * repeats)
    vectors = np.ones((len(sources), 2), dtype=np.float32)
    audit = audit_edge_support(
        vectors,
        endpoints,
        sources,
        split,
        thresholds=SupportThresholds(
            fit_occurrences_per_endpoint=2,
            confirm_occurrences_per_endpoint=2,
            fit_paired_sources=2,
            confirm_paired_sources=2,
            maximum_source_endpoint_fraction=0.7,
        ),
    )
    assert not audit.passed
    assert "fit_target_source_concentration" in audit.failures
    assert "fit_confusion_source_concentration" in audit.failures


def test_weighted_probe_uses_weighted_scaler_and_is_deterministic() -> None:
    vectors = np.asarray(
        [[-2.0, 0], [2.0, 0], [-1.5, 1], [1.5, 1], [-1.0, 2], [1.0, 2]],
        dtype=np.float32,
    )
    endpoints = np.asarray(
        ["target", "confusion", "target", "confusion", "target", "confusion"]
    )
    sources = np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"])
    weights = source_endpoint_balanced_weights(endpoints, sources)
    expected_mean = np.average(vectors, axis=0, weights=weights)

    first = fit_weighted_probe(vectors, endpoints, sources, edge_id="edge-a")
    second = fit_weighted_probe(vectors, endpoints, sources, edge_id="edge-a")

    assert first.converged and second.converged
    assert first.scaler.mean_.dtype == np.float32
    assert first.scaler.var_.dtype == np.float32
    assert first.scaler.scale_.dtype == np.float32
    np.testing.assert_allclose(first.scaler.mean_, expected_mean)
    np.testing.assert_array_equal(
        first.confusion_scores(vectors), second.confusion_scores(vectors)
    )


def test_exact_mann_whitney_ties_and_source_macro_exclusion() -> None:
    assert mann_whitney_auc(
        ["target", "target", "confusion", "confusion"],
        [0.0, 1.0, 1.0, 2.0],
    ) == pytest.approx(0.875)

    result = source_macro_auc(
        [
            "target",
            "target",
            "confusion",
            "confusion",
            "target",
            "confusion",
            "target",
        ],
        [0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 9.0],
        ["a", "a", "a", "a", "b", "b", "one_class"],
    )
    assert result.source_ids == ("a", "b")
    assert result.source_aucs == pytest.approx((0.875, 0.0))
    assert result.theta == pytest.approx(0.4375)
    assert result.excluded_one_class_sources == ("one_class",)


def test_within_source_permutation_is_add_one_reproducible_and_batch_invariant() -> None:
    endpoints = ["target", "target", "confusion", "confusion"] * 3
    sources = ["s1"] * 4 + ["s2"] * 4 + ["s3"] * 4
    scores = [0.0, 0.2, 0.8, 1.0, 0.1, 0.3, 0.7, 0.9, 0.0, 0.4, 0.6, 1.0]

    first = within_source_permutation_test(
        endpoints,
        scores,
        sources,
        edge_id="edge-permutation",
        iterations=127,
        batch_size=7,
        return_distribution=True,
    )
    second = within_source_permutation_test(
        endpoints,
        scores,
        sources,
        edge_id="edge-permutation",
        iterations=127,
        batch_size=64,
        return_distribution=True,
    )

    assert first.p_value == (1 + first.exceedances) / 128
    assert 1 / 128 <= first.p_value <= 1
    np.testing.assert_array_equal(first.null_thetas, second.null_thetas)
    assert np.all((first.null_thetas >= 0) & (first.null_thetas <= 1))


def test_source_bootstrap_is_reproducible_and_batch_invariant() -> None:
    first = source_bootstrap_ci(
        [0.1, 0.4, 0.7, 1.0],
        edge_id="edge-bootstrap",
        iterations=257,
        batch_size=9,
    )
    second = source_bootstrap_ci(
        [0.1, 0.4, 0.7, 1.0],
        edge_id="edge-bootstrap",
        iterations=257,
        batch_size=128,
    )
    assert isinstance(first, BootstrapResult)
    assert first == second
    assert 0 <= first.lower <= first.upper <= 1


def test_tost_requires_both_one_sided_tests() -> None:
    equivalent = one_sample_auc_tost([0.5] * 20)
    separated = one_sample_auc_tost([0.75, 0.8, 0.85, 0.9] * 5)

    assert equivalent.p_equivalence == 0
    assert separated.p_equivalence > 0.99
    assert separated.p_equivalence == max(separated.p_lower, separated.p_upper)


def test_bh_is_monotone_and_skipped_edges_are_part_of_family() -> None:
    q_values = complete_non_tone_bh(
        {"edge-a": 0.001, "edge-c": 0.02},
        ["edge-a", "edge-b", "edge-c"],
    )
    assert q_values == pytest.approx(
        {"edge-a": 0.003, "edge-b": 1.0, "edge-c": 0.03}
    )
    tied = benjamini_hochberg({"b": 0.02, "a": 0.02, "c": 0.03})
    assert tied["a"] <= tied["c"]
    assert tied["b"] <= tied["c"]
    by = benjamini_yekutieli(
        {"edge-a": 0.001, "edge-b": 1.0, "edge-c": 0.02}
    )
    assert by["edge-a"] == pytest.approx(q_values["edge-a"] * (1 + 1 / 2 + 1 / 3))
    assert by["edge-c"] == pytest.approx(q_values["edge-c"] * (1 + 1 / 2 + 1 / 3))


def _partition(passed: bool = True) -> PartitionSupport:
    return PartitionSupport(
        partition="fixture",
        observed_source_count=20,
        paired_sources=tuple(f"s{index}" for index in range(20)),
        one_class_sources=(),
        raw_endpoint_occurrences=(100, 100),
        included_endpoint_occurrences=(100, 100),
        maximum_source_endpoint_fractions=(0.05, 0.05),
        included_indices=tuple(range(200)),
        passed=passed,
        failures=() if passed else ("fixture_support",),
    )


def _evidence(
    edge_id: str,
    *,
    tone: bool = False,
    support: bool = True,
    converged: bool = True,
    theta: float | None = 0.7,
    ci: tuple[float, float] | None = (0.62, 0.78),
    p_e: float = 0.001,
    p_d: float = 1.0,
) -> EdgeEvidence:
    partition = _partition(support)
    return EdgeEvidence(
        edge_id=edge_id,
        tone_only=tone,
        support=EdgeSupport(
            fit=partition,
            confirm=partition,
            passed=support,
            failures=() if support else ("fixture_support",),
        ),
        converged=converged,
        seed=edge_seed(edge_id),
        theta=theta,
        source_aucs=() if theta is None else (theta,) * 20,
        source_auc_ids=tuple(f"s{index}" for index in range(20)),
        ci95=ci,
        p_e=p_e,
        p_d=p_d,
        permutation_exceedances=0,
        warning_messages=(),
    )


def test_route_semantics_keep_tone_equivalence_and_reversal_separate() -> None:
    support_failure = decide_route(
        _evidence("support", tone=True, support=False, theta=None, ci=None),
        q_e=None,
        q_d=None,
    )
    tone = decide_route(
        _evidence("tone", tone=True, theta=None, ci=None),
        q_e=None,
        q_d=None,
    )
    encoder = decide_route(_evidence("encoder"), q_e=0.01, q_d=1.0)
    equivalent = decide_route(
        _evidence(
            "equiv",
            theta=0.5,
            ci=(0.46, 0.54),
            p_e=1.0,
            p_d=0.001,
        ),
        q_e=1.0,
        q_d=0.01,
    )
    reversed_edge = decide_route(
        _evidence(
            "reversed",
            theta=0.4,
            ci=(0.35, 0.44),
            p_e=1.0,
            p_d=0.001,
        ),
        q_e=1.0,
        q_d=0.001,
    )
    uncertain = decide_route(
        _evidence(
            "uncertain",
            theta=0.48,
            ci=(0.41, 0.49),
            p_e=1.0,
            p_d=0.001,
        ),
        q_e=1.0,
        q_d=0.001,
    )

    assert (support_failure.route, support_failure.reason) == ("A", "A_support")
    assert (tone.route, tone.reason) == ("D", "D_tone")
    assert (encoder.route, encoder.reason) == ("E", "E_confirmed")
    assert (equivalent.route, equivalent.reason) == ("D", "D_equiv")
    assert (reversed_edge.route, reversed_edge.reason) == ("A", "A_reversed")
    # CI entirely below chance bars D, but theta >= .45 keeps the conservative
    # label A_uncertain rather than overstating reversal.
    assert (uncertain.route, uncertain.reason) == ("A", "A_uncertain")


def test_family_correction_contains_support_skips_at_p_one() -> None:
    evidence = {
        "edge-e": _evidence("edge-e", p_e=0.001),
        "edge-d": _evidence(
            "edge-d",
            theta=0.5,
            ci=(0.46, 0.54),
            p_e=1.0,
            p_d=0.001,
        ),
        "edge-skip": _evidence(
            "edge-skip", support=False, theta=None, ci=None, p_e=1.0, p_d=1.0
        ),
        "edge-tone": _evidence("edge-tone", tone=True, theta=None, ci=None),
    }
    family = finalize_route_family(
        evidence,
        {
            "edge-d": False,
            "edge-e": False,
            "edge-skip": False,
            "edge-tone": True,
        },
    )

    assert set(family.q_e) == {"edge-d", "edge-e", "edge-skip"}
    assert set(family.q_d) == {"edge-d", "edge-e", "edge-skip"}
    assert family.q_e["edge-e"] == pytest.approx(0.003)
    assert family.q_e["edge-skip"] == 1
    assert family.q_d["edge-d"] == pytest.approx(0.003)
    assert family.q_d["edge-skip"] == 1
    assert set(family.sensitivity_q_e_by) == {"edge-d", "edge-e", "edge-skip"}
    assert set(family.sensitivity_q_d_by) == {"edge-d", "edge-e", "edge-skip"}
    assert family.decisions["edge-d"].reason == "D_equiv"
    assert family.decisions["edge-tone"].reason == "D_tone"
    assert family.decisions["edge-skip"].reason == "A_support"


def test_route_mass_gate_passes_exact_contract_and_blocks_any_failed_check() -> None:
    decisions: dict[str, RouteDecision] = {}
    support: dict[str, float] = {}
    vc_mass: dict[str, float] = {}
    for route in ("E", "D"):
        for index in range(10):
            edge_id = f"{route}-{index}"
            decisions[edge_id] = RouteDecision(
                edge_id,
                route,
                "E_confirmed" if route == "E" else "D_equiv",
                0.01 if route == "E" else 1.0,
                0.01 if route == "D" else 1.0,
            )
            support[edge_id] = 1_500
            vc_mass[edge_id] = 25
    audit = audit_route_mass(
        decisions,
        support,
        vc_mass,
        vc_total_substitution_mass=10_000,
    )
    assert audit.status == "PASS"
    assert audit.eligible_for_artifacts_and_training
    assert audit.active_stable_support == 30_000
    assert audit.e_vc_mass_fraction == pytest.approx(0.025)
    assert audit.d_vc_mass_fraction == pytest.approx(0.025)
    assert audit.active_vc_mass_fraction == pytest.approx(0.05)

    vc_mass["D-0"] = 0
    blocked = audit_route_mass(
        decisions,
        support,
        vc_mass,
        vc_total_substitution_mass=10_000,
    )
    assert blocked.status == "BLOCKED"
    assert "minimum_active_vc_mass_fraction" in blocked.blockers
    assert not blocked.eligible_for_artifacts_and_training


def test_bounded_end_to_end_edge_evidence_is_deterministic() -> None:
    split = SourceSplit(
        fit_sources=tuple(f"fit-{index}" for index in range(4)),
        confirm_sources=tuple(f"confirm-{index}" for index in range(4)),
    )
    rng = np.random.default_rng(8)
    vectors, endpoints, sources = [], [], []
    for source in split.fit_sources + split.confirm_sources:
        for endpoint, center in (("target", -2.0), ("confusion", 2.0)):
            for _ in range(2):
                value = rng.normal(0, 0.05, size=4)
                value[0] += center
                vectors.append(value)
                endpoints.append(endpoint)
                sources.append(source)
    thresholds = SupportThresholds(
        fit_occurrences_per_endpoint=8,
        confirm_occurrences_per_endpoint=8,
        fit_paired_sources=4,
        confirm_paired_sources=4,
        maximum_source_endpoint_fraction=0.25,
    )
    kwargs = dict(
        edge_id="bounded-edge",
        tone_only=False,
        support_thresholds=thresholds,
        permutation_iterations=31,
        bootstrap_iterations=64,
    )
    first = evaluate_edge(
        np.asarray(vectors), endpoints, sources, split, **kwargs
    )
    second = evaluate_edge(
        np.asarray(vectors), endpoints, sources, split, **kwargs
    )

    assert first == second
    assert first.support.passed and first.converged
    assert first.theta == 1.0
    assert first.ci95 == (1.0, 1.0)
    assert first.permutation_exceedances is not None


def _heldout_fixture(*, within_source_signal: bool):
    split = SourceSplit(
        fit_sources=tuple(f"fit-{index}" for index in range(4)),
        confirm_sources=tuple(f"confirm-{index}" for index in range(4)),
    )
    vectors, endpoints, sources = [], [], []
    all_sources = split.fit_sources + split.confirm_sources
    for source_index, source in enumerate(all_sources):
        for endpoint, endpoint_value in (("target", -2.0), ("confusion", 2.0)):
            for repeat in range(2):
                value = np.zeros(len(all_sources) + 2, dtype=np.float32)
                # The first block is pure source identity and is constant
                # within a source. Held-out source columns are unseen at fit.
                value[source_index] = 5.0
                if within_source_signal:
                    value[-2] = endpoint_value
                value[-1] = repeat * 0.01
                vectors.append(value)
                endpoints.append(endpoint)
                sources.append(source)
    thresholds = SupportThresholds(
        fit_occurrences_per_endpoint=8,
        confirm_occurrences_per_endpoint=8,
        fit_paired_sources=4,
        confirm_paired_sources=4,
        maximum_source_endpoint_fraction=0.25,
    )
    return np.asarray(vectors), endpoints, sources, split, thresholds


def test_source_identity_only_cannot_create_heldout_within_source_e_route() -> None:
    vectors, endpoints, sources, split, thresholds = _heldout_fixture(
        within_source_signal=False
    )
    evidence = evaluate_edge(
        vectors,
        endpoints,
        sources,
        split,
        edge_id="source-identity-only",
        tone_only=False,
        support_thresholds=thresholds,
        permutation_iterations=127,
        bootstrap_iterations=64,
    )
    family = finalize_route_family(
        {evidence.edge_id: evidence},
        {evidence.edge_id: False},
    )

    assert evidence.theta == pytest.approx(0.5)
    assert evidence.p_e == 1.0
    assert family.decisions[evidence.edge_id].route != "E"


def test_true_within_source_separation_can_confirm_e_in_bounded_family() -> None:
    vectors, endpoints, sources, split, thresholds = _heldout_fixture(
        within_source_signal=True
    )
    evidence = evaluate_edge(
        vectors,
        endpoints,
        sources,
        split,
        edge_id="true-within-source",
        tone_only=False,
        support_thresholds=thresholds,
        permutation_iterations=127,
        bootstrap_iterations=64,
    )
    family = finalize_route_family(
        {evidence.edge_id: evidence},
        {evidence.edge_id: False},
    )

    assert evidence.theta == 1.0
    assert evidence.ci95 == (1.0, 1.0)
    assert evidence.p_e == pytest.approx(1 / 128)
    assert family.decisions[evidence.edge_id].reason == "E_confirmed"


def test_insufficient_paired_sources_is_a_support_and_stays_in_bh_family() -> None:
    split = SourceSplit(
        fit_sources=("fit-paired", "fit-one"),
        confirm_sources=("confirm-paired", "confirm-one"),
    )
    endpoints = ["target", "confusion", "target", "target", "confusion", "target"]
    sources = [
        "fit-paired",
        "fit-paired",
        "fit-one",
        "confirm-paired",
        "confirm-paired",
        "confirm-one",
    ]
    evidence = evaluate_edge(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        endpoints,
        sources,
        split,
        edge_id="insufficient-paired",
        tone_only=False,
        support_thresholds=SupportThresholds(
            fit_occurrences_per_endpoint=1,
            confirm_occurrences_per_endpoint=1,
            fit_paired_sources=2,
            confirm_paired_sources=2,
            maximum_source_endpoint_fraction=1.0,
        ),
        permutation_iterations=7,
        bootstrap_iterations=8,
    )
    family = finalize_route_family(
        {evidence.edge_id: evidence},
        {evidence.edge_id: False},
    )

    assert not evidence.support.passed
    assert evidence.p_e == evidence.p_d == 1.0
    assert family.q_e[evidence.edge_id] == family.q_d[evidence.edge_id] == 1.0
    assert family.decisions[evidence.edge_id].reason == "A_support"
