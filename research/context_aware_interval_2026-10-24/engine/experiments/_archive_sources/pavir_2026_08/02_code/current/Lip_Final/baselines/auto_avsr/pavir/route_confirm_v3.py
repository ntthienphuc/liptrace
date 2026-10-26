"""Confirmatory source-held-out routing statistics for PAViR.

This module defines the immutable statistical core of
``PAViR-RouteConfirm-v3``.  It deliberately does not read or write PAViR-v2
artifacts.  A caller must compute the global source split once, pass that same
split to every edge, and include every non-tone graph edge in both
Benjamini-Hochberg families (unsupported/skipped edges have p=1).

Endpoint semantics are fixed throughout:

* ``target`` is class 0;
* ``confusion`` is class 1;
* AUC therefore measures higher confusion scores versus target scores.

The only production Monte-Carlo defaults are 199,999 within-source
permutations and 10,000 source-bootstrap draws.  The counts are arguments so
bounded tests can exercise the identical algorithm cheaply.

This file is the single-process statistical core.  Equivalence of 1/2/3-worker
artifact estimators is a separate integration contract and is intentionally
not claimed by these unit-level routines.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, t as student_t
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


PROTOCOL = "PAViR-RouteConfirm-v3"
PROTOCOL_PRECOMMIT_SHA256 = (
    "63d9a2513bd66f35833db92329a98c5672620a56fd0087c48ae322fc5b85359d"
)
SOURCE_SPLIT_DOMAIN = b"pavir-route-confirm-v3\0"
EDGE_SEED_DOMAIN = b"pavir-route-v3-seed\0"

EXPECTED_SOURCE_COUNT = 2_714
FIT_FRACTION = 0.70
PRODUCTION_PERMUTATIONS = 199_999
PRODUCTION_BOOTSTRAP_ITERATIONS = 10_000
PROBE_REGULARIZATION_C = 1.0
PROBE_MAX_ITERATIONS = 2_000

E_FDR = 0.025
D_FDR = 0.025
E_CI_LOWER = 0.60
EQUIVALENCE_LOWER = 0.45
EQUIVALENCE_UPPER = 0.55
CHANCE_AUC = 0.50

_ENDPOINT_TO_LABEL = {"target": 0, "confusion": 1}
_LABEL_TO_ENDPOINT = ("target", "confusion")


@dataclass(frozen=True)
class SourceAssignment:
    """One source-only row suitable for a separately hash-bound split file."""

    source_id: str
    source_digest_sha256: str
    rank: int
    partition: str


@dataclass(frozen=True)
class SourceSplit:
    """One global, endpoint-blind deterministic 70/30 source split."""

    fit_sources: tuple[str, ...]
    confirm_sources: tuple[str, ...]

    @property
    def source_count(self) -> int:
        return len(self.fit_sources) + len(self.confirm_sources)

    def to_record(self) -> dict[str, object]:
        return {
            "protocol": PROTOCOL,
            "protocol_precommit_sha256": PROTOCOL_PRECOMMIT_SHA256,
            "split_method": (
                "sort (SHA256(b'pavir-route-confirm-v3\\0' + UTF8(source_id)), "
                "UTF8(source_id)); first floor(0.70*N) fit"
            ),
            "source_count": self.source_count,
            "fit_count": len(self.fit_sources),
            "confirm_count": len(self.confirm_sources),
            "fit_sources": list(self.fit_sources),
            "confirm_sources": list(self.confirm_sources),
        }


@dataclass(frozen=True)
class SupportThresholds:
    """Precommitted per-edge support and concentration thresholds."""

    fit_occurrences_per_endpoint: int = 100
    confirm_occurrences_per_endpoint: int = 40
    fit_paired_sources: int = 20
    confirm_paired_sources: int = 20
    maximum_source_endpoint_fraction: float = 0.20

    def __post_init__(self) -> None:
        if (
            self.fit_occurrences_per_endpoint <= 0
            or self.confirm_occurrences_per_endpoint <= 0
            or self.fit_paired_sources <= 0
            or self.confirm_paired_sources <= 0
            or not 0 < self.maximum_source_endpoint_fraction <= 1
        ):
            raise ValueError("invalid RouteConfirm-v3 support thresholds")


@dataclass(frozen=True)
class PartitionSupport:
    """Auditable support after excluding sources missing either endpoint."""

    partition: str
    observed_source_count: int
    paired_sources: tuple[str, ...]
    one_class_sources: tuple[str, ...]
    raw_endpoint_occurrences: tuple[int, int]
    included_endpoint_occurrences: tuple[int, int]
    maximum_source_endpoint_fractions: tuple[float, float]
    included_indices: tuple[int, ...]
    passed: bool
    failures: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        payload = asdict(self)
        payload["raw_endpoint_occurrences"] = {
            endpoint: int(self.raw_endpoint_occurrences[label])
            for label, endpoint in enumerate(_LABEL_TO_ENDPOINT)
        }
        payload["included_endpoint_occurrences"] = {
            endpoint: int(self.included_endpoint_occurrences[label])
            for label, endpoint in enumerate(_LABEL_TO_ENDPOINT)
        }
        payload["maximum_source_endpoint_fractions"] = {
            endpoint: float(self.maximum_source_endpoint_fractions[label])
            for label, endpoint in enumerate(_LABEL_TO_ENDPOINT)
        }
        payload["included_indices"] = list(self.included_indices)
        return payload


@dataclass(frozen=True)
class EdgeSupport:
    fit: PartitionSupport
    confirm: PartitionSupport
    passed: bool
    failures: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "fit": self.fit.to_record(),
            "confirm": self.confirm.to_record(),
        }


@dataclass(frozen=True)
class WeightedFP32Scaler:
    """Locked population-variance scaler with FP32 parameters and arithmetic."""

    mean_: np.ndarray
    var_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(
        cls, vectors: np.ndarray, sample_weight: np.ndarray
    ) -> WeightedFP32Scaler:
        matrix = _validate_vectors(vectors)
        weights = np.asarray(sample_weight, dtype=np.float32)
        if (
            weights.ndim != 1
            or len(weights) != len(matrix)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError("FP32 scaler received invalid sample weights")
        weight_sum = np.sum(weights, dtype=np.float32)
        mean = (
            np.sum(
                matrix * weights[:, None],
                axis=0,
                dtype=np.float32,
            )
            / weight_sum
        ).astype(np.float32, copy=False)
        centered = (matrix - mean).astype(np.float32, copy=False)
        variance = (
            np.sum(
                centered * centered * weights[:, None],
                axis=0,
                dtype=np.float32,
            )
            / weight_sum
        ).astype(np.float32, copy=False)
        scale = np.sqrt(variance).astype(np.float32, copy=False)
        scale = np.where(scale > 0, scale, np.float32(1.0)).astype(
            np.float32, copy=False
        )
        return cls(mean_=mean, var_=variance, scale_=scale)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        matrix = _validate_vectors(vectors)
        if matrix.shape[1] != len(self.mean_):
            raise ValueError("FP32 scaler feature dimension mismatch")
        return ((matrix - self.mean_) / self.scale_).astype(np.float32, copy=False)


@dataclass(frozen=True)
class WeightedProbe:
    """A fitted deterministic FP32 weighted scaler and linear probe."""

    scaler: WeightedFP32Scaler
    classifier: LogisticRegression
    seed: int
    converged: bool
    warning_messages: tuple[str, ...]

    def confusion_scores(self, vectors: np.ndarray) -> np.ndarray:
        matrix = _validate_vectors(vectors)
        transformed = self.scaler.transform(matrix).astype(np.float32, copy=False)
        return self.classifier.predict_proba(transformed)[:, 1].astype(
            np.float64, copy=False
        )


@dataclass(frozen=True)
class SourceMacroAUC:
    theta: float
    source_ids: tuple[str, ...]
    source_aucs: tuple[float, ...]
    excluded_one_class_sources: tuple[str, ...]


@dataclass(frozen=True)
class PermutationResult:
    p_value: float
    exceedances: int
    iterations: int
    null_mean: float
    null_std: float
    null_thetas: np.ndarray | None = None


@dataclass(frozen=True)
class BootstrapResult:
    lower: float
    upper: float
    iterations: int


@dataclass(frozen=True)
class EquivalenceResult:
    mean: float
    standard_error: float
    p_lower: float
    p_upper: float
    p_equivalence: float


@dataclass(frozen=True)
class EdgeEvidence:
    edge_id: str
    tone_only: bool
    support: EdgeSupport
    converged: bool
    seed: int
    theta: float | None
    source_aucs: tuple[float, ...]
    source_auc_ids: tuple[str, ...]
    ci95: tuple[float, float] | None
    p_e: float
    p_d: float
    permutation_exceedances: int | None
    warning_messages: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        payload = asdict(self)
        payload["support"] = self.support.to_record()
        return payload


@dataclass(frozen=True)
class RouteDecision:
    edge_id: str
    route: str
    reason: str
    q_e: float | None
    q_d: float | None


@dataclass(frozen=True)
class RouteFamily:
    decisions: Mapping[str, RouteDecision]
    q_e: Mapping[str, float]
    q_d: Mapping[str, float]
    sensitivity_q_e_by: Mapping[str, float]
    sensitivity_q_d_by: Mapping[str, float]


@dataclass(frozen=True)
class RouteMassThresholds:
    minimum_e_edges: int = 10
    minimum_d_edges: int = 10
    minimum_active_stable_support: float = 29_362
    minimum_e_stable_support: float = 2_937
    minimum_d_stable_support: float = 2_937
    minimum_active_vc_mass_fraction: float = 0.05
    minimum_e_vc_mass_fraction: float = 0.005
    minimum_d_vc_mass_fraction: float = 0.005

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(float(value) < 0 for value in values.values()):
            raise ValueError("route-mass thresholds must be non-negative")


@dataclass(frozen=True)
class RouteMassAudit:
    status: str
    eligible_for_artifacts_and_training: bool
    e_edge_count: int
    d_edge_count: int
    e_stable_support: float
    d_stable_support: float
    active_stable_support: float
    vc_total_substitution_mass: float
    vc_graph_covered_mass: float
    e_vc_mass: float
    d_vc_mass: float
    active_vc_mass: float
    e_vc_mass_fraction: float
    d_vc_mass_fraction: float
    active_vc_mass_fraction: float
    checks: Mapping[str, bool]
    blockers: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def _validate_source_id(value: object) -> str:
    if not isinstance(value, (str, np.str_)):
        raise TypeError("source IDs must be strings")
    source = str(value)
    if not source:
        raise ValueError("source IDs must be non-empty")
    source.encode("utf-8", errors="strict")
    return source


def deterministic_source_partition(
    source_ids: Sequence[str] | np.ndarray,
    *,
    expected_source_count: int | None = EXPECTED_SOURCE_COUNT,
) -> tuple[SourceAssignment, ...]:
    """Return the complete source-only assignment table in protocol order.

    Each exact UTF-8 source ID is keyed by
    ``SHA256(b'pavir-route-confirm-v3\\0' + source_id_utf8)``.  Unique sources
    are sorted by ``(digest_bytes, source_id)`` and the first
    ``floor(0.70 * N)`` are fit sources.  For N=2,714 this is exactly
    1,899 fit and 815 confirmation sources.  The production default fails
    closed unless the frozen 2,714-source inventory is present; bounded
    synthetic tests may explicitly pass ``expected_source_count=None``.
    """

    sources = sorted({_validate_source_id(value) for value in source_ids})
    if (
        expected_source_count is not None
        and len(sources) != int(expected_source_count)
    ):
        raise ValueError(
            "global source inventory mismatch: "
            f"expected {int(expected_source_count)}, observed {len(sources)}"
        )
    if len(sources) < 2:
        raise ValueError("global source split requires at least two sources")
    keyed = sorted(
        (
            hashlib.sha256(SOURCE_SPLIT_DOMAIN + source.encode("utf-8")).digest(),
            source.encode("utf-8"),
            source,
        )
        for source in sources
    )
    fit_count = math.floor(FIT_FRACTION * len(keyed))
    if not 0 < fit_count < len(keyed):
        raise ValueError("70/30 source split produced an empty partition")
    return tuple(
        SourceAssignment(
            source_id=source,
            source_digest_sha256=digest.hex(),
            rank=rank,
            partition="route_fit" if rank < fit_count else "route_confirm",
        )
        for rank, (digest, _, source) in enumerate(keyed)
    )


def deterministic_source_split(
    source_ids: Sequence[str] | np.ndarray,
    *,
    expected_source_count: int | None = EXPECTED_SOURCE_COUNT,
) -> SourceSplit:
    """Build a fit/confirm view of :func:`deterministic_source_partition`."""

    assignments = deterministic_source_partition(
        source_ids,
        expected_source_count=expected_source_count,
    )
    return SourceSplit(
        fit_sources=tuple(
            row.source_id for row in assignments if row.partition == "route_fit"
        ),
        confirm_sources=tuple(
            row.source_id for row in assignments if row.partition == "route_confirm"
        ),
    )


def edge_seed(edge_id: str) -> int:
    """Return the protocol's deterministic unsigned 32-bit edge seed.

    The seed is the first four SHA-256 digest bytes interpreted as an unsigned
    big-endian integer.  This explicitly fixes the uint conversion and yields
    a value accepted by scikit-learn and NumPy.
    """

    if not isinstance(edge_id, str) or not edge_id:
        raise ValueError("edge_id must be a non-empty string")
    digest = hashlib.sha256(EDGE_SEED_DOMAIN + edge_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _encode_endpoints(endpoints: Sequence[str] | np.ndarray) -> np.ndarray:
    values = np.asarray(endpoints).astype(str)
    if values.ndim != 1:
        raise ValueError("endpoints must be rank 1")
    unknown = sorted(set(values.tolist()) - set(_ENDPOINT_TO_LABEL))
    if unknown:
        raise ValueError(f"unknown RouteConfirm-v3 endpoint: {unknown[0]}")
    return np.asarray([_ENDPOINT_TO_LABEL[value] for value in values], dtype=np.int8)


def _validate_vectors(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("vectors must be a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain non-finite values")
    return matrix


def _validate_aligned(
    vectors: np.ndarray,
    endpoints: Sequence[str] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _validate_vectors(vectors)
    labels = _encode_endpoints(endpoints)
    raw_sources = np.asarray(source_ids)
    if raw_sources.ndim != 1:
        raise ValueError("source_ids must be rank 1")
    sources = np.asarray(
        [_validate_source_id(value) for value in raw_sources], dtype=str
    )
    if not (len(matrix) == len(labels) == len(sources)):
        raise ValueError("vectors/endpoints/source_ids are not aligned")
    return matrix, labels, sources


def _partition_support(
    labels: np.ndarray,
    sources: np.ndarray,
    allowed_sources: frozenset[str],
    *,
    partition: str,
    minimum_endpoint_occurrences: int,
    minimum_paired_sources: int,
    maximum_source_endpoint_fraction: float,
) -> PartitionSupport:
    partition_mask = np.fromiter(
        (source in allowed_sources for source in sources),
        dtype=np.bool_,
        count=len(sources),
    )
    partition_indices = np.flatnonzero(partition_mask)
    observed_sources = sorted(set(sources[partition_indices].tolist()))
    paired: list[str] = []
    one_class: list[str] = []
    for source in observed_sources:
        source_labels = set(labels[(sources == source) & partition_mask].tolist())
        if source_labels == {0, 1}:
            paired.append(source)
        elif source_labels:
            one_class.append(source)
    paired_set = frozenset(paired)
    included_mask = partition_mask & np.fromiter(
        (source in paired_set for source in sources),
        dtype=np.bool_,
        count=len(sources),
    )
    included_indices = np.flatnonzero(included_mask)
    raw_counts = tuple(
        int(np.sum(labels[partition_indices] == label)) for label in (0, 1)
    )
    included_counts = tuple(
        int(np.sum(labels[included_indices] == label)) for label in (0, 1)
    )
    concentration: list[float] = []
    for label, total in enumerate(included_counts):
        largest = max(
            (
                int(np.sum((sources[included_indices] == source) & (labels[included_indices] == label)))
                for source in paired
            ),
            default=0,
        )
        concentration.append(float(largest / total) if total else math.inf)

    failures: list[str] = []
    for label, endpoint in enumerate(_LABEL_TO_ENDPOINT):
        if included_counts[label] < minimum_endpoint_occurrences:
            failures.append(f"{partition}_{endpoint}_occurrences")
        if concentration[label] > maximum_source_endpoint_fraction:
            failures.append(f"{partition}_{endpoint}_source_concentration")
    if len(paired) < minimum_paired_sources:
        failures.append(f"{partition}_paired_sources")
    return PartitionSupport(
        partition=partition,
        observed_source_count=len(observed_sources),
        paired_sources=tuple(paired),
        one_class_sources=tuple(one_class),
        raw_endpoint_occurrences=raw_counts,
        included_endpoint_occurrences=included_counts,
        maximum_source_endpoint_fractions=tuple(concentration),
        included_indices=tuple(int(index) for index in included_indices),
        passed=not failures,
        failures=tuple(failures),
    )


def audit_edge_support(
    vectors: np.ndarray,
    endpoints: Sequence[str] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
    split: SourceSplit,
    *,
    thresholds: SupportThresholds = SupportThresholds(),
) -> EdgeSupport:
    """Apply paired-source exclusion and all precommitted support floors."""

    _, labels, sources = _validate_aligned(vectors, endpoints, source_ids)
    fit_set = frozenset(split.fit_sources)
    confirm_set = frozenset(split.confirm_sources)
    if fit_set & confirm_set:
        raise ValueError("fit and confirmation source partitions overlap")
    unknown = sorted(set(sources.tolist()) - fit_set - confirm_set)
    if unknown:
        raise ValueError(f"edge contains source outside the global split: {unknown[0]}")
    fit = _partition_support(
        labels,
        sources,
        fit_set,
        partition="fit",
        minimum_endpoint_occurrences=thresholds.fit_occurrences_per_endpoint,
        minimum_paired_sources=thresholds.fit_paired_sources,
        maximum_source_endpoint_fraction=thresholds.maximum_source_endpoint_fraction,
    )
    confirm = _partition_support(
        labels,
        sources,
        confirm_set,
        partition="confirm",
        minimum_endpoint_occurrences=thresholds.confirm_occurrences_per_endpoint,
        minimum_paired_sources=thresholds.confirm_paired_sources,
        maximum_source_endpoint_fraction=thresholds.maximum_source_endpoint_fraction,
    )
    failures = fit.failures + confirm.failures
    return EdgeSupport(
        fit=fit,
        confirm=confirm,
        passed=not failures,
        failures=failures,
    )


def source_endpoint_balanced_weights(
    endpoints: Sequence[str] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Return normalized ``1 / n_(source,endpoint)`` occurrence weights.

    Every non-empty source×endpoint cell has equal raw weight one.  The raw
    weights are then divided by their mean, so the returned FP32 weights have
    mean one.  All included sources must contain both endpoints; callers must
    perform paired-source filtering first.
    """

    labels = _encode_endpoints(endpoints)
    sources = np.asarray(
        [_validate_source_id(value) for value in np.asarray(source_ids)], dtype=str
    )
    if labels.ndim != 1 or sources.ndim != 1 or len(labels) != len(sources):
        raise ValueError("endpoints and source_ids are not aligned")
    if not len(labels):
        raise ValueError("cannot weight an empty probe population")
    raw = np.empty(len(labels), dtype=np.float64)
    for source in sorted(set(sources.tolist())):
        source_mask = sources == source
        if set(labels[source_mask].tolist()) != {0, 1}:
            raise ValueError("probe weights require paired endpoint sources")
        for label in (0, 1):
            cell = source_mask & (labels == label)
            raw[cell] = 1.0 / int(np.sum(cell))
    normalized = raw / float(np.mean(raw))
    return normalized.astype(np.float32)


def fit_weighted_probe(
    vectors: np.ndarray,
    endpoints: Sequence[str] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
    *,
    edge_id: str,
) -> WeightedProbe:
    """Fit the locked FP32 weighted scaler + liblinear logistic probe."""

    matrix, labels, sources = _validate_aligned(vectors, endpoints, source_ids)
    weights = source_endpoint_balanced_weights(endpoints, sources)
    seed = edge_seed(edge_id)
    classifier = LogisticRegression(
        C=PROBE_REGULARIZATION_C,
        solver="liblinear",
        max_iter=PROBE_MAX_ITERATIONS,
        random_state=seed,
    )
    caught: list[warnings.WarningMessage]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        scaler = WeightedFP32Scaler.fit(matrix, weights)
        transformed = scaler.transform(matrix).astype(np.float32, copy=False)
        classifier.fit(transformed, labels, sample_weight=weights)
    convergence_messages = tuple(
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    )
    return WeightedProbe(
        scaler=scaler,
        classifier=classifier,
        seed=seed,
        converged=not convergence_messages,
        warning_messages=convergence_messages,
    )


def mann_whitney_auc(
    endpoints: Sequence[str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Exact rank AUC with confusion positive and score ties worth 0.5."""

    labels = _encode_endpoints(endpoints)
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(labels) or not np.isfinite(values).all():
        raise ValueError("scores/endpoints are invalid or unaligned")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if not positives or not negatives:
        raise ValueError("Mann-Whitney AUC requires both endpoints")
    ranks = rankdata(values, method="average")
    u_statistic = float(np.sum(ranks[labels == 1])) - positives * (positives + 1) / 2
    return float(u_statistic / (positives * negatives))


def source_macro_auc(
    endpoints: Sequence[str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
) -> SourceMacroAUC:
    """Unweighted mean of exact within-source AUCs.

    One-class sources are excluded rather than pooled across sources, and are
    returned explicitly for audit.  In the production path they should
    already have been removed by :func:`audit_edge_support`.
    """

    labels = _encode_endpoints(endpoints)
    endpoint_values = np.asarray(endpoints).astype(str)
    values = np.asarray(scores, dtype=np.float64)
    sources = np.asarray(
        [_validate_source_id(value) for value in np.asarray(source_ids)], dtype=str
    )
    if (
        values.ndim != 1
        or sources.ndim != 1
        or not (len(labels) == len(values) == len(sources))
        or not np.isfinite(values).all()
    ):
        raise ValueError("source-macro AUC inputs are invalid or unaligned")
    included: list[str] = []
    excluded: list[str] = []
    aucs: list[float] = []
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        if set(labels[mask].tolist()) != {0, 1}:
            excluded.append(source)
            continue
        included.append(source)
        aucs.append(mann_whitney_auc(endpoint_values[mask], values[mask]))
    if not aucs:
        raise ValueError("source-macro AUC has no paired sources")
    return SourceMacroAUC(
        theta=float(np.mean(aucs, dtype=np.float64)),
        source_ids=tuple(included),
        source_aucs=tuple(aucs),
        excluded_one_class_sources=tuple(excluded),
    )


def _paired_rank_specs(
    endpoints: Sequence[str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
) -> tuple[list[tuple[str, np.ndarray, int, int]], float]:
    observed = source_macro_auc(endpoints, scores, source_ids)
    labels = _encode_endpoints(endpoints)
    values = np.asarray(scores, dtype=np.float64)
    sources = np.asarray(source_ids).astype(str)
    specs: list[tuple[str, np.ndarray, int, int]] = []
    for source in observed.source_ids:
        mask = sources == source
        source_labels = labels[mask]
        positives = int(np.sum(source_labels == 1))
        negatives = int(np.sum(source_labels == 0))
        specs.append(
            (
                source,
                rankdata(values[mask], method="average").astype(np.float64),
                positives,
                negatives,
            )
        )
    return specs, observed.theta


def within_source_permutation_test(
    endpoints: Sequence[str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
    *,
    edge_id: str,
    iterations: int = PRODUCTION_PERMUTATIONS,
    batch_size: int = 2_048,
    return_distribution: bool = False,
) -> PermutationResult:
    """One-sided fixed-score null preserving each source's endpoint counts.

    Sampling a uniformly random subset of ``n_confusion`` ranks is exactly
    equivalent to permuting endpoint labels within that source.  The
    implementation is vectorized in deterministic batches; batch size affects
    memory only, not the random stream or result.  The p-value is the
    Monte-Carlo add-one estimate ``(1 + #null>=observed) / (B + 1)``.
    """

    if iterations <= 0 or batch_size <= 0:
        raise ValueError("permutation iterations and batch_size must be positive")
    specs, observed_theta = _paired_rank_specs(endpoints, scores, source_ids)
    rng = np.random.default_rng(edge_seed(edge_id))
    null_sum = np.zeros(iterations, dtype=np.float64)
    for _, ranks, positives, negatives in specs:
        count = len(ranks)
        choose_positive = positives <= negatives
        selected_count = positives if choose_positive else negatives
        rank_total = float(np.sum(ranks))
        offset = positives * (positives + 1) / 2
        denominator = positives * negatives
        start = 0
        while start < iterations:
            stop = min(start + batch_size, iterations)
            priorities = rng.random((stop - start, count))
            chosen = np.argpartition(
                priorities, kth=selected_count - 1, axis=1
            )[:, :selected_count]
            selected_rank_sum = np.sum(ranks[chosen], axis=1)
            positive_rank_sum = (
                selected_rank_sum
                if choose_positive
                else rank_total - selected_rank_sum
            )
            null_sum[start:stop] += (positive_rank_sum - offset) / denominator
            start = stop
    null_thetas = null_sum / len(specs)
    exceedances = int(np.sum(null_thetas >= observed_theta))
    return PermutationResult(
        p_value=float((1 + exceedances) / (iterations + 1)),
        exceedances=exceedances,
        iterations=iterations,
        null_mean=float(np.mean(null_thetas)),
        null_std=float(np.std(null_thetas)),
        null_thetas=null_thetas if return_distribution else None,
    )


def source_bootstrap_ci(
    source_aucs: Sequence[float] | np.ndarray,
    *,
    edge_id: str,
    iterations: int = PRODUCTION_BOOTSTRAP_ITERATIONS,
    batch_size: int = 2_048,
) -> BootstrapResult:
    """Two-sided 95% percentile CI from equal-weight source resampling."""

    values = np.asarray(source_aucs, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
        or iterations <= 0
        or batch_size <= 0
    ):
        raise ValueError("invalid source-bootstrap inputs")
    rng = np.random.default_rng(edge_seed(edge_id))
    draws = np.empty(iterations, dtype=np.float64)
    start = 0
    while start < iterations:
        stop = min(start + batch_size, iterations)
        indices = rng.integers(
            0, len(values), size=(stop - start, len(values)), endpoint=False
        )
        draws[start:stop] = np.mean(values[indices], axis=1, dtype=np.float64)
        start = stop
    return BootstrapResult(
        lower=float(np.quantile(draws, 0.025)),
        upper=float(np.quantile(draws, 0.975)),
        iterations=iterations,
    )


def one_sample_auc_tost(
    source_aucs: Sequence[float] | np.ndarray,
    *,
    lower: float = EQUIVALENCE_LOWER,
    upper: float = EQUIVALENCE_UPPER,
) -> EquivalenceResult:
    """One-sample TOST for mean source AUC equivalence to ``[.45,.55]``."""

    values = np.asarray(source_aucs, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
        or not 0 <= lower < upper <= 1
    ):
        raise ValueError("invalid source-AUC TOST inputs")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    if standard_error == 0:
        p_lower = 0.0 if mean > lower else 0.5 if mean == lower else 1.0
        p_upper = 0.0 if mean < upper else 0.5 if mean == upper else 1.0
    else:
        degrees = len(values) - 1
        lower_statistic = (mean - lower) / standard_error
        upper_statistic = (mean - upper) / standard_error
        p_lower = float(student_t.sf(lower_statistic, degrees))
        p_upper = float(student_t.cdf(upper_statistic, degrees))
    return EquivalenceResult(
        mean=mean,
        standard_error=standard_error,
        p_lower=p_lower,
        p_upper=p_upper,
        p_equivalence=max(p_lower, p_upper),
    )


def _step_up_adjusted_p(
    p_values: Mapping[str, float], *, family_factor: float
) -> dict[str, float]:
    if not p_values:
        raise ValueError("multiple-testing family cannot be empty")
    if not math.isfinite(family_factor) or family_factor < 1:
        raise ValueError("multiple-testing family factor must be at least one")
    validated: list[tuple[str, float]] = []
    for edge_id, value in p_values.items():
        if not isinstance(edge_id, str) or not edge_id:
            raise ValueError("BH edge IDs must be non-empty strings")
        probability = float(value)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"invalid p-value for {edge_id}")
        validated.append((edge_id, probability))
    ordered = sorted(validated, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for index in range(count - 1, -1, -1):
        rank = index + 1
        candidate = min(
            1.0, ordered[index][1] * count * family_factor / rank
        )
        running = min(running, candidate)
        adjusted[index] = running
    by_edge = {
        edge_id: float(adjusted[index])
        for index, (edge_id, _) in enumerate(ordered)
    }
    return {edge_id: by_edge[edge_id] for edge_id in sorted(by_edge)}


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Deterministic monotone Benjamini-Hochberg adjusted p-values."""

    return _step_up_adjusted_p(p_values, family_factor=1.0)


def benjamini_yekutieli(p_values: Mapping[str, float]) -> dict[str, float]:
    """Dependence-robust BY q-values for the precommitted sensitivity report.

    These values are report-only and never replace BH q-values in route
    decisions.
    """

    count = len(p_values)
    if not count:
        raise ValueError("BY family cannot be empty")
    harmonic = math.fsum(1.0 / rank for rank in range(1, count + 1))
    return _step_up_adjusted_p(p_values, family_factor=harmonic)


def complete_non_tone_bh(
    observed_p_values: Mapping[str, float],
    non_tone_edge_ids: Sequence[str],
) -> dict[str, float]:
    """Fill every skipped non-tone graph edge with p=1 before BH."""

    family = tuple(sorted(set(non_tone_edge_ids)))
    if not family:
        raise ValueError("non-tone BH family cannot be empty")
    extras = sorted(set(observed_p_values) - set(family))
    if extras:
        raise ValueError(f"p-value supplied for edge outside BH family: {extras[0]}")
    return benjamini_hochberg(
        {edge_id: float(observed_p_values.get(edge_id, 1.0)) for edge_id in family}
    )


def evaluate_edge(
    vectors: np.ndarray,
    endpoints: Sequence[str] | np.ndarray,
    source_ids: Sequence[str] | np.ndarray,
    split: SourceSplit,
    *,
    edge_id: str,
    tone_only: bool,
    support_thresholds: SupportThresholds = SupportThresholds(),
    permutation_iterations: int = PRODUCTION_PERMUTATIONS,
    bootstrap_iterations: int = PRODUCTION_BOOTSTRAP_ITERATIONS,
) -> EdgeEvidence:
    """Compute one edge's held-out evidence without making a family decision."""

    matrix, _, sources = _validate_aligned(vectors, endpoints, source_ids)
    endpoint_values = np.asarray(endpoints).astype(str)
    support = audit_edge_support(
        matrix,
        endpoint_values,
        sources,
        split,
        thresholds=support_thresholds,
    )
    seed = edge_seed(edge_id)
    if not support.passed or tone_only:
        return EdgeEvidence(
            edge_id=edge_id,
            tone_only=bool(tone_only),
            support=support,
            converged=True,
            seed=seed,
            theta=None,
            source_aucs=(),
            source_auc_ids=(),
            ci95=None,
            p_e=1.0,
            p_d=1.0,
            permutation_exceedances=None,
            warning_messages=(),
        )

    fit_indices = np.asarray(support.fit.included_indices, dtype=np.int64)
    confirm_indices = np.asarray(support.confirm.included_indices, dtype=np.int64)
    probe = fit_weighted_probe(
        matrix[fit_indices],
        endpoint_values[fit_indices],
        sources[fit_indices],
        edge_id=edge_id,
    )
    if not probe.converged:
        return EdgeEvidence(
            edge_id=edge_id,
            tone_only=False,
            support=support,
            converged=False,
            seed=seed,
            theta=None,
            source_aucs=(),
            source_auc_ids=(),
            ci95=None,
            p_e=1.0,
            p_d=1.0,
            permutation_exceedances=None,
            warning_messages=probe.warning_messages,
        )
    scores = probe.confusion_scores(matrix[confirm_indices])
    macro = source_macro_auc(
        endpoint_values[confirm_indices],
        scores,
        sources[confirm_indices],
    )
    if macro.excluded_one_class_sources:
        raise RuntimeError("paired confirmation filter admitted a one-class source")
    permutation = within_source_permutation_test(
        endpoint_values[confirm_indices],
        scores,
        sources[confirm_indices],
        edge_id=edge_id,
        iterations=permutation_iterations,
    )
    bootstrap = source_bootstrap_ci(
        macro.source_aucs,
        edge_id=edge_id,
        iterations=bootstrap_iterations,
    )
    equivalence = one_sample_auc_tost(macro.source_aucs)
    return EdgeEvidence(
        edge_id=edge_id,
        tone_only=False,
        support=support,
        converged=True,
        seed=seed,
        theta=macro.theta,
        source_aucs=macro.source_aucs,
        source_auc_ids=macro.source_ids,
        ci95=(bootstrap.lower, bootstrap.upper),
        p_e=permutation.p_value,
        p_d=equivalence.p_equivalence,
        permutation_exceedances=permutation.exceedances,
        warning_messages=(),
    )


def decide_route(
    evidence: EdgeEvidence,
    *,
    q_e: float | None,
    q_d: float | None,
) -> RouteDecision:
    """Apply the precommitted E/D/A order after family correction.

    ``D_tone`` is a separate linguistic prior but must pass the same support
    floor.  Non-tone D requires both TOST BH significance and a 95% source
    bootstrap CI entirely inside [.45,.55].  A CI wholly below chance is
    treated as reversed evidence and bars D.  To keep the requested labels
    unambiguous, the reason is ``A_reversed`` exactly when theta < .45;
    otherwise a non-confirmed statistical edge is ``A_uncertain``.
    """

    if not evidence.support.passed:
        return RouteDecision(evidence.edge_id, "A", "A_support", q_e, q_d)
    if evidence.tone_only:
        return RouteDecision(evidence.edge_id, "D", "D_tone", None, None)
    if not evidence.converged or evidence.theta is None or evidence.ci95 is None:
        return RouteDecision(evidence.edge_id, "A", "A_support", q_e, q_d)
    if q_e is None or q_d is None:
        raise ValueError("non-tone decision requires both BH q-values")
    ci_lower, ci_upper = evidence.ci95
    if q_e <= E_FDR and ci_lower >= E_CI_LOWER:
        return RouteDecision(evidence.edge_id, "E", "E_confirmed", q_e, q_d)
    ci_reversed = ci_upper < CHANCE_AUC
    d_allowed = evidence.theta >= EQUIVALENCE_LOWER and not ci_reversed
    if (
        d_allowed
        and q_d <= D_FDR
        and ci_lower >= EQUIVALENCE_LOWER
        and ci_upper <= EQUIVALENCE_UPPER
    ):
        return RouteDecision(evidence.edge_id, "D", "D_equiv", q_e, q_d)
    reason = "A_reversed" if evidence.theta < EQUIVALENCE_LOWER else "A_uncertain"
    return RouteDecision(evidence.edge_id, "A", reason, q_e, q_d)


def finalize_route_family(
    evidence_by_edge: Mapping[str, EdgeEvidence],
    tone_only_by_edge: Mapping[str, bool],
) -> RouteFamily:
    """Correct both non-tone families, including every skipped edge at p=1."""

    graph_edges = set(tone_only_by_edge)
    if not graph_edges or set(evidence_by_edge) != graph_edges:
        raise ValueError("route evidence must cover every graph edge exactly once")
    for edge_id, evidence in evidence_by_edge.items():
        if evidence.edge_id != edge_id:
            raise ValueError("route evidence edge key mismatch")
        if evidence.tone_only != bool(tone_only_by_edge[edge_id]):
            raise ValueError("route evidence tone flag differs from graph")
    non_tone = sorted(
        edge_id for edge_id, tone in tone_only_by_edge.items() if not tone
    )
    if non_tone:
        p_e = {edge_id: evidence_by_edge[edge_id].p_e for edge_id in non_tone}
        p_d = {edge_id: evidence_by_edge[edge_id].p_d for edge_id in non_tone}
        q_e = complete_non_tone_bh(p_e, non_tone)
        q_d = complete_non_tone_bh(p_d, non_tone)
        q_e_by = benjamini_yekutieli(p_e)
        q_d_by = benjamini_yekutieli(p_d)
    else:
        q_e, q_d, q_e_by, q_d_by = {}, {}, {}, {}
    decisions: dict[str, RouteDecision] = {}
    for edge_id in sorted(graph_edges):
        decisions[edge_id] = decide_route(
            evidence_by_edge[edge_id],
            q_e=q_e.get(edge_id),
            q_d=q_d.get(edge_id),
        )
    return RouteFamily(
        decisions=decisions,
        q_e=q_e,
        q_d=q_d,
        sensitivity_q_e_by=q_e_by,
        sensitivity_q_d_by=q_d_by,
    )


def audit_route_mass(
    decisions: Mapping[str, RouteDecision | str],
    stable_support_by_edge: Mapping[str, float],
    vc_substitution_mass_by_edge: Mapping[str, float],
    *,
    vc_total_substitution_mass: float,
    thresholds: RouteMassThresholds = RouteMassThresholds(),
) -> RouteMassAudit:
    """Gate whether confirmed routes carry enough training-relevant mass.

    The VC fractions use the caller-supplied *full* VC-calib substitution mass
    as denominator; ``vc_graph_covered_mass`` is also reported so a caller
    cannot silently substitute the smaller covered-only denominator.
    Only a PASS authorizes downstream artifact extraction or training.
    """

    edge_ids = set(decisions)
    if not edge_ids:
        raise ValueError("route-mass audit received no decisions")
    if set(stable_support_by_edge) != edge_ids:
        raise ValueError("stable support must cover every route edge exactly")
    if set(vc_substitution_mass_by_edge) - edge_ids:
        raise ValueError("VC mass references an edge outside the route family")
    total_vc = float(vc_total_substitution_mass)
    if not math.isfinite(total_vc) or total_vc <= 0:
        raise ValueError("VC total substitution mass must be finite and positive")

    routes: dict[str, str] = {}
    for edge_id, value in decisions.items():
        route = value.route if isinstance(value, RouteDecision) else str(value)
        if route not in {"E", "D", "A"}:
            raise ValueError(f"invalid route for {edge_id}: {route}")
        routes[edge_id] = route
    support = {edge_id: float(value) for edge_id, value in stable_support_by_edge.items()}
    vc_mass = {edge_id: float(vc_substitution_mass_by_edge.get(edge_id, 0.0)) for edge_id in edge_ids}
    if any(not math.isfinite(value) or value < 0 for value in (*support.values(), *vc_mass.values())):
        raise ValueError("route masses must be finite and non-negative")
    graph_covered = float(sum(vc_mass.values()))
    if graph_covered > total_vc + 1e-12:
        raise ValueError("graph-covered VC mass exceeds total substitution mass")

    e_edges = sorted(edge_id for edge_id, route in routes.items() if route == "E")
    d_edges = sorted(edge_id for edge_id, route in routes.items() if route == "D")
    e_support = float(sum(support[edge_id] for edge_id in e_edges))
    d_support = float(sum(support[edge_id] for edge_id in d_edges))
    e_vc = float(sum(vc_mass[edge_id] for edge_id in e_edges))
    d_vc = float(sum(vc_mass[edge_id] for edge_id in d_edges))
    checks = {
        "minimum_e_edges": len(e_edges) >= thresholds.minimum_e_edges,
        "minimum_d_edges": len(d_edges) >= thresholds.minimum_d_edges,
        "minimum_active_stable_support": e_support + d_support
        >= thresholds.minimum_active_stable_support,
        "minimum_e_stable_support": e_support
        >= thresholds.minimum_e_stable_support,
        "minimum_d_stable_support": d_support
        >= thresholds.minimum_d_stable_support,
        "minimum_active_vc_mass_fraction": (e_vc + d_vc) / total_vc
        >= thresholds.minimum_active_vc_mass_fraction,
        "minimum_e_vc_mass_fraction": e_vc / total_vc
        >= thresholds.minimum_e_vc_mass_fraction,
        "minimum_d_vc_mass_fraction": d_vc / total_vc
        >= thresholds.minimum_d_vc_mass_fraction,
    }
    blockers = tuple(name for name, passed in checks.items() if not passed)
    passed = not blockers
    return RouteMassAudit(
        status="PASS" if passed else "BLOCKED",
        eligible_for_artifacts_and_training=passed,
        e_edge_count=len(e_edges),
        d_edge_count=len(d_edges),
        e_stable_support=e_support,
        d_stable_support=d_support,
        active_stable_support=e_support + d_support,
        vc_total_substitution_mass=total_vc,
        vc_graph_covered_mass=graph_covered,
        e_vc_mass=e_vc,
        d_vc_mass=d_vc,
        active_vc_mass=e_vc + d_vc,
        e_vc_mass_fraction=e_vc / total_vc,
        d_vc_mass_fraction=d_vc / total_vc,
        active_vc_mass_fraction=(e_vc + d_vc) / total_vc,
        checks=checks,
        blockers=blockers,
    )
