"""Grouped visual-identifiability estimation for frozen PAViR edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GroupedAUC:
    auc: float
    probabilities: np.ndarray
    folds: np.ndarray
    n_splits: int


def _validate_binary_inputs(
    vectors: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(vectors, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int64)
    source = np.asarray(groups).astype(str)
    if matrix.ndim != 2 or target.ndim != 1 or source.ndim != 1:
        raise ValueError("vectors, labels, and groups must be rank 2/1/1")
    if not (len(matrix) == len(target) == len(source)):
        raise ValueError("vectors, labels, and groups have different lengths")
    if len(matrix) < 4 or not np.isfinite(matrix).all():
        raise ValueError("identifiability vectors are too few or non-finite")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("identifiability labels must contain exactly 0 and 1")
    if any(not value for value in source):
        raise ValueError("identifiability groups must be non-empty")
    return matrix, target, source


def grouped_oof_auc(
    vectors: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    regularization_c: float = 1.0,
) -> GroupedAUC:
    """Fit train-fold-only logistic probes and return grouped OOF AUC."""

    matrix, target, source = _validate_binary_inputs(vectors, labels, groups)
    if n_splits < 2 or regularization_c <= 0:
        raise ValueError("n_splits must be >=2 and C must be positive")
    class_group_counts = [
        len(np.unique(source[target == label])) for label in (0, 1)
    ]
    folds = min(int(n_splits), *class_group_counts, len(np.unique(source)))
    if folds < 2:
        raise ValueError("each endpoint needs occurrences in at least two groups")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=int(seed)
    )
    probabilities = np.full(len(target), np.nan, dtype=np.float64)
    assignments = np.full(len(target), -1, dtype=np.int16)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(matrix, target, groups=source)
    ):
        if len(np.unique(target[train_indices])) != 2:
            raise ValueError(f"fold {fold} training partition has one class")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(regularization_c),
                class_weight="balanced",
                max_iter=2000,
                random_state=int(seed) + fold,
                solver="liblinear",
            ),
        )
        model.fit(matrix[train_indices], target[train_indices])
        probabilities[test_indices] = model.predict_proba(matrix[test_indices])[:, 1]
        assignments[test_indices] = fold
    if not np.isfinite(probabilities).all() or np.any(assignments < 0):
        raise RuntimeError("grouped OOF probe did not cover every occurrence")
    return GroupedAUC(
        auc=float(roc_auc_score(target, probabilities)),
        probabilities=probabilities,
        folds=assignments,
        n_splits=folds,
    )


def source_bootstrap_auc(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Source-cluster percentile CI for an already out-of-fold prediction."""

    target = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    source = np.asarray(groups).astype(str)
    if not (len(target) == len(scores) == len(source)) or iterations <= 0:
        raise ValueError("invalid bootstrap inputs")
    unique = np.unique(source)
    indices = {group: np.flatnonzero(source == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    attempts = 0
    maximum_attempts = max(iterations * 20, 100)
    while len(draws) < iterations and attempts < maximum_attempts:
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        attempts += 1
        if len(np.unique(target[selected])) != 2:
            continue
        draws.append(float(roc_auc_score(target[selected], scores[selected])))
    if len(draws) < max(100, iterations // 2):
        raise ValueError("too few valid source-bootstrap draws")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def refit_label_null(
    vectors: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    permutations: int = 200,
    n_splits: int = 5,
    seed: int = 42,
    regularization_c: float = 1.0,
) -> np.ndarray:
    """Refit the full grouped probe after every global label permutation."""

    matrix, target, source = _validate_binary_inputs(vectors, labels, groups)
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    rng = np.random.default_rng(int(seed))
    values = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = rng.permutation(target)
        result = grouped_oof_auc(
            matrix,
            shuffled,
            source,
            n_splits=n_splits,
            seed=int(seed) + index + 1,
            regularization_c=regularization_c,
        )
        values[index] = result.auc
    return values
