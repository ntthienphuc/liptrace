from __future__ import annotations

import numpy as np

from pavir.identifiability import (
    grouped_oof_auc,
    refit_label_null,
    source_bootstrap_auc,
)


def separable_fixture():
    rng = np.random.default_rng(7)
    vectors, labels, groups = [], [], []
    for source in range(10):
        for label in (0, 1):
            for _ in range(4):
                value = rng.normal(0, 0.15, size=6)
                value[0] += -2.0 if label == 0 else 2.0
                vectors.append(value)
                labels.append(label)
                groups.append(f"s{source}")
    return np.asarray(vectors), np.asarray(labels), np.asarray(groups)


def test_grouped_oof_probe_and_source_bootstrap_are_deterministic():
    vectors, labels, groups = separable_fixture()
    first = grouped_oof_auc(vectors, labels, groups, n_splits=5, seed=42)
    second = grouped_oof_auc(vectors, labels, groups, n_splits=5, seed=42)
    assert first.auc > 0.99
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert source_bootstrap_auc(
        labels, first.probabilities, groups, iterations=200, seed=42
    )[0] > 0.99


def test_refit_null_preserves_count_and_is_reproducible():
    vectors, labels, groups = separable_fixture()
    first = refit_label_null(
        vectors, labels, groups, permutations=8, n_splits=5, seed=13
    )
    second = refit_label_null(
        vectors, labels, groups, permutations=8, n_splits=5, seed=13
    )
    assert first.shape == (8,)
    np.testing.assert_array_equal(first, second)
    assert np.all((0 <= first) & (first <= 1))
