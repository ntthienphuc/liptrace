"""Deterministic offline utilities shared by the PAViR artifact pipeline.

This module intentionally contains no model code.  It freezes the text
normalization, word alignment, Vietnamese tone comparison, and source-group
splitting rules used before any PAViR training arm is started.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence


_SPACE_RE = re.compile(r"\s+")
_TONE_MARKS = frozenset(
    {
        "\N{COMBINING GRAVE ACCENT}",
        "\N{COMBINING ACUTE ACCENT}",
        "\N{COMBINING TILDE}",
        "\N{COMBINING HOOK ABOVE}",
        "\N{COMBINING DOT BELOW}",
    }
)


def normalize_text(text: str) -> str:
    """Return the same NFC word representation as ``benchmark_v1``."""

    normalized = unicodedata.normalize("NFC", str(text)).lower()
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        characters.append(character if category[0] in {"L", "N", "M"} else " ")
    return _SPACE_RE.sub(" ", "".join(characters)).strip()


def strip_vietnamese_tone(text: str) -> str:
    """Remove only Vietnamese tone marks, retaining vowel-quality marks.

    Breve, circumflex, and horn are deliberately retained.  Thus ``ma`` and
    ``má`` are tone-only, whereas ``a`` and ``ă`` are not.
    """

    decomposed = unicodedata.normalize("NFD", normalize_text(text))
    kept = "".join(character for character in decomposed if character not in _TONE_MARKS)
    return unicodedata.normalize("NFC", kept)


def is_tone_only_pair(left: str, right: str) -> bool:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    return (
        bool(left_normalized)
        and left_normalized != right_normalized
        and strip_vietnamese_tone(left_normalized)
        == strip_vietnamese_tone(right_normalized)
    )


@dataclass(frozen=True)
class WordEdit:
    """One deterministic Levenshtein operation."""

    operation: str
    reference_index: int | None
    hypothesis_index: int | None
    reference_word: str | None
    hypothesis_word: str | None


def align_words(reference: str, hypothesis: str) -> tuple[WordEdit, ...]:
    """Return a deterministic minimum-edit word alignment.

    Ties are frozen as match/substitution, deletion, then insertion.  This
    makes the confusion graph reproducible across edit-distance libraries.
    """

    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    rows, columns = len(reference_words), len(hypothesis_words)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    back: list[list[str | None]] = [
        [None] * (columns + 1) for _ in range(rows + 1)
    ]
    for row in range(1, rows + 1):
        costs[row][0] = row
        back[row][0] = "delete"
    for column in range(1, columns + 1):
        costs[0][column] = column
        back[0][column] = "insert"

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            same = reference_words[row - 1] == hypothesis_words[column - 1]
            diagonal = costs[row - 1][column - 1] + (0 if same else 1)
            deletion = costs[row - 1][column] + 1
            insertion = costs[row][column - 1] + 1
            best = min(diagonal, deletion, insertion)
            costs[row][column] = best
            if diagonal == best:
                back[row][column] = "equal" if same else "substitute"
            elif deletion == best:
                back[row][column] = "delete"
            else:
                back[row][column] = "insert"

    edits: list[WordEdit] = []
    row, column = rows, columns
    while row or column:
        operation = back[row][column]
        if operation in {"equal", "substitute"}:
            edits.append(
                WordEdit(
                    operation=operation,
                    reference_index=row - 1,
                    hypothesis_index=column - 1,
                    reference_word=reference_words[row - 1],
                    hypothesis_word=hypothesis_words[column - 1],
                )
            )
            row -= 1
            column -= 1
        elif operation == "delete":
            edits.append(
                WordEdit(
                    operation=operation,
                    reference_index=row - 1,
                    hypothesis_index=None,
                    reference_word=reference_words[row - 1],
                    hypothesis_word=None,
                )
            )
            row -= 1
        elif operation == "insert":
            edits.append(
                WordEdit(
                    operation=operation,
                    reference_index=None,
                    hypothesis_index=column - 1,
                    reference_word=None,
                    hypothesis_word=hypothesis_words[column - 1],
                )
            )
            column -= 1
        else:  # pragma: no cover - protects the frozen DP implementation.
            raise RuntimeError(f"invalid alignment back-pointer at {row},{column}")
    edits.reverse()
    return tuple(edits)


def stable_hash(value: str, *, seed: int = 0) -> str:
    return hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).hexdigest()


def balanced_group_partition(
    group_statistics: Iterable[tuple[str, Sequence[float]]],
    *,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Greedily split groups into two deterministic, approximately balanced arms.

    The helper is a dependency-free fallback for audits and tests.  The
    production ViCocktail firewall may use an exact MILP, but both methods use
    this stable hash as the final tie-break.
    """

    rows = [
        (str(group), tuple(float(value) for value in values))
        for group, values in group_statistics
    ]
    if len(rows) < 2:
        raise ValueError("at least two groups are required")
    dimensions = len(rows[0][1])
    if dimensions == 0 or any(len(values) != dimensions for _, values in rows):
        raise ValueError("group statistics must have one fixed non-empty shape")
    scales = [
        max(sum(values[index] for _, values in rows), 1.0)
        for index in range(dimensions)
    ]
    rows.sort(
        key=lambda item: (
            -sum(item[1][index] / scales[index] for index in range(dimensions)),
            stable_hash(item[0], seed=seed),
        )
    )
    assignments: list[list[str]] = [[], []]
    totals = [[0.0] * dimensions, [0.0] * dimensions]
    target_sizes = (len(rows) // 2, len(rows) - len(rows) // 2)
    for group, values in rows:
        candidates = []
        for side in (0, 1):
            if len(assignments[side]) >= target_sizes[side]:
                continue
            hypothetical = [
                totals[side][index] + values[index] for index in range(dimensions)
            ]
            other = totals[1 - side]
            imbalance = sum(
                abs(hypothetical[index] - other[index]) / scales[index]
                for index in range(dimensions)
            )
            candidates.append((imbalance, len(assignments[side]), side))
        _, _, selected = min(candidates)
        assignments[selected].append(group)
        for index, value in enumerate(values):
            totals[selected][index] += value
    return tuple(sorted(assignments[0])), tuple(sorted(assignments[1]))
