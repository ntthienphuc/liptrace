"""Greedy CTC and historical sample metrics without training dependencies."""
import numpy as np


def decode_path(path, chars, blank):
    previous, result = None, []
    for value in path:
        value = int(value)
        if value < 0 or value > blank:
            raise ValueError('CTC path index outside charset')
        if value != blank and value != previous:
            result.append(chars[value])
        previous = value
    return ''.join(result)


def levenshtein(a, b):
    """Unit-cost edit distance in one row; no normalization is implicit."""
    row = np.arange(len(b) + 1, dtype=np.int64)
    for i, left in enumerate(a, 1):
        previous, row[0] = int(row[0]), i
        for j, right in enumerate(b, 1):
            old = int(row[j])
            row[j] = min(previous + (left != right), old + 1, int(row[j - 1]) + 1)
            previous = old
    return int(row[-1])


def cer_score(reference, hypothesis):
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer_score(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)
