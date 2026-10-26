from __future__ import annotations

from pavir.offline import (
    align_words,
    balanced_group_partition,
    is_tone_only_pair,
    normalize_text,
    strip_vietnamese_tone,
)


def test_normalization_matches_frozen_benchmark_contract():
    assert normalize_text("  Xin—CHÀO, Việt-Nam! 42 ") == "xin chào việt nam 42"


def test_tone_only_retains_vowel_quality_marks():
    assert strip_vietnamese_tone("mắng mẵng mặng") == "măng măng măng"
    assert is_tone_only_pair("ma", "má")
    assert is_tone_only_pair("mắng", "mẵng")
    assert not is_tone_only_pair("ma", "mă")
    assert not is_tone_only_pair("ban", "băn")


def test_word_alignment_emits_only_true_one_to_one_substitution():
    edits = align_words("tôi đi học hôm nay", "tôi đã học nay")
    assert [(edit.operation, edit.reference_word, edit.hypothesis_word) for edit in edits] == [
        ("equal", "tôi", "tôi"),
        ("substitute", "đi", "đã"),
        ("equal", "học", "học"),
        ("delete", "hôm", None),
        ("equal", "nay", "nay"),
    ]


def test_word_alignment_is_deterministic_on_ties():
    assert align_words("a b", "b a") == align_words("a b", "b a")


def test_balanced_group_partition_is_disjoint_complete_and_stable():
    statistics = [(f"g{index}", (index + 1, (index + 1) * 2)) for index in range(10)]
    left, right = balanced_group_partition(statistics, seed=42)
    assert len(left) == len(right) == 5
    assert not set(left) & set(right)
    assert set(left) | set(right) == {f"g{index}" for index in range(10)}
    assert (left, right) == balanced_group_partition(statistics, seed=42)
