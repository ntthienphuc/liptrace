from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "build_pavir_occurrences",
    ROOT / "tools" / "build_pavir_occurrences.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_word_token_spans_use_prefix_stable_boundaries():
    vocabulary = {"▁xin": (1,), "▁chào": (2, 3), "▁bạn": (4,)}

    def encode(text):
        output = []
        for word in text.split():
            output.extend(vocabulary[f"▁{word}"])
        return tuple(output)

    assert MODULE.word_token_spans("xin chào bạn", encode) == (
        ("xin", 0, 1),
        ("chào", 1, 3),
        ("bạn", 3, 4),
    )


def test_word_token_spans_reject_prefix_retokenization():
    def encode(text):
        return {"a": (1,), "a b": (9, 2)}[text]

    try:
        MODULE.word_token_spans("a b", encode)
    except ValueError as error:
        assert "retokenized" in str(error)
    else:
        raise AssertionError("prefix retokenization was not rejected")
