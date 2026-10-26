"""Independent exhaustive CPU checks; run directly or through unittest."""

import itertools
import math
import unittest

import numpy as np

from confirmation_common import edit_counts
from ctc_decode import prefix_beam
from geometry import allocation, make_interval, make_sliding, reference_for_interval, select_units


def exhaustive(logp, eos=None):
    """Sum every legal CTC path using ordinary probabilities, not beam recurrence."""
    frames, vocabulary = logp.shape
    totals = {}
    for path in itertools.product(range(vocabulary), repeat=frames):
        if eos is not None and eos in path:
            continue
        tokens = []
        previous = None
        probability = 1.0
        for frame, token in enumerate(path):
            probability *= math.exp(float(logp[frame, token]))
            if token != previous and token != 0:
                tokens.append(token)
            previous = token
        key = tuple(tokens)
        totals[key] = totals.get(key, 0.0) + probability
    return totals


class GeometryTests(unittest.TestCase):
    def test_balanced_exhaustive_coverage(self):
        for total in range(1, 301):
            for target in (1, 7, 50, 301):
                for context in (0, 16, 400):
                    windows = make_sliding(total, target, context)
                    self.assertEqual(len(windows), math.ceil(total / target))
                    covered = []
                    lengths = []
                    for index, window in enumerate(windows):
                        lower = window["target_start"]
                        upper = window["target_end"]
                        covered.extend(range(lower, upper))
                        lengths.append(upper - lower)
                        self.assertEqual(lower, index * total // len(windows))
                        self.assertEqual(upper, (index + 1) * total // len(windows))
                        self.assertEqual(window["input_start"], max(0, lower - context))
                        self.assertEqual(window["input_end"], min(total, upper + context))
                        self.assertEqual(window["local_start"], lower - window["input_start"])
                        self.assertEqual(window["local_end"], upper - window["input_start"])
                        self.assertEqual(window["left_context"], lower - window["input_start"])
                        self.assertEqual(window["right_context"], window["input_end"] - upper)
                    self.assertEqual(covered, list(range(total)))
                    self.assertLessEqual(max(lengths), target)
                    self.assertLessEqual(max(lengths) - min(lengths), 1)

    def test_edges_and_nominal_length(self):
        expected = {"start": (0, 50), "center": (25, 75), "end": (51, 101)}
        for position, bounds in expected.items():
            window = make_interval(101, 50, 16, position)
            self.assertEqual((window["target_start"], window["target_end"]), bounds)
        for position in expected:
            window = make_interval(7, 50, 16, position)
            self.assertEqual(window["nominal_target_frames"], 50)
            self.assertEqual(window["actual_target_frames"], 7)
            self.assertEqual(window["left_context"], 0)
            self.assertEqual(window["right_context"], 0)

    def test_midpoint_and_overlap_ties(self):
        unit = {"text": "tie", "start": 4, "end": 6, "metadata": 42}
        self.assertEqual(select_units([unit], 0, 5), [])
        self.assertEqual(select_units([unit], 5, 10), [unit])
        self.assertEqual(select_units([unit], 0, 5, "overlap"), [unit])
        self.assertEqual(select_units([unit], 6, 10, "overlap"), [])
        self.assertEqual(select_units([unit], 5, 5, "overlap"), [])
        selected = select_units([unit], 0, 10)
        selected[0]["text"] = "changed"
        self.assertEqual(unit["text"], "tie")

    def test_reference_timing_and_empty_reference(self):
        words = [
            {"text": " XIN ", "start": 0.0, "end": 0.4},
            {"text": "Chào", "start": 0.4, "end": 0.8},
        ]
        self.assertEqual(reference_for_interval(words, 0, 10), "xin")
        self.assertEqual(reference_for_interval(words, 10, 20), "chào")
        self.assertEqual(reference_for_interval(words, 0, 20, metric="cer"), "xinchào")
        self.assertEqual(reference_for_interval(words, 0, 5), "")
        self.assertEqual(reference_for_interval(words, 0, 5, rule="overlap"), "xin")
        self.assertEqual(reference_for_interval([], 0, 20), "")
        self.assertEqual(edit_counts("", "xin chào")["insertions"], 2)
        self.assertEqual(edit_counts("", "")["errors"], 0)
        self.assertEqual(edit_counts("", "你 好", "cer")["units"], 0)

    def test_sentencepiece_unicode_and_whole_word_ownership(self):
        vocabulary = ["<blank>", "▁Xin", "▁cha", "\u0300o", "▁thế", "▁giới"]
        tokens = [1, 2, 3, 4, 5]
        spans = [(0, 2), (3, 4), (4, 6), (6, 8), (8, 10)]
        result = allocation(tokens, spans, vocabulary, 0, 10)
        self.assertEqual(result["text"], "xin chào thế giới")
        self.assertEqual(result["units"][1]["token_positions"], [1, 2])
        self.assertEqual(result["units"][1]["start"], 3)
        self.assertEqual(result["units"][1]["end"], 6)
        self.assertEqual(allocation(tokens, spans, vocabulary, 0, 4.5)["text"], "xin")
        self.assertEqual(allocation(tokens, spans, vocabulary, 4.5, 10)["text"], "chào thế giới")
        self.assertEqual(allocation(tokens, spans, vocabulary, 0, 10, metric="cer")["text"], "xinchàothếgiới")

    def test_native_characters_and_composition(self):
        vocabulary = ["<blank>", "你", "好", "a", "\u0301", "<space>", "b"]
        tokens = [1, 2, 3, 4, 5, 6]
        spans = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        result = allocation(tokens, spans, vocabulary, 0, 6, metric="cer")
        self.assertEqual(result["text"], "你好áb")
        self.assertEqual([unit["text"] for unit in result["units"]], ["你", "好", "á", "b"])
        self.assertEqual(allocation(tokens, spans, vocabulary, 1, 2, metric="cer")["text"], "好")
        accent = result["units"][2]
        self.assertEqual((accent["start"], accent["end"]), (2, 4))
        self.assertEqual(accent["token_positions"], [2, 3])
        self.assertEqual(allocation(tokens, spans, vocabulary, 0, 6)["text"], "你好á b")
        multi = allocation([1], [(2, 6)], ["<blank>", "▁你好"], 0, 10, metric="cer")
        self.assertEqual([(unit["start"], unit["end"]) for unit in multi["units"]], [(2, 6), (2, 6)])

    def test_byte_fallback(self):
        vocabulary = ["<blank>", "▁", "<0xE4>", "<0xBD>", "<0xA0>"]
        result = allocation([1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)], vocabulary, 0, 4)
        self.assertEqual(result["text"], "你")
        self.assertEqual(result["units"][0]["token_positions"], [1, 2, 3])

    def test_full_ownership_and_partition_invariants(self):
        vocabulary = ["<blank>", "▁one", "▁two", "▁two", "▁four"]
        spans = [(0, 1), (2, 5), (5, 9), (9, 12)]
        for rule in ("midpoint", "overlap"):
            result = allocation([1, 2, 3, 4], spans, vocabulary, 0, 12, rule=rule)
            self.assertEqual(result["units"], result["selected_units"])
            self.assertEqual(result["text"], "one two two four")
        units = allocation([1, 2, 3, 4], spans, vocabulary, 0, 12)["units"]
        gathered = []
        for window in make_sliding(12, 5, 0):
            gathered.extend(select_units(units, window["target_start"], window["target_end"]))
        self.assertEqual(gathered, units)
        self.assertEqual(allocation([], [], vocabulary, 0, 0)["text"], "")
        for position in ("start", "center", "end"):
            window = make_interval(120, 50, 0, position)
            result = allocation([1], [(0, 50)], vocabulary, window["local_start"], window["local_end"])
            self.assertEqual(result["text"], "one")

    def test_geometry_validation(self):
        for invalid in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                make_interval(invalid, 50, 0)
            with self.assertRaises(ValueError):
                make_sliding(100, invalid, 0)
        for invalid in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                make_sliding(100, 50, invalid)
        with self.assertRaises(ValueError):
            reference_for_interval([], 0, 1, fps=0)
        with self.assertRaises(ValueError):
            select_units([], 0, 1, rule="unknown")
        with self.assertRaises(ValueError):
            select_units([{"text": "x", "start": 1, "end": 1}], 0, 2)
        with self.assertRaises(ValueError):
            allocation([1], [], ["<blank>", "x"], 0, 10)
        with self.assertRaises(ValueError):
            allocation([0], [(0, 1)], ["<blank>"], 0, 10)


class CTCDecodeTests(unittest.TestCase):
    def test_exhaustive_random_path_sums(self):
        random = np.random.default_rng(20261019)
        for vocabulary in (2, 3, 4):
            for frames in range(0, 7):
                for eos in (None, vocabulary - 1):
                    for repeat in range(4):
                        raw = random.uniform(0.01, 1, size=(frames, vocabulary))
                        logp = np.log(raw / raw.sum(axis=1, keepdims=True))
                        totals = exhaustive(logp, eos)
                        expected = min(totals, key=lambda tokens: (-totals[tokens], tokens))
                        actual = prefix_beam(logp, beam_size=4096, eos=eos, token_prune=None)
                        self.assertEqual(tuple(actual["tokens"]), expected)
                        self.assertAlmostEqual(actual["score"], math.log(totals[expected]), places=11)

    def test_sum_is_not_argmax_path(self):
        logp = np.log([[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]])
        result = prefix_beam(logp, beam_size=30, token_prune=None)
        self.assertEqual(result["tokens"], [1])
        self.assertAlmostEqual(math.exp(result["score"]), 0.4025)
        self.assertEqual(np.argmax(logp, axis=1).tolist(), [0, 0])

    def test_repetition_blank_and_terminal(self):
        for path, expected in (([1, 1], [1]), ([1, 0, 1], [1, 1]), ([0, 0], []), ([1, 0], [1])):
            logp = np.full((len(path), 3), -np.inf)
            logp[np.arange(len(path)), path] = 0
            result = prefix_beam(logp, eos=2, token_prune=None)
            self.assertEqual(result["tokens"], expected)
            self.assertEqual(result["score"], 0)
        result = prefix_beam(np.log([[0.2, 0.3, 0.5]]), eos=2, token_prune=None)
        self.assertEqual(result["tokens"], [1])
        self.assertAlmostEqual(result["score"], math.log(0.3))
        with self.assertRaises(ValueError):
            prefix_beam([[-np.inf, -np.inf, 0]], eos=2)

    def test_pruning_preserves_prefix_last_and_blank(self):
        result = prefix_beam(np.log([[0.1, 0.9, 1e-30], [0.25, 0.35, 0.4]]), beam_size=1, token_prune=1)
        self.assertEqual(result["tokens"], [1])
        self.assertAlmostEqual(math.exp(result["score"]), 0.54)
        self.assertEqual(result["token_prune"], 1)
        self.assertEqual(result["algorithm"], "ctc_prefix_beam")

    def test_default_top64_matches_exact_on_small_vocab(self):
        logp = np.log([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])
        pruned = prefix_beam(logp)
        exact = prefix_beam(logp, token_prune=None)
        self.assertEqual(pruned["tokens"], exact["tokens"])
        self.assertEqual(pruned["score"], exact["score"])
        self.assertEqual(pruned["token_prune"], 64)

    def test_numerical_stability_and_empty_time(self):
        result = prefix_beam(np.tile([-10000.0, -10001.0], (10, 1)), token_prune=None, beam_size=100)
        self.assertTrue(math.isfinite(result["score"]))
        result = prefix_beam(np.empty((0, 2)))
        self.assertEqual(result["tokens"], [])
        self.assertEqual(result["score"], 0)

    def test_invalid_inputs(self):
        for values in ([0, 1], [[math.nan, 0]], [[math.inf, 0]], [[0, 1]], np.empty((1, 0))):
            with self.assertRaises(ValueError):
                prefix_beam(values)
        for options in ({"beam_size": 0}, {"token_prune": 0}, {"blank": 1}, {"eos": 0}, {"eos": 2}):
            with self.assertRaises(ValueError):
                prefix_beam([[-1, -1]], **options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
