"""CPU whole-clip integration tests with observable, reference-blind runtimes."""

import copy
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from geometry import make_sliding
from sliding_inference import evaluate_clip, frozen_geometry_queries
from test_inference_core import FakeRuntime


def fixture(total=11, target=4, context=2):
    pixels = np.arange(total, dtype=np.uint8)[:, None, None]
    row = {"frames": total, "metric": "wer", "reference": "xin chào", "fps": 25.0}
    decoders = [{"name": "hybrid", "kind": "hybrid", "ctc_weight": 0.5},
                {"name": "greedy", "kind": "greedy"},
                {"name": "beam", "kind": "ctc", "beam_size": 10}]
    job = {
        "arms": ["B", "C", "CTC_stitched", "F"],
        "ownership_rule": "midpoint",
        "cells": [{"cell_id": "test", "target_frames": target, "context_frames": context}],
        "decoders": decoders,
        "arm_decoders": {"B": ["hybrid", "greedy", "beam"],
                         "C": ["hybrid", "greedy", "beam"],
                         "CTC_stitched": ["greedy", "beam"],
                         "F": ["hybrid", "greedy", "beam"]},
    }
    return pixels, row, job


def record_for(result, arm, decoder="hybrid", cell="test"):
    return next(record for record in result["records"]
                if (record["arm"], record["decoder"], record["cell_id"]) == (arm, decoder, cell))


class SlidingTests(unittest.TestCase):
    def test_balanced_partitions_cover_original_frames_once(self):
        for total in (1, 2, 24, 25, 26, 49, 50, 51, 74, 75, 76, 151):
            for target in (25, 50, 75):
                with self.subTest(total=total, target=target):
                    windows = make_sliding(total, target, 16)
                    count = math.ceil(total / target)
                    self.assertEqual(len(windows), count)
                    coverage = np.zeros(total, dtype=int)
                    widths = []
                    for index, window in enumerate(windows):
                        start = index * total // count
                        end = (index + 1) * total // count
                        self.assertEqual((window["target_start"], window["target_end"]), (start, end))
                        self.assertEqual(window["input_start"], max(0, start - 16))
                        self.assertEqual(window["input_end"], min(total, end + 16))
                        self.assertEqual(window["local_start"] + window["input_start"], start)
                        self.assertEqual(window["local_end"] + window["input_start"], end)
                        coverage[start:end] += 1
                        widths.append(end - start)
                    np.testing.assert_array_equal(coverage, np.ones(total, dtype=int))
                    self.assertLessEqual(max(widths) - min(widths), 1)

    def test_matched_inputs_caps_and_readout_geometry(self):
        pixels, row, job = fixture()
        runtime = FakeRuntime()
        result = evaluate_clip(runtime, pixels, row, job)
        left = record_for(result, "B")
        right = record_for(result, "C")
        self.assertEqual(left["target_schedule"], right["target_schedule"])
        self.assertEqual(left["joins"], [3, 7])
        self.assertEqual(left["window_count"], 3)
        expected = [pixels] + [pixels[item["input_start"]:item["input_end"]]
                               for item in left["target_schedule"]]
        self.assertEqual(len(runtime.inputs), len(expected))
        for actual, wanted in zip(runtime.inputs, expected):
            np.testing.assert_array_equal(actual, wanted)
        for bounded, contextual in zip(left["windows"], right["windows"]):
            geometry = bounded["geometry"]
            length = geometry["input_end"] - geometry["input_start"]
            self.assertEqual(bounded["geometry"], contextual["geometry"])
            self.assertEqual(bounded["input_pixel_sha256"], contextual["input_pixel_sha256"])
            self.assertEqual(bounded["output_cap"], length)
            self.assertEqual(contextual["output_cap"], length)
            self.assertEqual(bounded["decoded_state_frames"], geometry["actual_target_frames"])
            self.assertEqual(contextual["decoded_state_frames"], length)
            core = pixels[geometry["target_start"]:geometry["target_end"], 0, 0]
            calls = [call for call in runtime.calls if not call["native"]
                     and call["cap"] == length and np.array_equal(call["states"][:, 0], core)]
            self.assertEqual(len(calls), 1)
            context = pixels[geometry["input_start"]:geometry["input_end"]]
            np.testing.assert_array_equal(calls[0]["states"][:, 1], np.full(len(core), context.mean()))
            self.assertEqual(bounded["input_pixel_sha256"], hashlib.sha256(context.tobytes()).hexdigest())

    def test_stitched_logits_are_original_frame_order_without_context_duplicates(self):
        pixels, row, job = fixture(total=13)
        from inference_core import _Session

        class ContextLogitsRuntime(FakeRuntime):
            def encode(self, pixels):
                encoded = super().encode(pixels)
                logp = encoded['logp']
                logp[:, 1] += float(pixels.mean()) / 10
                logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
                return encoded

        class RecordingSession(_Session):
            stitched = []

            def decode(self, states, logp, decoder, cap, native=False):
                if states is logp:
                    self.stitched.append(logp.copy())
                return super().decode(states, logp, decoder, cap, native)

        with patch("sliding_inference._Session", RecordingSession):
            evaluate_clip(ContextLogitsRuntime(), pixels, row, job)
        expected_parts = []
        for window in make_sliding(len(pixels), 4, 2):
            context = pixels[window['input_start']:window['input_end']]
            logp = ContextLogitsRuntime().encode(context)['logp']
            expected_parts.append(logp[window['local_start']:window['local_end']])
        expected = np.concatenate(expected_parts, axis=0)
        full = ContextLogitsRuntime().encode(pixels)['logp']
        self.assertFalse(np.array_equal(expected, full))
        self.assertEqual(len(RecordingSession.stitched), 2)
        for actual in RecordingSession.stitched:
            np.testing.assert_array_equal(actual, expected)

    def test_ctc_seam_repetition_collapses_globally_but_preserves_blank_separation(self):
        for sequence, stitched_text in (([1, 1, 1, 1], "xin"), ([1, 0, 1, 1], "xin xin")):
            with self.subTest(sequence=sequence):
                pixels, row, job = fixture(total=4, target=2, context=1)
                pixels[:, 0, 0] = sequence
                result = evaluate_clip(FakeRuntime(), pixels, row, job)
                for decoder in ("greedy", "beam"):
                    self.assertEqual(record_for(result, "B", decoder)["hypothesis"], "xin xin")
                    self.assertEqual(record_for(result, "CTC_stitched", decoder)["hypothesis"], stitched_text)

    def test_projection_uses_own_tokens_and_half_open_midpoints(self):
        pixels, row, job = fixture(total=8, target=4, context=2)
        runtime = FakeRuntime()
        runtime.forced_tokens = [1, 2]
        alignments = []

        def align(logp, tokens, blank):
            alignments.append((tokens[:], len(logp), blank))
            return {"spans": [[0, 2], [2, 6]]}

        job["arm_decoders"]["B"] = ["hybrid"]
        job["arm_decoders"]["C"] = ["hybrid"]
        with patch("inference_core._alignment", return_value=(align, {"source": "cpu-test"})):
            result = evaluate_clip(runtime, pixels, row, job)
        contextual = record_for(result, "C")
        self.assertEqual([window["hypothesis"] for window in contextual["windows"]], ["xin", "chào"])
        self.assertEqual(contextual["hypothesis"], "xin chào")
        self.assertEqual(record_for(result, "B")["hypothesis"], "xin chào xin chào")
        self.assertEqual(alignments, [([1, 2], 6, 0), ([1, 2], 6, 0)])
        selected = contextual["windows"][1]["allocation"]["selected_units"]
        self.assertEqual(selected[0]["start"], 2)
        self.assertEqual(selected[0]["end"], 6)

    def test_reference_is_accessed_once_after_all_inference_and_never_words(self):
        pixels, row, job = fixture()
        events = []

        class Runtime(FakeRuntime):
            def encode(self, pixels):
                events.append("encode")
                return super().encode(pixels)

            def _decode(self, *args, **kwargs):
                events.append("decode")
                return super()._decode(*args, **kwargs)

        class GuardedRow(dict):
            def __getitem__(self, key):
                if key == "words":
                    raise AssertionError("Whole-clip inference must never inspect reference timings")
                if key == "reference":
                    events.append("reference")
                return super().__getitem__(key)

        result = evaluate_clip(Runtime(), pixels, GuardedRow(row), job)
        self.assertEqual(events[-1], "reference")
        self.assertEqual(events.count("reference"), 1)
        alternate = dict(row, reference="unrelated transcript", words=object())
        replay = evaluate_clip(FakeRuntime(), pixels, alternate, job)
        self.assertEqual([record["hypothesis"] for record in result["records"]],
                         [record["hypothesis"] for record in replay["records"]])
        self.assertTrue(result["diagnostics"]["references_accessed_after_all_predictions"])

    def test_one_window_short_clip_and_native_full_clip(self):
        for total in (1, 7, 25):
            with self.subTest(total=total):
                pixels, row, job = fixture(total=total, target=25, context=16)
                runtime = FakeRuntime()
                result = evaluate_clip(runtime, pixels, row, job)
                self.assertEqual(result["diagnostics"]["encode_calls"], 1)
                self.assertEqual(result["diagnostics"]["encoded_frames"], total)
                self.assertEqual(result["diagnostics"]["alignment_calls"], 0)
                for record in result["records"]:
                    self.assertEqual(record["window_count"], 1)
                    self.assertEqual(record["joins"], [])
                    self.assertEqual(record["frames_encoded"], total)
                for decoder in ("hybrid", "greedy", "beam"):
                    self.assertEqual(record_for(result, "B", decoder)["hypothesis"],
                                     record_for(result, "C", decoder)["hypothesis"])
                    self.assertEqual(record_for(result, "B", decoder)["hypothesis"],
                                     record_for(result, "F", decoder)["hypothesis"])
                native_calls = [call for call in runtime.calls if call["native"]]
                self.assertEqual(len(native_calls), 1)
                self.assertEqual(len(native_calls[0]["states"]), total)

    def test_frame_accounting_timing_and_json_across_cells(self):
        pixels, row, job = fixture()
        job["cells"].append({"cell_id": "short", "target_frames": 50, "context_frames": 16})
        runtime = FakeRuntime()
        result = evaluate_clip(runtime, pixels, row, job)
        self.assertEqual(result["diagnostics"]["encoded_frames"], sum(len(value) for value in runtime.inputs))
        self.assertGreaterEqual(result["diagnostics"]["total_seconds"], result["diagnostics"]["prediction_seconds"])
        for record in result["records"]:
            expected = len(pixels) if record["arm"] == "F" else sum(
                window["input_end"] - window["input_start"] for window in record["target_schedule"])
            self.assertEqual(record["frames_encoded"], expected)
            timing = record["timing"]
            for field in ("encode_seconds", "decode_seconds", "alignment_seconds", "projection_seconds"):
                self.assertTrue(math.isfinite(timing[field]))
                self.assertGreaterEqual(timing[field], 0)
            self.assertIn("shared GPU load", timing["semantics"])
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_failed_self_alignment_is_retained_with_explicit_fallback(self):
        pixels, row, job = fixture()
        runtime = FakeRuntime()
        runtime.forced_tokens = [1] * 20
        result = evaluate_clip(runtime, pixels, row, job)
        projected = record_for(result, 'C')
        self.assertEqual(len(projected['windows']), 3)
        for window in projected['windows']:
            self.assertEqual(window['allocation']['fallback']['method'], 'uniform_token')
            self.assertTrue(window['allocation']['fallback']['reason'])
        self.assertIn('counts', projected)

    def test_empty_references_keep_insertions_and_cer_uses_full_transcript(self):
        pixels, row, job = fixture(total=8, target=25)
        runtime = FakeRuntime()
        runtime.metric = 'cer'
        row['metric'] = 'cer'
        row['reference'] = ''
        result = evaluate_clip(runtime, pixels, row, job)
        full = record_for(result, 'F')
        self.assertEqual(full['reference'], '')
        self.assertEqual(full['counts']['units'], 0)
        self.assertEqual(full['counts']['insertions'], len('xinchàoniño世界'))
        self.assertEqual(full['counts']['errors'], full['counts']['insertions'])

    def test_invalid_contracts_fail_before_encoding(self):
        pixels, row, job = fixture()
        invalid = []
        for field in ("cells", "decoders"):
            changed = copy.deepcopy(job)
            changed[field] = []
            invalid.append(changed)
        changed = copy.deepcopy(job)
        changed["decoders"].append(copy.deepcopy(changed["decoders"][0]))
        invalid.append(changed)
        changed = copy.deepcopy(job)
        changed["cells"].append(copy.deepcopy(changed["cells"][0]))
        invalid.append(changed)
        changed = copy.deepcopy(job)
        changed["cells"][0]["target_frames"] = 0
        invalid.append(changed)
        for names in ([], ["greedy", "greedy"], ["unknown"], ["hybrid"]):
            changed = copy.deepcopy(job)
            changed["arm_decoders"]["CTC_stitched"] = names
            invalid.append(changed)
        for index, changed in enumerate(invalid):
            with self.subTest(case=index):
                runtime = FakeRuntime()
                with self.assertRaises(ValueError):
                    evaluate_clip(runtime, pixels, row, changed)
                self.assertEqual(runtime.inputs, [])

    def test_pixel_and_manifest_validation(self):
        pixels, row, job = fixture()
        for payload, metadata in ((pixels.astype(np.float32), row),
                                  (pixels[:0], dict(row, frames=0)),
                                  (pixels, dict(row, frames=12)),
                                  (pixels, dict(row, metric="cer"))):
            with self.subTest(shape=payload.shape, metric=metadata["metric"]):
                runtime = FakeRuntime()
                with self.assertRaises(ValueError):
                    evaluate_clip(runtime, payload, metadata, job)
                self.assertEqual(runtime.inputs, [])

    def test_sparse_queries_are_distinct_geometry_only(self):
        self.assertEqual(frozen_geometry_queries(12, target=50, count=5), [(0, 12)])
        self.assertEqual(frozen_geometry_queries(100, target=50, count=3), [(0, 50), (25, 75), (50, 100)])
        self.assertEqual(frozen_geometry_queries(101, target=50), [(25, 75)])


if __name__ == "__main__":
    unittest.main()
