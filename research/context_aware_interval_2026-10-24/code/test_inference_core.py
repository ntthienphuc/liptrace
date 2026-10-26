"""CPU integration tests of inference contracts using an observable fake model."""

import copy
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from confirmation_common import normalize
from inference_core import _Session, _input, _position, evaluate_clip
from geometry import make_interval


HERE = Path(__file__).resolve().parent


class FakeRuntime:
    blank_id = 0
    eos_id = 5
    metric = "wer"
    vocabulary = ["<blank>", "▁xin", "▁chào", "▁niño", "▁世界", "<eos>"]
    root = HERE.parent
    registry = {"alignment_source": "code/projection.py"}

    def __init__(self):
        self.inputs = []
        self.calls = []
        self.forced_tokens = None
        self.forced_text = None

    def text(self, tokens):
        return normalize("".join(self.vocabulary[token] for token in tokens).replace("▁", " "))

    def encode(self, pixels):
        self.inputs.append(pixels.copy())
        values = pixels.reshape(len(pixels), -1)[:, 0].astype(np.float32)
        # Context dependence ensures cropped and contextual states really differ.
        states = np.column_stack((values, np.full(len(values), values.mean(), dtype=np.float32)))
        logp = np.full((len(pixels), len(self.vocabulary)), -8.0, dtype=np.float32)
        for frame, value in enumerate(values):
            logp[frame, int(value) % 5] = 0.0
        logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
        return {"states": states, "logp": logp, "metadata": {"frames": len(pixels)}}

    def _decode(self, states, cap, native, **settings):
        self.calls.append({"states": states.copy(), "cap": cap, "native": native, **settings})
        tokens = self.forced_tokens if self.forced_tokens is not None else [1, 2, 3, 4]
        text = self.forced_text if self.forced_text is not None else self.text(tokens)
        return {"tokens": list(tokens), "text": text, "score": -1.0,
                "trace": {"cap_reached": False, "native": native}, "cap": cap, "seconds": 0.0}

    def decode(self, states, ctc_weight=0.5, cap=None, beam_size=40, penalty=0):
        return self._decode(states, cap, False, ctc_weight=ctc_weight, beam_size=beam_size, penalty=penalty)

    def decode_native(self, states, ctc_weight=0.5, beam_size=40, penalty=0):
        return self._decode(states, len(states), True,
                            ctc_weight=ctc_weight, beam_size=beam_size, penalty=penalty)


def fixture():
    pixels = np.arange(12, dtype=np.uint8)[:, None, None]
    row = {"clip_id": "clip-α", "source_id": "speaker_unknown", "frames": len(pixels),
           "metric": "wer", "fps": 25.0, "reference": "xin chào niño 世界",
           "words": [{"text": text, "start": index * 3 / 25, "end": (index + 1) * 3 / 25}
                     for index, text in enumerate(["xin", "chào", "niño", "世界"])]}
    job = {"cells": [{"cell_id": "middle", "target_frames": 4, "context_frames": 2,
                       "position": "center"}],
           "decoders": [{"name": "h05", "kind": "hybrid", "ctc_weight": 0.5}],
           "arms": ["A", "B", "C", "U_word", "U_token", "F"]}
    return pixels, row, job


class InferenceTests(unittest.TestCase):
    def test_context_caps_and_exact_encoder_inputs(self):
        pixels, row, job = fixture()
        runtime = FakeRuntime()
        output = evaluate_clip(runtime, pixels, row, job)
        records = {record["arm"]: record for record in output["records"]}
        self.assertEqual([records[arm]["output_cap"] for arm in job["arms"]], [4, 8, 8, 8, 8, 12])
        self.assertEqual([records[arm]["decoded_state_frames"] for arm in job["arms"]], [4, 4, 8, 8, 8, 12])
        self.assertEqual([len(value) for value in runtime.inputs], [4, 8, 12])
        np.testing.assert_array_equal(runtime.inputs[0], pixels[4:8])
        np.testing.assert_array_equal(runtime.inputs[1], pixels[2:10])
        self.assertEqual(output["diagnostics"]["decode_calls"], 4)
        self.assertEqual(output["diagnostics"]["encoded_frames"], 24)
        for arm in ["U_word", "U_token"]:
            self.assertEqual(records[arm]["tokens"], records["C"]["tokens"])
            self.assertEqual(records[arm]["native_text"], records["C"]["native_text"])
            self.assertTrue(records[arm]["timing"]["decoder_cache_hit"])
        self.assertIn("sha256", output["diagnostics"]["alignment_source"])
        json.dumps(output, ensure_ascii=False, allow_nan=False)

    def test_zero_context_identity_keeps_native_text(self):
        pixels, row, job = fixture()
        job["cells"][0]["context_frames"] = 0
        job["arms"].remove("F")
        runtime = FakeRuntime()
        runtime.forced_text = "xin chào niño 世界 boundary"
        output = evaluate_clip(runtime, pixels, row, job)
        self.assertEqual({record["hypothesis"] for record in output["records"]}, {runtime.forced_text})
        self.assertEqual(output["diagnostics"]["encode_calls"], 1)
        self.assertEqual(output["diagnostics"]["decode_calls"], 1)
        self.assertEqual(output["diagnostics"]["alignment_calls"], 0)

    def test_short_whole_input_and_full_endpoint(self):
        pixels, row, job = fixture()
        job["cells"][0]["target_frames"] = 100
        job["full_clip"] = True
        output = evaluate_clip(FakeRuntime(), pixels, row, job)
        records = output["records"]
        self.assertEqual(len(records), 7)
        self.assertEqual({record["hypothesis"] for record in records}, {"xin chào niño 世界"})
        self.assertEqual(records[-1]["reference"], row["reference"])
        self.assertTrue(records[-1]["timing"]["decoder_cache_hit"])
        self.assertEqual(output["diagnostics"]["encode_calls"], 1)

    def test_interventions_preserve_target_and_lengths(self):
        pixels, _, _ = fixture()
        base = make_interval(12, 4, 3, "center")
        for intervention in ["real", "repeat", "left", "right", "reverse"]:
            context, geometry = _input(pixels, base, intervention)
            lower = geometry["local_start"]
            upper = geometry["local_end"]
            np.testing.assert_array_equal(context[lower:upper], pixels[4:8])
            self.assertEqual(len(context), 7 if intervention in ("left", "right") else 10)
            if intervention == "repeat":
                self.assertEqual(context[:, 0, 0].tolist(), [4, 4, 4, 4, 5, 6, 7, 7, 7, 7])
            if intervention == "reverse":
                self.assertEqual(context[:, 0, 0].tolist(), [3, 2, 1, 4, 5, 6, 7, 10, 9, 8])
        np.testing.assert_array_equal(pixels[:, 0, 0], np.arange(12))

    def test_reference_access_occurs_after_every_prediction(self):
        pixels, row, job = fixture()
        runtime = FakeRuntime()

        class GuardedRow(dict):
            def __getitem__(self, key):
                if key in ("reference", "words"):
                    if len(runtime.calls) != 4:
                        raise AssertionError("Reference accessed before all predictions")
                return super().__getitem__(key)

        result = evaluate_clip(runtime, pixels, GuardedRow(row), job)
        alternate = copy.deepcopy(row)
        alternate["reference"] = "unrelated reference"
        alternate["words"] = [{"text": "different", "start": 0, "end": 12 / 25}]
        replay = evaluate_clip(FakeRuntime(), pixels, alternate, job)
        self.assertEqual([record["hypothesis"] for record in result["records"]],
                         [record["hypothesis"] for record in replay["records"]])

    def test_alignment_failure_retained_and_marked(self):
        pixels, row, job = fixture()
        runtime = FakeRuntime()
        runtime.forced_tokens = [1] * 13
        output = evaluate_clip(runtime, pixels, row, job)
        projected = [record for record in output["records"] if record["arm"] in ("C", "F")]
        self.assertEqual(len(projected), 2)
        for record in projected:
            self.assertEqual(record["allocation_fallback"]["method"], "uniform_token")
            self.assertIn("counts", record)
        self.assertEqual(output["diagnostics"]["allocation_fallback_records"], 2)

    def test_greedy_collapse_before_removing_blank_and_eos(self):
        runtime = FakeRuntime()
        sequence = [1, 1, 0, 1, 5, 1, 2, 2]
        logp = np.full((len(sequence), 6), -9.0)
        logp[np.arange(len(sequence)), sequence] = 0.0
        session = _Session(runtime)
        result, _ = session.decode(np.zeros((len(sequence), 2)), logp,
                                   {"name": "g", "kind": "greedy"}, len(sequence))
        self.assertEqual(result["tokens"], [1, 1, 1, 2])
        self.assertEqual(result["text"], "xin xin xin chào")

    def test_ctc_is_summed_prefix_beam_not_argmax_path(self):
        runtime = FakeRuntime()
        probabilities = np.array([[0.6, 0.4, 0, 0, 0, 0], [0.6, 0.4, 0, 0, 0, 0]])
        with np.errstate(divide="ignore"):
            logp = np.log(probabilities)
        session = _Session(runtime)
        result, _ = session.decode(np.zeros((2, 2)), logp,
                                   {"name": "q", "kind": "ctc", "token_prune": None}, 2)
        self.assertEqual(result["tokens"], [1])
        self.assertAlmostEqual(np.exp(result["score"]), 0.64)
        self.assertEqual(result["trace"]["algorithm"], "ctc_prefix_beam")
        self.assertEqual(runtime.calls, [])

    def test_unicode_projected_words_and_character_scoring(self):
        pixels, row, job = fixture()
        job["arms"] = ["U_word"]
        result = evaluate_clip(FakeRuntime(), pixels, row, job)
        self.assertEqual(result["records"][0]["hypothesis"], "chào niño")
        runtime = FakeRuntime()
        runtime.metric = "cer"
        row["metric"] = "cer"
        result = evaluate_clip(runtime, pixels, row, job)
        self.assertEqual(result["records"][0]["counts"]["errors"], 0)

    def test_hash_position_uses_only_clip_id(self):
        clip_id = "nguồn-世界"
        index = int(hashlib.sha256(clip_id.encode("utf-8")).hexdigest(), 16) % 3
        self.assertEqual(_position(clip_id, "hash"), ("start", "center", "end")[index])

    def test_cache_does_not_mix_caps_settings_or_context(self):
        pixels, row, job = fixture()
        job["arms"] = ["A", "B", "C"]
        other = dict(job["cells"][0], cell_id="repeat", intervention="repeat")
        job["cells"].append(other)
        job["decoders"].append({"name": "h09", "kind": "hybrid", "ctc_weight": 0.9})
        runtime = FakeRuntime()
        output = evaluate_clip(runtime, pixels, row, job)
        self.assertEqual(output["diagnostics"]["encode_calls"], 3)
        self.assertEqual({call["ctc_weight"] for call in runtime.calls}, {0.5, 0.9})
        target_calls = [call for call in runtime.calls if len(call["states"]) == 4]
        self.assertEqual({call["cap"] for call in target_calls}, {4, 8})

    def test_missing_alignment_registry_fails_closed(self):
        pixels, row, job = fixture()
        runtime = FakeRuntime()
        runtime.registry = {}
        with self.assertRaisesRegex(ValueError, "alignment_source"):
            evaluate_clip(runtime, pixels, row, job)


if __name__ == "__main__":
    unittest.main()
