"""Reference-blind bounded inference, prediction alignment, and delayed scoring."""

import ast
import hashlib
import json
import math
import time

import numpy as np

from confirmation_common import ROOT, edit_counts, normalize, resolve
from ctc_decode import prefix_beam
from geometry import allocation, make_interval, reference_for_interval, select_units


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(value)


def _identity(value):
    array = _array(value)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    return (str(array.dtype), tuple(array.shape), digest)


def _sync(runtime):
    synchronize = getattr(runtime, "_synchronize", None)
    if synchronize is not None:
        synchronize()


def _alignment(runtime):
    registry = runtime.registry
    if "alignment_source" not in registry:
        raise ValueError("model_registry.json must register alignment_source before projection")
    path = resolve(registry["alignment_source"], getattr(runtime, "root", ROOT))
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    expected = registry.get("alignment_source_sha256")
    if expected is not None and digest != expected:
        raise ValueError("Historical alignment source SHA-256 changed")
    tree = ast.parse(source.decode("utf-8-sig"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "align_prediction"]
    if len(nodes) != 1:
        raise ValueError("Expected one historical align_prediction definition")
    scope = {"np": np, "__name__": "confirmation_prediction_alignment"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["align_prediction"], {"source": registry["alignment_source"], "sha256": digest}


def _uniform(count, frames):
    if not count:
        return []
    return [[index * frames / count, (index + 1) * frames / count] for index in range(count)]


def _position(clip_id, requested):
    if requested != "hash":
        return requested
    number = int(hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest(), 16)
    return ("start", "center", "end")[number % 3]


def _input(pixels, geometry, intervention):
    """Left/right means only that context side; repeat keeps both lengths."""
    geometry = dict(geometry)
    if intervention == "left":
        geometry["input_end"] = geometry["target_end"]
        geometry["right_context"] = 0
    elif intervention == "right":
        geometry["input_start"] = geometry["target_start"]
        geometry["left_context"] = 0
    elif intervention not in ("real", "repeat", "reverse"):
        raise ValueError(f"Unknown intervention: {intervention}")
    geometry["local_start"] = geometry["target_start"] - geometry["input_start"]
    geometry["local_end"] = geometry["target_end"] - geometry["input_start"]
    context = pixels[geometry["input_start"]:geometry["input_end"]].copy()
    lower = geometry["local_start"]
    upper = geometry["local_end"]
    if intervention == "repeat":
        context[:lower] = context[lower]
        context[upper:] = context[upper - 1]
    elif intervention == "reverse":
        context[:lower] = context[:lower][::-1].copy()
        context[upper:] = context[upper:][::-1].copy()
    target = pixels[geometry["target_start"]:geometry["target_end"]]
    if not np.array_equal(context[lower:upper], target):
        raise AssertionError("Context intervention changed target pixels")
    geometry["intervention"] = intervention
    geometry["target_pixels_identical"] = True
    return np.ascontiguousarray(context), geometry


class _Session:
    """Clip-local caches; input bytes and actual states define every reuse."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.encodings = {}
        self.decodings = {}
        self.alignments = {}
        self.aligner = None
        self.alignment_identity = None
        self.encode_calls = 0
        self.decode_calls = 0
        self.alignment_calls = 0
        self.encoded_frames = 0

    def encode(self, pixels):
        key = _identity(pixels)
        cached = key in self.encodings
        if not cached:
            _sync(self.runtime)
            began = time.perf_counter()
            result = dict(self.runtime.encode(pixels))
            _sync(self.runtime)
            result["measured_seconds"] = time.perf_counter() - began
            if len(result["states"]) != len(pixels) or len(result["logp"]) != len(pixels):
                raise ValueError("Encoder time mapping must be one state/logit row per input frame")
            logp = np.asarray(result["logp"])
            if logp.shape != (len(pixels), len(self.runtime.vocabulary)):
                raise ValueError("Unexpected CTC shape")
            if not np.isfinite(logp).all():
                raise FloatingPointError("Nonfinite encoder CTC probabilities")
            result["pixel_sha256"] = key[-1]
            self.encodings[key] = result
            self.encode_calls += 1
            self.encoded_frames += len(pixels)
        return self.encodings[key], cached

    def decode(self, states, logp, decoder, cap, native=False):
        kind = decoder["kind"]
        settings = {
            "kind": kind,
            "ctc_weight": decoder.get("ctc_weight", 0.5),
            "beam_size": decoder.get("beam_size", 40),
            "penalty": decoder.get("penalty", 0),
            "blank_bias": decoder.get("blank_bias", 0),
            "token_prune": decoder.get("token_prune", 64),
            "cap": cap,
            "native": native and kind == "hybrid",
        }
        key = (_identity(states), _identity(logp), json.dumps(settings, sort_keys=True, allow_nan=False))
        cached = key in self.decodings
        if cached:
            return self.decodings[key], True
        values = np.array(logp, dtype=np.float64, copy=True)
        if settings["blank_bias"]:
            if kind == "hybrid":
                raise ValueError("Hybrid blank_bias is unsupported by the native runtime")
            values[:, self.runtime.blank_id] += settings["blank_bias"]
            maximum = values.max(axis=1, keepdims=True)
            values -= maximum + np.log(np.exp(values - maximum).sum(axis=1, keepdims=True))
        _sync(self.runtime)
        began = time.perf_counter()
        if kind == "hybrid":
            arguments = {name: settings[name] for name in ("ctc_weight", "beam_size", "penalty")}
            if native:
                result = dict(self.runtime.decode_native(states, **arguments))
            else:
                result = dict(self.runtime.decode(states, cap=cap, **arguments))
        elif kind == "greedy":
            path = values.argmax(axis=1).tolist()
            tokens = []
            previous = None
            for token in path:
                if token != previous and token not in (self.runtime.blank_id, self.runtime.eos_id):
                    tokens.append(int(token))
                previous = token
            score = float(values[np.arange(len(path)), path].sum())
            result = {"tokens": tokens, "text": self.runtime.text(tokens), "score": score,
                      "trace": {"implementation": "ctc_greedy_collapse_then_lexical"}, "cap": cap}
        elif kind == "ctc":
            beam = prefix_beam(values, beam_size=settings["beam_size"], blank=self.runtime.blank_id,
                               eos=self.runtime.eos_id, token_prune=settings["token_prune"])
            result = {"tokens": beam["tokens"], "text": self.runtime.text(beam["tokens"]),
                      "score": beam["score"], "trace": beam, "cap": cap}
        else:
            raise ValueError(f"Unknown decoder kind: {kind}")
        _sync(self.runtime)
        result["measured_seconds"] = time.perf_counter() - began
        result["settings"] = settings
        result["tokens"] = [int(token) for token in result["tokens"]]
        if not math.isfinite(float(result["score"])):
            raise FloatingPointError("Nonfinite decoded hypothesis score")
        if any(token in (self.runtime.blank_id, self.runtime.eos_id) for token in result["tokens"]):
            raise ValueError("Decoded tokens must be lexical")
        self.decodings[key] = result
        self.decode_calls += 1
        return result, False

    def align(self, logp, tokens):
        key = (_identity(logp), tuple(tokens))
        if key in self.alignments:
            return self.alignments[key], True
        if self.aligner is None:
            self.aligner, self.alignment_identity = _alignment(self.runtime)
        began = time.perf_counter()
        fallback = None
        try:
            result = self.aligner(logp, tokens, blank=self.runtime.blank_id)
            if result is None:
                raise ValueError("CTC predicted sequence has no feasible alignment")
            spans = result["spans"]
            if len(spans) != len(tokens):
                raise ValueError("CTC alignment token/span mismatch")
            if any(not (0 <= start < end <= len(logp)) for start, end in spans):
                raise ValueError("CTC alignment has invalid span")
        except Exception as error:
            spans = _uniform(len(tokens), len(logp))
            fallback = {"method": "uniform_token", "reason": f"{type(error).__name__}: {error}"}
        output = {"spans": spans, "fallback": fallback, "seconds": time.perf_counter() - began}
        self.alignments[key] = output
        self.alignment_calls += 1
        return output, False


def _project(session, decoded, logp, start, end, arm, metric, rule):
    began = time.perf_counter()
    details = {"fallback": None, "alignment_seconds": 0.0, "alignment_cache_hit": False}
    if start == 0 and end == len(logp):
        text = decoded["text"]
        details["full_input_ownership"] = True
    elif arm == "U_word":
        words = decoded["text"].split()
        spans = _uniform(len(words), len(logp))
        units = [{"text": word, "start": lower, "end": upper}
                 for word, (lower, upper) in zip(words, spans)]
        selected = select_units(units, start, end, rule)
        text = normalize(" ".join(unit["text"] for unit in selected))
        details["selected_units"] = selected
    else:
        if arm == "U_token":
            spans = _uniform(len(decoded["tokens"]), len(logp))
        else:
            alignment, cached = session.align(logp, decoded["tokens"])
            spans = alignment["spans"]
            details["fallback"] = alignment["fallback"]
            details["alignment_seconds"] = alignment["seconds"]
            details["alignment_cache_hit"] = cached
        allocated = allocation(decoded["tokens"], spans, session.runtime.vocabulary,
                               start, end, metric=metric, rule=rule)
        text = allocated["text"]
        details["selected_units"] = allocated["selected_units"]
    details["projection_seconds"] = time.perf_counter() - began
    return text, details


def evaluate_clip(runtime, pixels, row, job):
    """Produce every hypothesis first, then access references solely for scoring.

    A encodes I. B decodes I states encoded within J with cap=len(J). C and
    uniform controls share one J hypothesis. F uses the native full-clip decoder.
    CTC/greedy controls consume all supplied states and do not truncate to cap.
    Left/right interventions remove the unavailable side, while reverse reverses
    each context margin independently. Repeat uses the nearest target edge frame.
    All caching is local to this call, so model identity cannot cross runtimes.
    """
    began = time.perf_counter()
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or len(pixels) < 1:
        raise ValueError("Expected a nonempty uint8 pixel array")
    if len(pixels) != row["frames"]:
        raise ValueError("Manifest frame count differs from pixel payload")
    metric = row["metric"]
    if metric != runtime.metric:
        raise ValueError("Manifest and model metrics differ")
    if metric not in ("wer", "cer"):
        raise ValueError("Unsupported metric")
    arms = list(job.get("arms", ["A", "B", "C", "U_word", "U_token", "F"]))
    if len(set(arms)) != len(arms) or any(arm not in ("A", "B", "C", "U_word", "U_token", "F")
                                          for arm in arms):
        raise ValueError("Unknown or duplicate arm")
    decoders = job["decoders"]
    if not decoders or len({decoder["name"] for decoder in decoders}) != len(decoders):
        raise ValueError("Decoder names must be nonempty and unique")
    allowed = job.get("arm_decoders", {})
    names = {decoder["name"] for decoder in decoders}
    if any(arm not in (*arms, "F") or not values or not set(values).issubset(names)
           for arm, values in allowed.items()):
        raise ValueError("Invalid frozen arm-specific decoder selection")
    cells = job["cells"]
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("Cell IDs must be unique")
    rule = job.get("ownership_rule", "midpoint")
    if rule not in ("midpoint", "overlap"):
        raise ValueError("Unknown ownership rule")
    session = _Session(runtime)
    records = []

    def append(cell_id, arm, decoder, geometry, encoded, encode_hit, states, logp,
               cap, start=None, end=None, native=False, full_clip=False):
        decoded, decode_hit = session.decode(states, logp, decoder, cap, native=native)
        projection = {"fallback": None, "alignment_seconds": 0.0, "projection_seconds": 0.0,
                      "alignment_cache_hit": False}
        hypothesis = decoded["text"]
        if start is not None:
            hypothesis, projection = _project(session, decoded, logp, start, end, arm, metric, rule)
        records.append({
            "cell_id": cell_id,
            "arm": arm,
            "decoder": decoder["name"],
            "hypothesis": hypothesis,
            "geometry": dict(geometry),
            "tokens": list(decoded["tokens"]),
            "native_text": decoded["text"],
            "score": float(decoded["score"]),
            "decoder_settings": dict(decoded["settings"]),
            "decode_trace": dict(decoded.get("trace", {})),
            "allocation_fallback": projection["fallback"],
            "allocation": projection,
            "input_pixel_sha256": encoded["pixel_sha256"],
            "input_frames": len(encoded["states"]),
            "decoded_state_frames": len(states),
            "output_cap": cap,
            "cap_formula": "len(recognizer_input): A=len(I), B=C=U=len(J), F=len(full_clip)",
            "ownership_rule": rule,
            "unrestricted_context": arm == "F",
            "full_clip_endpoint": full_clip,
            "timing": {
                "encode_seconds": encoded["measured_seconds"],
                "decode_seconds": decoded["measured_seconds"],
                "alignment_seconds": projection["alignment_seconds"],
                "projection_seconds": projection["projection_seconds"],
                "encoder_cache_hit": encode_hit,
                "decoder_cache_hit": decode_hit,
                "alignment_cache_hit": projection["alignment_cache_hit"],
                "cost_semantics": "component standalone time; cache flags identify actual reuse",
            },
        })

    for cell in cells:
        position = _position(row["clip_id"], cell.get("position", "hash"))
        geometry = make_interval(len(pixels), cell["target_frames"], cell["context_frames"], position)
        context, geometry = _input(pixels, geometry, cell.get("intervention", "real"))
        geometry["requested_position"] = cell.get("position", "hash")
        geometry["position_hash_rule"] = "sha256(utf8(clip_id)) modulo 3: start,center,end"
        target = pixels[geometry["target_start"]:geometry["target_end"]]
        for decoder in decoders:
            for arm in arms:
                if decoder["name"] not in allowed.get(arm, names):
                    continue
                if arm == "A":
                    encoded, cached = session.encode(target)
                    append(cell["cell_id"], arm, decoder, geometry, encoded, cached,
                           encoded["states"], encoded["logp"], len(target))
                elif arm == "F":
                    encoded, cached = session.encode(pixels)
                    append(cell["cell_id"], arm, decoder, geometry, encoded, cached,
                           encoded["states"], encoded["logp"], len(pixels),
                           geometry["target_start"], geometry["target_end"], native=True)
                else:
                    encoded, cached = session.encode(context)
                    lower = geometry["local_start"]
                    upper = geometry["local_end"]
                    if arm == "B":
                        append(cell["cell_id"], arm, decoder, geometry, encoded, cached,
                               encoded["states"][lower:upper], encoded["logp"][lower:upper], len(context))
                    else:
                        append(cell["cell_id"], arm, decoder, geometry, encoded, cached,
                               encoded["states"], encoded["logp"], len(context), lower, upper)
    if job.get("full_clip", False):
        if any(cell["cell_id"] == "full_clip" for cell in cells):
            raise ValueError("full_clip is a reserved cell ID when the full clip endpoint is enabled")
        geometry = make_interval(len(pixels), len(pixels), 0, "start")
        geometry["intervention"] = "real"
        for decoder in decoders:
            if decoder["name"] not in allowed.get("F", names):
                continue
            encoded, cached = session.encode(pixels)
            append("full_clip", "F", decoder, geometry, encoded, cached,
                   encoded["states"], encoded["logp"], len(pixels), 0, len(pixels),
                   native=True, full_clip=True)
    prediction_seconds = time.perf_counter() - began
    # This boundary is intentionally after every requested model/alignment call.
    for record in records:
        geometry = record["geometry"]
        if (record["full_clip_endpoint"] or
                (geometry["target_start"] == 0 and geometry["target_end"] == len(pixels))):
            reference = normalize(row["reference"])
        else:
            reference = reference_for_interval(row["words"], geometry["target_start"],
                                               geometry["target_end"], row["fps"], rule, metric)
        record["reference"] = reference
        record["counts"] = edit_counts(reference, record["hypothesis"], metric)
    diagnostics = {
        "reference_used_for_inference": False,
        "references_accessed_after_all_predictions": True,
        "encode_calls": session.encode_calls,
        "decode_calls": session.decode_calls,
        "alignment_calls": session.alignment_calls,
        "encoded_frames": session.encoded_frames,
        "alignment_source": session.alignment_identity,
        "allocation_fallback_records": sum(record["allocation_fallback"] is not None for record in records),
        "prediction_seconds": prediction_seconds,
        "total_seconds": time.perf_counter() - began,
        "cache_scope": "one clip, one immutable runtime; exact pixels/states/logp/settings/cap",
        "interventions": {
            "repeat": "each context margin repeats its nearest target boundary frame",
            "left": "retain left context; remove right context",
            "right": "retain right context; remove left context",
            "reverse": "reverse each context margin independently; preserve target",
        },
    }
    return {"records": records, "diagnostics": diagnostics}
