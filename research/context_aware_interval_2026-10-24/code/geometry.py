"""Half-open frame geometry and timestamp-only lexical ownership.

References carry measured word timestamps in seconds. Predicted token spans are
in frames of the permitted recognizer input. Multi-character tokens and reference
words share their observed span; no independent character timings are invented.
"""

import math
import unicodedata
from numbers import Integral, Real

from confirmation_common import normalize


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _coordinate(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real coordinate")
    return float(value)


def _bounds(start, end):
    start = _coordinate(start, "start")
    end = _coordinate(end, "end")
    if start < 0 or end < start:
        raise ValueError("Bounds must satisfy 0 <= start <= end")
    return start, end


def _metric(metric):
    if metric not in ("wer", "cer"):
        raise ValueError("metric must be wer or cer")


def _geometry(total, nominal, context, start, end, position):
    lower = max(0, start - context)
    upper = min(total, end + context)
    return {
        "target_start": start,
        "target_end": end,
        "input_start": lower,
        "input_end": upper,
        "local_start": start - lower,
        "local_end": end - lower,
        "left_context": start - lower,
        "right_context": upper - end,
        "nominal_target_frames": nominal,
        "actual_target_frames": end - start,
        "position": position,
    }


def make_interval(total_frames, target_frames, context_frames, position="center"):
    """Center odd residuals with floor division, and clip real context at edges."""
    total = _integer(total_frames, "total_frames", 1)
    nominal = _integer(target_frames, "target_frames", 1)
    context = _integer(context_frames, "context_frames")
    target = min(total, nominal)
    if position not in ("start", "center", "end"):
        raise ValueError("position must be start, center or end")
    starts = {"start": 0, "center": (total - target) // 2, "end": total - target}
    start = starts[position]
    return _geometry(total, nominal, context, start, start + target, position)


def make_sliding(total_frames, target_frames, context_frames):
    """Return K=ceil(T/L) balanced targets with exact, unique frame coverage."""
    total = _integer(total_frames, "total_frames", 1)
    nominal = _integer(target_frames, "target_frames", 1)
    context = _integer(context_frames, "context_frames")
    count = (total + nominal - 1) // nominal
    windows = []
    for index in range(count):
        start = index * total // count
        end = (index + 1) * total // count
        window = _geometry(total, nominal, context, start, end, "sliding")
        window["window_index"] = index
        windows.append(window)
    return windows


def select_units(units, start, end, rule="midpoint"):
    """Preserve lexical order/metadata; exact midpoint ties belong to the right."""
    start, end = _bounds(start, end)
    if rule not in ("midpoint", "overlap"):
        raise ValueError("rule must be midpoint or overlap")
    selected = []
    for unit in units:
        lower, upper = _bounds(unit["start"], unit["end"])
        if lower == upper:
            raise ValueError("Unit spans must have positive duration")
        if not isinstance(unit.get("text"), str):
            raise ValueError("Unit text must be a string")
        midpoint = lower + (upper - lower) / 2
        owned = start <= midpoint < end
        if rule == "overlap":
            owned = start < end and lower < end and upper > start
        if owned:
            selected.append(dict(unit))
    return selected


def _render(units, metric):
    text = normalize(" ".join(unit["text"] for unit in units))
    if metric == "cer":
        return "".join(character for character in text if not character.isspace())
    return text


def reference_for_interval(words, start, end, fps=25.0, rule="midpoint", metric="wer"):
    """Select measured reference words; CER retains their shared time ownership."""
    _metric(metric)
    fps = _coordinate(fps, "fps")
    if fps <= 0:
        raise ValueError("fps must be positive")
    units = []
    for word in words:
        lower, upper = _bounds(word["start"], word["end"])
        units.append({**word, "start": lower * fps, "end": upper * fps})
    # Some supplied alignments represent a word by a point. Preserve that
    # observed time for midpoint ownership; do not fabricate a duration.
    selected = []
    for unit in units:
        if unit["start"] == unit["end"]:
            if rule == "midpoint" and start <= unit["start"] < end:
                selected.append(unit)
        else:
            selected.extend(select_units([unit], start, end, rule))
    return _render(selected, metric)


def _word_units(tokens, spans, vocabulary, metric):
    if len(tokens) != len(spans):
        raise ValueError("Token and span counts differ")
    validated = []
    for span in spans:
        if isinstance(span, dict):
            lower, upper = _bounds(span["start"], span["end"])
        else:
            if len(span) != 2:
                raise ValueError("Each span must contain start and end")
            lower, upper = _bounds(*span)
        if lower == upper:
            raise ValueError("Token spans must have positive duration")
        validated.append((lower, upper))
    words = []
    letters = []
    positions = []
    byte_values = []
    byte_positions = []

    def flush_word():
        if letters:
            words.append({
                "text": normalize("".join(letters)),
                "start": min(validated[index][0] for index in positions),
                "end": max(validated[index][1] for index in positions),
                "token_positions": sorted(set(positions)),
            })
            letters.clear()
            positions.clear()

    def append_text(piece, owners):
        for character in piece:
            if character.isspace():
                flush_word()
            else:
                previous = "".join(letters)
                composed = unicodedata.normalize("NFC", previous + character)
                joins_previous = unicodedata.combining(character) or len(composed) <= len(previous)
                if metric == "cer" and letters and not joins_previous:
                    flush_word()
                letters.append(character)
                positions.extend(owners)

    def flush_bytes():
        if byte_values:
            decoded = bytes(byte_values).decode("utf-8", errors="strict")
            append_text(decoded, byte_positions)
            byte_values.clear()
            byte_positions.clear()

    for position, token in enumerate(tokens):
        token = _integer(token, "token")
        if token >= len(vocabulary):
            raise ValueError("Token ID outside vocabulary")
        piece = vocabulary[token]
        if not isinstance(piece, str):
            raise ValueError("Vocabulary entries must be strings")
        if piece in ("<blank>", "<s>", "</s>", "<sos>", "<eos>", "<sos/eos>"):
            raise ValueError("Allocation requires lexical tokens without blank/SOS/EOS")
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            byte_values.append(int(piece[3:5], 16))
            byte_positions.append(position)
            continue
        flush_bytes()
        append_text(piece.replace("\u2581", " ").replace("<space>", " "), [position])
    flush_bytes()
    flush_word()
    return words


def allocation(tokens, spans, vocabulary, start, end, metric="wer", rule="midpoint"):
    """Allocate reconstructed words, or characters with observed token spans.

    Native character vocabularies concatenate until an explicit whitespace token.
    CER characters within one token share that token's span; a composed character
    spanning tokens owns their union. Native character tokens retain their actual
    individual spans. Reference characters still share measured word timestamps.
    UTF-8 byte fallback runs share their union and decode strictly before NFC.
    """
    _metric(metric)
    units = _word_units(tokens, spans, vocabulary, metric)
    if metric == "cer":
        units = [
            {**word, "text": character, "timing_granularity": "observed_token_span_union"}
            for word in units
            for character in word["text"]
            if not character.isspace()
        ]
    selected = select_units(units, start, end, rule)
    return {"text": _render(selected, metric), "units": units, "selected_units": selected}
