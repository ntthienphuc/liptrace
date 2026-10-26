"""Strict full-clip inputs and shared manifest validation; never resize or fabricate timing."""

import hashlib
import io
import math
import unicodedata
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from confirmation_common import ROOT, normalize, resolve, sha


@lru_cache(maxsize=8)
def _archive(path, size, modified_ns):
    return zipfile.ZipFile(path)


def open_archive(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _archive(str(path), stat.st_size, stat.st_mtime_ns)


def checked_payload(content, expected):
    actual = hashlib.sha256(content).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"Payload SHA256 mismatch: expected={expected}, actual={actual}")
    return content


def payload_shape(content):
    """Validate an NPZ array header without expanding its entire full-clip pixel array."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if archive.namelist() != ["frames.npy"]:
            raise ValueError("NPZ must contain exactly frames.npy")
        with archive.open("frames.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY header {version}")
            expected_size = stream.tell() + math.prod(shape) * dtype.itemsize
            if expected_size != archive.getinfo("frames.npy").file_size:
                raise ValueError("NPY header shape does not match payload byte count")
    return shape, dtype


def validate_shape(shape, dtype, frames, color_space):
    wanted = (frames, 3, 96, 96) if color_space == "rgb" else (frames, 96, 96)
    if color_space not in ("rgb", "gray") or frames < 1:
        raise ValueError("Invalid frame count or explicit color space")
    if tuple(shape) != wanted or np.dtype(dtype) != np.uint8:
        raise ValueError(f"Expected uint8 {wanted}; received {dtype} {shape}")


def load_pixels(row, root=ROOT):
    """Return contiguous uint8 [T,96,96] or RGB [T,3,96,96], checking every read."""
    source = row["input"]
    path = resolve(source["path"], root)
    kind = source["kind"]
    color = source.get("color_space", "gray")
    if abs(float(row["fps"]) - 25.0) > 0.001:
        raise ValueError("Only verified 25fps manifests are admitted")
    if kind == "zip_npz":
        payload = open_archive(path).read(source["member"])
        checked_payload(payload, source["sha256"])
        shape, dtype = payload_shape(payload)
        validate_shape(shape, dtype, int(row["frames"]), color)
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            pixels = data["frames"]
    elif kind == "npy":
        payload = checked_payload(path.read_bytes(), source["sha256"])
        pixels = np.load(io.BytesIO(payload), allow_pickle=False)
    elif kind == "video":
        import av

        if sha(path) != source["sha256"]:
            raise ValueError(f"Video payload SHA256 mismatch: {path}")
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise ValueError("Expected exactly one video stream")
            stream = container.streams.video[0]
            if stream.average_rate is None or abs(float(stream.average_rate) - 25) > 0.001:
                raise ValueError("Video frame rate mismatch; no implicit resampling")
            stream.thread_count = 1
            pixel_format = "rgb24" if color == "rgb" else "gray"
            frames = [frame.to_ndarray(format=pixel_format) for frame in container.decode(stream)]
        if not frames:
            raise ValueError("Video contains no decoded frames")
        pixels = np.stack(frames)
        if color == "rgb":
            pixels = pixels.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported full-clip input: {kind}")
    validate_shape(pixels.shape, pixels.dtype, int(row["frames"]), color)
    pixels = np.ascontiguousarray(pixels)
    expected_pixels = row.get("metadata", {}).get("pixels_sha256")
    if expected_pixels and hashlib.sha256(pixels.tobytes()).hexdigest() != expected_pixels:
        raise ValueError("Decoded pixel SHA256 mismatch")
    return pixels


def reference_text(text, language):
    """Preserve phase5 Vietnamese whitespace/syllable normalization exactly."""
    value = normalize(text)
    if language == "vi":
        value = "".join(character if unicodedata.category(character)[0] in "LNM" else " "
                        for character in value)
    return " ".join(value.split())


def canonical_words(original, frames, reference, language):
    """Retain supplied seconds and word spans; diagnose unscorable timing without fixing it."""
    words = []
    failures = []
    previous = -math.inf
    for item in original:
        text = reference_text(item["text"], language)
        start = float(item["start"])
        end = float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            failures.append("nonfinite_word_time")
            continue
        if start < -0.04 - 1e-7 or end < start or start < previous - 1e-7:
            failures.append("negative_reversed_or_unordered_word_time")
        if end > frames / 25.0 + 0.04 + 1e-7:
            failures.append("word_end_exceeds_clip_by_more_than_one_frame")
        previous = start
        if text:
            words.append({"text": text, "start": start, "end": end})
    if not words:
        failures.append("word_timing_unavailable")
    if normalize(" ".join(word["text"] for word in words)) != reference_text(reference, language):
        failures.append("timed_text_does_not_reconstruct_full_reference")
    return words, sorted(set(failures))
