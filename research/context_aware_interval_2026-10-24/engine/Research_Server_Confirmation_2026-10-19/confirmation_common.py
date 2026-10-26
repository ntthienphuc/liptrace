"""Portable identities, durable records and language-aware scoring for confirmation."""

import hashlib
import json
import os
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIP_PROJECT_ROOT", str(HERE.parent))).resolve()


def resolve(path, root=ROOT):
    candidate = Path(str(path).replace("\\", "/"))
    return candidate.resolve() if candidate.is_absolute() else (Path(root) / candidate).resolve()


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete JSONL line {number}: {path}")
            if line.strip():
                yield json.loads(line)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", str(text)).lower().split())


def scoring_units(text, metric="wer"):
    normalized = normalize(text)
    if metric == "wer":
        return normalized.split()
    if metric == "cer":
        return [character for character in normalized if not character.isspace()]
    raise ValueError(f"Unsupported metric: {metric}")


def edit_counts(reference, hypothesis, metric="wer"):
    left = scoring_units(reference, metric)
    right = scoring_units(hypothesis, metric)
    previous = [(index, 0, 0, index) for index in range(len(right) + 1)]
    for row_index, word in enumerate(left, 1):
        current = [(row_index, 0, row_index, 0)]
        for column, predicted in enumerate(right, 1):
            diagonal = previous[column - 1]
            deleted = previous[column]
            inserted = current[-1]
            mismatch = int(word != predicted)
            choices = [
                (diagonal[0] + mismatch, diagonal[1] + mismatch, diagonal[2], diagonal[3]),
                (deleted[0] + 1, deleted[1], deleted[2] + 1, deleted[3]),
                (inserted[0] + 1, inserted[1], inserted[2], inserted[3] + 1),
            ]
            current.append(min(choices, key=lambda value: value[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "units": len(left),
        "metric": metric,
    }
