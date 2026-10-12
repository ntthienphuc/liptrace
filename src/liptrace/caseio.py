"""Reviewed migration cases, with explicit opt-in for local reference code."""
import importlib.util
import json
from pathlib import Path
from .contracts import sha256

LABEL_STATES = {"reviewed", "historical", "unreviewed", "absent"}


def resolve_path(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Case paths must be nonempty strings")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_case(path):
    source = Path(path).resolve()
    doc = json.loads(source.read_text(encoding="utf-8-sig"))
    if doc.get("schema") != "liptrace-migration-case-v1":
        raise ValueError("Unsupported migration case schema")
    root = source.parent
    doc["bundle"] = str(resolve_path(root, doc.get("bundle")))
    for field in ("deployment", "checkpoint"):
        if doc.get(field) is not None:
            doc[field] = str(resolve_path(root, doc[field]))
    samples = doc.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 10000:
        raise ValueError("Case needs 1..10000 samples")
    ids = set()
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Samples must be objects")
        identity = row.get("sample_id")
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 256 or identity in ids:
            raise ValueError("Case sample ids must be unique, nonempty strings")
        ids.add(identity)
        if not isinstance(row.get("source_group"), str) or not row["source_group"].strip():
            raise ValueError("Every sample needs a source_group; identity is not inferred")
        row["video"] = str(resolve_path(root, row.get("video")))
        ref = row.get("reference")
        state = row.get("reference_status", "unreviewed" if ref is not None else "absent")
        if state not in LABEL_STATES or (ref is None) != (state == "absent"):
            raise ValueError("Reference and reference_status disagree")
        if ref is not None and (not isinstance(ref, str) or len(ref) > 4096):
            raise ValueError("Reference must be a string of at most 4096 characters")
        row["reference_status"] = state
        digest = row.get("expected_video_sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid expected_video_sha256")
    doc["case_path"] = str(source)
    doc["case_sha256"] = sha256(source)
    return doc


def original_predictor(case, profile, allow_original_adapter=False):
    spec = case.get("original_adapter")
    if not spec:
        return None, None
    if not allow_original_adapter:
        raise ValueError("Original adapters execute local code; pass --allow-original-adapter explicitly")
    filename, factory_name = spec.rsplit(":", 1)
    if not factory_name.isidentifier():
        raise ValueError("Original adapter factory must be a Python identifier")
    path = resolve_path(Path(case["case_path"]).parent, filename)
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("Original adapter must be an existing local Python file")
    checkpoint = case.get("checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        raise ValueError("Original adapter requires a checkpoint")
    source = importlib.util.spec_from_file_location("liptrace_user_reference_" + sha256(path)[:16], path)
    module = importlib.util.module_from_spec(source)
    source.loader.exec_module(module)
    predictor = getattr(module, factory_name)(checkpoint, profile)
    if not callable(predictor):
        raise ValueError("Original adapter factory must return a callable")
    receipt = {"adapter_sha256": sha256(path), "factory": factory_name,
               "checkpoint_sha256": sha256(checkpoint),
               "implementation_receipt": getattr(predictor, "receipt", None),
               "scope": "explicit trusted local code; source hashes are consistency evidence, not authentication"}
    return predictor, receipt
