"""Run the preserved recognizers on licensed, prepared full-clip pixels.

One model family per process. References are optional and used only for scoring.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def compatibility_digest(stream, name):
    result = hashlib.new(name)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["VLR_T0", "EN_B1", "ES_NATIVE", "VC_NATIVE"], required=True)
    parser.add_argument("--pixels", type=Path, required=True, help="Prepared uint8 .npy full-clip frames")
    parser.add_argument("--engine-root", type=Path, default=HERE / "engine")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--position", choices=["start", "center", "end"], default="center")
    parser.add_argument("--target-frames", type=int, default=50)
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--decoder", choices=["greedy", "ctc40"], default="ctc40")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.engine_root.resolve()
    if not hasattr(hashlib, "file_digest"):
        hashlib.file_digest = compatibility_digest
    os.environ["LIP_PROJECT_ROOT"] = str(root)
    sys.path.insert(0, str(root / "Research_Server_Confirmation_2026-10-19"))
    from data_io import validate_shape
    from inference_core import evaluate_clip
    if args.model == "VC_NATIVE":
        from vicocktail_runtime import Runtime
    else:
        from native_runtime import Runtime
    runtime = Runtime(args.model, device=args.device, root=root)
    pixels = np.load(args.pixels, allow_pickle=False)
    color = "gray" if pixels.ndim == 3 else "rgb"
    validate_shape(pixels.shape, pixels.dtype, len(pixels), color)
    row = {"clip_id": args.pixels.stem, "frames": len(pixels), "fps": 25.0,
           "metric": "wer", "reference": "", "words": []}
    decoder = {"name": args.decoder, "kind": "greedy" if args.decoder == "greedy" else "ctc",
               "beam_size": 40, "token_prune": 64}
    job = {"arms": ["A", "B", "C"], "decoders": [decoder], "cells": [{
        "cell_id": "requested_interval", "target_frames": args.target_frames,
        "context_frames": args.context_frames, "position": args.position}]}
    result = evaluate_clip(runtime, pixels, row, job)
    for record in result["records"]:
        record.pop("reference", None)
        record.pop("counts", None)
    result["scope"] = "Reference-free inference; no evaluation reference supplied"
    result["runtime_identity"] = runtime.identity
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "predictions": {
        record["arm"]: record["hypothesis"] for record in result["records"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
