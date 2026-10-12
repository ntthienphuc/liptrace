"""Explicit adapter to a user-provided original trainer; not part of model profiles."""
import hashlib
import importlib.util
import os
from pathlib import Path
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_predictor(checkpoint, profile):
    path = Path(os.environ["LIPTRACE_LEGACY_TRAINER"])
    spec = importlib.util.spec_from_file_location("liptrace_pinned_original", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    document = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = {key: document["args"][key] for key in ("img_size", "max_frames", "rnn_units", "dropout")}
    if cfg != profile["model"] or document["charset"] != profile["charset"]:
        raise ValueError("Original checkpoint and package profile disagree")
    if document["blank_index"] != profile["blank_index"]:
        raise ValueError("Original blank index differs")
    model = legacy.LipNetBackbone(len(document["charset"]) + 1, **cfg).eval()
    model.load_state_dict(document["model_state"], strict=True)

    def predict(video):
        tensor = legacy.read_video_gray_resize(str(video), cfg["max_frames"], cfg["img_size"])[None]
        with torch.inference_mode():
            logits = model(torch.from_numpy(tensor))
            transcript = legacy.greedy_decode(logits.log_softmax(-1).transpose(0, 1),
                                             document["blank_index"], dict(enumerate(document["charset"])))[0]
        return {"tensor": tensor, "logits": logits.numpy(), "transcript": transcript}

    predict.receipt = {"trainer_sha256": digest(path), "torch": torch.__version__,
                       "scope": "original model, preprocessing and greedy decode imported from provided trainer"}
    return predict
