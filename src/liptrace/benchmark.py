"""Loaded-backend host measurements; no trace serialization in timed loops."""
import platform
import time
from pathlib import Path
import numpy as np
from .bundle import read_bundle
from .contracts import json_digest
from .media import decode_frames, pack_frames
from .replay import decode_path


def distribution(values):
    return {"n": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(np.min(values)), "max_ms": float(np.max(values))}


class LoadedBackend:
    def __init__(self, bundle, runtime, threads=2):
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        start = time.perf_counter()
        self.document = read_bundle(bundle)
        self.profile = self.document["profile"]
        self.runtime = runtime
        self.threads = threads
        if runtime == "torch":
            import torch
            from .runtime import load_recognizer
            torch.set_num_threads(threads)
            self.model, _, _ = load_recognizer(Path(bundle) / "model.pt")
            self.version = torch.__version__
            self.providers = ["torch-cpu"]
        elif runtime == "onnx":
            import onnxruntime as ort
            if "model.onnx" not in self.document["artifacts"]:
                raise ValueError("Bundle has no ONNX artifact")
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.model = ort.InferenceSession(str(Path(bundle) / "model.onnx"),
                                             sess_options=options, providers=["CPUExecutionProvider"])
            self.version = ort.__version__
            self.providers = self.model.get_providers()
        else:
            raise ValueError("runtime must be torch or onnx")
        self.loading_ms = (time.perf_counter() - start) * 1000

    def forward(self, tensor):
        if self.runtime == "torch":
            import torch
            with torch.inference_mode():
                logits = self.model(torch.from_numpy(tensor)).numpy()
        else:
            logits = self.model.run(["logits"], {"video": tensor})[0]
        shape = (1, self.profile["model"]["max_frames"], len(self.profile["charset"]) + 1)
        if logits.shape != shape or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError("Benchmark output contract mismatch")
        return logits

    def measure(self, video, warmup=1, repetitions=3):
        if type(warmup) is not int or warmup < 0 or type(repetitions) is not int or repetitions < 1:
            raise ValueError("warmup must be nonnegative and repetitions positive")
        cfg = self.profile["model"]
        values = {key: [] for key in ("decode", "pack", "forward", "ctc_decode", "pipeline")}
        transcripts = []
        for iteration in range(warmup + repetitions):
            start = time.perf_counter()
            frames = decode_frames(video)
            decoded = time.perf_counter()
            tensor, _ = pack_frames(frames, cfg["max_frames"], cfg["img_size"])
            packed = time.perf_counter()
            logits = self.forward(tensor)
            forwarded = time.perf_counter()
            transcript = decode_path(logits.argmax(-1)[0], self.profile["charset"], self.profile["blank_index"])
            ended = time.perf_counter()
            if iteration < warmup:
                continue
            times = {"decode": decoded - start, "pack": packed - decoded,
                     "forward": forwarded - packed, "ctc_decode": ended - forwarded,
                     "pipeline": ended - start}
            for key, value in times.items():
                values[key].append(value * 1000)
            transcripts.append(transcript)
        return {"schema": "liptrace-host-timing-v1", "runtime": self.runtime,
                "version": self.version, "providers": self.providers,
                "threads": self.threads, "warmup": warmup, "repetitions": repetitions,
                "loading_ms": self.loading_ms, "stages": {k: distribution(v) for k, v in values.items()},
                "observations_ms": values, "transcript": transcripts[0],
                "serial_clips_per_second": 1000 / np.mean(values["pipeline"]),
                "stable_transcript": len(set(transcripts)) == 1,
                "profile_sha256": json_digest(self.profile),
                "decoded_frames": len(frames), "input_frame_shape": list(frames.shape[1:]),
                "environment": {"python": platform.python_version(), "platform": platform.platform()},
                "scope": "host CPU; reused loaded backend; decode+pack+forward+CTC per clip; "
                         "excludes model loading, ROI extraction, camera, network, UI and trace writes"}
