"""Reproduce original/package/export behavior on a hash-bound cohort."""
import json
import time
from pathlib import Path
import numpy as np
from . import __version__
from .benchmark import LoadedBackend
from .bundle import read_bundle
from .caseio import load_case, original_predictor
from .contracts import json_digest, sha256, write_json
from .migration_report import summarize, write_predictions
from .metrics import reference_diagnostics
from .original import compare_original
from .parity import compare_task_traces
from .replay import trace_video


def failure(row, stage, error):
    row["errors"].append({"stage": stage, "type": type(error).__name__, "message": str(error)})


def run_migration(case, out, warmup=1, repetitions=3, atol=1e-5, rtol=1e-4,
                  threads=2, allow_original_adapter=False):
    if type(warmup) is not int or warmup < 0 or type(repetitions) is not int or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    if not np.isfinite(atol) or not np.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    configuration = load_case(case)
    if configuration.get("original_adapter") and not allow_original_adapter:
        raise ValueError("Original adapters execute local code; pass --allow-original-adapter")
    root = Path(out)
    if root.exists():
        raise ValueError("Migration destination already exists; choose a new directory")
    doc = read_bundle(configuration["bundle"])
    profile = doc["profile"]
    expected = []
    runtimes = {"torch": configuration["bundle"]}
    if configuration.get("deployment"):
        deployed = read_bundle(configuration["deployment"])
        if deployed["profile"] != profile:
            raise ValueError("Package and deployment profile differ")
        if deployed["provenance"].get("source_checkpoint_sha256") != doc["provenance"].get(
                "source_checkpoint_sha256"):
            raise ValueError("Package and deployment source checkpoint differ")
        pinned_bundle = deployed["provenance"].get("source_bundle_manifest_sha256")
        if pinned_bundle and pinned_bundle != sha256(Path(configuration["bundle"]) / "manifest.json"):
            raise ValueError("Deployment was exported from a different bundle manifest")
        expected.append("torch_onnx")
        runtimes["onnx"] = configuration["deployment"]
    if configuration.get("original_adapter"):
        expected.append("original_torch")
        if sha256(configuration["checkpoint"]) != doc["provenance"].get("source_checkpoint_sha256"):
            raise ValueError("Original checkpoint does not match bundle source identity")
    if not expected:
        raise ValueError("Supply deployment or an explicit original adapter for a migration comparison")
    original, original_receipt = original_predictor(configuration, profile, allow_original_adapter)
    root.mkdir(parents=True)
    sources = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    receipt = {"schema": "liptrace-migration-receipt-v1", "software_version": __version__,
               "software_source_sha256": json_digest(sources), "software_source_files": sources,
               "case_sha256": configuration["case_sha256"], "profile_sha256": json_digest(profile),
               "bundle_manifest_sha256": sha256(Path(configuration["bundle"]) / "manifest.json"),
               "bundle_artifacts": doc["artifacts"], "source_checkpoint_sha256": doc["provenance"].get(
                   "source_checkpoint_sha256"), "original": original_receipt,
               "atol": atol, "rtol": rtol, "warmup": warmup, "repetitions": repetitions, "threads": threads}
    if configuration.get("deployment"):
        receipt["deployment_manifest_sha256"] = sha256(Path(configuration["deployment"]) / "manifest.json")
        receipt["deployment_artifacts"] = deployed["artifacts"]
    write_json(root / "receipt.json", receipt)
    engines = {}
    initialization_errors = {}
    for runtime, bundle in runtimes.items():
        try:
            engines[runtime] = LoadedBackend(bundle, runtime, threads)
        except Exception as error:
            initialization_errors[runtime] = error
    rows = []
    for index, sample in enumerate(configuration["samples"]):
        row = {key: sample.get(key) for key in ("sample_id", "source_group", "reference", "reference_status")}
        row.update({"executions": {}, "comparisons": {}, "benchmarks": {}, "errors": []})
        if sample.get("reference") is not None:
            row["reference_diagnostics"] = reference_diagnostics(sample["reference"], profile)
        directory = root / "samples" / f"{index:05d}"
        directory.mkdir(parents=True)
        try:
            digest = sha256(sample["video"])
            if sample.get("expected_video_sha256") and digest != sample["expected_video_sha256"]:
                raise ValueError("Source video identity mismatch")
            row["video_sha256"] = digest
        except Exception as error:
            failure(row, "input", error)
            rows.append(row)
            write_json(directory / "result.json", row)
            write_json(root / "progress.json", {"state": "running", "processed": len(rows),
                                               "planned": len(configuration["samples"])})
            continue
        traces = {}
        for runtime, bundle in runtimes.items():
            try:
                reference = sample.get("reference") if sample["reference_status"] in ("reviewed", "historical") else None
                start = time.perf_counter()
                trace = trace_video(bundle, sample["video"], directory / runtime, runtime, reference, threads=threads)
                if trace["video_sha256"] != digest:
                    raise ValueError("Video changed during migration")
                row["executions"][runtime] = {"transcript": trace["transcript"],
                                              "trace_manifest_sha256": sha256(directory / runtime / "trace.json"),
                                              "trace_wall_ms": (time.perf_counter() - start) * 1000,
                                              "model_forward_ms": trace["model_forward_ms"]}
                traces[runtime] = directory / runtime
                if runtime in initialization_errors:
                    raise initialization_errors[runtime]
                row["benchmarks"][runtime] = engines[runtime].measure(sample["video"], warmup, repetitions)
                measured = row["benchmarks"][runtime]
                if not measured["stable_transcript"] or measured["transcript"] != trace["transcript"]:
                    raise ValueError("Repeated benchmark output differs from recorded trace")
            except Exception as error:
                failure(row, runtime, error)
        if original is not None:
            try:
                start = time.perf_counter()
                value = original(sample["video"])
                if sha256(sample["video"]) != digest:
                    raise ValueError("Video changed during original execution")
                if not isinstance(value, dict) or not isinstance(value.get("transcript"), str):
                    raise ValueError("Original predictor must return a transcript and arrays")
                row["executions"]["original"] = {"transcript": value["transcript"],
                                                  "pipeline_call_ms": (time.perf_counter() - start) * 1000}
                np.savez_compressed(directory / "original_stages.npz", tensor=value["tensor"], logits=value["logits"])
                row["original_stages_sha256"] = sha256(directory / "original_stages.npz")
                if "torch" in traces:
                    row["comparisons"]["original_torch"] = compare_original(value, traces["torch"], profile, atol, rtol)
            except Exception as error:
                failure(row, "original", error)
        if "torch" in traces and "onnx" in traces:
            try:
                row["comparisons"]["torch_onnx"] = compare_task_traces(traces["torch"], traces["onnx"], atol, rtol)
            except Exception as error:
                failure(row, "torch_onnx", error)
        write_json(directory / "result.json", row)
        rows.append(row)
        write_json(root / "progress.json", {"state": "running", "processed": len(rows),
                                           "planned": len(configuration["samples"])})
    summary = summarize(rows, expected, profile["normalization"])
    report = {"schema": "liptrace-migration-report-v1", "state": "complete", "receipt": receipt,
              "summary": summary, "samples": rows,
              "scope": "host replay of declared pre-cropped inputs; label declarations not reviewed by software; "
                       "no speaker-generalization, live-camera, Jetson or educational claims"}
    write_json(root / "report.json", report)
    write_predictions(root, rows)
    write_json(root / "progress.json", {"state": "complete", "processed": len(rows),
                                       "planned": len(configuration["samples"])})
    return report
