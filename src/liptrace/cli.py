"""Public CLI: no hidden training data or implicitly downloaded weights."""
import argparse
import json
import sys
from pathlib import Path
from .contracts import write_json


def emit(value, out=None):
    if out:
        write_json(out, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        from .training import main as train
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train()
        return
    parser = argparse.ArgumentParser(description='LipTrace: traceable character-CTC lip reading')
    parser.add_argument('--threads', type=int, default=2, help='Inference/demo CPU threads (default 2); training has its own --threads flag')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train', help='Train CNN + two BiGRUs with audited train/val/test CSVs')
    b = sub.add_parser('bundle', help='Create a sanitized, hash-bound model bundle')
    b.add_argument('--checkpoint', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--normalization', choices=['legacy-lower-v1', 'nfc-lower-v1'],
                   help='Inherit checkpoint declaration; legacy fallback when absent. Conflicting overrides fail.')
    inspect = sub.add_parser('inspect', help='Validate and inspect every bundle artifact')
    inspect.add_argument('--bundle', required=True)
    inspect.add_argument('--expect-manifest-sha256')
    for name in ('trace', 'predict'):
        p = sub.add_parser(name, help='Predict and save a full replay trace for a mouth clip')
        p.add_argument('--bundle', required=True)
        p.add_argument('--video', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--runtime', choices=['torch', 'onnx'], default='torch')
        p.add_argument('--reference')
        p.add_argument('--expect-manifest-sha256')
    a = sub.add_parser('audit', help='Collect split, media, normalization and CTC issues')
    for role in ('train', 'val', 'test'):
        a.add_argument('--' + role, required=True)
    a.add_argument('--max-frames', type=int, default=32)
    a.add_argument('--normalization', choices=['legacy-lower-v1', 'nfc-lower-v1'], default='legacy-lower-v1')
    a.add_argument('--out', required=True)
    a.add_argument('--skip-decode', action='store_true', help='Hash-only preflight; report marks decoding unchecked')
    c = sub.add_parser('compare', help='Compare complete traces and find first divergence')
    c.add_argument('--left', required=True)
    c.add_argument('--right', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--atol', type=float, default=1e-5)
    c.add_argument('--rtol', type=float, default=1e-4)
    e = sub.add_parser('export', help='Export fixed-geometry ONNX into a new bundle')
    e.add_argument('--bundle', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--expect-manifest-sha256')
    m = sub.add_parser('metrics', help='Recompute historical sample-mean CER/WER')
    m.add_argument('--predictions', required=True)
    m.add_argument('--out', required=True)
    demo = sub.add_parser('demo', help='Synthetic train -> bundle -> replay, optionally ONNX')
    demo.add_argument('--out', default='demo_run')
    demo.add_argument('--onnx', action='store_true')
    args = parser.parse_args()
    try:
        import torch
        if args.threads <= 0:
            raise ValueError('threads must be positive')
        torch.set_num_threads(args.threads)
        if args.command == 'bundle':
            from .bundle import create_bundle
            emit(create_bundle(args.checkpoint, args.out, args.normalization))
        elif args.command == 'inspect':
            from .bundle import read_bundle
            emit(read_bundle(args.bundle, args.expect_manifest_sha256))
        elif args.command in ('trace', 'predict'):
            from .replay import trace_video
            emit(trace_video(args.bundle, args.video, args.out, args.runtime, args.reference, args.expect_manifest_sha256))
        elif args.command == 'audit':
            from .dataset import audit_dataset
            report = audit_dataset([args.train, args.val, args.test], args.max_frames, args.normalization, not args.skip_decode)
            emit(report, args.out)
            if not report['valid']:
                parser.exit(2, 'Dataset audit failed; all detected issues saved in report.\n')
        elif args.command == 'compare':
            from .replay import compare_traces
            report = compare_traces(args.left, args.right, args.atol, args.rtol)
            emit(report, args.out)
            if not report['pass']:
                parser.exit(2, f"Replay diverged at {report['first_divergence']}\n")
        elif args.command == 'export':
            from .bundle import export_onnx
            emit(export_onnx(args.bundle, args.out, args.expect_manifest_sha256))
        elif args.command == 'metrics':
            from .audit import prediction_metrics
            emit(prediction_metrics(args.predictions), args.out)
        elif args.command == 'demo':
            from .demo import run_demo
            emit(run_demo(args.out, args.onnx))
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
