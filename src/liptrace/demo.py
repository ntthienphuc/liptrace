"""Redistributable synthetic fixture; demonstrates execution, not recognition."""
import csv
import os
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from .contracts import write_json
from .replay import trace_video, compare_traces
from .bundle import export_onnx


def run_demo(out='demo_run', onnx=False):
    root = Path(out).resolve()
    if root.exists():
        raise ValueError('Demo destination already exists; use a new directory')
    root.mkdir(parents=True)
    rng = np.random.default_rng(20261006)
    paths = []
    for role in ('train', 'val', 'test'):
        rows = []
        for i in range(4):
            clip = root / f'{role}_{i}.avi'
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
            if not writer.isOpened():
                raise RuntimeError('MJPG fixture encoder unavailable')
            try:
                for t in range(6):
                    frame = rng.integers(0, 50, (16, 16, 3), dtype=np.uint8)
                    cv2.rectangle(frame, (3, 5), (12, 6 + (t + i) % 3), (220, 220, 220), -1)
                    writer.write(frame)
            finally:
                writer.release()
            rows.append({'clip_path': clip.name, 'text': 'a' if i % 2 == 0 else 'b', 'group_key': f'{role}_{i}'})
        manifest = root / f'{role}.csv'
        with manifest.open('w', newline='', encoding='utf-8') as handle:
            w = csv.DictWriter(handle, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        paths.append(manifest)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '2'
    cmd = [sys.executable, '-m', 'liptrace.training', '--train_csv', str(paths[0]),
           '--val_csv', str(paths[1]), '--test_csv', str(paths[2]), '--out_dir', str(root / 'run'),
           '--img_size', '16', '--max_frames', '6', '--rnn_units', '4', '--batch_size', '2',
           '--epochs', '1', '--num_workers', '0']
    run = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (root / 'training_stdout.txt').write_text(run.stdout, encoding='utf-8')
    (root / 'training_stderr.txt').write_text(run.stderr, encoding='utf-8')
    if run.returncode:
        raise RuntimeError(run.stderr)
    bundle = root / 'run' / 'bundle'
    trace_video(bundle, root / 'test_0.avi', root / 'torch_trace', reference='a')
    trace_video(bundle, root / 'test_0.avi', root / 'torch_repeat', reference='a')
    repeat = compare_traces(root / 'torch_trace', root / 'torch_repeat')
    write_json(root / 'repeat_comparison.json', repeat)
    parity = None
    if onnx:
        export_onnx(bundle, root / 'onnx_bundle')
        trace_video(root / 'onnx_bundle', root / 'test_0.avi', root / 'onnx_trace', 'onnx', 'a')
        parity = compare_traces(root / 'torch_trace', root / 'onnx_trace')
        write_json(root / 'onnx_comparison.json', parity)
    result = {'status': 'PASS' if repeat['pass'] and (parity is None or parity['pass']) else 'FAIL',
              'scope': '12 synthetic mouth-like clips; one CPU training epoch; no recognition accuracy claim',
              'repeat': repeat, 'onnx': parity}
    write_json(root / 'receipt.json', result)
    if result['status'] != 'PASS':
        raise RuntimeError('Demo replay parity failed; see receipt.json')
    return result
