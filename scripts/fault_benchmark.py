"""Small independent, deterministic fault experiment; no natural accuracy claim."""
import argparse
import csv
import json
import platform
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
from liptrace.bundle import create_bundle, read_bundle
from liptrace.contracts import write_json
from liptrace.dataset import audit_dataset
from liptrace.model import LipNetBackbone


def fixture(root):
    paths = []
    for i, role in enumerate(('train', 'val', 'test')):
        clip = root / f'{role}.avi'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
        if not writer.isOpened():
            raise RuntimeError('Fixture encoder unavailable')
        for frame in range(4):
            writer.write(np.full((16, 16, 3), 20 + i * 50 + frame * 5, np.uint8))
        writer.release()
        row = {'clip_path': clip.name, 'text': 'ab', 'group_key': role,
               'start_s': '0', 'end_s': '0.4', 'source_duration_s': '1'}
        path = root / f'{role}.csv'
        write_csv(path, row)
        paths.append(path)
    return paths


def write_csv(path, row):
    with path.open('w', encoding='utf-8', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=row)
        w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='fault_report.json')
    args = ap.parse_args()
    torch.set_num_threads(2)
    cases = []
    specifications = [('healthy', None, None, None), ('missing_group', 'group_key', '', 'missing_group'),
        ('empty_label', 'text', '', 'empty_label'), ('unseen_char', 'text', 'z', 'unseen_character'),
        ('repeated_ctc_label', 'text', 'aaaab', 'ctc_infeasible'),
        ('negative_timestamp', 'start_s', '-1', 'invalid_timestamp'),
        ('nonfinite_timestamp', 'end_s', 'nan', 'invalid_timestamp'),
        ('out_of_source_range', 'end_s', '2', 'invalid_timestamp'),
        ('missing_video', 'clip_path', 'missing.avi', 'missing_media'),
        ('group_leakage', 'group_key', 'train', 'group_overlap'),
        ('renamed_duplicate', None, None, 'content_overlap'), ('corrupt_video', None, None, 'invalid_media')]
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        for name, key, value, expected in specifications:
            root = base / name
            root.mkdir()
            paths = fixture(root)
            if key:
                with paths[1].open(encoding='utf-8') as h:
                    row = next(csv.DictReader(h))
                row[key] = value
                write_csv(paths[1], row)
            if name == 'renamed_duplicate':
                shutil.copyfile(root / 'train.avi', root / 'val.avi')
            if name == 'corrupt_video':
                (root / 'val.avi').write_bytes(b'invalid movie')
            report = audit_dataset(paths, max_frames=6)
            detected = sorted({i['code'] for i in report['issues']})
            passed = report['valid'] if expected is None else not report['valid'] and expected in detected
            cases.append({'case': name, 'expected': expected or 'accept', 'detected': detected, 'pass': passed})
        cfg = {'img_size': 16, 'max_frames': 6, 'rnn_units': 4, 'dropout': 0.0}
        torch.manual_seed(5)
        checkpoint = base / 'model.pt'
        torch.save({'model_state': LipNetBackbone(3, **cfg).state_dict(), 'charset': ['a', 'b'],
                    'blank_index': 2, 'args': cfg}, checkpoint)
        for name in ('swapped_charset', 'modified_weights', 'changed_manifest'):
            root = base / name
            identity = create_bundle(checkpoint, root)
            if name in ('swapped_charset', 'modified_weights'):
                ckpt = torch.load(root / 'model.pt', weights_only=True)
                if name == 'swapped_charset':
                    ckpt['charset'].reverse()
                else:
                    ckpt['model_state']['classifier.bias'][0] += 1
                torch.save(ckpt, root / 'model.pt')
            else:
                doc = json.loads((root / 'manifest.json').read_text())
                doc['profile']['charset'].reverse()
                write_json(root / 'manifest.json', doc)
            rejected = False
            try:
                read_bundle(root, identity['manifest_sha256'])
            except ValueError:
                rejected = True
            cases.append({'case': name, 'expected': 'reject', 'pass': rejected})
    faults = [c for c in cases if c['expected'] != 'accept']
    healthy = [c for c in cases if c['expected'] == 'accept']
    summary = {'schema': 'liptrace-fault-benchmark-v1', 'cases': cases,
               'faults_expected': len(faults), 'faults_detected': sum(c['pass'] for c in faults),
               'healthy_expected': len(healthy), 'healthy_accepted': sum(c['pass'] for c in healthy),
               'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
               'opencv': cv2.__version__,
               'scope': 'deliberately injected faults and one healthy fixture; not population detection performance'}
    write_json(args.out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, indent=2))
    if not all(c['pass'] for c in cases):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
