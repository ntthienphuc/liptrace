"""Generate a small untrained model and synthetic clips for migration replay."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from liptrace.bundle import create_bundle, export_onnx
from liptrace.contracts import sha256, write_json
from liptrace.migration import run_migration
from liptrace.model import LipNetBackbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Choose a new directory; existing experiments are not overwritten')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(12)
    cfg = {'img_size': 16, 'max_frames': 6, 'rnn_units': 4, 'dropout': 0.0}
    model = LipNetBackbone(3, **cfg).eval()
    checkpoint = args.out / 'source.pt'
    torch.save({'model_state': model.state_dict(), 'charset': ['a', 'b'], 'blank_index': 2,
                'args': cfg}, checkpoint)
    create_bundle(checkpoint, args.out / 'bundle')
    export_onnx(args.out / 'bundle', args.out / 'deployment', runtime_only=True)
    samples = []
    for index, frames in enumerate((4, 6, 10)):
        path = args.out / f'generated-{index}.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
        if not writer.isOpened():
            raise RuntimeError('OpenCV could not create the generated video')
        for frame in range(frames):
            writer.write(np.full((16, 16, 3), (index * 50 + frame * 10) % 256, dtype=np.uint8))
        writer.release()
        samples.append({'sample_id': f'generated-{index}', 'source_group': f'generated-source-{index}',
                        'video': path.name, 'expected_video_sha256': sha256(path),
                        'reference': 'ab', 'reference_status': 'unreviewed'})
    case = {'schema': 'liptrace-migration-case-v1', 'bundle': 'bundle', 'deployment': 'deployment',
            'samples': samples, 'scope': 'generated engineering fixture; untrained model; no accuracy claim'}
    write_json(args.out / 'case.json', case)
    report = run_migration(args.out / 'case.json', args.out / 'report', repetitions=args.repetitions)
    summary = report['summary']
    print(f"Migration complete: {summary['planned_samples']} samples; strict pass={summary['pass']}")
    print(args.out / 'report' / 'report.json')
    raise SystemExit(0 if summary['pass'] else 2)


if __name__ == '__main__':
    main()
