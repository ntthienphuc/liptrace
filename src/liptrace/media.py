"""Decode actual frames before sampling, rather than trusting container counts."""
from pathlib import Path
import cv2
import numpy as np


def decode_frames(path, max_decoded=4096, max_bytes=256 * 1024 * 1024):
    if not Path(path).is_file():
        raise RuntimeError(f'Cannot open video: {path}')
    cap = cv2.VideoCapture(str(path))
    frames, byte_count = [], 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            byte_count += frame.nbytes
            if len(frames) >= max_decoded or byte_count > max_bytes:
                raise ValueError('Decoded media exceeds resource limit; supply a shorter mouth clip')
            if frames and frame.shape != frames[0].shape:
                raise ValueError('Video changes frame geometry')
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f'Empty video: {path}')
    return np.stack(frames)


def pack_frames(frames, count, size):
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or not len(frames):
        raise ValueError('Expected nonempty uint8 BGR frames')
    if type(count) is not int or type(size) is not int or count <= 0 or size <= 0:
        raise ValueError('Invalid packing geometry')
    n = len(frames)
    indices = np.linspace(0, n - 1, num=count).round().astype(np.int64) if n >= count else np.concatenate(
        [np.arange(n), np.full(count - n, n - 1, dtype=np.int64)])
    gray = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (size, size),
                       interpolation=cv2.INTER_AREA) for i in indices]
    tensor = (np.stack(gray).astype(np.float32) / 255.0 * 2.0 - 1.0)[None, None]
    return tensor, indices


def read_video_gray_resize(path, max_frames, size):
    return pack_frames(decode_frames(path), max_frames, size)[0][0]
