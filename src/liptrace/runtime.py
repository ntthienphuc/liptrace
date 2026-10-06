"""Strict CPU/CUDA inference for pre-cropped mouth clips."""
import hashlib
import json
import platform
import time
from pathlib import Path
import torch
from .model import LipNetBackbone
from .training import read_video_gray_resize, greedy_decode


def load_recognizer(path, device='cpu'):
    path = Path(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    for key in ['model_state', 'charset', 'blank_index', 'args']:
        if key not in ckpt:
            raise ValueError(f'Checkpoint missing {key}')
    chars = ckpt['charset']
    if not isinstance(chars, list) or not chars or any(not isinstance(c,str) or len(c)!=1 for c in chars):
        raise ValueError('Invalid character inventory')
    if len(set(chars)) != len(chars) or ckpt['blank_index'] != len(chars):
        raise ValueError('Checkpoint charset/CTC blank mismatch')
    args = ckpt['args']
    cfg = {k: args[k] for k in ['img_size','max_frames','rnn_units','dropout']}
    from .contracts import profile_from_checkpoint
    profile_from_checkpoint(ckpt, args.get('normalization', 'legacy-lower-v1'))
    if cfg['img_size'] < 8 or cfg['img_size'] % 8 or cfg['max_frames'] <= 0 or cfg['rnn_units'] <= 0:
        raise ValueError('Invalid checkpoint input geometry')
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    model = LipNetBackbone(len(chars)+1, **cfg)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.to(target).eval()
    return model, ckpt, target


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def predict_video(checkpoint, video, device='cpu', repetitions=1, warmup=0):
    if repetitions < 1 or warmup < 0:
        raise ValueError('repetitions must be >=1 and warmup >=0')
    model, ckpt, target = load_recognizer(checkpoint, device)
    start = time.perf_counter()
    arr = read_video_gray_resize(str(video), ckpt['args']['max_frames'], ckpt['args']['img_size'])
    x = torch.from_numpy(arr).unsqueeze(0).to(target)
    sync(target)
    prep_ms = (time.perf_counter()-start)*1000
    timings = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync(target)
        for _ in range(repetitions):
            start = time.perf_counter()
            logits = model(x)
            sync(target)
            timings.append((time.perf_counter()-start)*1000)
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite model logits')
        prediction = greedy_decode(logits.log_softmax(-1).transpose(0,1), len(ckpt['charset']), dict(enumerate(ckpt['charset'])))[0]
    return {'prediction': prediction, 'video': str(video), 'input_kind': 'pre_cropped_mouth',
            'input_shape': list(x.shape), 'output_shape': list(logits.shape),
            'charset_size_without_blank': len(ckpt['charset']), 'output_classes_with_blank': len(ckpt['charset'])+1,
            'checkpoint_epoch': ckpt.get('epoch'),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'device': str(target), 'python': platform.python_version(), 'torch': torch.__version__,
            'preprocessing_ms': prep_ms, 'model_forward_ms': timings,
            'timing_scope': 'decoded mouth clip once; repeated model forwards; excludes model loading, mouth localization and UI',
            'warmup': warmup, 'repetitions': repetitions, 'accuracy_evaluation': False}
