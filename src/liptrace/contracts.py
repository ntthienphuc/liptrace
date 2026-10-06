"""Versioned, closed preprocessing and model contracts; no executable plugins."""
import hashlib
import json
import math
import unicodedata
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def normalize(text, mode='legacy-lower-v1'):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    if mode == 'legacy-lower-v1':
        return ' '.join(text.strip().lower().split())
    if mode == 'nfc-lower-v1':
        return unicodedata.normalize('NFC', ' '.join(text.strip().lower().split()))
    raise ValueError(f'Unsupported normalization: {mode}')


def profile_from_checkpoint(ckpt, normalization='legacy-lower-v1'):
    profile = {
        'architecture': 'lipnet-3d-2bigru-v1',
        'model': {key: ckpt['args'][key] for key in
                  ('img_size', 'max_frames', 'rnn_units', 'dropout')},
        'charset': ckpt['charset'], 'blank_index': ckpt['blank_index'],
        'preprocessing': {'input_kind': 'pre_cropped_mouth', 'color': 'opencv-bgr-to-gray',
                          'resize': 'INTER_AREA', 'sampling': 'uniform-round-repeat-last-v1',
                          'decode': 'all-frames-v1', 'layout': 'BCTHW', 'dtype': 'float32',
                          'scale': 'uint8/255*2-1'},
        'decoder': 'greedy-ctc-v1', 'normalization': normalization,
        'metrics': 'per-sample-cer-wer-v1',
    }
    validate_profile(profile)
    return profile


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {
        'architecture', 'model', 'charset', 'blank_index', 'preprocessing',
        'decoder', 'normalization', 'metrics'}:
        raise ValueError('Invalid profile schema')
    cfg = profile['model']
    if not isinstance(cfg, dict) or set(cfg) != {'img_size', 'max_frames', 'rnn_units', 'dropout'}:
        raise ValueError('Invalid model configuration')
    for key in ('img_size', 'max_frames', 'rnn_units'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if cfg['img_size'] % 8 or cfg['img_size'] > 512 or cfg['max_frames'] > 4096 or cfg['rnn_units'] > 2048:
        raise ValueError('Unsupported geometry or capacity')
    if type(cfg['dropout']) not in (int, float) or not math.isfinite(cfg['dropout']) or not 0 <= cfg['dropout'] < 1:
        raise ValueError('Invalid dropout')
    chars = profile['charset']
    if not isinstance(chars, list) or not chars or any(not isinstance(c, str) or len(c) != 1 for c in chars):
        raise ValueError('Invalid ordered charset')
    if len(set(chars)) != len(chars) or type(profile['blank_index']) is not int or profile['blank_index'] != len(chars):
        raise ValueError('Invalid CTC blank or duplicate characters')
    normalize('', profile['normalization'])
    if any(normalize(c, profile['normalization']) != c for c in chars if c != ' '):
        raise ValueError('Charset is incompatible with normalization')
    expected = {'input_kind': 'pre_cropped_mouth', 'color': 'opencv-bgr-to-gray',
                'resize': 'INTER_AREA', 'sampling': 'uniform-round-repeat-last-v1',
                'decode': 'all-frames-v1', 'layout': 'BCTHW', 'dtype': 'float32', 'scale': 'uint8/255*2-1'}
    if profile['architecture'] != 'lipnet-3d-2bigru-v1' or profile['preprocessing'] != expected:
        raise ValueError('Unsupported architecture or preprocessing')
    if profile['decoder'] != 'greedy-ctc-v1' or profile['metrics'] != 'per-sample-cer-wer-v1':
        raise ValueError('Unsupported decoder or metrics')
