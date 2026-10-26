"""Small portable IO and identity utilities; no workspace-specific imports."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def resolve(config, key):
    value = os.path.expandvars(config[key])
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def config_file(path):
    config = read_json(path)
    config['config_sha256'] = digest(path)
    return config


def rank(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def code_identity():
    files = sorted((ROOT / 'phase5').glob('*.py'))
    files += sorted(p for p in (ROOT / 'vendor').rglob('*') if p.suffix in {'.py', '.npy'})
    files += sorted((ROOT / 'assets/tokenizers').glob('*'))
    values = {path.relative_to(ROOT).as_posix(): digest(path) for path in files if path.is_file()}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
