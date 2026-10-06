"""Validate release archives do not contain models, data, environments or secrets."""
import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default='dist')
    args = parser.parse_args()
    folder = Path(args.dist)
    reports, checksums = [], []
    files = sorted(folder.glob('*.whl')) + sorted(folder.glob('*.tar.gz'))
    if len(files) != 2:
        raise RuntimeError('Expected exactly one wheel and one source distribution')
    for artifact in files:
        if artifact.suffix == '.whl':
            with zipfile.ZipFile(artifact) as handle:
                contents = {name: handle.read(name) for name in handle.namelist() if not name.endswith('/')}
            source = Path(__file__).resolve().parents[1] / 'src' / 'liptrace'
            expected = {'liptrace/' + p.name: p.read_text(encoding='utf-8') for p in source.glob('*.py')}
            packaged = {name for name in contents if name.startswith('liptrace/') and name.endswith('.py')}
            if packaged != set(expected):
                raise RuntimeError('Wheel module set differs from source checkout')
            for name, text in expected.items():
                if contents[name].decode('utf-8').replace('\r\n', '\n') != text.replace('\r\n', '\n'):
                    raise RuntimeError(f'Wheel source differs from checkout: {name}')
        else:
            with tarfile.open(artifact) as handle:
                contents = {member.name: handle.extractfile(member).read() for member in handle.getmembers() if member.isfile()}
        forbidden_suffixes = ('.pt', '.pth', '.onnx', '.npz', '.avi', '.mp4', '.pem', '.key')
        for name, payload in contents.items():
            parts = name.split('/')
            if name.lower().endswith(forbidden_suffixes) or any(p.startswith('.venv') or p in ('recovery', '.git', '__pycache__') for p in parts):
                raise RuntimeError(f'Forbidden release file: {name}')
            credential = rb'\bghp_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----'
            if re.search(credential, payload):
                raise RuntimeError(f'Credential marker in {name}')
        if not any(name.endswith('LICENSE') for name in contents):
            raise RuntimeError('Missing source license')
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        reports.append({'file': artifact.name, 'sha256': digest, 'bytes': artifact.stat().st_size,
                        'entries': len(contents), 'status': 'PASS'})
        checksums.append(f'{digest}  {artifact.name}')
    (folder / 'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n', encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
