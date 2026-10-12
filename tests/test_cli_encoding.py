"""Check Vietnamese CLI output through a non Unicode Windows style pipe."""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
from liptrace.bundle import create_bundle
from liptrace.model import LipNetBackbone


@pytest.mark.parametrize('valid', [True, False])
def test_cli_uses_utf8_for_vietnamese_piped_output(tmp_path, valid):
    bundle = tmp_path / 'thiếu'
    if valid:
        cfg = {'img_size': 16, 'max_frames': 6, 'rnn_units': 4, 'dropout': 0.0}
        model = LipNetBackbone(2, **cfg).eval()
        checkpoint = tmp_path / 'source.pt'
        torch.save({'model_state': model.state_dict(), 'charset': ['ệ'], 'blank_index': 1,
                    'args': cfg}, checkpoint)
        create_bundle(checkpoint, bundle)
    command = [sys.executable, '-m', 'liptrace', 'inspect', '--bundle', str(bundle)]
    result = subprocess.run(command, capture_output=True,
                            env={**os.environ, 'PYTHONIOENCODING': 'cp1252'})
    output = result.stdout.decode('utf-8')
    error = result.stderr.decode('utf-8')
    if valid:
        assert result.returncode == 0, error
        assert 'ệ' in output
        assert isinstance(json.loads(output), dict)
    else:
        assert result.returncode == 2
        assert 'thiếu' in error
        assert 'charmap' not in error
