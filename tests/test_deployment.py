"""Deployment security/compatibility and a training-dependency-blocked process."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from liptrace.bundle import create_bundle, export_onnx, read_bundle
from liptrace.contracts import json_digest, sha256, write_json
from liptrace.ctc import cer_score, wer_score, decode_path
from liptrace.deployment import read_deployment, SCHEMA
from liptrace.replay import trace_video, compare_traces


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from liptrace.model import LipNetBackbone
    torch.set_num_threads(2)
    torch.manual_seed(17)
    root = tmp_path_factory.mktemp('deployment')
    cfg = dict(img_size=16, max_frames=6, rnn_units=4, dropout=0.0)
    model = LipNetBackbone(3, **cfg)
    checkpoint = root / 'source.pt'
    torch.save(dict(model_state=model.state_dict(), charset=['a', 'b'], blank_index=2,
                    args=cfg), checkpoint)
    create_bundle(checkpoint, root / 'bundle')
    receipt = export_onnx(root / 'bundle', root / 'deployment', runtime_only=True)
    writer = cv2.VideoWriter(str(root / 'clip.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
    assert writer.isOpened()
    for value in (0, 60, 180, 240):
        writer.write(np.full((16, 16, 3), value, np.uint8))
    writer.release()
    trace_video(root / 'bundle', root / 'clip.avi', root / 'torch', reference='ab')
    return root, receipt


def test_roundtrip_exact_source_identities_and_strict_parity(artifact, tmp_path):
    root, receipt = artifact
    deployment = root / 'deployment'
    assert {p.name for p in deployment.iterdir()} == {'manifest.json', 'model.onnx'}
    doc = read_bundle(deployment, receipt['manifest_sha256'])
    assert doc['schema'] == SCHEMA
    assert doc['verification']['checkpoint_verified'] is False
    assert doc['provenance']['source_checkpoint_sha256'] == sha256(root / 'source.pt')
    assert doc['provenance']['bundled_checkpoint_sha256'] == sha256(root / 'bundle/model.pt')
    assert doc['provenance']['source_bundle_manifest_sha256'] == sha256(root / 'bundle/manifest.json')
    trace = trace_video(deployment, root / 'clip.avi', tmp_path / 'onnx', 'onnx', 'ab')
    assert trace['verification']['checkpoint_verified'] is False
    assert trace['timings_ms']['forward'] == trace['model_forward_ms']
    assert compare_traces(root / 'torch', tmp_path / 'onnx')['pass']
    with pytest.raises(ValueError, match='requires runtime'):
        trace_video(deployment, root / 'clip.avi', tmp_path / 'bad')


@pytest.mark.parametrize('mutation', ['graph_bytes', 'profile', 'checkpoint_hash', 'source_manifest_hash',
                                     'io', 'verification', 'metadata', 'input_shape', 'extra_output'])
def test_tamper_rejected_even_after_unpinned_json_updates(artifact, tmp_path, mutation):
    root, receipt = artifact
    target = tmp_path / 'deployment'
    shutil.copytree(root / 'deployment', target)
    manifest = target / 'manifest.json'
    doc = json.loads(manifest.read_text())
    if mutation == 'graph_bytes':
        with (target / 'model.onnx').open('ab') as handle:
            handle.write(b'changed')
    elif mutation == 'profile':
        doc['profile']['charset'].reverse()
        doc['provenance']['profile_sha256'] = json_digest(doc['profile'])
    elif mutation == 'checkpoint_hash':
        doc['provenance']['source_checkpoint_sha256'] = '0' * 64
    elif mutation == 'source_manifest_hash':
        doc['provenance']['source_bundle_manifest_sha256'] = '0' * 64
    elif mutation == 'io':
        doc['io']['input']['shape'][0] = 2
    elif mutation == 'verification':
        doc['verification']['checkpoint_verified'] = True
    else:
        import onnx
        graph = onnx.load(str(target / 'model.onnx'))
        if mutation == 'metadata':
            onnx.helper.set_model_props(graph, {})
        elif mutation == 'input_shape':
            graph.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
        else:
            graph.graph.output.append(graph.graph.input[0])
        onnx.save(graph, str(target / 'model.onnx'))
        doc['artifacts']['model.onnx'] = sha256(target / 'model.onnx')
    write_json(manifest, doc)
    with pytest.raises(ValueError):
        read_deployment(target)
    with pytest.raises(ValueError):
        read_bundle(target, receipt['manifest_sha256'])


def test_existing_mixed_bundle_cannot_drop_checkpoint_verification(artifact, tmp_path):
    root, _ = artifact
    target = tmp_path / 'legacy'
    shutil.copytree(root / 'bundle', target)
    (target / 'model.pt').unlink()
    with pytest.raises(ValueError, match='Artifact identity'):
        read_bundle(target)


@pytest.mark.parametrize('ref,hyp', [('', ''), ('', 'abc'), ('abc', ''), ('book', 'bok'),
                                   ('á à', 'á a'), ('a b', 'a b c'), ('  a ', 'a')])
def test_numpy_ctc_metrics_keep_legacy_semantics(ref, hyp):
    pytest.importorskip('torch')
    from liptrace.training import cer_score as legacy_cer, wer_score as legacy_wer
    assert cer_score(ref, hyp) == legacy_cer(ref, hyp)
    assert wer_score(ref, hyp) == legacy_wer(ref, hyp)
    assert decode_path([0, 0, 2, 0, 1, 1], ['a', 'b'], 2) == 'aab'


def test_cli_and_prediction_with_training_imports_blocked(artifact, tmp_path):
    root, receipt = artifact
    script = r"""
import contextlib, importlib.abc, io, json, sys
from pathlib import Path
blocked = {'torch', 'onnx', 'pandas', 'tqdm'}
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Training dependency import blocked: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
from liptrace.cli import main
from liptrace.bundle import read_bundle
from liptrace.replay import read_trace, compare_traces
root, out, digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def run(args, expected=0):
    sys.argv = ['liptrace'] + [str(x) for x in args]
    stderr = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
        try:
            main()
            code = 0
        except SystemExit as exc:
            code = exc.code
    if expected == 0:
        assert code == 0, (code, stderr.getvalue())
    else:
        assert code != 0 and 'liptrace[training]' in str(code) + stderr.getvalue()
run(['inspect', '--bundle', root / 'deployment', '--expect-manifest-sha256', digest])
run(['predict', '--bundle', root / 'deployment', '--video', root / 'clip.avi',
     '--runtime', 'onnx', '--reference', 'ab', '--out', out / 'onnx'])
run(['compare', '--left', root / 'torch', '--right', out / 'onnx', '--out', out / 'comparison.json'])
run(['train'], 2)
run(['export', '--bundle', root / 'bundle', '--out', out / 'export'], 2)
run(['trace', '--bundle', root / 'bundle', '--video', root / 'clip.avi', '--out', out / 'torch'], 2)
assert not any(key.split('.')[0] in blocked for key in sys.modules)
assert json.loads((out / 'comparison.json').read_text())['pass']
print('PASS: inspect, prediction, compare; torch/onnx/pandas/tqdm imports blocked')
"""
    result = subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path),
                             receipt['manifest_sha256']], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS:' in result.stdout
