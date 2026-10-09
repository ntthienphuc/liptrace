import csv
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from liptrace.bundle import create_bundle, read_bundle, export_onnx
from liptrace.contracts import normalize, sha256, write_json, json_digest
from liptrace.dataset import audit_dataset
from liptrace.media import decode_frames, pack_frames
from liptrace.model import LipNetBackbone
from liptrace.replay import trace_video, compare_traces, decode_path


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(12)
    cfg = {'img_size': 16, 'max_frames': 6, 'rnn_units': 4, 'dropout': 0.0}
    model = LipNetBackbone(3, **cfg)
    p = tmp_path / 'original.pt'
    torch.save({'model_state': model.state_dict(), 'charset': ['a', 'b'], 'blank_index': 2,
                'args': {**cfg, 'train_csv': 'private-local-path'}, 'optimizer_state': {'secret': 'not-public'}}, p)
    return p


def video(path, value=60):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
    assert writer.isOpened()
    for i in range(4):
        writer.write(np.full((16, 16, 3), value + i * 10, np.uint8))
    writer.release()
    return path


def manifests(root, mutation=None):
    paths = []
    for i, role in enumerate(['train', 'val', 'test']):
        clip = video(root / f'{role}.avi', i * 60)
        row = {'clip_path': clip.name, 'text': 'ab', 'group_key': role,
               'start_s': '0', 'end_s': '0.4', 'source_duration_s': '1'}
        if mutation:
            mutation(i, row, clip)
        p = root / f'{role}.csv'
        with p.open('w', encoding='utf-8', newline='') as h:
            w = csv.DictWriter(h, fieldnames=row)
            w.writeheader()
            w.writerow(row)
        paths.append(p)
    return paths


def codes(report):
    return {x['code'] for x in report['issues']}


def test_bundle_sanitizes_training_state_and_binds_order(checkpoint, tmp_path):
    root = tmp_path / 'bundle'
    receipt = create_bundle(checkpoint, root)
    doc = read_bundle(root, receipt['manifest_sha256'])
    c = torch.load(root / 'model.pt', weights_only=True)
    assert set(c) == {'model_state', 'args', 'charset', 'blank_index'}
    assert 'train_csv' not in c['args']
    doc['profile']['charset'].reverse()
    write_json(root / 'manifest.json', doc)
    with pytest.raises(ValueError, match='profile mismatch|profile identity'):
        read_bundle(root)
    with pytest.raises(ValueError, match='Manifest identity'):
        read_bundle(root, receipt['manifest_sha256'])


def test_checkpoint_same_shape_swap_rejected_by_bundle_hash(checkpoint, tmp_path):
    root = tmp_path / 'bundle'
    create_bundle(checkpoint, root)
    c = torch.load(root / 'model.pt', weights_only=True)
    c['charset'].reverse()
    torch.save(c, root / 'model.pt')
    with pytest.raises(ValueError, match='Artifact identity'):
        read_bundle(root)


def test_bundle_inherits_and_preserves_training_normalization(checkpoint, tmp_path):
    ckpt = torch.load(checkpoint, weights_only=True)
    ckpt['args']['normalization'] = 'nfc-lower-v1'
    torch.save(ckpt, checkpoint)
    root = tmp_path / 'bundle'
    create_bundle(checkpoint, root)
    doc = read_bundle(root)
    assert doc['profile']['normalization'] == 'nfc-lower-v1'
    assert torch.load(root / 'model.pt', weights_only=True)['args']['normalization'] == 'nfc-lower-v1'
    with pytest.raises(ValueError, match='conflicts'):
        create_bundle(checkpoint, tmp_path / 'conflict', 'legacy-lower-v1')
    doc['profile']['normalization'] = 'legacy-lower-v1'
    doc['provenance']['profile_sha256'] = json_digest(doc['profile'])
    write_json(root / 'manifest.json', doc)
    with pytest.raises(ValueError, match='normalization mismatch'):
        read_bundle(root)


def test_old_bundle_without_recorded_normalization_remains_readable(checkpoint, tmp_path):
    root = tmp_path / 'bundle'
    create_bundle(checkpoint, root)
    ckpt = torch.load(root / 'model.pt', weights_only=True)
    del ckpt['args']['normalization']
    torch.save(ckpt, root / 'model.pt')
    doc = json.loads((root / 'manifest.json').read_text())
    doc['artifacts']['model.pt'] = sha256(root / 'model.pt')
    write_json(root / 'manifest.json', doc)
    assert read_bundle(root)['profile']['normalization'] == 'legacy-lower-v1'


@pytest.mark.parametrize('field,value', [('blank_index', 0), ('decoder', 'beam'),
    ('normalization', 'unknown'), ('charset', ['a', 'a']), ('charset', ['ab']),
    ('architecture', 'arbitrary-python-plugin')])
def test_profile_rejects_unsupported_contract(checkpoint, tmp_path, field, value):
    root = tmp_path / 'bundle'
    create_bundle(checkpoint, root)
    doc = json.loads((root / 'manifest.json').read_text())
    doc['profile'][field] = value
    write_json(root / 'manifest.json', doc)
    with pytest.raises(ValueError):
        read_bundle(root)


def test_exact_duplicate_different_names_and_groups(tmp_path):
    paths = manifests(tmp_path)
    shutil.copyfile(tmp_path / 'train.avi', tmp_path / 'val.avi')
    report = audit_dataset(paths, 6)
    assert not report['valid']
    assert report['overlaps']['train:val']['path'] == 0
    assert report['overlaps']['train:val']['group'] == 0
    assert 'content_overlap' in codes(report)


@pytest.mark.parametrize('key,value,code', [
    ('group_key', '', 'missing_group'), ('clip_path', 'missing.avi', 'missing_media'),
    ('text', '', 'empty_label'), ('text', 'aaaabbbb', 'ctc_infeasible'),
    ('text', 'z', 'unseen_character'), ('start_s', '-1', 'invalid_timestamp'),
    ('start_s', 'nan', 'invalid_timestamp'), ('end_s', '0', 'invalid_timestamp'),
    ('end_s', '2', 'invalid_timestamp')])
def test_dataset_faults_collect_reasons(tmp_path, key, value, code):
    paths = manifests(tmp_path, lambda i, row, clip: row.update({key: value}) if i == 1 else None)
    report = audit_dataset(paths, 6)
    assert not report['valid']
    assert code in codes(report)


def test_group_overlap_and_corrupt_media(tmp_path):
    paths = manifests(tmp_path, lambda i, row, clip: row.update(group_key='same'))
    (tmp_path / 'test.avi').write_bytes(b'not a video')
    report = audit_dataset(paths, 6)
    assert {'group_overlap', 'invalid_media'} <= codes(report)


@pytest.mark.parametrize('csv_text,code', [
    ('clip_path,text,group_key,text\ntrain.avi,ab,train,ab\n', 'manifest_schema'),
    ('clip_path,text,group_key\ntrain.avi,ab,train,unexpected\n', 'manifest_row_shape'),
    ('clip_path,text,group_key\ntrain.avi,ab\n', 'manifest_row_shape')])
def test_malformed_csv_cannot_silently_discard_values(tmp_path, csv_text, code):
    paths = manifests(tmp_path)
    paths[0].write_text(csv_text, encoding='utf-8')
    report = audit_dataset(paths, 6)
    assert not report['valid'] and code in codes(report)


def test_healthy_split_and_manifest_relative_paths(tmp_path, monkeypatch):
    paths = manifests(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    report = audit_dataset(paths, 6)
    assert report['valid'], report['issues']
    assert report['charset'] == ['a', 'b']
    assert all(Path(r['path']).is_absolute() for r in report['records'])


def test_normalization_preserves_accents_and_legacy_is_explicit():
    assert normalize(' MÁ ', 'nfc-lower-v1') == 'má'
    assert normalize('ma\u0301', 'nfc-lower-v1') == 'má'
    assert normalize('ma\u0301') != normalize('má')
    assert normalize(' MÁ ') != 'ma'


def test_normalization_collisions_reported(tmp_path):
    paths = manifests(tmp_path, lambda i, row, clip: row.update(text='má' if i == 0 else 'ma\u0301'))
    report = audit_dataset(paths, 6, 'nfc-lower-v1')
    assert report['valid']
    assert 'normalization_collision' in codes(report)


def test_ctc_repeats_require_blank():
    assert decode_path([0, 0, 2, 0, 1, 1], ['a', 'b'], 2) == 'aab'
    with pytest.raises(ValueError):
        decode_path([3], ['a', 'b'], 2)


def traces(tmp_path, checkpoint):
    root = tmp_path / 'bundle'
    create_bundle(checkpoint, root)
    clip = video(tmp_path / 'clip.avi')
    for name in ['a', 'b']:
        trace_video(root, clip, tmp_path / name, reference='ab')
    return tmp_path / 'a', tmp_path / 'b'


def rewrite_array(root, transform):
    with np.load(root / 'stages.npz', allow_pickle=False) as h:
        data = {k: h[k] for k in h.files}
    transform(data)
    np.savez_compressed(root / 'stages.npz', **data)
    doc = json.loads((root / 'trace.json').read_text())
    doc['stages_sha256'] = sha256(root / 'stages.npz')
    write_json(root / 'trace.json', doc)


def test_repeat_replay_is_equal(tmp_path, checkpoint):
    a, b = traces(tmp_path, checkpoint)
    assert compare_traces(a, b)['pass']


@pytest.mark.parametrize('stage', ['decoded_frames', 'sample_indices', 'tensor', 'logits', 'ctc_path'])
def test_first_divergent_array(tmp_path, checkpoint, stage):
    a, b = traces(tmp_path, checkpoint)
    def modify(data):
        data[stage] = data[stage].copy()
        data[stage].flat[0] += 1
    rewrite_array(b, modify)
    result = compare_traces(a, b)
    assert result['first_divergence'] == stage
    assert not result['pass']


def test_tiny_numeric_difference_can_change_ctc_decision(tmp_path, checkpoint):
    a, b = traces(tmp_path, checkpoint)
    for root, winner in [(a, 0), (b, 1)]:
        def modify(data):
            data['logits'] = np.zeros_like(data['logits'])
            data['logits'][:, :, winner] = 1e-7
            data['ctc_path'] = data['logits'].argmax(-1)[0].astype(np.int64)
        rewrite_array(root, modify)
        doc = json.loads((root / 'trace.json').read_text())
        doc['transcript'] = 'a' if winner == 0 else 'b'
        write_json(root / 'trace.json', doc)
    result = compare_traces(a, b)
    assert next(c['pass'] for c in result['checks'] if c['stage'] == 'logits')
    assert result['first_divergence'] == 'ctc_path'


def test_trace_array_tampering_rejected(tmp_path, checkpoint):
    a, b = traces(tmp_path, checkpoint)
    with (b / 'stages.npz').open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError, match='identity'):
        compare_traces(a, b)


@pytest.mark.parametrize('stage', ['sample_indices', 'tensor', 'ctc_path'])
def test_identically_inconsistent_traces_cannot_pass(tmp_path, checkpoint, stage):
    a, b = traces(tmp_path, checkpoint)
    for root in (a, b):
        def modify(data):
            if stage == 'ctc_path':
                data[stage][0] = (data[stage][0] + 1) % 3
            else:
                data[stage].flat[0] += 1
        rewrite_array(root, modify)
    result = compare_traces(a, b)
    assert not result['pass']
    assert result['first_divergence'] == f'left_trace.{stage}'


@pytest.mark.parametrize('stage', ['transcript', 'metrics'])
def test_stale_json_stages_cannot_pass_by_matching_each_other(tmp_path, checkpoint, stage):
    a, b = traces(tmp_path, checkpoint)
    for root in (a, b):
        doc = json.loads((root / 'trace.json').read_text())
        if stage == 'transcript':
            doc[stage] = 'not the CTC decode'
        else:
            doc[stage]['cer'] = -1.0
        write_json(root / 'trace.json', doc)
    result = compare_traces(a, b)
    assert not result['pass']
    assert result['first_divergence'] == f'left_trace.{stage}'


@pytest.mark.parametrize('stage', ['tensor', 'logits', 'ctc_path'])
def test_trace_arrays_must_match_profile_geometry(tmp_path, checkpoint, stage):
    a, b = traces(tmp_path, checkpoint)
    rewrite_array(b, lambda data: data.update({stage: data[stage].reshape(-1)[:1]}))
    with pytest.raises(ValueError, match='shape'):
        compare_traces(a, b)


def test_decode_limits_and_padding(tmp_path):
    clip = video(tmp_path / 'clip.avi')
    with pytest.raises(ValueError, match='resource limit'):
        decode_frames(clip, max_decoded=2)
    frames = decode_frames(clip)
    tensor, indices = pack_frames(frames, 6, 16)
    assert indices.tolist() == [0, 1, 2, 3, 3, 3]
    np.testing.assert_array_equal(tensor[:, :, 3], tensor[:, :, 5])


@pytest.mark.parametrize('atol,rtol', [(float('nan'), 0), (-1, 0), (0, float('inf'))])
def test_bad_tolerances_rejected(tmp_path, atol, rtol):
    with pytest.raises(ValueError, match='Tolerances'):
        compare_traces(tmp_path, tmp_path, atol, rtol)


def test_onnx_export_replay(tmp_path, checkpoint):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    source, target = tmp_path / 'source', tmp_path / 'exported'
    create_bundle(checkpoint, source)
    export_onnx(source, target)
    clip = video(tmp_path / 'clip.avi')
    trace_video(source, clip, tmp_path / 'torch')
    trace_video(target, clip, tmp_path / 'onnx', 'onnx')
    assert compare_traces(tmp_path / 'torch', tmp_path / 'onnx')['pass']
