import json
import numpy as np
import pytest
import torch
from liptrace.bundle import create_bundle, export_onnx
from liptrace.caseio import load_case
from liptrace.contracts import sha256, write_json
from liptrace.ctc import cer_score, wer_score
from liptrace.metrics import corpus_metrics, edit_counts, sample_metrics
from liptrace.migration import run_migration
from liptrace.parity import compare_task_traces
from liptrace.original import compare_original
from liptrace.replay import read_trace
from test_trace_contracts import checkpoint, video, traces, rewrite_array


def test_weighted_metrics_do_not_equal_mean_and_include_empty_hypothesis():
    report = corpus_metrics([{'reference': 'a', 'hypothesis': ''},
                             {'reference': 'book', 'hypothesis': 'bok'}])
    assert report['sample_mean']['cer'] == .625
    assert report['corpus_weighted']['cer'] == .4
    assert report['counts']['characters']['deletions'] == 2
    assert report['counts']['characters']['edits'] == 2
    assert report['sample_mean']['exact_match'] == 0
    assert corpus_metrics([])['sample_mean']['cer'] is None


@pytest.mark.parametrize('reference,hypothesis,counts', [
    ('abc', 'axc', (1, 0, 0)), ('abc', 'ab', (0, 1, 0)),
    ('ab', 'abc', (0, 0, 1)), ('', 'abc', (0, 0, 3)), ('abc', '', (0, 3, 0))])
def test_edit_counts_and_legacy_empty_reference(reference, hypothesis, counts):
    report = edit_counts(reference, hypothesis)
    assert tuple(report[key] for key in ['substitutions', 'deletions', 'insertions']) == counts
    sample = sample_metrics(reference, hypothesis)
    assert sample['cer'] == cer_score(reference, hypothesis)
    assert sample['wer'] == wer_score(reference, hypothesis)


def test_benign_ctc_path_change_is_separate_from_strict_fidelity(tmp_path, checkpoint):
    a, b = traces(tmp_path, checkpoint)
    paths = [[0, 0, 2, 2, 2, 2], [2, 0, 0, 2, 2, 2]]
    for root, path in zip((a, b), paths):
        def modify(data):
            logits = np.zeros_like(data['logits'])
            for i, winner in enumerate(path):
                logits[0, i, winner] = 1
            data['logits'] = logits
            data['ctc_path'] = np.asarray(path, dtype=np.int64)
        rewrite_array(root, modify)
        doc = json.loads((root / 'trace.json').read_text())
        doc['transcript'] = 'a'
        doc['metrics']['hypothesis'] = 'a'
        doc['metrics']['cer'] = cer_score('ab', 'a')
        doc['metrics']['wer'] = wer_score('ab', 'a')
        write_json(root / 'trace.json', doc)
    report = compare_task_traces(a, b)
    assert report['internal_consistency']
    assert not report['strict_fidelity']
    assert not report['path_agreement']
    assert report['task_output_preserved'] and report['path_changed_output_preserved']


def test_task_parity_cannot_accept_shared_inconsistency(tmp_path, checkpoint):
    a, b = traces(tmp_path, checkpoint)
    for root in (a, b):
        doc = json.loads((root / 'trace.json').read_text())
        doc['transcript'] = 'unsupported by recorded path'
        write_json(root / 'trace.json', doc)
    assert not compare_task_traces(a, b)['task_output_preserved']


def case_document(tmp_path, checkpoint):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    create_bundle(checkpoint, tmp_path / 'bundle')
    export_onnx(tmp_path / 'bundle', tmp_path / 'deployment', runtime_only=True)
    clip = video(tmp_path / 'clip.avi')
    case = {'schema': 'liptrace-migration-case-v1', 'bundle': 'bundle', 'deployment': 'deployment',
            'samples': [{'sample_id': 'valid', 'source_group': 'generated-1', 'video': clip.name,
                         'reference': 'ab', 'reference_status': 'unreviewed',
                         'expected_video_sha256': sha256(clip)}]}
    write_json(tmp_path / 'case.json', case)
    return case


def test_corpus_retains_failed_inputs_in_denominator(tmp_path, checkpoint):
    case = case_document(tmp_path, checkpoint)
    case['samples'] += [{'sample_id': 'missing', 'source_group': 'generated-2', 'video': 'missing.mp4'},
                        {'sample_id': 'wrong-hash', 'source_group': 'generated-3', 'video': 'clip.avi',
                         'expected_video_sha256': '0' * 64}]
    write_json(tmp_path / 'case.json', case)
    report = run_migration(tmp_path / 'case.json', tmp_path / 'output', warmup=0, repetitions=1)
    summary = report['summary']
    assert summary['planned_samples'] == 3 and summary['rows_with_errors'] == 2
    assert not summary['pass']
    comparison = summary['comparisons']['torch_onnx']
    assert comparison['evaluated'] == 1 and comparison['missing'] == 2
    assert comparison['strict_pass_over_all_planned'] == pytest.approx(1 / 3)
    assert len((tmp_path / 'output' / 'predictions.csv').read_text().splitlines()) == 4
    assert summary['metrics_by_label_status']['torch']['reviewed']['successful_labeled_samples'] == 0
    assert summary['metrics_by_label_status']['torch']['unreviewed']['successful_labeled_samples'] == 1
    assert report['samples'][0]['benchmarks']['onnx']['stable_transcript']
    with pytest.raises(ValueError, match='already exists'):
        run_migration(tmp_path / 'case.json', tmp_path / 'output')


def test_duplicate_ids_and_untrusted_adapter_rejected(tmp_path, checkpoint):
    case = case_document(tmp_path, checkpoint)
    case['samples'].append(case['samples'][0].copy())
    write_json(tmp_path / 'case.json', case)
    with pytest.raises(ValueError, match='unique'):
        load_case(tmp_path / 'case.json')
    case['samples'].pop()
    case['original_adapter'] = 'trusted.py:factory'
    case['checkpoint'] = 'original.pt'
    write_json(tmp_path / 'case.json', case)
    with pytest.raises(ValueError, match='allow-original-adapter'):
        run_migration(tmp_path / 'case.json', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_same_profile_different_source_model_is_not_a_valid_export(tmp_path, checkpoint):
    case = case_document(tmp_path, checkpoint)
    second = torch.load(checkpoint, weights_only=True)
    first_key = next(iter(second['model_state']))
    second['model_state'][first_key] += 1
    second_checkpoint = tmp_path / 'second.pt'
    torch.save(second, second_checkpoint)
    create_bundle(second_checkpoint, tmp_path / 'other_bundle')
    export_onnx(tmp_path / 'other_bundle', tmp_path / 'other_deployment', runtime_only=True)
    case['deployment'] = 'other_deployment'
    write_json(tmp_path / 'case.json', case)
    with pytest.raises(ValueError, match='source checkpoint differ'):
        run_migration(tmp_path / 'case.json', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_original_comparison_checks_package_internal_consistency(tmp_path, checkpoint):
    a, b = traces(tmp_path, checkpoint)
    document, arrays = read_trace(a)
    original = {'tensor': arrays['tensor'], 'logits': arrays['logits'],
                'transcript': document['transcript']}
    assert compare_original(original, a, document['profile'])['pass']
    document = json.loads((b / 'trace.json').read_text())
    document['transcript'] = 'inconsistent'
    write_json(b / 'trace.json', document)
    result = compare_original(original, b, document['profile'])
    assert not result['pass'] and not result['task_output_preserved']
    assert result['first_divergence'] == 'packaged_trace_consistency'
