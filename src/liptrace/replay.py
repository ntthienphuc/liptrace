"""Record and compare observable stages, including discrete CTC decisions."""
import json
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from .bundle import read_bundle
from .contracts import sha256, json_digest, write_json, normalize
from .media import decode_frames, pack_frames
from .ctc import decode_path, cer_score, wer_score
from .deployment import require_training, require_onnxruntime, open_session

STAGES = ('decoded_frames', 'sample_indices', 'tensor', 'logits', 'ctc_path', 'transcript', 'metrics')



def trace_video(bundle, video, out, runtime='torch', reference=None, expected_manifest_sha256=None, threads=2):
    started = time.perf_counter()
    if runtime not in ('torch', 'onnx'):
        raise ValueError('Runtime must be torch or onnx')
    doc = read_bundle(bundle, expected_manifest_sha256)
    validation_ms = (time.perf_counter() - started) * 1000
    profile = doc['profile']
    target = Path(out)
    if target.exists():
        raise ValueError('Trace destination already exists')
    prep_start = time.perf_counter()
    frames = decode_frames(video)
    tensor, indices = pack_frames(frames, profile['model']['max_frames'], profile['model']['img_size'])
    preparation_ms = (time.perf_counter() - prep_start) * 1000
    load_start = time.perf_counter()
    if runtime == 'torch':
        if 'model.pt' not in doc['artifacts']:
            raise ValueError('ONNX-only deployment requires runtime="onnx" (CLI: --runtime onnx)')
        torch = require_training()
        from .runtime import load_recognizer
        runtime_version = torch.__version__
        model, _, _ = load_recognizer(Path(bundle) / 'model.pt')
        loading_ms = (time.perf_counter() - load_start) * 1000
        with torch.inference_mode():
            t0 = time.perf_counter()
            logits = model(torch.from_numpy(tensor)).numpy()
            forward_ms = (time.perf_counter() - t0) * 1000
    elif runtime == 'onnx':
        if 'model.onnx' not in doc['artifacts']:
            raise ValueError('Bundle has no ONNX export')
        ort = require_onnxruntime()
        session = open_session(bundle, doc, threads=threads)
        loading_ms = (time.perf_counter() - load_start) * 1000
        t0 = time.perf_counter()
        logits = session.run(['logits'], {'video': tensor})[0]
        forward_ms = (time.perf_counter() - t0) * 1000
        runtime_version = ort.__version__
    else:
        raise ValueError('Runtime must be torch or onnx')
    if logits.dtype != np.float32 or logits.shape != (1, profile['model']['max_frames'], len(profile['charset']) + 1) or not np.isfinite(logits).all():
        raise ValueError('Output shape or finite-logit contract failed')
    ctc_start = time.perf_counter()
    path = logits.argmax(-1)[0].astype(np.int64)
    transcript = decode_path(path, profile['charset'], profile['blank_index'])
    ctc_ms = (time.perf_counter() - ctc_start) * 1000
    metrics = None
    if reference is not None:
        ref = normalize(reference, profile['normalization'])
        hyp = normalize(transcript, profile['normalization'])
        metrics = {'reference': ref, 'hypothesis': hyp, 'cer': cer_score(ref, hyp),
                   'wer': wer_score(ref, hyp), 'exact_match': ref == hyp,
                   'aggregation': 'one_sample; not corpus-weighted'}
    target.mkdir(parents=True)
    np.savez_compressed(target / 'stages.npz', decoded_frames=frames, sample_indices=indices,
                        tensor=tensor, logits=logits, ctc_path=path)
    trace = {'schema': 'liptrace-replay-v1', 'profile': profile,
             'profile_sha256': json_digest(profile), 'manifest_sha256': sha256(Path(bundle) / 'manifest.json'),
             'video_sha256': sha256(video), 'stages_sha256': sha256(target / 'stages.npz'),
             'transcript': transcript, 'metrics': metrics, 'runtime': runtime,
             'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                             'runtime_version': runtime_version, 'numpy': np.__version__, 'opencv': cv2.__version__},
             'model_forward_ms': forward_ms, 'timing_scope': 'one CPU forward; excludes decoding and loading; no warmup',
             'accuracy_evaluation': reference is not None}
    trace['artifact_schema'] = doc['schema']
    trace['verification'] = doc.get('verification', {
        'scope': 'bundle artifact hashes and checkpoint/profile metadata', 'checkpoint_verified': True})
    trace['model_identity'] = {'source_checkpoint_sha256': doc['provenance']['source_checkpoint_sha256'],
                               'artifacts': doc['artifacts']}
    trace['timings_ms'] = {'validation': validation_ms, 'preparation': preparation_ms,
                           'loading': loading_ms, 'forward': forward_ms, 'ctc': ctc_ms,
                           'full_execution': (time.perf_counter() - started) * 1000}
    trace['timings_scope'] = 'one execution including artifact validation and array persistence; excludes final trace JSON write; no warmup'
    write_json(target / 'trace.json', trace)
    return trace


def read_trace(directory):
    root = Path(directory)
    doc = json.loads((root / 'trace.json').read_text(encoding='utf-8'))
    if doc.get('schema') != 'liptrace-replay-v1' or sha256(root / 'stages.npz') != doc.get('stages_sha256'):
        raise ValueError('Trace schema or array identity mismatch')
    from .contracts import validate_profile
    validate_profile(doc['profile'])
    if doc.get('profile_sha256') != json_digest(doc['profile']):
        raise ValueError('Trace profile identity mismatch')
    with np.load(root / 'stages.npz', allow_pickle=False) as data:
        if set(data.files) != set(STAGES[:5]):
            raise ValueError('Trace arrays missing stages')
        arrays = {key: data[key] for key in data.files}
    cfg = doc['profile']['model']
    frames = arrays['decoded_frames']
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or not all(frames.shape):
        raise ValueError('Invalid decoded frame geometry or dtype')
    shapes = {'sample_indices': ((cfg['max_frames'],), np.dtype('int64')),
              'tensor': ((1, 1, cfg['max_frames'], cfg['img_size'], cfg['img_size']), np.dtype('float32')),
              'logits': ((1, cfg['max_frames'], len(doc['profile']['charset']) + 1), np.dtype('float32')),
              'ctc_path': ((cfg['max_frames'],), np.dtype('int64'))}
    for key, (shape, dtype) in shapes.items():
        if arrays[key].shape != shape or arrays[key].dtype != dtype or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid trace stage shape, dtype or finite values: {key}')
    if not isinstance(doc.get('transcript'), str):
        raise ValueError('Invalid trace transcript')
    if 'metrics' not in doc or (doc['metrics'] is not None and not isinstance(doc['metrics'], dict)):
        raise ValueError('Invalid trace metrics')
    return doc, arrays


def trace_consistency(doc, arrays):
    """Check derived stages independently of another trace's agreement.

    This validates recorded relationships, not that logits really came from the
    named model. Trust still requires an externally recorded artifact identity.
    """
    profile = doc['profile']
    tensor, indices = pack_frames(arrays['decoded_frames'], profile['model']['max_frames'],
                                  profile['model']['img_size'])
    expected_path = arrays['logits'].argmax(-1)[0].astype(np.int64)
    checks = [
        {'stage': 'sample_indices', 'pass': bool(np.array_equal(indices, arrays['sample_indices']))},
        {'stage': 'tensor', 'pass': bool(np.array_equal(tensor, arrays['tensor']))},
        {'stage': 'ctc_path', 'pass': bool(np.array_equal(expected_path, arrays['ctc_path']))},
    ]
    try:
        transcript = decode_path(arrays['ctc_path'], profile['charset'], profile['blank_index'])
    except ValueError:
        transcript = None
    checks.append({'stage': 'transcript', 'pass': transcript is not None and transcript == doc['transcript']})
    metrics = doc['metrics']
    metrics_ok = metrics is None
    if metrics is not None:
        ref = metrics.get('reference')
        if isinstance(ref, str):
            normalized = normalize(ref, profile['normalization'])
            hyp = normalize(doc['transcript'], profile['normalization'])
            expected = {'reference': normalized, 'hypothesis': hyp, 'cer': cer_score(normalized, hyp),
                        'wer': wer_score(normalized, hyp), 'exact_match': normalized == hyp,
                        'aggregation': 'one_sample; not corpus-weighted'}
            metrics_ok = metrics == expected
    checks.append({'stage': 'metrics', 'pass': metrics_ok and doc.get('accuracy_evaluation') is (metrics is not None)})
    return {'pass': all(c['pass'] for c in checks), 'checks': checks}


def compare_traces(left, right, atol=1e-5, rtol=1e-4):
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    a, aa = read_trace(left)
    b, bb = read_trace(right)
    checks = []
    # Ordered charset and preprocessing are semantic identity; ONNX artifacts may differ.
    checks.append({'stage': 'profile', 'pass': a['profile'] == b['profile']})
    checks.append({'stage': 'source_video', 'pass': a['video_sha256'] == b['video_sha256']})
    for stage in STAGES:
        if stage in aa:
            x, y = aa[stage], bb[stage]
            same_shape = x.shape == y.shape and x.dtype == y.dtype
            numeric = stage in ('tensor', 'logits')
            finite = np.isfinite(x).all() and np.isfinite(y).all()
            passed = bool(same_shape and finite and
                          (np.allclose(x, y, atol=atol, rtol=rtol) if numeric else np.array_equal(x, y)))
            check = {'stage': stage, 'pass': passed, 'same_shape_dtype': same_shape}
            if numeric and same_shape and finite:
                check['max_abs_error'] = float(np.max(np.abs(x.astype(np.float64) - y)))
        else:
            check = {'stage': stage, 'pass': a[stage] == b[stage]}
        checks.append(check)
    failed = next((c['stage'] for c in checks if not c['pass']), None)
    consistency = {'left': trace_consistency(a, aa), 'right': trace_consistency(b, bb)}
    if failed is None:
        failed = next((f'{side}_trace.{check["stage"]}' for side, report in consistency.items()
                       for check in report['checks'] if not check['pass']), None)
    return {'schema': 'liptrace-comparison-v1', 'pass': failed is None,
            'first_divergence': failed, 'checks': checks, 'trace_consistency': consistency,
            'atol': atol, 'rtol': rtol,
            'scope': 'two recorded executions; internal consistency and discrete path/transcript equality required'}
