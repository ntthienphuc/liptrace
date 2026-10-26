"""Matched whole-clip endpoints with per-window and globally collapsed CTC controls."""

import time

import numpy as np

from confirmation_common import edit_counts, normalize
from geometry import make_sliding
from inference_core import _Session, _project


def evaluate_clip(runtime, pixels, row, job):
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim == 0 or len(pixels) < 1:
        raise ValueError('Expected a nonempty uint8 pixel array')
    if len(pixels) != row['frames'] or row['metric'] != runtime.metric:
        raise ValueError('Pixel/manifest/model contract differs')
    if runtime.metric not in ('wer', 'cer'):
        raise ValueError('Unsupported metric')
    if job.get('ownership_rule') != 'midpoint':
        raise ValueError('Whole-clip allocation requires unique midpoint ownership')
    if job['arms'] != ['B', 'C', 'CTC_stitched', 'F']:
        raise ValueError('All matched whole-clip controls are required')
    decoders = job['decoders']
    names = [item['name'] for item in decoders]
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Decoder names must be nonempty and unique')
    decoder_map = {item['name']: item for item in decoders}
    if any(item['kind'] not in ('hybrid', 'greedy', 'ctc') for item in decoders):
        raise ValueError('Unknown decoder kind')
    allowed = job.get('arm_decoders', {})
    if set(allowed) != set(job['arms']):
        raise ValueError('Every whole-clip arm requires an explicit decoder selection')
    if any(not isinstance(values, list) or not values or len(set(values)) != len(values)
           or not set(values).issubset(decoder_map) for values in allowed.values()):
        raise ValueError('Invalid frozen arm-specific decoder selection')
    if any(decoder_map[name]['kind'] not in ('greedy', 'ctc')
           for name in job['arm_decoders']['CTC_stitched']):
        raise ValueError('Stitched control must decode CTC logits globally')
    cells = job['cells']
    cell_ids = [cell['cell_id'] for cell in cells]
    if (not cell_ids or any(not isinstance(name, str) or not name for name in cell_ids)
            or len(set(cell_ids)) != len(cell_ids)):
        raise ValueError('Cell IDs must be nonempty and unique')
    # Validate every schedule before invoking the encoder, including later cells.
    schedules = [make_sliding(len(pixels), cell['target_frames'], cell['context_frames']) for cell in cells]
    session = _Session(runtime)
    began = time.perf_counter()
    full, _ = session.encode(pixels)
    records = []
    for cell, windows in zip(cells, schedules):
        encoded_windows = []
        retained_logits = []
        for geometry in windows:
            context = pixels[geometry['input_start']:geometry['input_end']]
            encoded, _ = session.encode(context)
            lower, upper = geometry['local_start'], geometry['local_end']
            retained_logits.append(encoded['logp'][lower:upper])
            encoded_windows.append((geometry, encoded))
        stitched = np.concatenate(retained_logits, axis=0)
        if len(stitched) != len(pixels):
            raise AssertionError('Retained core logits do not cover each frame exactly once')
        for arm in job['arms']:
            for name in job['arm_decoders'][arm]:
                decoder = decoder_map[name]
                details = []
                encode_seconds = 0.0
                decode_seconds = 0.0
                alignment_seconds = 0.0
                projection_seconds = 0.0
                frames_encoded = len(pixels) if arm == 'F' else sum(
                    geometry['input_end'] - geometry['input_start'] for geometry in windows)
                if arm == 'F':
                    decoded, _ = session.decode(full['states'], full['logp'], decoder, len(pixels), native=True)
                    hypothesis = decoded['text']
                    encode_seconds = full['measured_seconds']
                    decode_seconds = decoded['measured_seconds']
                elif arm == 'CTC_stitched':
                    # CTC uses no attention states; the logits identify this exact global sequence.
                    decoded, _ = session.decode(stitched, stitched, decoder, len(pixels))
                    hypothesis = decoded['text']
                    encode_seconds = sum(encoded['measured_seconds'] for _, encoded in encoded_windows)
                    decode_seconds = decoded['measured_seconds']
                else:
                    hypotheses = []
                    for geometry, encoded in encoded_windows:
                        lower, upper = geometry['local_start'], geometry['local_end']
                        logp = encoded['logp']
                        states = encoded['states']
                        if arm == 'B':
                            states = states[lower:upper]
                            logp = logp[lower:upper]
                        decoded, _ = session.decode(states, logp, decoder, len(encoded['states']))
                        allocation = {'fallback': None, 'alignment_seconds': 0.0, 'projection_seconds': 0.0}
                        hypothesis = decoded['text']
                        if arm == 'C':
                            hypothesis, allocation = _project(session, decoded, logp, lower, upper,
                                                               'C', runtime.metric, 'midpoint')
                        hypotheses.append(hypothesis)
                        encode_seconds += encoded['measured_seconds']
                        decode_seconds += decoded['measured_seconds']
                        alignment_seconds += allocation['alignment_seconds']
                        projection_seconds += allocation['projection_seconds']
                        details.append({'geometry': geometry, 'hypothesis': hypothesis,
                                        'native_text': decoded['text'], 'tokens': decoded['tokens'],
                                        'input_pixel_sha256': encoded['pixel_sha256'],
                                        'input_frames': len(encoded['states']), 'decoded_state_frames': len(states),
                                        'output_cap': len(encoded['states']), 'allocation': allocation,
                                        'decoder_settings': decoded['settings'],
                                        'decode_trace': decoded.get('trace', {})})
                    hypothesis = normalize(' '.join(hypotheses))
                records.append({
                    'cell_id': cell['cell_id'], 'arm': arm, 'decoder': name,
                    'hypothesis': hypothesis, 'full_clip_endpoint': True,
                    'window_count': len(windows), 'windows': details,
                    'joins': [geometry['target_end'] for geometry in windows[:-1]],
                    'target_schedule': windows, 'frames_encoded': frames_encoded,
                    'nominal_target_frames': cell['target_frames'], 'context_frames': cell['context_frames'],
                    'ownership_rule': 'midpoint', 'unrestricted_context': arm == 'F',
                    'timing': {'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
                               'alignment_seconds': alignment_seconds,
                               'projection_seconds': projection_seconds,
                               'semantics': 'Component timings under shared GPU load; not isolated latency. '
                                            'Cached components retain their original timings; projection includes '
                                            'uncached alignment and is not additive with alignment_seconds'},
                })
    prediction_seconds = time.perf_counter() - began
    # Full-clip truth is accessed only after all recognition/allocation operations.
    reference = normalize(row['reference'])
    for record in records:
        record['reference'] = reference
        record['counts'] = edit_counts(reference, record['hypothesis'], runtime.metric)
    return {'records': records, 'diagnostics': {
        'reference_used_for_inference': False, 'references_accessed_after_all_predictions': True,
        'encode_calls': session.encode_calls, 'decode_calls': session.decode_calls,
        'alignment_calls': session.alignment_calls, 'encoded_frames': session.encoded_frames,
        'prediction_seconds': prediction_seconds, 'total_seconds': time.perf_counter() - began,
        'alignment_source': session.alignment_identity,
        'schedule': 'K=ceil(T/C); targets [floor(i*T/K), floor((i+1)*T/K))',
        'merging': 'Join predicted window strings; no reference-based repair or overlap deduplication',
        'ctc_stitched': 'Retain contextual core logits once per frame, then global greedy/prefix beam',
    }}


def frozen_geometry_queries(total, target=50, count=1):
    """Distinct sparse target intervals, independent of transcript and prediction scores."""
    if total < 1 or count not in (1, 3, 5):
        raise ValueError('Expected positive frames and one, three or five requested queries')
    length = min(total, target)
    gap = total - length
    starts = [gap // 2] if count == 1 else [index * gap // (count - 1) for index in range(count)]
    return sorted({(start, start + length) for start in starts})
