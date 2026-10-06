"""Collect every exclusion reason before training; groups are asserted metadata."""
import csv
import itertools
import math
from pathlib import Path
from .contracts import sha256, normalize
from .media import decode_frames


def audit_dataset(paths, max_frames=32, normalization='legacy-lower-v1', decode=True):
    if len(paths) != 3 or type(max_frames) is not int or max_frames <= 0:
        raise ValueError('Supply three split manifests and positive max_frames')
    roles = ['train', 'val', 'test']
    records, issues, manifests = [], [], []

    def issue(code, role, row, detail, severity='error'):
        issues.append({'code': code, 'role': role, 'row': row, 'detail': detail, 'severity': severity})

    for role, path in zip(roles, paths):
        path = Path(path).resolve()
        manifests.append({'role': role, 'sha256': sha256(path)})
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            label_col = next((c for c in ['text', 'label_ascii', 'label_text'] if c in fields), None)
            group_col = next((c for c in ['group_key', 'speaker_dir', 'stem'] if c in fields), None)
            if not label_col or 'clip_path' not in fields or not group_col:
                issue('manifest_schema', role, None, 'Need clip_path, text and group_key (or speaker_dir/stem)')
                continue
            rows = list(reader)
        if not rows:
            issue('empty_split', role, None, 'Empty manifest')
        for number, row in enumerate(rows, 2):
            raw = row.get(label_col) or ''
            text = normalize(raw, normalization)
            clip = (row.get('clip_path') or '').strip()
            group = (row.get(group_col) or '').strip()
            if not text:
                issue('empty_label', role, number, 'Label is empty after normalization')
            if not group:
                issue('missing_group', role, number, 'Missing source group; no person identity is inferred')
            resolved = Path(clip) if Path(clip).is_absolute() else path.parent / clip
            record = {'role': role, 'row': number, 'text': text, 'raw_text': raw,
                      'group': group, 'path': str(resolved.resolve()), 'content_sha256': None}
            records.append(record)
            required = len(text) + sum(a == b for a, b in zip(text, text[1:]))
            if required > max_frames:
                issue('ctc_infeasible', role, number, f'Need {required} CTC steps, model supplies {max_frames}')
            if 'start_s' in row or 'end_s' in row:
                try:
                    start, end = float(row.get('start_s', '')), float(row.get('end_s', ''))
                    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                        raise ValueError()
                    if 'source_duration_s' in row:
                        duration = float(row['source_duration_s'])
                        if not math.isfinite(duration) or end > duration:
                            raise ValueError()
                except (ValueError, TypeError):
                    issue('invalid_timestamp', role, number, 'Need finite 0 <= start < end <= supplied source duration')
            if not clip or not resolved.is_file():
                issue('missing_media', role, number, clip or 'Empty clip path')
                continue
            record['content_sha256'] = sha256(resolved)
            if decode:
                try:
                    record['decoded_frames'] = len(decode_frames(resolved))
                except (RuntimeError, ValueError) as exc:
                    issue('invalid_media', role, number, str(exc))
    charset = sorted(set(''.join(r['text'] for r in records if r['role'] == 'train')))
    for r in records:
        if r['role'] != 'train':
            unseen = sorted(set(r['text']) - set(charset))
            if unseen:
                issue('unseen_character', r['role'], r['row'], repr(unseen))
    overlaps = {}
    for a, b in itertools.combinations(roles, 2):
        ra, rb = [r for r in records if r['role'] == a], [r for r in records if r['role'] == b]
        counts = {}
        for key, code in [('group', 'group_overlap'), ('path', 'path_overlap'),
                          ('content_sha256', 'content_overlap')]:
            common = {r[key] for r in ra if r[key]} & {r[key] for r in rb if r[key]}
            counts[key] = len(common)
            if common:
                issue(code, f'{a}:{b}', None, f'{len(common)} shared {key} identities')
        overlaps[f'{a}:{b}'] = counts
    by_hash, by_label = {}, {}
    for r in records:
        if r['content_sha256']:
            by_hash.setdefault(r['content_sha256'], []).append(r)
        by_label.setdefault(r['text'], set()).add(r['raw_text'])
    for identical in by_hash.values():
        if len({r['text'] for r in identical}) > 1:
            issue('content_label_conflict', 'all', None, 'Identical encoded media has different normalized labels')
        if len(identical) > 1 and len({r['role'] for r in identical}) == 1:
            issue('duplicate_within_split', identical[0]['role'], None, f'{len(identical)} identical files', 'warning')
    for text, originals in by_label.items():
        if len(originals) > 1:
            issue('normalization_collision', 'all', None,
                  f'{len(originals)} raw labels normalize to {text!r}', 'warning')
    return {'schema': 'liptrace-dataset-audit-v1', 'valid': not any(i['severity'] == 'error' for i in issues),
            'max_frames': max_frames, 'normalization': normalization, 'charset_source': 'train_only',
            'charset': charset, 'manifest_identities': manifests, 'overlaps': overlaps,
            'rows': {role: sum(r['role'] == role for r in records) for role in roles},
            'issues': issues, 'records': records,
            'identity_scope': 'source-group metadata; speaker identity unverified',
            'duplicate_scope': 'exact encoded bytes; re-encoded or visually similar clips not detected',
            'decode_checked': decode}
