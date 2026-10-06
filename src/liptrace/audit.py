"""Manifest audits and historical sample-mean metric recomputation."""
import itertools
from pathlib import Path
import pandas as pd


def validate_splits(paths, require_files=False):
    """Legacy identifier-only check. Use dataset.audit_dataset for full media/CTC audit."""
    if len(paths) != 3:
        raise ValueError('Supply train, validation, and test manifests.')
    frames = [pd.read_csv(p, encoding='utf-8-sig', keep_default_na=False) for p in paths]
    names = ['train', 'val', 'test']
    report = {'rows': {}, 'overlap': {}, 'missing_files': {}}
    for name, df in zip(names, frames):
        if df.empty or 'clip_path' not in df:
            raise ValueError(f'{name}: nonempty clip_path column required')
        if not any(c in df for c in ['text', 'label_ascii', 'label_text']):
            raise ValueError(f'{name}: no text label column')
        if (df['clip_path'].astype(str).str.strip() == '').any():
            raise ValueError(f'{name}: empty clip_path')
        label_col = next(c for c in ['text', 'label_ascii', 'label_text'] if c in df)
        if (df[label_col].astype(str).str.strip() == '').any():
            raise ValueError(f'{name}: empty label')
        report['rows'][name] = len(df)
        if require_files:
            missing = sum(not Path(p).is_file() for p in df['clip_path'])
            report['missing_files'][name] = missing
            if missing:
                raise ValueError(f'{name}: {missing} missing clips; relocate manifests before training')
    for col in ['speaker_dir', 'group_key', 'stem', 'clip_path']:
        if all(col in df for df in frames):
            report['overlap'][col] = {}
            for i, j in itertools.combinations(range(3), 2):
                a, b = set(frames[i][col]), set(frames[j][col])
                if '' in a or '' in b:
                    raise ValueError(f'Empty group identifier in {col}')
                count = len(a & b)
                report['overlap'][col][names[i]+'_'+names[j]] = count
                if count:
                    raise ValueError(f'Split leakage in {col}: {count} overlapping identifiers')
    if not any(c in report['overlap'] for c in ['speaker_dir', 'group_key', 'stem']):
        raise ValueError('A group identity column (speaker_dir/group_key/stem) is required')
    return report


def prediction_metrics(csv_path):
    from .training import normalize_text, cer_score, wer_score
    df = pd.read_csv(csv_path, encoding='utf-8-sig', keep_default_na=False)
    if df.empty or not {'ref_text', 'hyp_text'}.issubset(df):
        raise ValueError('Prediction CSV needs nonempty ref_text and hyp_text rows')
    refs = [normalize_text(x) for x in df['ref_text']]
    hyps = [normalize_text(x) for x in df['hyp_text']]
    return {'rows': len(df), 'cer': sum(cer_score(a,b) for a,b in zip(refs,hyps))/len(df),
            'wer': sum(wer_score(a,b) for a,b in zip(refs,hyps))/len(df),
            'exact_match': sum(a==b for a,b in zip(refs,hyps))/len(df),
            'aggregation': 'mean_of_per_sample_error_ratios'}
