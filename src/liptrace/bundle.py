"""Bind ordered characters and preprocessing to exact checkpoint bytes."""
import json
import shutil
from pathlib import Path
from .contracts import sha256, json_digest, write_json, validate_profile, profile_from_checkpoint
from .deployment import require_training
from . import __version__


def read_bundle(directory, expected_manifest_sha256=None):
    root = Path(directory)
    manifest_path = root / 'manifest.json'
    if expected_manifest_sha256 and sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError('Manifest identity mismatch')
    doc = json.loads(manifest_path.read_text(encoding='utf-8'))
    from .deployment import SCHEMA, read_deployment
    if isinstance(doc, dict) and doc.get('schema') == SCHEMA:
        return read_deployment(root, expected_manifest_sha256)
    if set(doc) != {'schema', 'profile', 'artifacts', 'provenance'} or doc['schema'] != 'liptrace-bundle-v1':
        raise ValueError('Unsupported bundle schema')
    validate_profile(doc['profile'])
    if not isinstance(doc['provenance'], dict) or doc['provenance'].get('profile_sha256') != json_digest(doc['profile']):
        raise ValueError('Manifest profile identity mismatch')
    if not isinstance(doc['artifacts'], dict) or set(doc['artifacts']) not in ({'model.pt'}, {'model.pt', 'model.onnx'}):
        raise ValueError('Unsupported artifact set')
    for name, digest in doc['artifacts'].items():
        target = root / name
        if target.is_symlink() or not target.is_file() or sha256(target) != digest:
            raise ValueError(f'Artifact identity mismatch: {name}')
    require_training()
    from .runtime import load_recognizer
    _, ckpt, _ = load_recognizer(root / 'model.pt')
    # Legacy bundles omitted this field. New bundles preserve the training
    # declaration in the checkpoint so changing JSON alone cannot relabel it.
    recorded = ckpt['args'].get('normalization')
    if recorded is not None and recorded != doc['profile']['normalization']:
        raise ValueError('Checkpoint and manifest normalization mismatch')
    recovered = profile_from_checkpoint(ckpt, recorded or doc['profile']['normalization'])
    if recovered != doc['profile']:
        raise ValueError('Checkpoint and manifest profile mismatch')
    return doc


def create_bundle(checkpoint, directory, normalization=None):
    torch = require_training()
    from .runtime import load_recognizer
    root = Path(directory)
    if root.exists():
        raise ValueError('Bundle destination already exists; choose a new directory')
    model, ckpt, _ = load_recognizer(checkpoint)
    recorded = ckpt['args'].get('normalization')
    if recorded is not None and normalization is not None and normalization != recorded:
        raise ValueError('Requested normalization conflicts with checkpoint training metadata')
    normalization = recorded or normalization or 'legacy-lower-v1'
    profile = profile_from_checkpoint(ckpt, normalization)
    root.mkdir(parents=True)
    # Discard optimizer state, source paths and private training arguments.
    clean = {'model_state': model.state_dict(), 'charset': profile['charset'],
             'blank_index': profile['blank_index'],
             'args': {**profile['model'], 'normalization': normalization}}
    torch.save(clean, root / 'model.pt')
    doc = {'schema': 'liptrace-bundle-v1', 'profile': profile,
           'artifacts': {'model.pt': sha256(root / 'model.pt')},
           'provenance': {'source_checkpoint_sha256': sha256(checkpoint),
                          'profile_sha256': json_digest(profile), 'software': f'liptrace-{__version__}'}}
    write_json(root / 'manifest.json', doc)
    read_bundle(root)
    return {'manifest_sha256': sha256(root / 'manifest.json'), 'profile_sha256': json_digest(profile)}


def export_onnx(directory, destination, expected_manifest_sha256=None, runtime_only=False):
    torch = require_training()
    from .runtime import load_recognizer
    try:
        import onnx
    except ImportError as exc:
        raise ImportError('ONNX export requires "liptrace[training,onnx]".') from exc
    if runtime_only:
        from .deployment import require_onnxruntime
        require_onnxruntime()
    doc = read_bundle(directory, expected_manifest_sha256)
    if doc['schema'] != 'liptrace-bundle-v1':
        raise ValueError('Export requires a checkpoint bundle, not an ONNX-only deployment')
    source, target = Path(directory), Path(destination)
    if target.exists():
        raise ValueError('Export destination already exists')
    # Fixed single-clip geometry; verification belongs to replay, not export success.
    model, _, _ = load_recognizer(source / 'model.pt')
    cfg = doc['profile']['model']
    x = torch.zeros(1, 1, cfg['max_frames'], cfg['img_size'], cfg['img_size'])
    target.mkdir(parents=True)
    if not runtime_only:
        shutil.copyfile(source / 'model.pt', target / 'model.pt')
    torch.onnx.export(model, x, str(target / 'model.onnx'), input_names=['video'],
                      output_names=['logits'], opset_version=17, dynamo=False)
    graph = onnx.load(str(target / 'model.onnx'))
    if runtime_only:
        from .deployment import SCHEMA, IDENTITIES, VERIFICATION, expected_io
        provenance = {key: doc['provenance'][key] for key in
                      ('source_checkpoint_sha256', 'profile_sha256')}
        provenance.update(source_bundle_manifest_sha256=sha256(source / 'manifest.json'),
                          bundled_checkpoint_sha256=doc['artifacts']['model.pt'],
                          software=f'liptrace-{__version__}')
        onnx.helper.set_model_props(graph, {'liptrace.schema': SCHEMA,
            **{f'liptrace.{key}': provenance[key] for key in IDENTITIES}})
        onnx.save_model(graph, str(target / 'model.onnx'))
        doc = {'schema': SCHEMA, 'profile': doc['profile'], 'artifacts': {},
               'provenance': provenance, 'verification': dict(VERIFICATION),
               'io': expected_io(doc['profile'])}
    onnx.checker.check_model(graph)
    doc['artifacts']['model.onnx'] = sha256(target / 'model.onnx')
    write_json(target / 'manifest.json', doc)
    read_bundle(target)
    return {'manifest_sha256': sha256(target / 'manifest.json'),
            'schema': doc['schema'], 'verification': 'graph-checked; replay parity still required'}
