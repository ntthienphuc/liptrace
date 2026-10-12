"""ONNX-only deployment contracts, verified using ONNX Runtime alone.

Metadata binds a graph to declared source identities; it cannot establish the
truth of unavailable checkpoint bytes. Pin the manifest from a trusted receipt.
"""
import json
import re
from pathlib import Path
from .contracts import sha256, json_digest, validate_profile

SCHEMA = 'liptrace-onnx-deployment-v1'
VERIFICATION = {
    'scope': 'ONNX graph bytes, embedded identities and fixed input/output contract',
    'checkpoint_verified': False,
    'checkpoint_scope': 'source hashes preserved; checkpoint absent from deployment',
}
IDENTITIES = ('profile_sha256', 'source_checkpoint_sha256',
              'source_bundle_manifest_sha256', 'bundled_checkpoint_sha256')


def require_training():
    """Keep importing runtime-facing modules safe without the training extra."""
    try:
        import torch
        import pandas
        import tqdm
    except ImportError as exc:
        raise ImportError('This operation requires training dependencies; install "liptrace[training]".') from exc
    return torch


def require_onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError('ONNX execution/inspection requires "liptrace[runtime]".') from exc
    return onnxruntime


def expected_io(profile):
    cfg = profile['model']
    return {'input': {'name': 'video', 'type': 'tensor(float)',
                      'shape': [1, 1, cfg['max_frames'], cfg['img_size'], cfg['img_size']]},
            'output': {'name': 'logits', 'type': 'tensor(float)',
                       'shape': [1, cfg['max_frames'], len(profile['charset']) + 1]}}


def validate_session(session, profile, provenance=None):
    """Reject extra, renamed, dynamic, wrong-type or wrong-shape graph I/O."""
    expected = expected_io(profile)
    for kind, nodes in (('input', session.get_inputs()), ('output', session.get_outputs())):
        if len(nodes) != 1 or {'name': nodes[0].name, 'type': nodes[0].type,
                               'shape': nodes[0].shape} != expected[kind]:
            raise ValueError(f'ONNX {kind} contract mismatch')
    if session.get_overridable_initializers():
        raise ValueError('ONNX graph has overridable initializers')
    if provenance is not None:
        metadata = session.get_modelmeta().custom_metadata_map
        wanted = {'liptrace.schema': SCHEMA,
                  **{f'liptrace.{key}': provenance[key] for key in IDENTITIES}}
        if any(metadata.get(key) != value for key, value in wanted.items()):
            raise ValueError('ONNX embedded identity metadata mismatch')


def open_session(directory, doc, threads=2):
    ort = require_onnxruntime()
    options = ort.SessionOptions()
    if threads is not None:
        if type(threads) is not int or threads <= 0:
            raise ValueError('threads must be positive')
        options.intra_op_num_threads = threads
    # Loading graph bytes prevents ONNX external-data paths being resolved from
    # the artifact directory; this format contains one self-contained graph.
    try:
        session = ort.InferenceSession((Path(directory) / 'model.onnx').read_bytes(),
                                       sess_options=options, providers=['CPUExecutionProvider'])
    except Exception as exc:
        raise ValueError(f'Invalid self-contained ONNX graph: {exc}') from exc
    validate_session(session, doc['profile'],
                     doc['provenance'] if doc['schema'] == SCHEMA else None)
    return session


def read_deployment(directory, expected_manifest_sha256=None):
    root = Path(directory)
    manifest = root / 'manifest.json'
    if expected_manifest_sha256 and sha256(manifest) != expected_manifest_sha256:
        raise ValueError('Manifest identity mismatch')
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or set(doc) != {'schema', 'profile', 'artifacts', 'provenance', 'verification', 'io'} or doc['schema'] != SCHEMA:
        raise ValueError('Unsupported deployment schema')
    validate_profile(doc['profile'])
    provenance = doc['provenance']
    if not isinstance(provenance, dict) or set(provenance) != {*IDENTITIES, 'software'}:
        raise ValueError('Invalid deployment provenance')
    if any(not isinstance(provenance[k], str) or not re.fullmatch('[0-9a-f]{64}', provenance[k]) for k in IDENTITIES):
        raise ValueError('Invalid deployment identity digest')
    if provenance['profile_sha256'] != json_digest(doc['profile']):
        raise ValueError('Manifest profile identity mismatch')
    if doc['verification'] != VERIFICATION or doc['io'] != expected_io(doc['profile']):
        raise ValueError('Deployment verification scope or I/O declaration mismatch')
    if not isinstance(doc['artifacts'], dict) or set(doc['artifacts']) != {'model.onnx'}:
        raise ValueError('Unsupported deployment artifact set')
    graph = root / 'model.onnx'
    if graph.is_symlink() or not graph.is_file() or sha256(graph) != doc['artifacts']['model.onnx']:
        raise ValueError('Artifact identity mismatch: model.onnx')
    if (root / 'model.pt').exists():
        raise ValueError('Runtime-only deployment must not contain model.pt')
    open_session(root, doc)
    return doc
