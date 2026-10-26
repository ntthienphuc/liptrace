"""Isolated archived ViCocktail runtime and reference-blind native replay.

Fresh process required: do not mix this graph with the active EN/VLR runtime.
The CPU default is intentional; pass device='cuda' explicitly on the server.
"""

import argparse
import hashlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from confirmation_common import ROOT, read_json, read_jsonl, resolve, save_json, sha, utc
from native_runtime import Runtime as SharedRuntime
from native_runtime import claim_source, read_vocabulary, tensor_sha


HERE = Path(__file__).resolve().parent


def source_manifest(source):
    """Hash every Python source, including unimported graph branches."""
    source = Path(source)
    files = {path.relative_to(source).as_posix(): sha(path) for path in sorted(source.rglob('*.py'))}
    content = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return files, hashlib.sha256(content).hexdigest()


def validate_sources(root, registry, entry):
    source = resolve(entry['source'], root)
    files, digest = source_manifest(source)
    if digest != entry['source_tree_sha256'] or len(files) != entry['source_python_files']:
        raise ValueError('Archived ViCocktail Python source tree SHA-256/count changed')
    for relative, expected in registry['support_sources_sha256'].items():
        if sha(resolve(relative, root)) != expected:
            raise ValueError(f'Support source SHA-256 changed: {relative}')
    return files


def strict_state(payload):
    """Reject unknown entries rather than silently filtering model prefixes."""
    state = payload.get('state_dict')
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a nonempty Lightning state_dict')
    if not all(isinstance(key, str) and key.startswith('model.') and len(key) > 6
               and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError('Every state_dict entry must be a model. prefixed tensor')
    return {key.removeprefix('model.'): value for key, value in state.items()}


def prepare_pixels(pixels):
    """Archived test transform: divide, center88, RGB grayscale, normalize."""
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
        raise ValueError('pixels must be numpy uint8')
    if pixels.ndim not in (3, 4) or len(pixels) < 1:
        raise ValueError('Expected nonempty [T,96,96] or [T,3,96,96]')
    if tuple(pixels.shape[1:]) not in ((96, 96), (3, 96, 96)):
        raise ValueError('Expected grayscale or RGB 96x96 canonical pixels')
    value = torch.from_numpy(np.ascontiguousarray(pixels))
    if value.ndim == 3:
        value = value.unsqueeze(1)
    value = value / 255.0
    value = value[:, :, 4:92, 4:92]
    if value.shape[1] == 3:
        red, green, blue = value.unbind(dim=-3)
        value = (0.2989 * red + 0.587 * green + 0.114 * blue).unsqueeze(-3)
    value = value.clone()
    value.sub_(0.421).div_(0.165)
    return value.unsqueeze(0)


class Runtime(SharedRuntime):
    """Reuse verified beam/result interfaces, with a wholly separate loader."""

    def __init__(self, model_id='VC_NATIVE', device='cpu', threads=2, root=ROOT):
        began = time.monotonic()
        if model_id != 'VC_NATIVE':
            raise ValueError('This isolated runtime supports only VC_NATIVE')
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError('threads must be a positive integer')
        self.root = Path(root).resolve()
        self.registry = read_json(HERE / 'VC_Model_Registry.json')
        self.entry = dict(self.registry['models'][model_id])
        self.model_id = model_id
        self.device = torch.device(device)
        self.metric = self.entry['metric']
        self.spanish = False
        self.export_metadata = None
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise RuntimeError('Initialize runtime in a fresh process with interop_threads=1')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(17)
        self.source_files = validate_sources(self.root, self.registry, self.entry)
        self.asset_hashes = {}
        for field in ('checkpoint', 'tokenizer', 'units'):
            actual = sha(self.path(self.entry[field]))
            if actual != self.entry[field + '_sha256']:
                raise ValueError(f'Asset SHA-256 changed: {field}')
            self.asset_hashes[field] = actual
        source = self.path(self.entry['source'])
        claim_source(source)
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
        import sentencepiece

        if not Path(inspect.getfile(E2E)).resolve().is_relative_to(source):
            raise RuntimeError('Wrong native E2E graph imported')
        self.vocabulary = read_vocabulary(self.path(self.entry['units']))
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(self.path(self.entry['tokenizer'])))
        pieces = {self.processor.id_to_piece(index) for index in range(self.processor.get_piece_size())
                  if not self.processor.is_control(index)}
        if not pieces.issubset(set(self.vocabulary)) or self.processor.get_piece_size() != 2000:
            raise ValueError('Native SentencePiece/dictionary mismatch')
        payload = torch.load(self.path(self.entry['checkpoint']), map_location='cpu',
                             weights_only=True, mmap=True)
        configuration = self.entry['constructor']
        hyperparameters = payload['hyper_parameters']
        if any(payload.get(key) != self.entry[key] for key in ('epoch', 'global_step')):
            raise ValueError('Checkpoint epoch/step changed')
        if any(hyperparameters[key] != configuration[key] for key in ('modality', 'ctc_weight')):
            raise ValueError('Checkpoint native constructor changed')
        self.model = E2E(**configuration)
        self.model.load_state_dict(strict_state(payload), strict=True)
        self.export_metadata = {'epoch': payload['epoch'], 'global_step': payload['global_step'],
                                'state_field': 'state_dict', 'state_prefix': 'model.'}
        del payload
        if type(self.model.frontend.frontend3D[3]) is not torch.nn.MaxPool3d:
            raise ValueError('Native MaxPool3d was replaced')
        self.model.eval().requires_grad_(False).float().to(self.device)
        self.device = next(self.model.parameters()).device
        self.blank_id = int(self.model.blank)
        self.eos_id = int(self.model.eos)
        classes = self.entry['classes']
        dimensions = (len(self.vocabulary), self.model.odim, self.model.ctc.ctc_lo.out_features,
                      self.model.decoder.output_layer.out_features)
        if any(dimension != classes for dimension in dimensions):
            raise ValueError('Dictionary/CTC/attention head dimensions disagree')
        if self.blank_id != 0 or self.eos_id != classes - 1 or self.model.sos != self.eos_id:
            raise ValueError('Native blank/SOS/EOS conventions changed')
        self.factory_path = self.path(self.registry['factory_source'])
        self._load_search()
        claim_source(source)
        imported = self._source_identity()
        imported[Path(__file__).relative_to(self.root).as_posix()] = sha(__file__)
        self.identity = {
            'model_id': model_id, 'registry_sha256': sha(HERE / 'VC_Model_Registry.json'),
            'registry_entry': self.entry, 'asset_sha256': self.asset_hashes,
            'source_files': imported, 'full_source_tree_sha256': self.entry['source_tree_sha256'],
            'full_source_python_files': len(self.source_files), 'strict_state_load': True,
            'export_metadata': self.export_metadata, 'classes': classes, 'blank_id': self.blank_id,
            'eos_id': self.eos_id, 'frontend_pool': 'MaxPool3d', 'precision': 'float32',
            'tf32': False, 'deterministic_algorithms': True, 'device': str(self.device),
            'threads': threads, 'interop_threads': torch.get_num_interop_threads(),
            'torch': str(torch.__version__), 'numpy': np.__version__,
            'reference_used_for_inference': False, 'initialization_seconds': time.monotonic() - began,
        }

    @torch.inference_mode()
    def encode(self, pixels):
        self._synchronize()
        began = time.monotonic()
        prepared = prepare_pixels(pixels)
        prepared_hash = tensor_sha(prepared)
        with torch.autocast(self.device.type, enabled=False):
            features = self.model.frontend(prepared.to(self.device))
            projected = self.model.proj_encoder(features)
            states = self.model.encoder(projected, None)[0][0]
            logp = self.model.ctc.log_softmax(states.unsqueeze(0))[0]
        self._synchronize()
        if states.shape != (len(pixels), 768) or logp.shape != (len(pixels), len(self.vocabulary)):
            raise ValueError('Native encoder/CTC frame alignment or dimensions changed')
        if states.dtype != torch.float32 or logp.dtype != torch.float32:
            raise ValueError('Native encoder/CTC must remain FP32')
        if not torch.isfinite(states).all() or not torch.isfinite(logp).all():
            raise FloatingPointError('Nonfinite native encoder/CTC outputs')
        return {
            'states': states, 'logp': logp.detach().cpu().numpy().copy(),
            'metadata': {
                'model_id': self.model_id, 'frames': len(pixels), 'state_frames': len(states),
                'pixel_sha256': hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
                'input_tensor_sha256': prepared_hash, 'color_space': 'rgb' if pixels.ndim == 4 else 'gray',
                'transform': 'archived_div255_center88_gray_normalize_0.421_0.165',
                'frontend_pool': 'MaxPool3d', 'precision': 'float32',
                'encode_seconds': time.monotonic() - began,
            },
        }


def native_replay(runtime, row, manifest, beam_size=40):
    """Compare bounded decode to the actual archived native beam, without labels."""
    from data_io import load_pixels

    if row['corpus'] != 'VC' or row['language'] != 'vi':
        raise ValueError('Native replay requires a canonical VC Vietnamese clip')
    pixels = load_pixels(row, root=runtime.root)
    encoded = runtime.encode(pixels)
    states = encoded['states']
    comparisons = []
    for weight in (0.1, 0.5):
        native = runtime.decode_native(states, weight, beam_size=beam_size)
        bounded = runtime.decode(states, weight, cap=len(states), beam_size=beam_size)
        score_difference = abs(native['score'] - bounded['score'])
        equal = native['tokens'] == bounded['tokens'] and native['text'] == bounded['text']
        comparisons.append({
            'ctc_weight': weight, 'beam_size': beam_size, 'native': native, 'bounded': bounded,
            'tokens_and_text_equal': equal, 'absolute_score_difference': score_difference,
            'score_absolute_tolerance': 1e-5, 'passed': equal and score_difference <= 1e-5,
        })
    return {
        'status': 'passed' if all(item['passed'] for item in comparisons) else 'failed',
        'created_utc': utc(), 'clip_id': row['clip_id'], 'source_id': row['source_id'],
        'manifest': str(Path(manifest).resolve().relative_to(runtime.root).as_posix()),
        'manifest_sha256': sha(manifest), 'input': row['input'], 'frames': len(pixels),
        'device': str(runtime.device), 'encoding': encoded['metadata'], 'comparisons': comparisons,
        'reference_used': False, 'accuracy_measured': False,
        'scope': 'Current FP32 native-vs-bounded parity; not historical bf16 prediction reproduction.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('native-replay', help='Save real canonical-clip native/bounded parity receipt')
    replay.add_argument('--manifest', required=True)
    replay.add_argument('--clip-id', help='Default: shortest clip, tie-broken by clip ID; no label selection')
    replay.add_argument('--device', default='cpu')
    replay.add_argument('--threads', type=int, default=2)
    replay.add_argument('--beam-size', type=int, default=40)
    replay.add_argument('--root', type=Path, default=ROOT)
    replay.add_argument('--output', type=Path, default=HERE / 'VC_Runtime_Readiness.json')
    args = parser.parse_args()
    report = {'schema_version': 1, 'created_utc': utc(), 'status': 'running',
              'runtime_sha256': sha(__file__), 'registry_sha256': sha(HERE / 'VC_Model_Registry.json')}
    if args.output.exists():
        previous = read_json(args.output)
        if all(previous.get(key) == report[key] for key in ('runtime_sha256', 'registry_sha256')):
            report['cpu_tests'] = previous.get('cpu_tests')
    try:
        manifest = resolve(args.manifest, args.root)
        rows = list(read_jsonl(manifest))
        if args.clip_id:
            matches = [row for row in rows if row['clip_id'] == args.clip_id]
            if len(matches) != 1:
                raise ValueError('Requested clip ID must occur exactly once')
            row = matches[0]
        else:
            row = min(rows, key=lambda item: (item['frames'], item['clip_id']))
        runtime = Runtime(device=args.device, threads=args.threads, root=args.root)
        report['identity'] = runtime.identity
        report['native_replay'] = native_replay(runtime, row, manifest, args.beam_size)
        report['status'] = 'native_replay_passed' if report['native_replay']['status'] == 'passed' else 'failed'
        report['limits'] = [
            'No historical bf16 equivalence or recognition accuracy claim.',
            'Word timing eligibility remains separate; this runtime generates no reference timestamps.',
            'Main model_registry.json/run_jobs.py integration is intentionally left to the parent.',
        ]
        if runtime.device.type == 'cpu':
            report['limits'].append('GPU execution and GPU parity remain untested.')
        code = 0 if report['native_replay']['status'] == 'passed' else 1
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error),
                           'traceback': traceback.format_exc()}
        code = 1
    report['completed_utc'] = utc()
    save_json(args.output, report)
    print(json.dumps({'status': report['status'], 'output': str(args.output), 'exit_code': code}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
