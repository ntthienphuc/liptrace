"""Portable frozen recognizers. One ESPnet graph family per process, no references."""

import ast
import hashlib
import importlib.util
import inspect
import math
import os
import sys
import time
import zipfile
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from confirmation_common import ROOT, normalize, read_json, resolve, sha


HERE = Path(__file__).resolve().parent
PHASE = "Research_Phase_05_2026-10-13"
_ACTIVE_SOURCE = None


def load_definitions(path, names, namespace):
    """Execute only explicitly named top-level definitions, never recipe imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=str(path))
    selected = []
    for name in names:
        matches = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one definition {name}: {path}")
        selected.extend(matches)
    scope = {"__name__": "confirmation_native_definitions", **namespace}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def native_transform(video):
    """Archived Spanish test transform, preserving operation order and RGB weights."""
    value = video / 255.0
    value = value[:, :, 4:92, 4:92]
    red, green, blue = value.unbind(dim=-3)
    gray = (0.2989 * red + 0.587 * green + 0.114 * blue).unsqueeze(-3)
    gray = gray.clone()
    gray.sub_(0.421).div_(0.165)
    return gray


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_export(payload, entry):
    if payload.get("identity") != entry["training_identity"]:
        raise ValueError("Exported training identity differs from the frozen registry")
    if payload.get("step") != entry["selected_training_step"]:
        raise ValueError("Exported selected training step differs from the frozen registry")
    if payload.get("format") != "lossless-model-state-v1" or payload.get("optimizer_included") is not False:
        raise ValueError("Expected the lossless exported model state, not a training resume file")
    return {key: payload[key] for key in (
        "identity", "step", "format", "optimizer_included",
        "source_checkpoint_sha256", "model_tensor_sha256",
    )}


def read_vocabulary(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token, identifier = line.rsplit(maxsplit=1)
        identifier = int(identifier)
        if identifier in entries:
            raise ValueError(f"Duplicate vocabulary ID: {identifier}")
        entries[identifier] = token
    if set(entries) != set(range(1, len(entries) + 1)):
        raise ValueError("Vocabulary IDs must be contiguous from one")
    return ["<blank>"] + [entries[index] for index in sorted(entries)] + ["<eos>"]


def claim_source(source, fallback=None):
    global _ACTIVE_SOURCE
    source = Path(source).resolve()
    if _ACTIVE_SOURCE is not None and _ACTIVE_SOURCE != source:
        raise RuntimeError("A different ESPnet family was already loaded; start a fresh process")
    allowed = [source]
    if fallback is not None:
        allowed.append(Path(fallback).resolve())
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not name.startswith("espnet") or not filename:
            continue
        path = Path(filename).resolve()
        if not any(path.is_relative_to(base) for base in allowed):
            raise RuntimeError(f"Foreign ESPnet module already imported: {name}: {path}")
        if name in {"espnet", "espnet.nets"} and not path.is_relative_to(source):
            raise RuntimeError("ESPnet root belongs to another graph family; use a fresh process")
    _ACTIVE_SOURCE = source
    sys.path.insert(0, str(source))


class Runtime:
    def __init__(self, model_id, device="cuda", threads=2, root=ROOT):
        began = time.monotonic()
        self.root = Path(root).resolve()
        self.registry = read_json(HERE / "model_registry.json")
        if model_id not in self.registry["models"]:
            raise ValueError(f"Unregistered or pending model: {model_id}")
        self.entry = dict(self.registry["models"][model_id])
        self.model_id = model_id
        self.device = torch.device(device)
        self.metric = self.entry["metric"]
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise RuntimeError("Runtime requires interop_threads=1; initialize in a fresh process")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(17)
        self.asset_hashes = {}
        for field in ("checkpoint", "base_checkpoint", "tokenizer", "units", "config", "native_config"):
            if field not in self.entry:
                continue
            path = self.path(self.entry[field])
            expected = self.entry.get(field + "_sha256")
            if not expected or sha(path) != expected:
                raise ValueError(f"Asset SHA-256 mismatch or missing registry hash: {field}: {path}")
            self.asset_hashes[field] = expected
        self.spanish = self.entry["runtime_kind"] == "spanish_native"
        self.export_metadata = None
        if self.spanish:
            self._load_spanish()
        else:
            self._load_auto_avsr()
        self.model.eval().requires_grad_(False).float().to(self.device)
        self.device = next(self.model.parameters()).device
        self.blank_id = int(self.model.blank)
        self.eos_id = int(self.model.eos)
        classes = self.entry["classes"]
        if len(self.vocabulary) != classes or self.model.odim != classes:
            raise ValueError("Dictionary and model output dimensions disagree")
        if self.blank_id != 0 or self.eos_id != classes - 1 or self.model.sos != self.eos_id:
            raise ValueError("Native blank/SOS/EOS IDs changed")
        if self.model.ctc.ctc_lo.out_features != classes:
            raise ValueError("CTC head dimension differs from dictionary")
        if self.model.decoder.output_layer.out_features != classes:
            raise ValueError("Attention head dimension differs from dictionary")
        self._load_search()
        self.identity = {
            "model_id": model_id,
            "registry_sha256": sha(HERE / "model_registry.json"),
            "registry_entry": self.entry,
            "asset_sha256": self.asset_hashes,
            "strict_state_load": True,
            "export_metadata": self.export_metadata,
            "selected_training_step": self.entry.get("selected_training_step"),
            "training_budget_steps": self.entry.get("training_identity", {}).get("steps"),
            "classes": classes,
            "blank_id": self.blank_id,
            "eos_id": self.eos_id,
            "precision": "float32",
            "tf32": False,
            "deterministic_algorithms": True,
            "device": str(self.device),
            "threads": threads,
            "interop_threads": torch.get_num_interop_threads(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "reference_used_for_inference": False,
            "source_files": self._source_identity(),
            "initialization_seconds": time.monotonic() - began,
        }

    def path(self, path):
        return resolve(path, self.root)

    def _load_auto_avsr(self):
        source = self.path(self.entry["source"])
        claim_source(source)
        package_path = self.path(PHASE + "/phase5")
        package_name = "confirmation_frozen_phase5"
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package_name, package_path / "__init__.py", submodule_search_locations=[str(package_path)]
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
        from importlib import import_module
        recognizer_class = import_module(package_name + ".model").Recognizer
        config = read_json(self.path(self.entry["config"]))
        for field in ("source", "tokenizer", "units"):
            config[field] = str(self.path(self.entry[field]))
        exported = self.entry["state_field"] == "model"
        checkpoint_field = "base_checkpoint" if exported else "checkpoint"
        config["checkpoint"] = str(self.path(self.entry[checkpoint_field]))
        config["checkpoint_sha256"] = self.entry[checkpoint_field + "_sha256"]
        config["state_field"] = "raw"
        method = self.entry.get("training_identity", {}).get("method", "B0")
        self.recognizer = recognizer_class(config, method=method, device="cpu", scope="all")
        if exported:
            payload = torch.load(self.path(self.entry["checkpoint"]), map_location="cpu", weights_only=True)
            self.export_metadata = validate_export(payload, self.entry)
            self.recognizer.load_state_dict(payload["model"], strict=True)
            del payload
        self.recognizer.eval().requires_grad_(False).float().to(self.device)
        self.model = self.recognizer.native
        self.vocabulary = list(self.recognizer.tokenizer.tokens)
        if self.vocabulary != read_vocabulary(self.path(self.entry["units"])):
            raise ValueError("Recognizer dictionary differs from registry dictionary")
        if not Path(inspect.getfile(type(self.model))).resolve().is_relative_to(source):
            raise RuntimeError("Loaded the wrong Auto-AVSR graph")
        self.factory_path = self.path(self.registry["factory_source"])

    def _load_spanish(self):
        import sentencepiece
        source = self.path(self.entry["source"])
        fallback = self.path(PHASE + "/vendor/auto_avsr/espnet/nets")
        claim_source(source, fallback)
        import espnet.nets
        if str(fallback) not in espnet.nets.__path__:
            espnet.nets.__path__.append(str(fallback))
        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
        if not Path(inspect.getfile(E2E)).resolve().is_relative_to(source):
            raise RuntimeError("Loaded the wrong Spanish native graph")
        self.forward_receipt = read_json(self.path(self.entry["native_config"]))
        if self.forward_receipt["status"] != "passed":
            raise ValueError("Spanish native graph has no passed forward receipt")
        assets = self.path(self.entry["native_assets"])
        self.archive = assets / "Source_0114f5f1248a.zip"
        if sha(self.archive) != self.forward_receipt["source_sha256"]:
            raise ValueError("Spanish source archive hash changed")
        self.model = E2E(self.entry["classes"], Namespace(**self.forward_receipt["native_config"]))
        payload = torch.load(self.path(self.entry["checkpoint"]), map_location="cpu", weights_only=True)
        self.model.load_state_dict(payload, strict=True)
        del payload
        self.vocabulary = read_vocabulary(self.path(self.entry["units"]))
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(self.path(self.entry["tokenizer"])))
        pieces = {self.processor.id_to_piece(index) for index in range(self.processor.get_piece_size())
                  if not self.processor.is_control(index)}
        if not pieces.issubset(set(self.vocabulary)):
            raise ValueError("Spanish SentencePiece and native dictionary disagree")
        self.factory_path = assets / "original_recipe/lightning.py"

    def _load_search(self):
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.scorers.length_bonus import LengthBonus
        from espnet.nets.e2e_asr_common import end_detect
        from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
        namespace = {"BatchBeamSearch": BatchBeamSearch, "LengthBonus": LengthBonus}
        name = "get_beam_search_decoder"
        self.native_factory = load_definitions(self.factory_path, [name], namespace)[name]
        horizon = load_definitions(self.path(self.registry["horizon_source"]), ["HorizonBeamSearch"], {
            "BatchBeamSearch": BatchBeamSearch, "end_detect": end_detect,
        })["HorizonBeamSearch"]
        safe = load_definitions(self.path(self.registry["safe_search_source"]), [
            "TerminalCTCPrefixScoreTH", "SafeHorizonBeamSearch",
        ], {"torch": torch, "CTCPrefixScoreTH": CTCPrefixScoreTH, "HorizonBeamSearch": horizon})
        namespace["BatchBeamSearch"] = safe["SafeHorizonBeamSearch"]
        self.safe_factory = load_definitions(self.factory_path, [name], namespace)[name]
        namespace["BatchBeamSearch"] = horizon
        self.attention_factory = load_definitions(self.factory_path, [name], namespace)[name]

    def _source_identity(self):
        paths = {Path(__file__).resolve(), Path(inspect.getfile(type(self.model))).resolve(), self.factory_path}
        paths.update(self.path(self.registry[key]) for key in ("horizon_source", "safe_search_source"))
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if filename and filename.endswith(".py") and name.startswith(("espnet", "confirmation_frozen_phase5")):
                paths.add(Path(filename).resolve())
        result = {path.relative_to(self.root).as_posix(): sha(path) for path in sorted(paths)}
        if self.spanish:
            source = self.path(self.entry["source"])
            expected_interfaces = self.forward_receipt["framework_interface_dependencies"]
            with zipfile.ZipFile(self.archive) as archive:
                prefix = archive.namelist()[0].split("/")[0] + "/"
                for name, module in list(sys.modules.items()):
                    filename = getattr(module, "__file__", None)
                    if not name.startswith("espnet") or not filename or not filename.endswith(".py"):
                        continue
                    path = Path(filename).resolve()
                    actual = sha(path)
                    if path.is_relative_to(source):
                        original = archive.read(prefix + path.relative_to(source).as_posix())
                        if hashlib.sha256(original).hexdigest() != actual:
                            raise ValueError(f"Spanish upstream source changed: {name}")
                    elif name in expected_interfaces and actual != expected_interfaces[name]["sha256"]:
                        raise ValueError(f"Spanish interface dependency changed: {name}")
        return result

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def encode(self, pixels):
        began = time.monotonic()
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
            raise ValueError("pixels must be a numpy uint8 array")
        expected = (3, 96, 96) if self.spanish else (96, 96)
        if tuple(pixels.shape[1:]) != expected or len(pixels) < 1:
            raise ValueError(f"Expected nonempty [T,{expected}] pixels for {self.model_id}")
        pixels = np.ascontiguousarray(pixels)
        self._synchronize()
        with torch.autocast(self.device.type, enabled=False):
            if self.spanish:
                prepared = native_transform(torch.from_numpy(pixels)).unsqueeze(0)
                prepared_hash = tensor_sha(prepared)
                states = self.model.encoder(prepared.to(self.device), None)[0][0]
            else:
                states = self.recognizer.encode(pixels)[0]
                prepared_hash = None
            logp = self.model.ctc.log_softmax(states.unsqueeze(0))[0]
        self._synchronize()
        if states.shape != (len(pixels), 768) or states.dtype != torch.float32:
            raise ValueError("Encoder changed frame alignment, width or precision")
        if logp.shape != (len(pixels), len(self.vocabulary)) or logp.dtype != torch.float32:
            raise ValueError("CTC output dimensions or precision changed")
        if not torch.isfinite(states).all() or not torch.isfinite(logp).all():
            raise FloatingPointError("Nonfinite encoder states or CTC log probabilities")
        return {
            "states": states,
            "logp": logp.detach().cpu().numpy().copy(),
            "metadata": {
                "model_id": self.model_id,
                "frames": len(pixels),
                "state_frames": len(states),
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "input_tensor_sha256": prepared_hash,
                "color_space": "rgb" if self.spanish else "gray",
                "transform": "native_rgb_center88_gray_normalize" if self.spanish else "phase5_native_gray",
                "encode_seconds": time.monotonic() - began,
                "precision": "float32",
            },
        }

    def text(self, tokens):
        identifiers = [int(identifier) for identifier in tokens]
        if any(identifier < 0 or identifier >= len(self.vocabulary) for identifier in identifiers):
            raise ValueError("Out-of-range token ID")
        pieces = [self.vocabulary[identifier] for identifier in identifiers
                  if identifier not in {self.blank_id, self.eos_id}]
        return normalize("".join(pieces).replace("\u2581", " ").replace("<space>", " "))

    def _validate_decode(self, states, ctc_weight, cap, beam_size, penalty):
        if not isinstance(states, torch.Tensor) or states.ndim != 2 or states.shape[1] != 768:
            raise ValueError("Decoder states must have shape [T,768]")
        if len(states) < 1 or states.dtype != torch.float32 or states.device != self.device:
            raise ValueError("Decoder states must be nonempty FP32 on the runtime device")
        if not torch.isfinite(states).all():
            raise FloatingPointError("Nonfinite decoder states")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("cap must be a positive integer")
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
            raise ValueError("beam_size must be a positive integer")
        if not math.isfinite(ctc_weight) or not 0 <= ctc_weight <= 1 or not math.isfinite(penalty):
            raise ValueError("Invalid CTC weight or length penalty")

    def _result(self, hypotheses, began, trace, cap):
        if not hypotheses:
            raise RuntimeError("Beam search returned no ended hypothesis")
        hypothesis = hypotheses[0]
        sequence = [int(value) for value in hypothesis.yseq]
        score = float(hypothesis.score)
        if len(sequence) < 2 or sequence[0] != self.eos_id or sequence[-1] != self.eos_id:
            raise RuntimeError("Native hypothesis must begin with SOS and end with EOS")
        if not math.isfinite(score):
            raise FloatingPointError("Best ended hypothesis has nonfinite score")
        lexical = sequence[1:-1]
        if any(value < 0 or value >= len(self.vocabulary) for value in lexical):
            raise ValueError("Native decoder emitted an out-of-vocabulary token")
        tokens = [value for value in lexical if value not in {self.blank_id, self.eos_id}]
        self._synchronize()
        return {
            "tokens": tokens,
            "text": self.text(tokens),
            "score": score,
            "seconds": time.monotonic() - began,
            "trace": trace,
            "cap": cap,
        }

    @torch.inference_mode()
    def decode(self, states, ctc_weight=0.5, cap=None, beam_size=40, penalty=0):
        cap = len(states) if cap is None else cap
        self._validate_decode(states, ctc_weight, cap, beam_size, penalty)
        self._synchronize()
        began = time.monotonic()
        factory = self.attention_factory if ctc_weight == 0 else self.safe_factory
        with torch.autocast(self.device.type, enabled=False):
            search = factory(self.model, self.vocabulary, ctc_weight=ctc_weight,
                             beam_size=beam_size, penalty=penalty)
            search.eval()
            hypotheses = search(states, step_cap=cap)
        trace = dict(search.last_diagnostic)
        trace.update({"implementation": type(search).__name__, "ctc_weight": ctc_weight,
                      "beam_size": beam_size, "penalty": penalty})
        return self._result(hypotheses, began, trace, cap)

    @torch.inference_mode()
    def decode_native(self, states, ctc_weight, beam_size=40, penalty=0):
        cap = len(states)
        self._validate_decode(states, ctc_weight, cap, beam_size, penalty)
        self._synchronize()
        began = time.monotonic()
        trace = {"implementation": "upstream_BatchBeamSearch", "step_cap": cap,
                 "cap_reached": False, "iterations": 0, "ctc_weight": ctc_weight,
                 "beam_size": beam_size, "penalty": penalty}
        with torch.autocast(self.device.type, enabled=False):
            search = self.native_factory(self.model, self.vocabulary, ctc_weight=ctc_weight,
                                         beam_size=beam_size, penalty=penalty)
            original = search.post_process

            def observed_post_process(iteration, maxlen, minlen, running, ended):
                trace["iterations"] = iteration + 1
                trace["cap_reached"] = trace["cap_reached"] or iteration == maxlen - 1
                return original(iteration, maxlen, minlen, running, ended)

            search.post_process = observed_post_process
            search.eval()
            hypotheses = search(states)
        trace["ended_hypotheses"] = len(hypotheses)
        return self._result(hypotheses, began, trace, cap)
