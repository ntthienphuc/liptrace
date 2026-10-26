import logging
import os
import json
import hashlib
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping

from average_checkpoints import ensemble
from datamodule.data_module import DataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger


# Set environment variables and logger level
# logging.basicConfig(level=logging.WARNING)


PRESCREEN_EVIDENCE_SCHEMA = "pavir-prescreen-evidence-v2"
PRESCREEN_TELEMETRY_SCHEMA = "pavir-prescreen-telemetry-v2"
PRESCREEN_CHAIN_ORIGIN = "0" * 64
PRESCREEN_VRAM_LIMIT_BYTES = 72 * 1024**3


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one canonical, finite JSON value without a trailing newline."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash a complete model state, including buffers, without dtype coercion.

    This is intentionally public enough for the independent prescreen auditor.
    Keys are not stripped or rewritten.  Each tensor contributes canonical
    metadata, a NUL separator, its contiguous raw bytes viewed as uint8, and a
    newline separator.
    """

    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state value is not a tensor: {key}")
        digest.update(
            canonical_json_bytes(
                {
                    "key": str(key),
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                }
            )
        )
        digest.update(b"\0")
        value = tensor.detach().contiguous()
        if value.device.type != "cpu":
            value = value.cpu()
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_canonical_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish canonical JSON without overwriting an existing record."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        _write_all(descriptor, canonical_json_bytes(value) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _tensor_tree_finiteness(value: Any) -> dict[str, Any]:
    import torch

    floating_tensors = 0
    floating_elements = 0
    nonfinite_tensors = 0

    def visit(item: Any) -> None:
        nonlocal floating_tensors, floating_elements, nonfinite_tensors
        if isinstance(item, torch.Tensor):
            if item.is_floating_point() or item.is_complex():
                floating_tensors += 1
                floating_elements += int(item.numel())
                if not bool(torch.isfinite(item.detach()).all()):
                    nonfinite_tensors += 1
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return {
        "finite": nonfinite_tensors == 0,
        "floating_tensor_count": floating_tensors,
        "floating_element_count": floating_elements,
        "nonfinite_tensor_count": nonfinite_tensors,
    }


def _pre_clip_gradient_norm(module) -> tuple[float, int, int]:
    """Return the unmodified L2 norm/count of all populated gradients."""

    import torch

    gradients = []
    elements = 0
    for parameter in module.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        gradient = gradient.detach()
        gradients.append(gradient)
        elements += int(gradient.numel())
    if not gradients:
        return 0.0, 0, 0

    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for gradient in gradients:
        grouped.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    norms = []
    for values in grouped.values():
        try:
            norms.extend(torch._foreach_norm(values, 2.0))
        except (RuntimeError, TypeError):
            norms.extend(torch.linalg.vector_norm(value, 2.0) for value in values)
    first_device = norms[0].device
    total = torch.linalg.vector_norm(
        torch.stack(
            [norm.to(device=first_device, dtype=torch.float64) for norm in norms]
        ),
        2.0,
    )
    return float(total.detach().cpu()), len(gradients), elements


def _checkpoint_global_step(path: Path) -> int:
    import torch

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    step = payload.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError(f"fixed-final checkpoint has no integer global_step: {path}")
    return int(step)


def save_checkpoint_once(trainer, path: str | Path) -> Path:
    """Save a complete Lightning checkpoint atomically without clobbering."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        trainer.save_checkpoint(str(temporary), weights_only=False)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return path


def _prescreen_enabled(args) -> bool:
    names = (
        "prescreen_telemetry",
        "prescreen_evidence_journal",
        "prescreen_expected_w0",
        "prescreen_fixed_final_checkpoint",
        "prescreen_registry",
        "prescreen_run_id",
        "prescreen_arm",
        "prescreen_stage",
    )
    return any(getattr(args, name, None) is not None for name in names)


def validate_prescreen_args(args) -> bool:
    """Validate the opt-in v2 contract without changing the default path."""

    names = (
        "prescreen_telemetry",
        "prescreen_evidence_journal",
        "prescreen_expected_w0",
        "prescreen_fixed_final_checkpoint",
        "prescreen_registry",
        "prescreen_run_id",
        "prescreen_arm",
        "prescreen_stage",
    )
    enabled = _prescreen_enabled(args)
    if not enabled:
        return False
    missing = [name for name in names if not getattr(args, name, None)]
    if missing:
        raise ValueError(
            "prescreen telemetry v2 requires all identity/evidence arguments; "
            f"missing={missing}"
        )
    if args.accumulate_grad_batches != 1:
        raise ValueError("prescreen telemetry v2 requires accumulation=1")
    if args.ckpt_path:
        raise ValueError("prescreen telemetry v2 forbids checkpoint resume")
    if int(args.max_steps) <= 0:
        raise ValueError("prescreen telemetry v2 requires --max-steps > 0")
    if int(args.seed) != 42:
        raise ValueError("prescreen telemetry v2 requires seed 42")
    if str(args.precision) != "bf16-mixed":
        raise ValueError("prescreen telemetry v2 requires bf16-mixed precision")
    if int(args.num_sanity_val_steps) != 0:
        raise ValueError(
            "prescreen telemetry v2 requires --num-sanity-val-steps 0"
        )
    if args.limit_val_batches != 0:
        raise ValueError(
            "prescreen telemetry v2 requires --limit-val-batches 0"
        )
    if not args.disable_early_stopping:
        raise ValueError("prescreen telemetry v2 requires early stopping disabled")
    if int(args.gpus) != 1 or int(args.num_nodes) != 1:
        raise ValueError("prescreen telemetry v2 requires one GPU on one node")
    if str(args.exp_name) != str(args.prescreen_arm):
        raise ValueError("--exp-name must equal --prescreen-arm")
    if args.prescreen_arm == "native" and args.pavir_config:
        raise ValueError("native prescreen arm must not receive --pavir-config")
    if args.prescreen_arm == "full" and not args.pavir_config:
        raise ValueError("full prescreen arm requires --pavir-config")

    expected_w0 = Path(args.prescreen_expected_w0).resolve()
    registry = Path(args.prescreen_registry).resolve()
    if not expected_w0.is_file():
        raise FileNotFoundError(expected_w0)
    if not registry.is_file():
        raise FileNotFoundError(registry)
    if (
        not args.init_lightning_checkpoint
        or Path(args.init_lightning_checkpoint).resolve() != expected_w0
    ):
        raise ValueError(
            "prescreen telemetry v2 requires weights-only initialization from "
            "the exact --prescreen-expected-w0"
        )
    output_paths = [
        Path(args.prescreen_telemetry).resolve(),
        Path(args.prescreen_evidence_journal).resolve(),
        Path(args.prescreen_fixed_final_checkpoint).resolve(),
    ]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("prescreen telemetry, journal, and checkpoint must differ")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"prescreen outputs are no-clobber: {existing}")
    for name in ("prescreen_run_id", "prescreen_arm", "prescreen_stage"):
        value = str(getattr(args, name))
        if not value or "\n" in value or "\r" in value:
            raise ValueError(f"invalid --{name.replace('_', '-')}")
    return True


class PrescreenTelemetryV2Callback(Callback):
    """Fail-closed, hash-chained evidence for the guarded 328-update gate."""

    def __init__(self, args):
        self.args = args
        self.telemetry_path = Path(args.prescreen_telemetry).resolve()
        self.journal_path = Path(args.prescreen_evidence_journal).resolve()
        self.expected_w0 = Path(args.prescreen_expected_w0).resolve()
        self.fixed_checkpoint = Path(
            args.prescreen_fixed_final_checkpoint
        ).resolve()
        self.registry = Path(args.prescreen_registry).resolve()
        self.expected_updates = int(args.max_steps)
        self.execution = {
            "registry_path": str(self.registry),
            "registry_sha256": sha256_file(self.registry),
            "run_id": str(args.prescreen_run_id),
            "arm": str(args.prescreen_arm),
            "stage": str(args.prescreen_stage),
            "seed": int(args.seed),
            "precision": str(args.precision),
            "accumulate_grad_batches": int(args.accumulate_grad_batches),
            "resumed": False,
        }
        self._descriptor = None
        self._previous_sha256 = PRESCREEN_CHAIN_ORIGIN
        self._total_records = 0
        self._updates = []
        self._current = None
        self._parameter_hook_handles = []
        self._optimizer_hook_handles = []
        self._trainable_parameter_names = set()
        self._parameters_with_hook_calls = set()
        self._optimizer_step_pre_hooks = 0
        self._optimizer_step_post_hooks = 0
        self._validation_batches = 0
        self._sanity_validation_batches = 0
        self._oom_events = 0
        self._initialization = None
        self._fit_snapshot = None
        self._summary_written = False
        self._cuda_enabled = False

    def _append(self, payload: Mapping[str, Any]) -> None:
        if self._descriptor is None:
            raise RuntimeError("prescreen evidence journal is not open")
        material = {
            "previous_sha256": self._previous_sha256,
            "payload": payload,
        }
        record_sha256 = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        envelope = {**material, "record_sha256": record_sha256}
        _write_all(self._descriptor, canonical_json_bytes(envelope) + b"\n")
        os.fsync(self._descriptor)
        self._previous_sha256 = record_sha256
        self._total_records += 1

    def _close_journal(self) -> None:
        if self._descriptor is not None:
            os.fsync(self._descriptor)
            os.close(self._descriptor)
            self._descriptor = None
            _fsync_directory(self.journal_path.parent)

    def _load_and_verify_w0(self, pl_module) -> dict[str, Any]:
        import torch

        checkpoint = torch.load(
            self.expected_w0,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        expected_state = checkpoint.get("state_dict")
        if not isinstance(expected_state, Mapping) or not expected_state:
            raise ValueError("expected W0 has no state_dict")
        current_state = pl_module.state_dict()
        keys_exact = set(current_state) == set(expected_state)
        values_exact = keys_exact
        if keys_exact:
            for key in sorted(expected_state):
                expected = expected_state[key]
                current = current_state[key]
                if (
                    not isinstance(expected, torch.Tensor)
                    or not isinstance(current, torch.Tensor)
                    or expected.dtype != current.dtype
                    or tuple(expected.shape) != tuple(current.shape)
                    or not torch.equal(expected, current.detach().cpu())
                ):
                    values_exact = False
                    break
        else:
            values_exact = False
        expected_hash = model_state_sha256(expected_state)
        current_hash = model_state_sha256(current_state)
        if not values_exact or current_hash != expected_hash:
            raise RuntimeError("initial model state is not exactly expected W0")
        return {
            "expected_w0": {
                "path": str(self.expected_w0),
                "sha256": sha256_file(self.expected_w0),
                "model_state_sha256": expected_hash,
            },
            "model_state_sha256": current_hash,
            "model_state_keys": len(current_state),
            "model_state_exact_w0": True,
        }

    @staticmethod
    def _scheduler_record(trainer) -> tuple[list[dict[str, Any]], bool]:
        records = []
        for config in trainer.lr_scheduler_configs:
            scheduler = config.scheduler
            state = scheduler.state_dict()
            records.append(
                {
                    "class": type(scheduler).__name__,
                    "last_epoch": int(state.get("last_epoch", -1)),
                    "step_count": int(state.get("_step_count", -1)),
                    "base_lrs": [float(value) for value in state.get("base_lrs", [])],
                }
            )
        fresh = (
            len(records) == 1
            and records[0]["class"] == "WarmupCosineScheduler"
            and records[0]["last_epoch"] == 0
            and records[0]["step_count"] == 1
        )
        return records, fresh

    def _register_hooks(self, trainer, pl_module) -> None:
        for name, parameter in pl_module.named_parameters():
            if not parameter.requires_grad:
                continue
            self._trainable_parameter_names.add(name)

            def gradient_hook(gradient, parameter_name=name):
                if self._current is None:
                    raise RuntimeError(
                        "gradient hook fired outside a prescreen update"
                    )
                self._current["gradient_hook_calls"] += 1
                self._current["gradient_hook_names"].add(parameter_name)
                self._parameters_with_hook_calls.add(parameter_name)
                return gradient

            self._parameter_hook_handles.append(
                parameter.register_hook(gradient_hook)
            )
        optimizer = trainer.optimizers[0]

        def optimizer_pre_hook(_optimizer, _args, _kwargs):
            self._optimizer_step_pre_hooks += 1

        def optimizer_post_hook(_optimizer, _args, _kwargs):
            self._optimizer_step_post_hooks += 1

        self._optimizer_hook_handles.extend(
            (
                optimizer.register_step_pre_hook(optimizer_pre_hook),
                optimizer.register_step_post_hook(optimizer_post_hook),
            )
        )

    def _remove_hooks(self) -> None:
        for handle in self._parameter_hook_handles + self._optimizer_hook_handles:
            handle.remove()
        self._parameter_hook_handles.clear()
        self._optimizer_hook_handles.clear()

    def on_fit_start(self, trainer, pl_module):
        import torch

        if int(trainer.global_step) != 0:
            raise RuntimeError("prescreen fit did not start at global_step zero")
        if len(trainer.optimizers) != 1:
            raise RuntimeError("prescreen requires exactly one optimizer")
        optimizer = trainer.optimizers[0]
        optimizer_state_entries = len(optimizer.state)
        if optimizer_state_entries != 0:
            raise RuntimeError("prescreen optimizer state is not fresh/empty")
        scheduler_states, scheduler_fresh = self._scheduler_record(trainer)
        if not scheduler_fresh:
            raise RuntimeError("prescreen scheduler state is not fresh")
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            raise RuntimeError("bf16 prescreen unexpectedly has a precision scaler")
        verified = self._load_and_verify_w0(pl_module)
        self._initialization = {
            **verified,
            "global_step": int(trainer.global_step),
            "global_step_zero": True,
            "optimizer_count": len(trainer.optimizers),
            "optimizer_state_entries": optimizer_state_entries,
            "fresh_optimizer_state_empty": True,
            "scheduler_count": len(scheduler_states),
            "scheduler_states": scheduler_states,
            "fresh_scheduler_state": True,
            "precision_scaler_state": "not_applicable_bf16",
            "precision_scaler_fresh_or_not_applicable": True,
        }
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(
            self.journal_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        self._append(
            {
                "schema_version": PRESCREEN_EVIDENCE_SCHEMA,
                "record_type": "header",
                "execution": self.execution,
                "expected_updates": self.expected_updates,
                "expected_w0": self._initialization["expected_w0"],
                "telemetry_path": str(self.telemetry_path),
                "fixed_final_checkpoint_path": str(self.fixed_checkpoint),
                "initialization": self._initialization,
                "validation": {
                    "num_sanity_val_steps": int(
                        self.args.num_sanity_val_steps
                    ),
                    "limit_val_batches": int(self.args.limit_val_batches),
                },
            }
        )
        self._register_hooks(trainer, pl_module)
        self._cuda_enabled = any(
            parameter.device.type == "cuda"
            for parameter in pl_module.parameters()
        )
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._current is not None:
            raise RuntimeError("prescreen update metadata overlapped")
        update_index = len(self._updates)
        global_step = int(trainer.global_step)
        if update_index >= self.expected_updates or global_step != update_index:
            raise RuntimeError(
                "prescreen update order drifted before backward: "
                f"index={update_index}, global_step={global_step}"
            )
        sample_ids = [str(value) for value in batch.get("sample_ids", [])]
        input_lengths = [
            int(value) for value in batch["input_lengths"].detach().cpu().tolist()
        ]
        transformed_hashes = [
            str(value)
            for value in batch.get("transformed_input_sha256s", [])
        ]
        if (
            not sample_ids
            or len(sample_ids) != len(input_lengths)
            or len(sample_ids) != len(transformed_hashes)
            or any(len(value) != 64 for value in transformed_hashes)
        ):
            raise RuntimeError(
                "prescreen batch lacks aligned sample IDs, lengths, or "
                "transformed-input hashes"
            )
        batch_identity = {
            "sample_ids": sample_ids,
            "input_lengths": input_lengths,
            "transformed_input_sha256s": transformed_hashes,
        }
        self._current = {
            "update_index": update_index,
            "global_step_before": global_step,
            "batch_idx": int(batch_idx),
            **batch_identity,
            "batch_identity_sha256": hashlib.sha256(
                canonical_json_bytes(batch_identity)
            ).hexdigest(),
            "gradient_hook_calls": 0,
            "gradient_hook_names": set(),
        }

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self._current is None:
            raise RuntimeError("missing prescreen batch at optimizer step")
        if "pre_clip_gradient_norm_l2" in self._current:
            raise RuntimeError("duplicate optimizer step for prescreen batch")
        norm, tensor_count, element_count = _pre_clip_gradient_norm(pl_module)
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite pre-clip gradient norm")
        self._current.update(
            {
                "pre_clip_gradient_norm_l2": norm,
                "pre_clip_gradient_norm_finite": True,
                "gradient_tensor_count": tensor_count,
                "gradient_element_count": element_count,
            }
        )

    @staticmethod
    def _loss_value(outputs) -> float:
        import torch

        value = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError("prescreen training output has no scalar loss")
        return float(value.detach().float().cpu())

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self._current is None:
            raise RuntimeError("missing prescreen update metadata at batch end")
        current = self._current
        self._current = None
        loss = self._loss_value(outputs)
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite prescreen loss")
        update_index = int(current["update_index"])
        global_step_after = int(trainer.global_step)
        if global_step_after != update_index + 1:
            raise RuntimeError(
                "prescreen global_step did not advance exactly once: "
                f"{global_step_after} != {update_index + 1}"
            )
        if "pre_clip_gradient_norm_l2" not in current:
            raise RuntimeError("prescreen pre-clip gradient observation is missing")
        if int(current["gradient_hook_calls"]) <= 0:
            raise RuntimeError("prescreen observed no parameter gradient hooks")
        payload = {
            "schema_version": PRESCREEN_EVIDENCE_SCHEMA,
            "record_type": "update",
            "update_index": update_index,
            "global_step_before": int(current["global_step_before"]),
            "global_step_after": global_step_after,
            "batch_idx": int(current["batch_idx"]),
            "sample_ids": current["sample_ids"],
            "input_lengths": current["input_lengths"],
            "transformed_input_sha256s": current[
                "transformed_input_sha256s"
            ],
            "batch_identity_sha256": current["batch_identity_sha256"],
            "loss": loss,
            "loss_finite": True,
            "pre_clip_gradient_norm_l2": current[
                "pre_clip_gradient_norm_l2"
            ],
            "pre_clip_gradient_norm_finite": True,
            "gradient_tensor_count": int(current["gradient_tensor_count"]),
            "gradient_element_count": int(current["gradient_element_count"]),
            "gradient_hook_calls": int(current["gradient_hook_calls"]),
            "gradient_hook_unique_parameters": len(
                current["gradient_hook_names"]
            ),
        }
        self._append(payload)
        self._updates.append(payload)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if trainer.sanity_checking:
            self._sanity_validation_batches += 1
        else:
            self._validation_batches += 1

    def _capture_fit_snapshot(self, trainer, pl_module) -> dict[str, Any]:
        import torch

        model_finiteness = _tensor_tree_finiteness(pl_module.state_dict())
        optimizer_finiteness = _tensor_tree_finiteness(
            [optimizer.state for optimizer in trainer.optimizers]
        )
        peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if self._cuda_enabled else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved()) if self._cuda_enabled else 0
        )
        return {
            "end_global_step": int(trainer.global_step),
            "final_finiteness": {
                "model": {
                    **model_finiteness,
                    "state_sha256": model_state_sha256(
                        pl_module.state_dict()
                    ),
                },
                "optimizer": optimizer_finiteness,
            },
            "resources": {
                "cuda_peak_allocated_bytes": peak_allocated,
                "cuda_peak_reserved_bytes": peak_reserved,
                "maximum_peak_allocated_bytes_exclusive": (
                    PRESCREEN_VRAM_LIMIT_BYTES
                ),
                "oom_events": self._oom_events,
            },
        }

    def on_fit_end(self, trainer, pl_module):
        self._close_journal()
        self._remove_hooks()
        self._fit_snapshot = self._capture_fit_snapshot(trainer, pl_module)
        preliminary_checks = self._checks(
            trainer,
            fixed_checkpoint_global_step=None,
        )
        for name, passed in preliminary_checks.items():
            if name == "fixed_final_checkpoint_exact_global_step":
                continue
            if not passed:
                raise RuntimeError(
                    f"prescreen telemetry v2 fit gate failed: {name}"
                )

    def _checks(
        self,
        trainer,
        *,
        fixed_checkpoint_global_step: int | None,
    ) -> dict[str, bool]:
        snapshot = self._fit_snapshot or {}
        final = snapshot.get("final_finiteness", {})
        resources = snapshot.get("resources", {})
        update_indices = [row["update_index"] for row in self._updates]
        return {
            "start_global_step_zero": bool(
                self._initialization
                and self._initialization["global_step_zero"]
            ),
            "fresh_optimizer_state_empty": bool(
                self._initialization
                and self._initialization["fresh_optimizer_state_empty"]
            ),
            "fresh_scheduler_state": bool(
                self._initialization
                and self._initialization["fresh_scheduler_state"]
            ),
            "precision_scaler_fresh_or_not_applicable": bool(
                self._initialization
                and self._initialization[
                    "precision_scaler_fresh_or_not_applicable"
                ]
            ),
            "initial_model_exact_expected_w0": bool(
                self._initialization
                and self._initialization["model_state_exact_w0"]
            ),
            "exact_physical_batch_count": (
                len(self._updates) == self.expected_updates
            ),
            "exact_optimizer_step_pre_hook_count": (
                self._optimizer_step_pre_hooks == self.expected_updates
            ),
            "exact_optimizer_step_post_hook_count": (
                self._optimizer_step_post_hooks == self.expected_updates
            ),
            "ordered_updates_zero_based_contiguous": (
                update_indices == list(range(self.expected_updates))
            ),
            "all_losses_finite": (
                len(self._updates) == self.expected_updates
                and all(row["loss_finite"] for row in self._updates)
            ),
            "all_pre_clip_gradient_norms_finite": (
                len(self._updates) == self.expected_updates
                and all(
                    row["pre_clip_gradient_norm_finite"]
                    for row in self._updates
                )
            ),
            "gradient_hooks_observed_every_update": (
                len(self._updates) == self.expected_updates
                and all(row["gradient_hook_calls"] > 0 for row in self._updates)
            ),
            "zero_sanity_validation_batches": (
                self._sanity_validation_batches == 0
            ),
            "zero_validation_batches": self._validation_batches == 0,
            "final_model_finite": bool(
                final.get("model", {}).get("finite")
            ),
            "final_optimizer_finite": bool(
                final.get("optimizer", {}).get("finite")
            ),
            "no_oom": resources.get("oom_events") == 0,
            "peak_allocated_vram_below_72_gib": (
                int(resources.get("cuda_peak_allocated_bytes", -1))
                < PRESCREEN_VRAM_LIMIT_BYTES
            ),
            "fixed_final_checkpoint_exact_global_step": (
                fixed_checkpoint_global_step == self.expected_updates
            ),
        }

    def _summary(
        self,
        *,
        status: str,
        trainer,
        exception: BaseException | None = None,
        fixed_checkpoint_record: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        journal_record = {
            "path": str(self.journal_path),
            "sha256": (
                sha256_file(self.journal_path)
                if self.journal_path.is_file()
                else None
            ),
            "total_records": self._total_records,
            "update_records": len(self._updates),
            "terminal_chain_sha256": self._previous_sha256,
        }
        fixed_step = (
            int(fixed_checkpoint_record["global_step"])
            if fixed_checkpoint_record is not None
            else None
        )
        checks = self._checks(
            trainer,
            fixed_checkpoint_global_step=fixed_step,
        )
        return {
            "schema_version": PRESCREEN_TELEMETRY_SCHEMA,
            "status": status,
            "execution": self.execution,
            "expected_updates": self.expected_updates,
            "initialization": self._initialization,
            "counts": {
                "physical_train_batches": len(self._updates),
                "optimizer_step_pre_hooks": self._optimizer_step_pre_hooks,
                "optimizer_step_post_hooks": self._optimizer_step_post_hooks,
                "pre_clip_observations": sum(
                    "pre_clip_gradient_norm_l2" in row
                    for row in self._updates
                ),
                "journal_update_records": len(self._updates),
                "start_global_step": (
                    self._initialization["global_step"]
                    if self._initialization
                    else None
                ),
                "end_global_step": (
                    self._fit_snapshot.get("end_global_step")
                    if self._fit_snapshot
                    else int(getattr(trainer, "global_step", -1))
                ),
            },
            "loss": {
                "observations": len(self._updates),
                "all_finite": all(
                    row["loss_finite"] for row in self._updates
                ),
                "first": (
                    self._updates[0]["loss"] if self._updates else None
                ),
                "last": (
                    self._updates[-1]["loss"] if self._updates else None
                ),
                "minimum": (
                    min(row["loss"] for row in self._updates)
                    if self._updates
                    else None
                ),
                "maximum": (
                    max(row["loss"] for row in self._updates)
                    if self._updates
                    else None
                ),
            },
            "gradients": {
                "observations": sum(
                    "pre_clip_gradient_norm_l2" in row
                    for row in self._updates
                ),
                "all_pre_clip_norms_finite": all(
                    row["pre_clip_gradient_norm_finite"]
                    for row in self._updates
                ),
                "gradient_hook_calls_total": sum(
                    row["gradient_hook_calls"] for row in self._updates
                ),
                "updates_with_gradient_hooks": sum(
                    row["gradient_hook_calls"] > 0 for row in self._updates
                ),
                "trainable_parameter_count": len(
                    self._trainable_parameter_names
                ),
                "parameters_with_gradient_hooks": len(
                    self._parameters_with_hook_calls
                ),
            },
            "crn": {
                "ordered_sample_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        [row["sample_ids"] for row in self._updates]
                    )
                ).hexdigest(),
                "ordered_input_lengths_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        [row["input_lengths"] for row in self._updates]
                    )
                ).hexdigest(),
                "ordered_transformed_input_sha256s_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        [
                            row["transformed_input_sha256s"]
                            for row in self._updates
                        ]
                    )
                ).hexdigest(),
                "ordered_batch_identity_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        [
                            row["batch_identity_sha256"]
                            for row in self._updates
                        ]
                    )
                ).hexdigest(),
            },
            "validation": {
                "num_sanity_val_steps": int(
                    self.args.num_sanity_val_steps
                ),
                "sanity_validation_batches": (
                    self._sanity_validation_batches
                ),
                "validation_batches": self._validation_batches,
            },
            "final_finiteness": (
                self._fit_snapshot.get("final_finiteness")
                if self._fit_snapshot
                else None
            ),
            "resources": (
                self._fit_snapshot.get("resources")
                if self._fit_snapshot
                else {
                    "cuda_peak_allocated_bytes": 0,
                    "cuda_peak_reserved_bytes": 0,
                    "maximum_peak_allocated_bytes_exclusive": (
                        PRESCREEN_VRAM_LIMIT_BYTES
                    ),
                    "oom_events": self._oom_events,
                }
            ),
            "evidence_journal": journal_record,
            "fixed_final_checkpoint": fixed_checkpoint_record,
            "checks": checks,
            "exception": None if exception is None else repr(exception),
        }

    def finalize_after_fit(self, trainer, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path).resolve()
        if checkpoint_path != self.fixed_checkpoint:
            raise ValueError("fixed-final checkpoint path changed")
        global_step = _checkpoint_global_step(checkpoint_path)
        checkpoint_record = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "global_step": global_step,
        }
        summary = self._summary(
            status="complete",
            trainer=trainer,
            fixed_checkpoint_record=checkpoint_record,
        )
        failed = [name for name, passed in summary["checks"].items() if not passed]
        if failed:
            raise RuntimeError(
                f"prescreen telemetry v2 final gate failed: {failed}"
            )
        _write_canonical_json_once(self.telemetry_path, summary)
        self._summary_written = True

    def record_external_failure(
        self, trainer, exception: BaseException
    ) -> None:
        if self._summary_written:
            return
        import torch

        if isinstance(exception, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(exception).lower()
        ):
            self._oom_events += 1
        self._close_journal()
        self._remove_hooks()
        if self.telemetry_path.exists():
            return
        try:
            summary = self._summary(
                status="failed",
                trainer=trainer,
                exception=exception,
            )
            _write_canonical_json_once(self.telemetry_path, summary)
            self._summary_written = True
        except Exception:
            # Never hide the original training failure with a best-effort
            # secondary telemetry write.
            return

    def on_exception(self, trainer, pl_module, exception):
        self.record_external_failure(trainer, exception)


class CurriculumEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        sampler = getattr(loader, "sampler", None)
        if sampler is None or not hasattr(sampler, "set_epoch"):
            raise RuntimeError("Curriculum was requested but its sampler is unavailable")
        sampler.set_epoch(trainer.current_epoch)
        fraction = float(sampler.current_fraction)
        pl_module.log("curriculum_fraction", fraction, on_step=False, on_epoch=True)
        print(f"Curriculum epoch={trainer.current_epoch} eligible_fraction={fraction:.3f}")


class CampaignTelemetryCallback(Callback):
    """Record physical batches, finite losses and peak CUDA allocation."""

    def __init__(self, output_path, expected_physical_batches):
        self.output_path = Path(output_path).resolve()
        self.expected_physical_batches = int(expected_physical_batches)
        if self.expected_physical_batches <= 0:
            raise ValueError("expected campaign physical batches must be positive")
        self.physical_batches = 0
        self.nonfinite_batches = 0
        self.losses = []
        self.start_global_step = 0

    def _write(self, trainer, status, exception=None):
        import torch

        payload = {
            "schema_version": "pavir-training-telemetry-v1",
            "status": status,
            "precision": str(trainer.precision),
            "start_global_step": self.start_global_step,
            "end_global_step": int(trainer.global_step),
            "physical_train_batches": self.physical_batches,
            "expected_physical_train_batches": self.expected_physical_batches,
            "finite": self.nonfinite_batches == 0,
            "nonfinite_batches": self.nonfinite_batches,
            "loss_first": self.losses[0] if self.losses else None,
            "loss_last": self.losses[-1] if self.losses else None,
            "loss_min": min(self.losses) if self.losses else None,
            "loss_max": max(self.losses) if self.losses else None,
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
            ),
            "exception": None if exception is None else repr(exception),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(self.output_path.name + ".tmp")
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.output_path)

    def on_fit_start(self, trainer, pl_module):
        import torch

        self.start_global_step = int(trainer.global_step)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        import torch

        self.physical_batches += 1
        value = outputs["loss"] if isinstance(outputs, dict) else outputs
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            self.nonfinite_batches += 1
        else:
            self.losses.append(float(value.detach()))

    def on_fit_end(self, trainer, pl_module):
        status = (
            "complete"
            if self.physical_batches == self.expected_physical_batches
            and self.nonfinite_batches == 0
            else "failed"
        )
        self._write(trainer, status)
        if status != "complete":
            raise RuntimeError(
                "campaign telemetry gate failed: "
                f"physical={self.physical_batches}/"
                f"{self.expected_physical_batches}, "
                f"nonfinite={self.nonfinite_batches}"
            )

    def on_exception(self, trainer, pl_module, exception):
        self._write(trainer, "failed", exception=exception)


def get_trainer(args):
    seed_everything(args.seed, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor=args.monitor_metric,
        mode="min",
        save_last=True,
        filename="{epoch:02d}-{step}",
        save_top_k=args.save_top_k,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop = EarlyStopping(
        monitor=args.monitor_metric,
        mode="min",
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        check_finite=True,
        verbose=True,
    )
    callbacks = [lr_monitor]
    if not args.disable_checkpointing:
        callbacks.insert(0, checkpoint)
    if not args.disable_early_stopping:
        callbacks.append(early_stop)
    if getattr(args, "curriculum_scores", None):
        callbacks.append(CurriculumEpochCallback())
    if getattr(args, "campaign_telemetry", None):
        callbacks.append(
            CampaignTelemetryCallback(
                args.campaign_telemetry,
                args.campaign_expected_physical_batches,
            )
        )
    if _prescreen_enabled(args):
        callbacks.append(PrescreenTelemetryV2Callback(args))

    return Trainer(
        sync_batchnorm=args.gpus > 1,
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else "auto",
        callbacks=callbacks,
        enable_checkpointing=not args.disable_checkpointing,
        # Dataset and sampler do not change between epochs. Keeping workers
        # alive preserves the decoded-video cache and avoids PyTorch's known
        # persistent_workers + dataloader-reload instability.
        reload_dataloaders_every_n_epochs=0,
        logger=CSVLogger(save_dir=args.exp_dir, name=f"{args.exp_name}_logs"),
        gradient_clip_val=10.0,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
    )


def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default="./exp",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
        required=True,
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name",
        required=True,
    )
    parser.add_argument(
        "--group-name",
        type=str,
        help="Group name of the task (wandb API)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Type of input modality",
        required=True,
        choices=["audio", "video"],
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory of preprocessed dataset",
        required=True,
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="Filename of training label list",
        required=True,
    )
    parser.add_argument(
        "--val-file",
        default="lrs3_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of validation label list. (Default: lrs3_test_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of testing label list. (Default: lrs3_test_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--num-nodes",
        default=4,
        type=int,
        help="Number of machines used. (Default: 4)",
        required=True,
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of gpus in each machine. (Default: 8)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--transfer-frontend",
        action="store_true",
        help="Flag to load the front-end only, works with `pretrained-model`",
    )
    parser.add_argument(
        "--transfer-encoder",
        action="store_true",
        help="Flag to load the weights of encoder, works with `pretrained-model`",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of epochs for warmup. (Default: 5)",
    )
    parser.add_argument(
        "--max-epochs",
        default=75,
        type=int,
        help="Number of epochs. (Default: 75)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this exact optimizer-update budget; -1 disables the limit.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1600,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Accumulate this many physical batches per optimizer update.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. (Default: 1e-3)",
    )
    parser.add_argument("--aux-lr", type=float, default=None, help="Optional higher LR for newly initialized ViSyll heads only.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.03,
        help="Weight decay",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.1,
        help="CTC weight",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--curriculum-scores", type=str, default=None,
        help="Train-only JSON mapping sample ids to fixed difficulty scores.",
    )
    parser.add_argument(
        "--curriculum-fractions", type=str, default="0.5,0.75,1.0,1.0,1.0",
        help="Comma-separated eligible fractions, one per epoch then clamped to the last value.",
    )
    parser.add_argument("--curriculum-seed", type=int, default=42)
    parser.add_argument(
        "--num-workers", type=int, default=12,
    )
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed",
    )
    parser.add_argument(
        "--log-every-n-steps", type=int, default=20,
    )
    parser.add_argument(
        "--campaign-telemetry",
        type=str,
        default=None,
        help="Write fail-closed physical-batch/finite/peak-VRAM telemetry.",
    )
    parser.add_argument(
        "--campaign-expected-physical-batches",
        type=int,
        default=None,
        help="Exact physical batch count required by --campaign-telemetry.",
    )
    parser.add_argument(
        "--prescreen-telemetry",
        type=str,
        default=None,
        help=(
            "Opt-in fail-closed prescreen telemetry-v2 summary. Enabling this "
            "requires the complete prescreen identity/evidence argument set."
        ),
    )
    parser.add_argument(
        "--prescreen-evidence-journal",
        type=str,
        default=None,
        help="No-clobber, hash-chained per-update prescreen JSONL evidence.",
    )
    parser.add_argument(
        "--prescreen-expected-w0",
        type=str,
        default=None,
        help="Exact Lightning W0 whose complete model state must match at step 0.",
    )
    parser.add_argument(
        "--prescreen-fixed-final-checkpoint",
        type=str,
        default=None,
        help="No-clobber complete checkpoint saved explicitly after fit.",
    )
    parser.add_argument(
        "--prescreen-registry",
        type=str,
        default=None,
        help="Hash-bound v3 arm registry for the prescreen run identity.",
    )
    parser.add_argument("--prescreen-run-id", type=str, default=None)
    parser.add_argument("--prescreen-arm", type=str, default=None)
    parser.add_argument("--prescreen-stage", type=str, default=None)
    parser.add_argument(
        "--limit-train-batches", type=int, default=None,
    )
    parser.add_argument(
        "--limit-val-batches", type=int, default=None,
    )
    parser.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=2,
        help="Lightning validation sanity steps. Default 2 preserves prior behavior.",
    )
    parser.add_argument(
        "--average-last", type=int, default=0,
        help="Average the last N checkpoints after training; disabled by default.",
    )
    parser.add_argument(
        "--save-top-k", type=int, default=3,
        help="Number of epoch checkpoints retained by Lightning; -1 retains every epoch.",
    )
    parser.add_argument(
        "--disable-early-stopping", action="store_true",
        help="Run the locked fixed epoch schedule and select checkpoints post hoc.",
    )
    parser.add_argument(
        "--disable-checkpointing", action="store_true",
        help="Disable checkpoint writes for implementation smoke tests only.",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=8,
        help="Stop after this many validation epochs without val-loss improvement.",
    )
    parser.add_argument(
        "--early-stop-min-delta", type=float, default=0.01,
        help="Minimum absolute validation-loss improvement.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path of the checkpoint from which training is resumed.",
    )
    parser.add_argument("--init-lightning-checkpoint", type=str, default=None, help="Load model weights only from a Lightning checkpoint; optimizer/scheduler restart fairly across screening arms.")
    parser.add_argument(
        "--pavir-config",
        type=str,
        default=None,
        help="Hash-locked PAViR artifact/config bundle; omit for the native arm.",
    )
    parser.add_argument(
        "--sample-id-map",
        type=str,
        default=None,
        help="Shared canonical loader-key to sample-ID map (automatically taken from --pavir-config for PAViR).",
    )
    parser.add_argument("--onset-vocab", type=str, default=None)
    parser.add_argument("--rime-vocab", type=str, default=None)
    parser.add_argument("--aux-layer", type=int, choices=[6, 12], default=6)
    parser.add_argument("--onset-weight", type=float, default=0.0)
    parser.add_argument("--rime-weight", type=float, default=0.0)
    parser.add_argument("--syllable-vocab", type=str, default=None)
    parser.add_argument("--syllable-weight", type=float, default=0.0)
    parser.add_argument("--syllable-ctc-weight", type=float, default=0.1)
    parser.add_argument("--syllable-decoder-blocks", type=int, default=3)
    parser.add_argument("--syllable-mode", choices=["unified", "rhyme_first"], default="unified")
    parser.add_argument("--count-weight", type=float, default=0.0)
    parser.add_argument("--count-bins", type=int, default=33)
    parser.add_argument("--count-hidden-dim", type=int, default=256)
    parser.add_argument("--count-dropout", type=float, default=0.1)
    parser.add_argument("--interctc-layer", type=int, choices=[6], default=6)
    parser.add_argument("--interctc-text-weight", type=float, default=0.0)
    parser.add_argument("--interctc-toneless-weight", type=float, default=0.0)
    parser.add_argument(
        "--interctc-shared-weight", type=float, default=0.0,
        help="Canonical InterCTC mix inside the existing outer CTC budget; reuses final CTC projection.",
    )
    parser.add_argument("--signer-vocab", type=str, default=None)
    parser.add_argument("--signer-classes", type=int, default=11)
    parser.add_argument("--signer-layer", type=int, choices=[6], default=6)
    parser.add_argument("--signer-hidden-dim", type=int, default=256)
    parser.add_argument("--signer-dropout", type=float, default=0.1)
    parser.add_argument("--signer-adv-weight", type=float, default=0.0)
    parser.add_argument("--signer-kd-weight", type=float, default=0.0)
    parser.add_argument("--signer-teacher-checkpoint", type=str, default=None)
    parser.add_argument("--signer-ramp-updates", type=int, default=491)
    parser.add_argument("--xbridge-self-weight", type=float, default=0.0)
    parser.add_argument("--xbridge-cross-weight", type=float, default=0.0)
    parser.add_argument("--xbridge-middle-layer", type=int, choices=[6], default=6)
    parser.add_argument("--xbridge-rank", type=int, default=192)
    parser.add_argument("--xbridge-teacher-root", type=str, default=None)
    parser.add_argument("--xbridge-freeze-base", action="store_true")
    parser.add_argument("--xbridge-ctc-only", action="store_true")
    parser.add_argument("--mdbridge-mode", choices=["none", "pointwise", "difference"], default="none")
    parser.add_argument("--mdbridge-middle-layer", type=int, choices=[6], default=6)
    parser.add_argument("--mdbridge-rank", type=int, default=192)
    parser.add_argument("--mdbridge-lags", type=str, default="1,2,4")
    parser.add_argument("--mdbridge-freeze-base", action="store_true")
    parser.add_argument("--monitor-metric", type=str, default="loss_val")
    parser.add_argument(
        "--slurm-job-id",
        type=float,
        default=0,
        help="Slurm job id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    prescreen_enabled = validate_prescreen_args(args)
    import torch
    if (args.campaign_telemetry is None) != (
        args.campaign_expected_physical_batches is None
    ):
        raise ValueError(
            "--campaign-telemetry and "
            "--campaign-expected-physical-batches must be supplied together"
        )
    if args.ckpt_path and args.init_lightning_checkpoint:
        raise ValueError(
            "--ckpt-path resumes optimizer/scheduler state and is mutually "
            "exclusive with weights-only --init-lightning-checkpoint"
        )
    if args.pavir_config:
        from pathlib import Path
        from pavir.artifacts import PAViRConfig

        if args.modality != "video":
            raise ValueError("PAViR-VSR is visual-only; --modality must be video")
        locked = PAViRConfig.load(args.pavir_config)
        if args.seed != locked.seed:
            raise ValueError(
                f"--seed {args.seed} disagrees with the hash-locked "
                f"PAViR seed {locked.seed}"
            )
        locked_map = locked.canonical_ids.path.resolve()
        if args.sample_id_map and Path(args.sample_id_map).resolve() != locked_map:
            raise ValueError(
                "--sample-id-map disagrees with the hash-locked PAViR config"
            )
        args.sample_id_map = str(locked_map)
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed, workers=True)
    #init_logger(args.debug)
    args.slurm_job_id = os.environ.get("SLURM_JOB_ID", str(args.slurm_job_id))
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args, train_num_buckets=args.train_num_buckets, num_workers=args.num_workers)
    trainer = get_trainer(args)
    prescreen_callback = next(
        (
            callback
            for callback in trainer.callbacks
            if isinstance(callback, PrescreenTelemetryV2Callback)
        ),
        None,
    )
    try:
        trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
        if prescreen_enabled:
            if prescreen_callback is None:
                raise RuntimeError("prescreen telemetry v2 callback is missing")
            fixed_checkpoint = save_checkpoint_once(
                trainer,
                args.prescreen_fixed_final_checkpoint,
            )
            prescreen_callback.finalize_after_fit(trainer, fixed_checkpoint)
    except BaseException as exception:
        if prescreen_callback is not None:
            prescreen_callback.record_external_failure(trainer, exception)
        raise
    if args.average_last:
        ensemble(args)


if __name__ == "__main__":
    cli_main()
