"""Auto-AVSR with training-only PAViR auxiliary routing.

This class deliberately owns the exact same parameters and persistent buffers
as the native :class:`E2E`.  PAViR artifacts are fixed plain-Python/NumPy data,
so a resulting checkpoint can be loaded by native inference without this
module or a PAViR config.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from pavir.artifacts import ArtifactStore, PAViRConfig
from pavir.routing import AuxiliaryResult, context_margin_loss, visual_margin_loss


class PAViRE2E(E2E):
    """Native Auto-AVSR plus hash-locked, gradient-isolated auxiliaries."""

    pavir_enabled = True

    def __init__(
        self,
        odim: int,
        modality: str,
        *,
        pavir_config: str,
        ctc_weight: float = 0.1,
        ignore_id: int = -1,
    ):
        if modality != "video":
            raise ValueError("PAViR-VSR is visual-only")
        super().__init__(
            odim,
            modality,
            ctc_weight=ctc_weight,
            ignore_id=ignore_id,
        )
        self.pavir_config = PAViRConfig.load(pavir_config)
        self.pavir_store = ArtifactStore(self.pavir_config)
        if self.pavir_config.block_index > len(self.encoder.encoders):
            raise ValueError(
                f"PAViR block {self.pavir_config.block_index} exceeds "
                f"{len(self.encoder.encoders)} encoder blocks"
            )

    @staticmethod
    def _zero_result(reference: torch.Tensor) -> AuxiliaryResult:
        return AuxiliaryResult(
            loss=reference.new_zeros(()),
            eligible=0,
            active=0,
            selected_batch=False,
        )

    @staticmethod
    def _normalize_loader_targets(label: torch.Tensor) -> torch.Tensor:
        """Bridge Auto-AVSR's legacy ``[B,1,U]`` target collation to ``[B,U]``.

        Native Auto-AVSR happens to flatten the singleton axis through boolean
        indexing.  PAViR's explicit span checks require the canonical rank-two
        representation, so normalize only that legacy singleton form and fail
        closed for every other target shape.
        """

        if label.ndim == 3 and label.size(1) == 1:
            label = label.squeeze(1)
        if label.ndim != 2:
            raise ValueError(
                "PAViR targets must be [B,U] or legacy Auto-AVSR [B,1,U]"
            )
        return label

    def _encode_native(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        capture_block: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        captured = {}
        handle = None
        if capture_block:
            layer = self.encoder.encoders[self.pavir_config.block_index - 1]

            def capture(_module, _inputs, output):
                value = output[0]
                if isinstance(value, tuple):
                    value = value[0]
                captured["features"] = value

            handle = layer.register_forward_hook(capture)
        try:
            encoded, _ = self.encoder(x, padding_mask)
        finally:
            if handle is not None:
                handle.remove()
        block_features = captured.get("features")
        if capture_block and block_features is None:
            raise RuntimeError("PAViR failed to capture the configured encoder block")
        return encoded, block_features

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        label: torch.Tensor,
        *,
        sample_ids: Sequence[str] | None = None,
        update: int = 0,
        context_gate_index: int | None = None,
        apply_pavir: bool = True,
    ):
        """Run the native objective, then optional training-only auxiliaries."""

        label = self._normalize_loader_targets(label)
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        x = self.frontend(x)
        x = self.proj_encoder(x)

        use_visual = (
            apply_pavir and self.pavir_config.lambda_visual > 0.0
        )
        use_context = (
            apply_pavir and self.pavir_config.lambda_context > 0.0
        )
        if (use_visual or use_context) and sample_ids is None:
            raise ValueError("PAViR training requires canonical sample_ids")
        if sample_ids is not None and len(sample_ids) != x.size(0):
            raise ValueError("sample_ids does not match batch size")

        encoded, block_features = self._encode_native(
            x,
            padding_mask,
            capture_block=use_visual,
        )

        # Keep this native section in the same order as E2E.forward so its RNG
        # stream and floating-point result are directly comparable to B0.
        loss_ctc, ys_hat = self.ctc(encoded, lengths, label)
        ys_in_pad, ys_out_pad = add_sos_eos(
            label, self.sos, self.eos, self.ignore_id
        )
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(
            ys_in_pad, ys_mask, encoded, padding_mask
        )
        loss_att = self.criterion(pred_pad, ys_out_pad)
        native_loss = (
            self.ctc_weight * loss_ctc
            + (1.0 - self.ctc_weight) * loss_att
        )
        acc = th_accuracy(
            pred_pad.view(-1, self.odim),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        visual = self._zero_result(native_loss)
        context = self._zero_result(native_loss)
        if use_visual:
            assert block_features is not None and sample_ids is not None
            # Reuse the exact stochastic CTC logits already consumed by the
            # native CTC loss, but sever alignment gradients explicitly.
            ctc_log_probs = F.log_softmax(
                ys_hat.transpose(0, 1).detach().float(),
                dim=-1,
            )
            visual = visual_margin_loss(
                block_features,
                ctc_log_probs,
                label,
                lengths,
                sample_ids,
                self.pavir_store,
                margin=self.pavir_config.visual_margin,
                blank=self.blank,
                ignore_id=self.ignore_id,
            )
        if use_context:
            assert sample_ids is not None
            context = context_margin_loss(
                self.decoder,
                encoded,
                padding_mask,
                label,
                sample_ids,
                self.pavir_store,
                self.pavir_config,
                update=int(update),
                gate_index=context_gate_index,
                sos=self.sos,
                eos=self.eos,
                ignore_id=self.ignore_id,
            )

        ramp = (
            self.pavir_config.ramp(int(update))
            if use_visual or use_context
            else 0.0
        )
        weighted_visual = visual.loss * (
            self.pavir_config.lambda_visual * ramp
        )
        weighted_context = context.loss * (
            self.pavir_config.lambda_context * ramp
        )
        if use_visual or use_context:
            loss = native_loss + weighted_visual + weighted_context
        else:
            # Exact tensor identity for the lambda-zero/native-equivalence gate.
            loss = native_loss

        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att,
            "decoder_acc": acc,
            "text_loss": native_loss,
            "pavir_visual_loss": visual.loss,
            "pavir_context_loss": context.loss,
            "pavir_weighted_visual_loss": weighted_visual,
            "pavir_weighted_context_loss": weighted_context,
            "pavir_visual_eligible": visual.eligible,
            "pavir_visual_active": visual.active,
            "pavir_visual_infeasible": visual.infeasible,
            "pavir_context_eligible": context.eligible,
            "pavir_context_active": context.active,
            "pavir_context_selected_batch": float(context.selected_batch),
            "pavir_ramp": ramp,
        }
