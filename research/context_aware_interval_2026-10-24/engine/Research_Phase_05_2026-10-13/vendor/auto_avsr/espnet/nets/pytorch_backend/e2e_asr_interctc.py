"""Matched intermediate-CTC variants for Vietnamese visual speech recognition.

The auxiliary heads supervise Conformer block 6 during training only. They do
not feed predictions back into the encoder and are not used by native inference.
"""
from __future__ import annotations

import torch

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.e2e_asr_structured import (
    ctc_error_stats,
    ctc_feasible_mask,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class InterCTCE2E(E2E):
    """Native Auto-AVSR plus text and optional toneless intermediate CTC."""

    def __init__(
        self,
        odim: int,
        modality: str,
        ctc_weight: float = 0.1,
        interctc_layer: int = 6,
        interctc_text_weight: float = 0.0,
        interctc_toneless_weight: float = 0.0,
        interctc_shared_weight: float = 0.0,
        toneless_vocab_size: int = 0,
    ):
        super().__init__(odim, modality, ctc_weight=ctc_weight)
        if interctc_layer != 6:
            raise ValueError("The locked C2 protocol requires InterCTC at block 6")
        if interctc_shared_weight > 0:
            if not 0.0 < interctc_shared_weight < 1.0:
                raise ValueError("Shared-head InterCTC mixing weight must be in (0, 1)")
            if interctc_text_weight != 0 or interctc_toneless_weight != 0:
                raise ValueError("C2a-R cannot mix canonical shared-head and additive auxiliary modes")
        elif interctc_text_weight <= 0:
            raise ValueError("Additive C2 requires a positive text InterCTC weight")
        if interctc_toneless_weight < 0:
            raise ValueError("Toneless InterCTC weight cannot be negative")
        if interctc_toneless_weight > 0 and toneless_vocab_size <= 1:
            raise ValueError("A valid toneless vocabulary is required for C2b")

        self.interctc_layer = int(interctc_layer)
        self.interctc_text_weight = float(interctc_text_weight)
        self.interctc_toneless_weight = float(interctc_toneless_weight)
        self.interctc_shared_weight = float(interctc_shared_weight)
        # Construct text first in both arms. With the seed fixed before model
        # construction, C2a and C2b therefore start from the exact same text head.
        self.mid_text_ctc = (
            None if self.interctc_shared_weight > 0
            else CTC(odim, 768, 0.1, reduce=True)
        )
        self.mid_toneless_ctc = (
            CTC(toneless_vocab_size, 768, 0.1, reduce=True)
            if self.interctc_toneless_weight > 0
            else None
        )

    def encode_interctc(self, x: torch.Tensor, mask: torch.Tensor | None):
        xs = self.encoder.embed(x)
        middle = None
        for index, layer in enumerate(self.encoder.encoders):
            xs, mask = layer(xs, mask)
            if index + 1 == self.interctc_layer:
                middle = xs[0] if isinstance(xs, tuple) else xs
        final = xs[0] if isinstance(xs, tuple) else xs
        if self.encoder.normalize_before:
            final = self.encoder.after_norm(final)
        if middle is None:
            raise RuntimeError("InterCTC block was not reached")
        # Match standard ESPnet InterCTC: intermediate outputs from a
        # pre-normalized encoder pass through the encoder's native after_norm
        # before the CTC projection. This is the same shared, already-existing
        # LayerNorm used on the final output; no arm-specific module is added.
        if self.encoder.normalize_before:
            middle = self.encoder.after_norm(middle)
        return final, middle, mask

    @staticmethod
    def _auxiliary_loss(
        head: CTC,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor,
        weight: float,
        compute_metrics: bool,
    ) -> dict[str, object]:
        feasible = ctc_feasible_mask(targets, lengths)
        count = int(feasible.sum())
        if count:
            kept_targets = targets[feasible]
            raw, logits = head(hidden[feasible], lengths[feasible], kept_targets)
            weighted = float(weight) * raw
            if compute_metrics:
                errors, units, blanks, frames = ctc_error_stats(
                    logits, kept_targets, lengths[feasible], feasible[feasible]
                )
            else:
                errors = units = blanks = frames = 0
        else:
            raw = hidden.sum() * 0.0
            weighted = raw
            errors = units = blanks = frames = 0
        return {
            "raw": raw,
            "weighted": weighted,
            "feasible": count,
            "infeasible": len(feasible) - count,
            "errors": errors,
            "units": units,
            "blank_frames": blanks,
            "frames": frames,
        }

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        label: torch.Tensor,
        toneless_label: torch.Tensor | None = None,
        compute_aux_metrics: bool = False,
    ) -> dict[str, torch.Tensor | float | int]:
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        encoded = self.proj_encoder(self.frontend(x))
        final, middle, padding_mask = self.encode_interctc(encoded, padding_mask)

        loss_ctc, _ = self.ctc(final, lengths, label)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        pred_pad, _ = self.decoder(
            ys_in_pad,
            target_mask(ys_in_pad, self.ignore_id),
            final,
            padding_mask,
        )
        loss_att = self.criterion(pred_pad, ys_out_pad)
        text_loss = self.ctc_weight * loss_ctc + (1.0 - self.ctc_weight) * loss_att
        output: dict[str, torch.Tensor | float | int] = {
            "text_loss": text_loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att,
            "decoder_acc": th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            ),
        }

        if self.interctc_shared_weight > 0:
            # Canonical budget-preserving InterCTC. The pretrained final CTC
            # projection is reused at block 6, while the existing outer CTC
            # coefficient is split between final and intermediate CTC losses.
            # For ctc_weight=.1 and alpha=.5 this is exactly:
            #   .9 L_att + .05 L_final_ctc + .05 L_mid_ctc.
            final_ctc_weight = self.ctc_weight * (1.0 - self.interctc_shared_weight)
            output["weighted_final_ctc_loss"] = final_ctc_weight * loss_ctc
            total = (1.0 - self.ctc_weight) * loss_att + output["weighted_final_ctc_loss"]
            tasks = [
                (
                    "mid_text",
                    self.ctc,
                    label,
                    self.ctc_weight * self.interctc_shared_weight,
                )
            ]
        else:
            total = text_loss
            tasks = [("mid_text", self.mid_text_ctc, label, self.interctc_text_weight)]
        if self.mid_toneless_ctc is not None:
            if toneless_label is None:
                raise ValueError("Missing unified toneless-syllable targets for C2b")
            tasks.append(
                (
                    "mid_toneless",
                    self.mid_toneless_ctc,
                    toneless_label,
                    self.interctc_toneless_weight,
                )
            )
        for name, head, targets, weight in tasks:
            stats = self._auxiliary_loss(
                head, middle, lengths, targets, weight, compute_aux_metrics
            )
            total = total + stats["weighted"]
            output.update(
                {
                    f"raw_{name}_loss": stats["raw"],
                    f"weighted_{name}_loss": stats["weighted"],
                    f"{name}_to_text_ratio": stats["weighted"].detach()
                    / text_loss.detach().clamp_min(1e-8),
                    f"{name}_feasible": stats["feasible"],
                    f"{name}_infeasible": stats["infeasible"],
                    f"{name}_errors": stats["errors"],
                    f"{name}_units": stats["units"],
                    f"{name}_blank_frames": stats["blank_frames"],
                    f"{name}_frames": stats["frames"],
                }
            )
        output["loss"] = total
        return output
