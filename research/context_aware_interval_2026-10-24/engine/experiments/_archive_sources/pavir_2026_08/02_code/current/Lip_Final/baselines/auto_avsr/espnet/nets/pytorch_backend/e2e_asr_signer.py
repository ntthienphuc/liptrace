"""Signer-adversarial Auto-AVSR and content-preserving CP-SAR.

The signer classifier is attached to the raw output of Conformer block 6 and
is discarded by the native inference path.  CP-SAR additionally preserves the
teacher-forced decoder distribution of a frozen native B0 teacher.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


def gradient_reverse(value: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(value, scale)


class SignerInvariantE2E(E2E):
    """Matched S3a/S3b model; native decoding is unchanged."""

    def __init__(
        self,
        odim: int,
        modality: str,
        ctc_weight: float = 0.1,
        signer_classes: int = 11,
        signer_layer: int = 6,
        signer_hidden_dim: int = 256,
        signer_dropout: float = 0.1,
        kd_weight: float = 0.0,
        teacher_checkpoint: str | None = None,
    ):
        super().__init__(odim, modality, ctc_weight=ctc_weight)
        if signer_layer != 6:
            raise ValueError("The locked S3 protocol requires the signer head at block 6")
        if signer_classes < 2:
            raise ValueError("Signer adversarial training requires at least two train signers")
        if kd_weight < 0:
            raise ValueError("KD weight cannot be negative")
        if kd_weight > 0 and not teacher_checkpoint:
            raise ValueError("CP-SAR requires a frozen teacher checkpoint")
        self.signer_layer = int(signer_layer)
        self.signer_classes = int(signer_classes)
        self.kd_weight = float(kd_weight)
        self.signer_head = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, signer_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(signer_dropout),
            torch.nn.Linear(signer_hidden_dim, signer_classes),
        )
        # Deliberately bypass nn.Module registration: the frozen teacher must
        # never enter the optimizer or student checkpoints.
        object.__setattr__(self, "_frozen_teacher", None)
        self.teacher_checkpoint = teacher_checkpoint
        if self.kd_weight > 0:
            object.__setattr__(self, "_frozen_teacher", self._load_teacher(teacher_checkpoint))

    def _load_teacher(self, checkpoint_path: str) -> E2E:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.") and not key.startswith("model.signer_head.")
        }
        # Teacher construction must not advance the student/data RNG stream;
        # otherwise S3b would silently receive a different augmentation/order
        # sequence from S3a despite the same nominal seed.
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        teacher = E2E(self.odim, self.modality, ctc_weight=self.ctc_weight)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        teacher.load_state_dict(state, strict=True)
        teacher.requires_grad_(False)
        return teacher.eval()

    def _teacher(self, device: torch.device) -> E2E:
        teacher = self._frozen_teacher
        if teacher is None:
            raise RuntimeError("Frozen CP-SAR teacher is unavailable")
        first = next(teacher.parameters())
        if first.device != device:
            teacher.to(device)
        teacher.eval()
        return teacher

    def encode_with_middle(self, x: torch.Tensor, mask: torch.Tensor | None):
        xs = self.encoder.embed(x)
        middle = None
        for index, layer in enumerate(self.encoder.encoders):
            xs, mask = layer(xs, mask)
            if index + 1 == self.signer_layer:
                middle = xs[0] if isinstance(xs, tuple) else xs
        final = xs[0] if isinstance(xs, tuple) else xs
        if self.encoder.normalize_before:
            final = self.encoder.after_norm(final)
        if middle is None:
            raise RuntimeError("Signer attachment block was not reached")
        return final, middle, mask

    @staticmethod
    def masked_mean(hidden: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if padding_mask is None:
            return hidden.mean(dim=1)
        valid = padding_mask.squeeze(1).to(dtype=hidden.dtype)
        return (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.0)

    def _teacher_logits(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_mask: torch.Tensor,
    ) -> torch.Tensor:
        teacher = self._teacher(x.device)
        with torch.no_grad():
            teacher_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
            encoded = teacher.proj_encoder(teacher.frontend(x))
            encoded, teacher_mask = teacher.encoder(encoded, teacher_mask)
            logits, _ = teacher.decoder(ys_in_pad, ys_mask, encoded, teacher_mask)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        label: torch.Tensor,
        signer_target: torch.Tensor | None,
        grl_scale: float = 1.0,
    ) -> dict[str, torch.Tensor | float]:
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        encoded = self.proj_encoder(self.frontend(x))
        final, middle, padding_mask = self.encode_with_middle(encoded, padding_mask)

        loss_ctc, _ = self.ctc(final, lengths, label)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, final, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        text_loss = self.ctc_weight * loss_ctc + (1.0 - self.ctc_weight) * loss_att

        pooled = self.masked_mean(middle, padding_mask)
        signer_logits = self.signer_head(gradient_reverse(pooled, grl_scale))
        if signer_target is None:
            raise ValueError("S3 training/evaluation requires signer targets")
        signer_target = signer_target.reshape(-1).long()
        known = signer_target >= 0
        if known.any():
            signer_loss = F.cross_entropy(signer_logits[known], signer_target[known])
            signer_acc = (signer_logits[known].argmax(-1) == signer_target[known]).float().mean()
        else:
            signer_loss = signer_logits.sum() * 0.0
            signer_acc = signer_loss.detach()

        kd_loss = pred_pad.sum() * 0.0
        if self.kd_weight > 0:
            teacher_logits = self._teacher_logits(x, lengths, ys_in_pad, ys_mask)
            token_mask = ys_out_pad.ne(self.ignore_id)
            per_token = F.kl_div(
                F.log_softmax(pred_pad.float(), dim=-1),
                F.softmax(teacher_logits.float(), dim=-1),
                reduction="none",
            ).sum(-1)
            kd_loss = (per_token * token_mask).sum() / pred_pad.shape[0]

        # Canonical GRL: the classifier receives full CE gradient while the
        # encoder receives its sign-reversed, grl_scale-scaled gradient.
        loss = text_loss + signer_loss + self.kd_weight * kd_loss
        return {
            "loss": loss,
            "text_loss": text_loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att,
            "decoder_acc": th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            ),
            "signer_loss": signer_loss,
            "signer_acc": signer_acc,
            "known_signers": known.sum(),
            "grl_scale": float(grl_scale),
            "raw_kd_loss": kd_loss,
            "weighted_kd_loss": self.kd_weight * kd_loss,
        }
