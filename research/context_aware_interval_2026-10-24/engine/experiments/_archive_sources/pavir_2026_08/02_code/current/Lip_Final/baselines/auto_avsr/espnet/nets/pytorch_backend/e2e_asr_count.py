"""Native Auto-AVSR plus a clip-level syllable-count distribution head."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class CountE2E(E2E):
    """Preserve the native decoder and supervise only a global count head."""

    def __init__(
        self,
        odim: int,
        modality: str,
        ctc_weight: float,
        count_bins: int = 33,
        count_hidden_dim: int = 256,
        count_dropout: float = 0.1,
        count_weight: float = 0.1,
    ):
        super().__init__(odim, modality, ctc_weight=ctc_weight)
        if count_bins < 2:
            raise ValueError("count_bins must be at least 2")
        if count_weight <= 0:
            raise ValueError("count_weight must be positive")
        self.count_bins = int(count_bins)
        self.count_weight = float(count_weight)
        self.count_head = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, count_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(count_dropout),
            torch.nn.Linear(count_hidden_dim, count_bins),
        )

    @staticmethod
    def _pool(encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(encoded.size(1), device=encoded.device).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)
        return (encoded * mask.unsqueeze(-1)).sum(1) / lengths.clamp_min(1).unsqueeze(1)

    def encode_count(self, x: torch.Tensor, mask: torch.Tensor | None):
        encoded, mask = self.encoder(x, mask)
        if mask is None:
            lengths = torch.full((encoded.size(0),), encoded.size(1), device=encoded.device, dtype=torch.long)
        else:
            lengths = mask.squeeze(1).sum(-1).long()
        logits = self.count_head(self._pool(encoded, lengths))
        return encoded, logits, mask

    def forward(self, x, lengths, label, syllable_label=None):
        if syllable_label is None:
            raise ValueError("Count supervision requires existing 6-field ViSyll targets")
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        encoded = self.proj_encoder(self.frontend(x))
        encoded, count_logits, padding_mask = self.encode_count(encoded, padding_mask)
        text_ctc, _ = self.ctc(encoded, lengths, label)
        text_in, text_out = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        text_logits, _ = self.decoder(text_in, target_mask(text_in, self.ignore_id), encoded, padding_mask)
        text_att = self.criterion(text_logits, text_out)
        text_loss = self.ctc_weight * text_ctc + (1 - self.ctc_weight) * text_att
        count_target = (syllable_label != self.ignore_id).reshape(syllable_label.size(0), -1).sum(1).long()
        if bool((count_target >= self.count_bins).any()):
            raise ValueError("Observed syllable count exceeds locked count_bins")
        count_nll = F.cross_entropy(count_logits, count_target)
        count_prediction = count_logits.argmax(-1)
        weighted_count_loss = self.count_weight * count_nll
        return {
            "loss": text_loss + weighted_count_loss,
            "text_loss": text_loss,
            "loss_ctc": text_ctc,
            "loss_att": text_att,
            "decoder_acc": th_accuracy(text_logits.view(-1, self.odim), text_out, ignore_label=self.ignore_id),
            "count_nll": count_nll,
            "weighted_count_loss": weighted_count_loss,
            "count_mae": (count_prediction - count_target).abs().float().mean(),
            "count_exact": (count_prediction == count_target).float().mean(),
        }
