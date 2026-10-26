"""Multi-lag difference bridge for frozen Auto-AVSR adaptation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class MDBridgeE2E(E2E):
    def __init__(
        self,
        odim,
        modality,
        ctc_weight=0.1,
        middle_layer=6,
        bridge_rank=192,
        mode="difference",
        lags=(1, 2, 4),
    ):
        super().__init__(odim, modality, ctc_weight=ctc_weight)
        if mode not in {"pointwise", "difference"}:
            raise ValueError(f"Unsupported MD-Bridge mode: {mode}")
        self.md_middle_layer = int(middle_layer)
        self.md_mode = mode
        self.md_lags = tuple(int(value) for value in lags)
        self.md_disable_fusion = False

        self.md_norm = torch.nn.LayerNorm(768)
        self.md_down = torch.nn.Linear(768, bridge_rank)
        self.md_branches = torch.nn.ModuleList(
            torch.nn.Linear(bridge_rank, bridge_rank) for _ in self.md_lags
        )
        self.md_up = torch.nn.Linear(bridge_rank, 768)
        gate_hidden = max(bridge_rank // 2, 32)
        self.md_gate = torch.nn.Sequential(
            torch.nn.Linear(bridge_rank + len(self.md_lags) + 1, gate_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(gate_hidden, 1),
        )
        self.md_scale = torch.nn.Parameter(torch.zeros(()))
        torch.nn.init.zeros_(self.md_gate[-1].weight)
        torch.nn.init.zeros_(self.md_gate[-1].bias)

    @staticmethod
    def _lag_difference(hidden, lag):
        shifted = torch.zeros_like(hidden)
        shifted[:, lag:] = hidden[:, :-lag]
        return hidden - shifted

    def _encode(self, x, padding_mask):
        captured = {}

        def capture_middle(_module, _inputs, output):
            hidden = output[0]
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            captured["middle"] = hidden

        layer = self.encoder.encoders[self.md_middle_layer - 1]
        handle = layer.register_forward_hook(capture_middle)
        try:
            final, _ = self.encoder(x, padding_mask)
        finally:
            handle.remove()

        middle = self.md_norm(captured["middle"])
        hidden = F.gelu(self.md_down(middle))
        branch_inputs = []
        for lag in self.md_lags:
            branch_inputs.append(
                self._lag_difference(hidden, lag) if self.md_mode == "difference" else hidden
            )
        branch_outputs = [
            F.gelu(projection(value))
            for projection, value in zip(self.md_branches, branch_inputs)
        ]
        aggregate = torch.stack(branch_outputs, dim=0).mean(dim=0)
        residual = self.md_up(aggregate)
        magnitudes = torch.cat(
            [value.abs().mean(dim=-1, keepdim=True) for value in branch_inputs], dim=-1
        )
        disagreement = (middle - final).abs().mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.md_gate(torch.cat((hidden, magnitudes, disagreement), dim=-1)))
        alpha = torch.tanh(self.md_scale) * gate
        if self.md_disable_fusion:
            alpha = torch.zeros_like(alpha)
        fused = final + alpha * residual
        return fused, middle, final, residual, alpha, magnitudes

    def forward(self, x, lengths, label):
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        x = self.frontend(x)
        x = self.proj_encoder(x)
        fused, _middle, _final, residual, alpha, magnitudes = self._encode(x, padding_mask)

        loss_ctc, _ = self.ctc(fused, lengths, label)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, fused, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        text_loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)
        return {
            "loss": text_loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att,
            "decoder_acc": acc,
            "text_loss": text_loss,
            "md_alpha_mean": alpha.abs().mean(),
            "md_scale": torch.tanh(self.md_scale).abs(),
            "md_residual_l2": residual.float().norm(dim=-1).mean(),
            "md_branch_magnitude": magnitudes.float().mean(),
        }
