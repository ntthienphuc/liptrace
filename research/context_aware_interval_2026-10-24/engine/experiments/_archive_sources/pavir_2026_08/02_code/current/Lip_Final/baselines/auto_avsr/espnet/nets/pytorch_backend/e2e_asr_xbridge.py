"""Middle-final predictive bridge for the frozen Auto-AVSR architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class XBridgeE2E(E2E):
    def __init__(
        self,
        odim,
        modality,
        ctc_weight=0.1,
        middle_layer=6,
        bridge_rank=192,
        self_weight=0.2,
        cross_weight=0.0,
        av_dim=1024,
        ctc_only=False,
    ):
        super().__init__(odim, modality, ctc_weight=ctc_weight)
        self.xbridge_middle_layer = middle_layer
        self.xbridge_self_weight = float(self_weight)
        self.xbridge_cross_weight = float(cross_weight)
        self.xbridge_ctc_only = bool(ctc_only)
        self.xbridge_disable_fusion = False
        self.bridge_norm = torch.nn.LayerNorm(768)
        self.bridge_down = torch.nn.Linear(768, bridge_rank)
        self.bridge_up = torch.nn.Linear(bridge_rank, 768)
        self.bridge_gate = torch.nn.Sequential(
            torch.nn.Linear(768 * 3, bridge_rank),
            torch.nn.GELU(),
            torch.nn.Linear(bridge_rank, 1),
        )
        self.bridge_scale = torch.nn.Parameter(torch.zeros(()))
        self.bridge_av = torch.nn.Linear(bridge_rank, av_dim) if cross_weight > 0 else None

        torch.nn.init.zeros_(self.bridge_gate[-1].weight)
        torch.nn.init.zeros_(self.bridge_gate[-1].bias)

    def _encode(self, x, padding_mask):
        captured = {}

        def capture_middle(_module, _inputs, output):
            hidden = output[0]
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            captured["middle"] = hidden

        layer = self.encoder.encoders[self.xbridge_middle_layer - 1]
        handle = layer.register_forward_hook(capture_middle)
        try:
            final, _ = self.encoder(x, padding_mask)
        finally:
            handle.remove()

        middle = self.bridge_norm(captured["middle"])
        hidden = F.gelu(self.bridge_down(middle))
        candidate = middle + self.bridge_up(hidden)
        gate_input = torch.cat((middle, final, (middle - final).abs()), dim=-1)
        content_gate = torch.sigmoid(self.bridge_gate(gate_input))
        alpha = torch.tanh(self.bridge_scale) * content_gate
        if self.xbridge_disable_fusion:
            alpha = torch.zeros_like(alpha)
        fused = final + alpha * (candidate - final)
        return fused, middle, candidate, hidden, alpha

    @staticmethod
    def _masked_cosine(left, right, lengths):
        steps = left.size(1)
        mask = torch.arange(steps, device=left.device)[None, :] < lengths[:, None]
        loss = 1.0 - F.cosine_similarity(left.float(), right.float(), dim=-1)
        return loss.masked_select(mask).mean()

    def forward(self, x, lengths, label, av_teacher=None):
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        x = self.frontend(x)
        x = self.proj_encoder(x)
        fused, middle, candidate, hidden, alpha = self._encode(x, padding_mask)

        loss_ctc, _ = self.ctc(fused, lengths, label)
        if self.xbridge_ctc_only:
            loss_att = loss_ctc.new_zeros(())
            text_loss = loss_ctc
            acc = loss_ctc.new_zeros(())
        else:
            ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, fused, padding_mask)
            loss_att = self.criterion(pred_pad, ys_out_pad)
            text_loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
            acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)

        self_loss = self._masked_cosine(candidate, fused.detach(), lengths)
        cross_loss = fused.new_zeros(())
        if self.bridge_av is not None and av_teacher is not None:
            teacher = av_teacher.to(device=fused.device, dtype=fused.dtype)
            if teacher.size(1) != hidden.size(1):
                teacher = F.interpolate(
                    teacher.transpose(1, 2), size=hidden.size(1), mode="linear", align_corners=False
                ).transpose(1, 2)
            cross_loss = self._masked_cosine(self.bridge_av(hidden), teacher.detach(), lengths)

        loss = text_loss + self.xbridge_self_weight * self_loss + self.xbridge_cross_weight * cross_loss
        return {
            "loss": loss,
            "loss_ctc": loss_ctc,
            "loss_att": loss_att,
            "decoder_acc": acc,
            "text_loss": text_loss,
            "xbridge_self_loss": self_loss,
            "xbridge_cross_loss": cross_loss,
            "xbridge_alpha_mean": alpha.abs().mean(),
            "xbridge_scale": torch.tanh(self.bridge_scale).abs(),
        }
