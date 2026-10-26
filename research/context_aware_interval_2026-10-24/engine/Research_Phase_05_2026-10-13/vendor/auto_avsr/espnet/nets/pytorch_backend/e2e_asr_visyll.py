"""Auto-AVSR with an aligned unified toneless-syllable CTC+CE decoder."""
from __future__ import annotations

import torch

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.e2e_asr_structured import ctc_error_stats, ctc_feasible_mask
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class ViSyllE2E(E2E):
    """Keep the text decoder intact and add one synchronized syllable path.

    Unlike the earlier independent onset/rime CTC proxy, every target step is a
    complete tone-free ``initial|rhyme`` unit and the same target is optimized
    with both monotonic CTC and autoregressive cross-entropy.
    """

    def __init__(
        self,
        odim: int,
        modality: str,
        ctc_weight: float,
        syllable_vocab_size: int,
        syllable_weight: float = 0.2,
        syllable_ctc_weight: float = 0.1,
        syllable_decoder_blocks: int = 3,
        syllable_mode: str = "unified",
        unit_to_initial: list[int] | None = None,
        unit_to_rhyme: list[int] | None = None,
        initial_vocab_size: int = 0,
        rhyme_vocab_size: int = 0,
    ):
        super().__init__(odim, modality, ctc_weight=ctc_weight)
        if syllable_vocab_size < 3:
            raise ValueError("syllable vocabulary is too small")
        if not 0 <= syllable_ctc_weight <= 1:
            raise ValueError("syllable_ctc_weight must be in [0,1]")
        self.syllable_weight = syllable_weight
        self.syllable_ctc_weight = syllable_ctc_weight
        self.syllable_odim = syllable_vocab_size
        self.syllable_mode = syllable_mode
        self.syllable_sos = syllable_vocab_size - 1
        self.syllable_eos = syllable_vocab_size - 1
        self.syllable_ctc = CTC(syllable_vocab_size, 768, 0.1, reduce=True)
        if syllable_mode == "unified":
            self.syllable_decoder = TransformerDecoder(
                odim=syllable_vocab_size, attention_dim=768, attention_heads=12,
                linear_units=3072, num_blocks=syllable_decoder_blocks,
            )
            self.syllable_criterion = LabelSmoothingLoss(syllable_vocab_size, self.ignore_id, 0.1, False)
        elif syllable_mode == "rhyme_first":
            if not unit_to_initial or not unit_to_rhyme:
                raise ValueError("rhyme_first mode requires unit-component mappings")
            if len(unit_to_initial) != syllable_vocab_size or len(unit_to_rhyme) != syllable_vocab_size:
                raise ValueError("unit-component mapping length mismatch")
            self.syllable_decoder = TransformerDecoder(
                odim=syllable_vocab_size, attention_dim=768, attention_heads=12,
                linear_units=3072, num_blocks=syllable_decoder_blocks,
                use_output_layer=False,
            )
            self.rhyme_head = torch.nn.Linear(768, rhyme_vocab_size)
            self.rhyme_condition = torch.nn.Embedding(rhyme_vocab_size, 768)
            self.initial_head = torch.nn.Sequential(
                torch.nn.LayerNorm(768), torch.nn.Linear(768, initial_vocab_size)
            )
            self.rhyme_criterion = LabelSmoothingLoss(rhyme_vocab_size, self.ignore_id, 0.1, False)
            self.initial_criterion = LabelSmoothingLoss(initial_vocab_size, self.ignore_id, 0.1, False)
            self.register_buffer("unit_to_initial", torch.tensor(unit_to_initial, dtype=torch.long), persistent=False)
            self.register_buffer("unit_to_rhyme", torch.tensor(unit_to_rhyme, dtype=torch.long), persistent=False)
        else:
            raise ValueError(f"Unknown syllable_mode: {syllable_mode}")

    def encode_visyll(self, x: torch.Tensor, mask: torch.Tensor | None):
        encoded, mask = self.encoder(x, mask)
        return encoded, mask

    def forward(self, x, lengths, label, syllable_label=None, compute_aux_metrics=False):
        if syllable_label is None:
            raise ValueError("Missing unified syllable targets")
        if self.modality == "audio":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        encoded = self.proj_encoder(self.frontend(x))
        encoded, padding_mask = self.encode_visyll(encoded, padding_mask)

        text_ctc, _ = self.ctc(encoded, lengths, label)
        text_in, text_out = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        text_logits, _ = self.decoder(text_in, target_mask(text_in, self.ignore_id), encoded, padding_mask)
        text_att = self.criterion(text_logits, text_out)
        text_loss = self.ctc_weight * text_ctc + (1 - self.ctc_weight) * text_att

        feasible = ctc_feasible_mask(syllable_label, lengths)
        feasible_count = int(feasible.sum())
        if feasible_count:
            syllable_ctc, syllable_logits_ctc = self.syllable_ctc(encoded[feasible], lengths[feasible], syllable_label[feasible])
            if compute_aux_metrics:
                errors, units, blanks, frames = ctc_error_stats(
                    syllable_logits_ctc,
                    syllable_label[feasible],
                    lengths[feasible],
                    feasible[feasible],
                )
            else:
                errors = units = blanks = frames = 0
        else:
            syllable_ctc = encoded.sum() * 0
            errors = units = blanks = frames = 0

        syllable_in, syllable_out = add_sos_eos(
            syllable_label, self.syllable_sos, self.syllable_eos, self.ignore_id
        )
        syllable_hidden, _ = self.syllable_decoder(
            syllable_in, target_mask(syllable_in, self.ignore_id), encoded, padding_mask,
        )
        component_metrics = {}
        if self.syllable_mode == "unified":
            syllable_logits_att = syllable_hidden
            syllable_att = self.syllable_criterion(syllable_logits_att, syllable_out)
            syllable_accuracy = th_accuracy(
                syllable_logits_att.view(-1, self.syllable_odim), syllable_out, ignore_label=self.ignore_id
            )
        else:
            safe = syllable_out.masked_fill(syllable_out == self.ignore_id, 0)
            rhyme_target = self.unit_to_rhyme[safe].masked_fill(syllable_out == self.ignore_id, self.ignore_id)
            initial_target = self.unit_to_initial[safe].masked_fill(syllable_out == self.ignore_id, self.ignore_id)
            rhyme_logits = self.rhyme_head(syllable_hidden)
            rhyme_for_condition = rhyme_target.masked_fill(rhyme_target == self.ignore_id, 0)
            initial_logits = self.initial_head(syllable_hidden + self.rhyme_condition(rhyme_for_condition))
            rhyme_loss = self.rhyme_criterion(rhyme_logits, rhyme_target)
            initial_loss = self.initial_criterion(initial_logits, initial_target)
            syllable_att = 0.5 * (rhyme_loss + initial_loss)
            valid = syllable_out != self.ignore_id
            rhyme_correct = rhyme_logits.argmax(-1) == rhyme_target
            initial_correct = initial_logits.argmax(-1) == initial_target
            denominator = valid.sum().clamp_min(1)
            syllable_accuracy = ((rhyme_correct & initial_correct) & valid).sum().float() / denominator
            component_metrics = {
                "rhyme_decoder_acc": (rhyme_correct & valid).sum().float() / denominator,
                "initial_decoder_acc": (initial_correct & valid).sum().float() / denominator,
                "rhyme_att_loss": rhyme_loss,
                "initial_att_loss": initial_loss,
            }
        syllable_loss = self.syllable_ctc_weight * syllable_ctc + (1 - self.syllable_ctc_weight) * syllable_att
        weighted_syllable_loss = self.syllable_weight * syllable_loss
        total = text_loss + weighted_syllable_loss
        output = {
            "loss": total,
            "text_loss": text_loss,
            "loss_ctc": text_ctc,
            "loss_att": text_att,
            "decoder_acc": th_accuracy(text_logits.view(-1, self.odim), text_out, ignore_label=self.ignore_id),
            "raw_syllable_loss": syllable_loss,
            "syllable_ctc_loss": syllable_ctc,
            "syllable_att_loss": syllable_att,
            "weighted_syllable_loss": weighted_syllable_loss,
            "syllable_to_text_ratio": weighted_syllable_loss.detach() / text_loss.detach().clamp_min(1e-8),
            "syllable_decoder_acc": syllable_accuracy,
            "syllable_feasible": feasible_count,
            "syllable_infeasible": len(feasible) - feasible_count,
            "syllable_errors": errors,
            "syllable_units": units,
            "syllable_blank_frames": blanks,
            "syllable_frames": frames,
        }
        output.update(component_metrics)
        return output
