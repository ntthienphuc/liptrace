"""Auto-AVSR with optional onset/rime CTC supervision at block 6 or final block 12."""
from __future__ import annotations

import torch

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask


def ctc_feasible_mask(targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    answer = []
    for target, frames in zip(targets, lengths):
        ids = target[target != -1]
        repeats = int((ids[1:] == ids[:-1]).sum()) if len(ids) > 1 else 0
        answer.append(int(frames) >= len(ids) + repeats)
    return torch.tensor(answer, dtype=torch.bool, device=targets.device)


def collapse_ctc(sequence: list[int], blank: int = 0) -> list[int]:
    """Collapse a greedy CTC path, preserving repeats separated by blanks."""
    answer = []
    previous = None
    for value in sequence:
        if value != previous and value != blank:
            answer.append(value)
        previous = value
    return answer


def collapse_ctc_with_confidence(sequence: list[int], confidence: list[float], blank: int = 0) -> tuple[list[int], list[float]]:
    """Collapse a greedy path and retain max confidence for each token run."""
    if len(sequence) != len(confidence):
        raise ValueError("sequence and confidence must have equal length")
    ids, scores = [], []
    previous = None
    for value, score in zip(sequence, confidence):
        if value != previous:
            if value != blank:
                ids.append(value); scores.append(float(score))
        elif value != blank and scores:
            scores[-1] = max(scores[-1], float(score))
        previous = value
    return ids, scores


def ctc_error_stats(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor, feasible: torch.Tensor) -> tuple[int, int, int, int]:
    """Greedy CTC edit counts and blank frames for descriptive monitoring."""
    prediction = logits.detach().argmax(-1).transpose(0, 1).cpu()  # B,T
    targets = targets.detach().cpu(); feasible_cpu = feasible.cpu()
    errors = total = blank = frames = 0
    for pred, target, valid_length, keep in zip(prediction, targets, lengths.detach().cpu(), feasible_cpu):
        if not bool(keep): continue
        pred = pred[:int(valid_length)]
        # Standard CTC greedy collapse: merge consecutive repeats first, then
        # remove blanks. A blank therefore separates two identical symbols.
        collapsed = collapse_ctc(pred.tolist())
        reference = target[target != -1].tolist()
        # Small exact Levenshtein DP; used only for epoch-level monitoring.
        previous = list(range(len(reference) + 1))
        for i, value in enumerate(collapsed, 1):
            current = [i]
            for j, other in enumerate(reference, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (value != other)))
            previous = current
        errors += previous[-1]; total += len(reference); blank += int((pred == 0).sum()); frames += len(pred)
    return errors, total, blank, frames


class StructuredE2E(E2E):
    def __init__(self, odim, modality, ctc_weight=0.1, onset_vocab_size=0, rime_vocab_size=0, aux_layer=6, onset_weight=0.0, rime_weight=0.0):
        super().__init__(odim, modality, ctc_weight=ctc_weight)
        if aux_layer not in (6, 12): raise ValueError("aux_layer must be 6 or 12")
        self.aux_layer = aux_layer; self.onset_weight = onset_weight; self.rime_weight = rime_weight
        self.onset_ctc = CTC(onset_vocab_size, 768, 0.1, reduce=True) if onset_weight > 0 else None
        self.rime_ctc = CTC(rime_vocab_size, 768, 0.1, reduce=True) if rime_weight > 0 else None

    def encode_structured(self, x, mask):
        xs = self.encoder.embed(x); selected = None
        for index, layer in enumerate(self.encoder.encoders):
            xs, mask = layer(xs, mask)
            if index + 1 == self.aux_layer:
                selected = xs[0] if isinstance(xs, tuple) else xs
        final = xs[0] if isinstance(xs, tuple) else xs
        if self.encoder.normalize_before: final = self.encoder.after_norm(final)
        return final, (final if self.aux_layer == 12 else selected), mask

    def forward(self, x, lengths, label, onset_label=None, rime_label=None, compute_aux_metrics=False):
        if self.modality == "audio": lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        x = self.proj_encoder(self.frontend(x)); final, auxiliary, padding_mask = self.encode_structured(x, padding_mask)
        loss_ctc, _ = self.ctc(final, lengths, label)
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, target_mask(ys_in_pad, self.ignore_id), final, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        text_loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        total_loss = text_loss; output = {"text_loss": text_loss, "loss_ctc": loss_ctc, "loss_att": loss_att, "decoder_acc": th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)}
        for name, head, target, weight in (("onset", self.onset_ctc, onset_label, self.onset_weight), ("rime", self.rime_ctc, rime_label, self.rime_weight)):
            if head is None: continue
            if target is None: raise ValueError(f"Missing {name} targets")
            feasible = ctc_feasible_mask(target, lengths); count = int(feasible.sum())
            if count:
                selected_target = target[feasible]
                raw, logits = head(auxiliary[feasible], lengths[feasible], selected_target)
                mean_target_units = (selected_target != -1).sum().float() / count
                normalized = raw / mean_target_units.clamp_min(1.0)
                weighted = weight * normalized; total_loss = total_loss + weighted
                if compute_aux_metrics:
                    errors, units, blanks, frames = ctc_error_stats(logits, target[feasible], lengths[feasible], feasible[feasible])
                else:
                    errors = units = blanks = frames = 0
            else:
                raw = auxiliary.sum() * 0; normalized = raw; weighted = raw; errors = units = blanks = frames = 0
            output.update({f"raw_{name}_loss": raw, f"normalized_{name}_loss": normalized, f"weighted_{name}_loss": weighted, f"{name}_to_text_ratio": weighted.detach() / text_loss.detach().clamp_min(1e-8), f"{name}_feasible": count, f"{name}_infeasible": len(feasible) - count, f"{name}_errors": errors, f"{name}_units": units, f"{name}_blank_frames": blanks, f"{name}_frames": frames})
        output["loss"] = total_loss
        return output
