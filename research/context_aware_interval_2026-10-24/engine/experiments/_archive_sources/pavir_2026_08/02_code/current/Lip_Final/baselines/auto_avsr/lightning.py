import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule


def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)


def ctc_uncertainty_stats(log_prob):
    """Summarize frame posteriors without retaining large T x V tensors."""
    probabilities = log_prob.float().exp()
    vocab_size = probabilities.size(-1)
    entropy = -(probabilities * log_prob.float()).sum(-1) / torch.log(
        torch.tensor(float(vocab_size), device=probabilities.device)
    )
    top_prob, top_id = probabilities.max(-1)
    top_two = probabilities.topk(k=2, dim=-1).values
    margin = top_two[:, 0] - top_two[:, 1]
    blank_prob = probabilities[:, 0]
    nonblank = top_id.ne(0)
    transitions = top_id[1:].ne(top_id[:-1]) if len(top_id) > 1 else top_id.new_zeros(0, dtype=torch.bool)

    def moments(prefix, values):
        values = values.float()
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std(unbiased=False)),
            f"{prefix}_p10": float(torch.quantile(values, 0.10)),
            f"{prefix}_p50": float(torch.quantile(values, 0.50)),
            f"{prefix}_p90": float(torch.quantile(values, 0.90)),
        }

    collapsed = []
    previous = None
    for token_id in top_id.tolist():
        if token_id != 0 and token_id != previous:
            collapsed.append(token_id)
        previous = token_id
    stats = {
        "ctc_frame_count": int(log_prob.size(0)),
        "ctc_vocab_size": int(vocab_size),
        "ctc_blank_argmax_ratio": float(top_id.eq(0).float().mean()),
        "ctc_spike_ratio": float(nonblank.float().mean()),
        "ctc_transition_ratio": float(transitions.float().mean()) if len(transitions) else 0.0,
        "ctc_collapsed_token_count": len(collapsed),
        "ctc_collapsed_token_ratio": len(collapsed) / max(int(log_prob.size(0)), 1),
        "ctc_nonblank_mass_mean": float((1.0 - blank_prob).mean()),
    }
    stats.update(moments("ctc_entropy", entropy))
    stats.update(moments("ctc_top1_prob", top_prob))
    stats.update(moments("ctc_margin", margin))
    stats.update(moments("ctc_blank_prob", blank_prob))
    if nonblank.any():
        spike_confidence = top_prob[nonblank]
        stats.update(moments("ctc_spike_conf", spike_confidence))
    else:
        stats.update({f"ctc_spike_conf_{name}": 0.0 for name in ("mean", "std", "p10", "p50", "p90")})
    return stats


def _ctc_collapse(frame_ids):
    collapsed = []
    previous = None
    for token_id in frame_ids.tolist():
        if token_id != 0 and token_id != previous:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def ctc_stability_stats(reference_log_prob, variants):
    reference_prob = reference_log_prob.float().exp()
    reference_ids = reference_prob.argmax(-1)
    reference_tokens = _ctc_collapse(reference_ids)
    output = {}
    for name, variant_log_prob in variants.items():
        variant_prob = variant_log_prob.float().exp()
        variant_ids = variant_prob.argmax(-1)
        mixture = 0.5 * (reference_prob + variant_prob)
        js = 0.5 * (
            (reference_prob * (reference_log_prob.float() - mixture.clamp_min(1e-12).log())).sum(-1)
            + (variant_prob * (variant_log_prob.float() - mixture.clamp_min(1e-12).log())).sum(-1)
        )
        variant_tokens = _ctc_collapse(variant_ids)
        token_distance = torchaudio.functional.edit_distance(reference_tokens, variant_tokens)
        output.update(
            {
                f"stability_{name}_js_mean": float(js.mean()),
                f"stability_{name}_js_p90": float(torch.quantile(js, 0.90)),
                f"stability_{name}_posterior_l1_mean": float((reference_prob - variant_prob).abs().sum(-1).mean()),
                f"stability_{name}_top1_agreement": float(reference_ids.eq(variant_ids).float().mean()),
                f"stability_{name}_blank_agreement": float(reference_ids.eq(0).eq(variant_ids.eq(0)).float().mean()),
                f"stability_{name}_token_edit_ratio": token_distance / max(len(reference_tokens), 1),
                f"stability_{name}_token_count_delta": (len(variant_tokens) - len(reference_tokens)) / max(len(reference_tokens), 1),
            }
        )
    return output


class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._pavir_context_eligible_batches_seen = 0
        self.save_hyperparameters(args)

        self.modality = args.modality
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        onset_weight, rime_weight = getattr(args, "onset_weight", 0.0), getattr(args, "rime_weight", 0.0)
        syllable_weight = getattr(args, "syllable_weight", 0.0)
        count_weight = getattr(args, "count_weight", 0.0)
        interctc_text_weight = getattr(args, "interctc_text_weight", 0.0)
        interctc_toneless_weight = getattr(args, "interctc_toneless_weight", 0.0)
        interctc_shared_weight = getattr(args, "interctc_shared_weight", 0.0)
        signer_adv_weight = getattr(args, "signer_adv_weight", 0.0)
        signer_kd_weight = getattr(args, "signer_kd_weight", 0.0)
        xbridge_self_weight = getattr(args, "xbridge_self_weight", 0.0)
        xbridge_cross_weight = getattr(args, "xbridge_cross_weight", 0.0)
        mdbridge_mode = getattr(args, "mdbridge_mode", "none")
        pavir_config = getattr(args, "pavir_config", None)
        if pavir_config:
            incompatible = {
                "onset_weight": onset_weight,
                "rime_weight": rime_weight,
                "syllable_weight": syllable_weight,
                "count_weight": count_weight,
                "interctc_text_weight": interctc_text_weight,
                "interctc_toneless_weight": interctc_toneless_weight,
                "interctc_shared_weight": interctc_shared_weight,
                "signer_adv_weight": signer_adv_weight,
                "signer_kd_weight": signer_kd_weight,
                "xbridge_self_weight": xbridge_self_weight,
                "xbridge_cross_weight": xbridge_cross_weight,
            }
            active = sorted(name for name, value in incompatible.items() if value)
            if mdbridge_mode != "none":
                active.append("mdbridge_mode")
            if active:
                raise ValueError(
                    "PAViR matched arms cannot combine historical auxiliaries: "
                    + ", ".join(active)
                )
            from espnet.nets.pytorch_backend.e2e_asr_pavir import PAViRE2E

            self.model = PAViRE2E(
                len(self.token_list),
                self.modality,
                ctc_weight=getattr(args, "ctc_weight", 0.1),
                pavir_config=pavir_config,
            )
        elif mdbridge_mode != "none":
            from espnet.nets.pytorch_backend.e2e_asr_mdbridge import MDBridgeE2E
            self.model = MDBridgeE2E(
                len(self.token_list), self.modality,
                ctc_weight=getattr(args, "ctc_weight", 0.1),
                middle_layer=getattr(args, "mdbridge_middle_layer", 6),
                bridge_rank=getattr(args, "mdbridge_rank", 192),
                mode=mdbridge_mode,
                lags=tuple(int(value) for value in getattr(args, "mdbridge_lags", "1,2,4").split(",")),
            )
        elif xbridge_self_weight > 0 or xbridge_cross_weight > 0:
            from espnet.nets.pytorch_backend.e2e_asr_xbridge import XBridgeE2E
            self.model = XBridgeE2E(
                len(self.token_list), self.modality,
                ctc_weight=getattr(args, "ctc_weight", 0.1),
                middle_layer=getattr(args, "xbridge_middle_layer", 6),
                bridge_rank=getattr(args, "xbridge_rank", 192),
                self_weight=xbridge_self_weight,
                cross_weight=xbridge_cross_weight,
                ctc_only=getattr(args, "xbridge_ctc_only", False),
            )
        elif signer_adv_weight > 0:
            from espnet.nets.pytorch_backend.e2e_asr_signer import SignerInvariantE2E
            self.model = SignerInvariantE2E(
                len(self.token_list), self.modality,
                ctc_weight=getattr(args, "ctc_weight", 0.1),
                signer_classes=getattr(args, "signer_classes", 11),
                signer_layer=getattr(args, "signer_layer", 6),
                signer_hidden_dim=getattr(args, "signer_hidden_dim", 256),
                signer_dropout=getattr(args, "signer_dropout", 0.1),
                kd_weight=signer_kd_weight,
                teacher_checkpoint=getattr(args, "signer_teacher_checkpoint", None),
            )
        elif interctc_text_weight > 0 or interctc_shared_weight > 0:
            from espnet.nets.pytorch_backend.e2e_asr_interctc import InterCTCE2E
            toneless_units = []
            if interctc_toneless_weight > 0:
                payload = json.loads(Path(args.syllable_vocab).read_text(encoding="utf-8"))
                toneless_units = payload["units"]
            self.interctc_toneless_units = toneless_units
            self.model = InterCTCE2E(
                len(self.token_list), self.modality,
                ctc_weight=getattr(args, "ctc_weight", 0.1),
                interctc_layer=getattr(args, "interctc_layer", 6),
                interctc_text_weight=interctc_text_weight,
                interctc_toneless_weight=interctc_toneless_weight,
                interctc_shared_weight=interctc_shared_weight,
                toneless_vocab_size=len(toneless_units),
            )
        elif count_weight > 0:
            from espnet.nets.pytorch_backend.e2e_asr_count import CountE2E
            self.model = CountE2E(
                len(self.token_list), self.modality,
                ctc_weight=getattr(args, "ctc_weight", 0.1),
                count_bins=getattr(args, "count_bins", 33),
                count_hidden_dim=getattr(args, "count_hidden_dim", 256),
                count_dropout=getattr(args, "count_dropout", 0.1),
                count_weight=count_weight,
            )
        elif syllable_weight > 0:
            from espnet.nets.pytorch_backend.e2e_asr_visyll import ViSyllE2E
            syllable_vocab = json.loads(Path(args.syllable_vocab).read_text(encoding="utf-8"))
            self.syllable_units = syllable_vocab["units"]
            self.model = ViSyllE2E(
                len(self.token_list), self.modality,
                ctc_weight=getattr(args, "ctc_weight", 0.1),
                syllable_vocab_size=len(self.syllable_units),
                syllable_weight=syllable_weight,
                syllable_ctc_weight=getattr(args, "syllable_ctc_weight", 0.1),
                syllable_decoder_blocks=getattr(args, "syllable_decoder_blocks", 3),
                syllable_mode=getattr(args, "syllable_mode", "unified"),
                unit_to_initial=syllable_vocab.get("unit_to_initial"),
                unit_to_rhyme=syllable_vocab.get("unit_to_rhyme"),
                initial_vocab_size=len(syllable_vocab.get("initial_units", [])),
                rhyme_vocab_size=len(syllable_vocab.get("rhyme_units", [])),
            )
        elif onset_weight > 0 or rime_weight > 0:
            from espnet.nets.pytorch_backend.e2e_asr_structured import StructuredE2E
            def vocab_size(path): return len(json.loads(Path(path).read_text(encoding="utf-8"))["units"])
            self.auxiliary_units = {
                "onset": json.loads(Path(args.onset_vocab).read_text(encoding="utf-8"))["units"],
                "rime": json.loads(Path(args.rime_vocab).read_text(encoding="utf-8"))["units"],
            }
            self.model = StructuredE2E(len(self.token_list), self.modality, ctc_weight=getattr(args, "ctc_weight", 0.1), onset_vocab_size=vocab_size(args.onset_vocab), rime_vocab_size=vocab_size(args.rime_vocab), aux_layer=args.aux_layer, onset_weight=onset_weight, rime_weight=rime_weight)
        else:
            self.model = E2E(len(self.token_list), self.modality, ctc_weight=getattr(args, "ctc_weight", 0.1))

        # -- initialise
        if getattr(args, "pretrained_model_path", None):
            ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            if getattr(args, "transfer_frontend", False):
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.frontend.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend component are loaded successfully.")
            elif getattr(args, "transfer_encoder", False):
                tmp_ckpt = {k.replace("frontend.",""):v for k,v in ckpt.items() if k.startswith("frontend.")}
                self.model.frontend.load_state_dict(tmp_ckpt)
                tmp_ckpt = {k.replace("proj_encoder.",""):v for k,v in ckpt.items() if k.startswith("proj_encoder.")}
                self.model.proj_encoder.load_state_dict(tmp_ckpt)
                tmp_ckpt = {k.replace("encoder.",""):v for k,v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend, proj_encoder and encoder component are loaded successfully.")
            else:
                self.model.load_state_dict(ckpt)
                print("Pretrained weights of the full model are loaded successfully.")
        if getattr(args, "init_lightning_checkpoint", None):
            checkpoint = torch.load(args.init_lightning_checkpoint, map_location="cpu", weights_only=False)
            state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            allowed_missing = {key for key in missing if key.startswith(("count_head.", "onset_ctc.", "rime_ctc.", "syllable_ctc.", "syllable_decoder.", "rhyme_head.", "rhyme_condition.", "initial_head.", "mid_text_ctc.", "mid_toneless_ctc.", "signer_head.", "bridge_", "md_"))}
            if set(missing) != allowed_missing or unexpected: raise RuntimeError(f"Unsafe warm start: missing={missing}, unexpected={unexpected}")
            print(f"Warm-started model weights from {args.init_lightning_checkpoint}; new auxiliary keys={sorted(allowed_missing)}")
        if hasattr(self.model, "xbridge_middle_layer") and getattr(args, "xbridge_freeze_base", False):
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = name.startswith("bridge_")
            trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            print(f"XBridge frozen-base mode: trainable_parameters={trainable}")
        if hasattr(self.model, "md_middle_layer") and getattr(args, "mdbridge_freeze_base", False):
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = name.startswith("md_")
            trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            print(f"MD-Bridge frozen-base mode: trainable_parameters={trainable}")

    def configure_optimizers(self):
        auxiliary_lr = getattr(self.args, "aux_lr", None)
        if auxiliary_lr is not None and (hasattr(self.model, "syllable_weight") or hasattr(self.model, "count_weight")):
            prefixes = ("count_head.", "syllable_ctc.", "syllable_decoder.", "rhyme_head.", "rhyme_condition.", "initial_head.")
            base, auxiliary = [], []
            for name, parameter in self.model.named_parameters():
                (auxiliary if name.startswith(prefixes) else base).append(parameter)
            optimizer = torch.optim.AdamW(
                [{"params": base, "lr": self.args.lr}, {"params": auxiliary, "lr": auxiliary_lr}],
                lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98),
            )
        else:
            optimizer = torch.optim.AdamW(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98),
            )
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def _encode_for_decode(self, x):
        if hasattr(self.model, "md_middle_layer"):
            return self.model._encode(x, None)[0]
        if hasattr(self.model, "xbridge_middle_layer"):
            return self.model._encode(x, None)[0]
        return self.model.encoder(x, None)[0]

    @staticmethod
    def _load_xbridge_teacher(sample_ids, teacher_root, device):
        arrays = []
        for sample_id in sample_ids:
            path = Path(teacher_root) / (str(sample_id).replace("\\", "/") + ".npy")
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen AV-HuBERT latent: {path}")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.ndim != 2 or value.shape[1] != 1024:
                raise ValueError(f"Invalid AV-HuBERT latent shape at {path}: {value.shape}")
            arrays.append(torch.from_numpy(np.asarray(value, dtype=np.float32)))
        max_steps = max(len(value) for value in arrays)
        padded = torch.zeros(len(arrays), max_steps, 1024, dtype=torch.float32)
        for index, value in enumerate(arrays):
            padded[index, : len(value)] = value
        return padded.to(device=device, non_blocking=True)

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        x = self.model.frontend(sample.unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat = self._encode_for_decode(x)
        enc_feat = enc_feat.squeeze(0)
        decoded_nbest = []
        if not getattr(self.args, "skip_text_decoding", False):
            beam_hyps = self.beam_search(enc_feat)
            selected = [h.asdict() for h in beam_hyps[: min(len(beam_hyps), getattr(self.args, "nbest", 1))]]
            for hypothesis in selected:
                token_ids = torch.tensor(list(map(int, hypothesis["yseq"][1:])))
                text = self.text_transform.post_process(token_ids).replace("<eos>", "")
                decoded_nbest.append({
                    "text": text,
                    "score": float(hypothesis["score"]),
                    "scores": {name: float(value) for name, value in hypothesis.get("scores", {}).items()},
                    "search_length": max(len(hypothesis["yseq"]) - 1, 0),
                    "text_token_count": max(len(hypothesis["yseq"]) - 2, 0),
                })
            predicted = decoded_nbest[0]["text"]
        else:
            predicted = ""
        return predicted

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        x = self.model.frontend(sample["input"].unsqueeze(0))
        x = self.model.proj_encoder(x)
        auxiliary = None
        count_logits = None
        if hasattr(self.model, "encode_count"):
            enc_feat, count_logits, _ = self.model.encode_count(x, None)
        elif hasattr(self.model, "encode_structured"):
            enc_feat, auxiliary, _ = self.model.encode_structured(x, None)
            auxiliary = auxiliary.squeeze(0)
        else:
            enc_feat = self._encode_for_decode(x)
        enc_feat = enc_feat.squeeze(0)
        decoded_nbest = []
        if not getattr(self.args, "skip_text_decoding", False):
            beam_hyps = self.beam_search(enc_feat)
            selected = [h.asdict() for h in beam_hyps[: min(len(beam_hyps), getattr(self.args, "nbest", 1))]]
            for hypothesis in selected:
                token_ids = torch.tensor(list(map(int, hypothesis["yseq"][1:])))
                text = self.text_transform.post_process(token_ids).replace("<eos>", "")
                decoded_nbest.append({
                    "text": text,
                    "score": float(hypothesis["score"]),
                    "scores": {name: float(value) for name, value in hypothesis.get("scores", {}).items()},
                    "search_length": max(len(hypothesis["yseq"]) - 1, 0),
                    "text_token_count": max(len(hypothesis["yseq"]) - 2, 0),
                })
            predicted = decoded_nbest[0]["text"]
        else:
            predicted = ""

        actual_token_id = sample["target"]
        actual = self.text_transform.post_process(actual_token_id)

        if not getattr(self.args, "skip_text_decoding", False):
            self.total_edit_distance += compute_word_level_distance(actual, predicted)
            self.total_length += len(actual.split())
        self.test_references.append(actual)
        self.test_hypotheses.append(predicted)
        self.test_nbest.append(decoded_nbest)
        self.test_ids.append(sample.get("sample_id", str(sample_idx)))
        if count_logits is not None:
            logits = count_logits.squeeze(0).float()
            probabilities = logits.softmax(-1)
            self.test_count_evidence.append({
                "sample_id": self.test_ids[-1],
                "logits": logits.cpu().tolist(),
                "target_count": int(sample["syllable_target"].numel()),
                "mode_count": int(probabilities.argmax()),
                "expected_count": float((probabilities * torch.arange(len(probabilities), device=probabilities.device)).sum()),
                "raw_confidence": float(probabilities.max()),
            })
        if getattr(self.args, "emit_structured_evidence", False):
            from espnet.nets.pytorch_backend.e2e_asr_structured import collapse_ctc_with_confidence
            evidence = {"sample_id": self.test_ids[-1]}
            for name in ("onset", "rime"):
                head = getattr(self.model, f"{name}_ctc", None)
                if head is None:
                    continue
                log_prob = head.log_softmax(auxiliary.unsqueeze(0)).squeeze(0)
                frame_confidence, frame_ids = log_prob.exp().max(-1)
                token_ids, token_confidence = collapse_ctc_with_confidence(frame_ids.tolist(), frame_confidence.tolist())
                units = self.auxiliary_units[name]
                evidence[name] = {
                    "token_ids": token_ids,
                    "tokens": [units[index] if index < len(units) else "<unk>" for index in token_ids],
                    "token_confidence": token_confidence,
                    "mean_token_confidence": sum(token_confidence) / len(token_confidence) if token_confidence else 0.0,
                    "blank_ratio": float((frame_ids == 0).float().mean()),
                    "frames": len(frame_ids),
                }
            self.test_structured_evidence.append(evidence)
        emit_ctc = getattr(self.args, "emit_ctc_uncertainty", False)
        emit_stability = getattr(self.args, "emit_perturbation_stability", False)
        if emit_ctc or emit_stability:
            log_prob = self.model.ctc.log_softmax(enc_feat.unsqueeze(0)).squeeze(0)
        if emit_ctc:
            self.test_ctc_uncertainty.append(
                {"sample_id": self.test_ids[-1], **ctc_uncertainty_stats(log_prob)}
            )
        if emit_stability:
            video = sample["input"]
            blurred = F.avg_pool2d(video, kernel_size=3, stride=1, padding=1)
            occluded = video.clone()
            height, width = occluded.shape[-2:]
            occluded[::4, :, height // 2 - 6 : height // 2 + 6, width // 2 - 10 : width // 2 + 10] = 0
            variant_log_prob = {}
            for name, perturbed in (("blur", blurred), ("occlusion", occluded)):
                variant_x = self.model.frontend(perturbed.unsqueeze(0))
                variant_x = self.model.proj_encoder(variant_x)
                variant_enc = self._encode_for_decode(variant_x)
                variant_log_prob[name] = self.model.ctc.log_softmax(variant_enc).squeeze(0)
            self.test_perturbation_stability.append(
                {"sample_id": self.test_ids[-1], **ctc_stability_stats(log_prob, variant_log_prob)}
            )
        return

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch["inputs"].size(0)
        if self.trainer.world_size > 1:
            batch_sizes = self.all_gather(torch.tensor(batch_size, device=self.device))
            loss *= batch_sizes.numel() / batch_sizes.sum()  # world size / global batch size

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def _step(self, batch, batch_idx, step_type):
        if getattr(self.model, "pavir_enabled", False):
            pavir_kwargs = {
                "sample_ids": batch["sample_ids"],
                "update": int(self.global_step),
                "apply_pavir": step_type == "train",
            }
            if step_type == "train" and hasattr(self.model, "pavir_store"):
                context_eligible = any(
                    self.model.pavir_store.events_for(str(sample_id), "D")
                    for sample_id in batch["sample_ids"]
                )
                if context_eligible:
                    pavir_kwargs["context_gate_index"] = (
                        self._pavir_context_eligible_batches_seen
                    )
                    self._pavir_context_eligible_batches_seen += 1
            output = self.model(
                batch["inputs"],
                batch["input_lengths"],
                batch["targets"],
                **pavir_kwargs,
            )
            loss, loss_ctc, loss_att, acc = (
                output["loss"],
                output["loss_ctc"],
                output["loss_att"],
                output["decoder_acc"],
            )
        elif hasattr(self.model, "md_middle_layer"):
            output = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
            loss, loss_ctc, loss_att, acc = output["loss"], output["loss_ctc"], output["loss_att"], output["decoder_acc"]
        elif hasattr(self.model, "xbridge_middle_layer"):
            teacher = None
            teacher_root = getattr(self.args, "xbridge_teacher_root", None)
            if teacher_root and step_type == "train":
                teacher = self._load_xbridge_teacher(batch["sample_ids"], teacher_root, batch["inputs"].device)
            output = self.model(
                batch["inputs"], batch["input_lengths"], batch["targets"], av_teacher=teacher
            )
            loss, loss_ctc, loss_att, acc = output["loss"], output["loss_ctc"], output["loss_att"], output["decoder_acc"]
        elif hasattr(self.model, "signer_classes"):
            locked = float(getattr(self.args, "signer_adv_weight", 0.0))
            ramp_updates = int(getattr(self.args, "signer_ramp_updates", 491))
            if step_type == "train" and ramp_updates > 0:
                grl_scale = locked * min(1.0, float(self.global_step + 1) / ramp_updates)
            else:
                grl_scale = locked
            output = self.model(
                batch["inputs"], batch["input_lengths"], batch["targets"],
                batch.get("signer_targets"), grl_scale=grl_scale,
            )
            loss, loss_ctc, loss_att, acc = output["loss"], output["loss_ctc"], output["loss_att"], output["decoder_acc"]
        elif hasattr(self.model, "interctc_text_weight"):
            output = self.model(
                batch["inputs"], batch["input_lengths"], batch["targets"],
                batch.get("syllable_targets"), compute_aux_metrics=step_type != "train",
            )
            loss, loss_ctc, loss_att, acc = output["loss"], output["loss_ctc"], output["loss_att"], output["decoder_acc"]
        elif hasattr(self.model, "count_weight"):
            output = self.model(
                batch["inputs"], batch["input_lengths"], batch["targets"], batch.get("syllable_targets")
            )
            loss, loss_ctc, loss_att, acc = output["loss"], output["loss_ctc"], output["loss_att"], output["decoder_acc"]
        elif hasattr(self.model, "syllable_weight"):
            output = self.model(
                batch["inputs"], batch["input_lengths"], batch["targets"],
                batch.get("syllable_targets"), compute_aux_metrics=step_type != "train",
            )
            loss, loss_ctc, loss_att, acc = output["loss"], output["loss_ctc"], output["loss_att"], output["decoder_acc"]
        elif hasattr(self.model, "onset_weight"):
            output = self.model(batch["inputs"], batch["input_lengths"], batch["targets"], batch.get("onset_targets"), batch.get("rime_targets"), compute_aux_metrics=step_type != "train")
            loss, loss_ctc, loss_att, acc = output["loss"], output["loss_ctc"], output["loss_att"], output["decoder_acc"]
        else:
            loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"]); output = {"text_loss": loss}
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        else:
            self.log("loss_val", loss, batch_size=batch_size, sync_dist=True)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size, sync_dist=True)
            self.log("loss_att_val", loss_att, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc_val", acc, batch_size=batch_size, sync_dist=True)

        self.log(f"raw_text_loss{'_val' if step_type != 'train' else ''}", output["text_loss"], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        for name in ("onset", "rime"):
            if f"raw_{name}_loss" not in output: continue
            suffix = "_val" if step_type != "train" else ""
            for metric in (f"raw_{name}_loss", f"normalized_{name}_loss", f"weighted_{name}_loss"):
                self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log(f"{name}_to_text_ratio" + suffix, output[f"{name}_to_text_ratio"], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            if step_type != "train":
                stats = self._aux_validation_stats[name]
                for field in ("errors", "units", "blank_frames", "frames"): stats[field] += output[f"{name}_{field}"]
            self.log(f"{name}_infeasible" + suffix, float(output[f"{name}_infeasible"]), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True, reduce_fx="sum")

        if "raw_syllable_loss" in output:
            suffix = "_val" if step_type != "train" else ""
            for metric in ("raw_syllable_loss", "syllable_ctc_loss", "syllable_att_loss", "weighted_syllable_loss", "syllable_to_text_ratio", "syllable_decoder_acc"):
                self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            for metric in ("rhyme_decoder_acc", "initial_decoder_acc", "rhyme_att_loss", "initial_att_loss"):
                if metric in output:
                    self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("syllable_infeasible" + suffix, float(output["syllable_infeasible"]), on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True, reduce_fx="sum")
            if step_type != "train":
                stats = self._aux_validation_stats["syllable"]
                for field in ("errors", "units", "blank_frames", "frames"):
                    stats[field] += output[f"syllable_{field}"]
        if "count_nll" in output:
            suffix = "_val" if step_type != "train" else ""
            for metric in ("count_nll", "weighted_count_loss", "count_mae", "count_exact"):
                self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        for name in ("mid_text", "mid_toneless"):
            if f"raw_{name}_loss" not in output:
                continue
            suffix = "_val" if step_type != "train" else ""
            for metric in (f"raw_{name}_loss", f"weighted_{name}_loss", f"{name}_to_text_ratio"):
                self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log(
                f"{name}_infeasible" + suffix,
                float(output[f"{name}_infeasible"]),
                on_step=False, on_epoch=True, batch_size=batch_size,
                sync_dist=True, reduce_fx="sum",
            )
            if step_type != "train":
                stats = self._aux_validation_stats[name]
                for field in ("errors", "units", "blank_frames", "frames"):
                    stats[field] += output[f"{name}_{field}"]
        if "weighted_final_ctc_loss" in output:
            suffix = "_val" if step_type != "train" else ""
            self.log(
                "weighted_final_ctc_loss" + suffix,
                output["weighted_final_ctc_loss"],
                on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True,
            )
        if "signer_loss" in output:
            suffix = "_val" if step_type != "train" else ""
            for metric in ("signer_loss", "signer_acc", "raw_kd_loss", "weighted_kd_loss", "grl_scale"):
                self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        if "xbridge_self_loss" in output:
            suffix = "_val" if step_type != "train" else ""
            for metric in ("xbridge_self_loss", "xbridge_cross_loss", "xbridge_alpha_mean", "xbridge_scale"):
                self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        if "md_alpha_mean" in output:
            suffix = "_val" if step_type != "train" else ""
            for metric in ("md_alpha_mean", "md_scale", "md_residual_l2", "md_branch_magnitude"):
                self.log(metric + suffix, output[metric], on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        if "pavir_visual_loss" in output:
            suffix = "_val" if step_type != "train" else ""
            on_step = step_type == "train"
            for metric in (
                "pavir_visual_loss",
                "pavir_context_loss",
                "pavir_weighted_visual_loss",
                "pavir_weighted_context_loss",
                "pavir_visual_eligible",
                "pavir_visual_active",
                "pavir_visual_infeasible",
                "pavir_context_eligible",
                "pavir_context_active",
                "pavir_context_selected_batch",
                "pavir_ramp",
            ):
                self.log(
                    metric + suffix,
                    output[metric],
                    on_step=on_step,
                    on_epoch=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["pavir_runtime_state"] = {
            "context_eligible_batches_seen": int(
                self._pavir_context_eligible_batches_seen
            )
        }

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("pavir_runtime_state", {})
        self._pavir_context_eligible_batches_seen = int(
            state.get("context_eligible_batches_seen", 0)
        )

    def on_validation_epoch_start(self):
        names = ["onset", "rime"]
        if hasattr(self.model, "syllable_weight"):
            names.append("syllable")
        if hasattr(self.model, "interctc_text_weight"):
            names.append("mid_text")
            if self.model.mid_toneless_ctc is not None:
                names.append("mid_toneless")
        self._aux_validation_stats = {name: {field: 0 for field in ("errors", "units", "blank_frames", "frames")} for name in names}

    def on_validation_epoch_end(self):
        for name, stats in self._aux_validation_stats.items():
            if stats["units"]:
                self.log(f"{name}_error_rate_val", stats["errors"] / stats["units"], sync_dist=True)
                self.log(f"{name}_blank_ratio_val", stats["blank_frames"] / max(stats["frames"], 1), sync_dist=True)

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.test_references = []
        self.test_hypotheses = []
        self.test_nbest = []
        self.test_ids = []
        self.test_structured_evidence = []
        self.test_count_evidence = []
        self.test_ctc_uncertainty = []
        self.test_perturbation_stability = []
        self.text_transform = TextTransform()
        if not getattr(self.args, "skip_text_decoding", False):
            word_lm = None
            if getattr(self.args, "lm_weight", 0.0) != 0.0:
                if not getattr(self.args, "word_lm_path", None):
                    raise ValueError("--word-lm-path is required when --lm-weight is non-zero")
                from espnet.nets.scorers.word_ngram import (
                    SentencePieceWordNGramScorer,
                    StupidBackoffWordNGram,
                )
                sp_model = getattr(self.args, "word_lm_sp_model", None) or os.environ.get("VSR_SPM_MODEL")
                if not sp_model:
                    raise ValueError(
                        "--word-lm-sp-model or VSR_SPM_MODEL is required for word-LM shallow fusion"
                    )
                word_lm = SentencePieceWordNGramScorer(
                    StupidBackoffWordNGram.load(self.args.word_lm_path),
                    self.token_list,
                    sp_model,
                    sos=self.model.odim - 1,
                    eos=self.model.odim - 1,
                )
            self.beam_search = get_beam_search_decoder(
                self.model,
                self.token_list,
                rnnlm=word_lm,
                beam_size=getattr(self.args, "beam_size", 10),
                ctc_weight=getattr(self.args, "ctc_weight", 0.1),
                lm_weight=getattr(self.args, "lm_weight", 0.0),
                penalty=getattr(self.args, "length_penalty", 0.0),
            )

    def on_test_epoch_end(self):
        if self.total_length:
            self.log("wer", self.total_edit_distance / self.total_length)
        results_path = getattr(self.args, "results_path", None)
        if results_path:
            output = Path(results_path)
            output.mkdir(parents=True, exist_ok=True)
            (output / "hypotheses.json").write_text(
                json.dumps({"ref": self.test_references, "hypo": self.test_hypotheses}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (output / "nbest.json").write_text(
                json.dumps({"sample_id": self.test_ids, "ref": self.test_references, "nbest": self.test_nbest}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if self.test_structured_evidence:
                (output / "structured_evidence.json").write_text(
                    json.dumps(self.test_structured_evidence, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if self.test_count_evidence:
                (output / "count_evidence.json").write_text(
                    json.dumps(self.test_count_evidence, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if self.test_ctc_uncertainty:
                (output / "ctc_uncertainty.json").write_text(
                    json.dumps(self.test_ctc_uncertainty, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            if self.test_perturbation_stability:
                (output / "perturbation_stability.json").write_text(
                    json.dumps(self.test_perturbation_stability, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    scorers["lm"] = rnnlm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
