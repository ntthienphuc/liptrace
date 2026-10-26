#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import hashlib
import numpy as np
import torch
import torchaudio
import torchvision

from pavir.artifacts import CanonicalIDMap


def sample_id_from_path(path):
    """Return the benchmark clip id without leaking machine-specific roots."""
    normalized = str(path).replace("\\", "/")
    for marker in ("/train/", "/val/", "/test/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1].rsplit(".", 1)[0]
    return normalized.rsplit(".", 1)[0]


def transformed_input_sha256(tensor):
    """Hash one transformed sample with dtype/shape domain separation."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("transformed input must be a tensor")
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    value = tensor.detach().contiguous()
    if value.device.type != "cpu":
        value = value.cpu()
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def load_video(path, cache_path=None):
    """
    rtype: torch, T x C x H x W
    """
    if cache_path:
        if os.path.exists(cache_path):
            # Frozen shared cache is uint8 [T,H,W]. Auto-AVSR expects [T,C,H,W].
            vid = torch.from_numpy(np.load(cache_path, mmap_mode="r", allow_pickle=False).copy())
            return vid.unsqueeze(1)
        if os.environ.get("VSR_CACHE_STRICT", "1") == "1":
            raise FileNotFoundError(f"Required video cache is missing: {cache_path}")
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    if os.environ.get("VSR_VISUAL_ONLY", "0") == "1":
        raise RuntimeError(
            "Audio loading is disabled by the visual-only campaign firewall"
        )
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        signer_vocab=None,
        sample_id_map=None,
        emit_transformed_input_sha256=False,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        self.subset = subset
        if sample_id_map is None or isinstance(sample_id_map, CanonicalIDMap):
            self.sample_id_map = sample_id_map
        else:
            self.sample_id_map = CanonicalIDMap.load(sample_id_map)
        self.emit_transformed_input_sha256 = bool(
            emit_transformed_input_sha256
        )

        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]

        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.signer_to_id = None
        if signer_vocab:
            payload = json.loads(open(signer_vocab, encoding="utf-8").read())
            signers = payload.get("signers", payload)
            self.signer_to_id = {
                str(item["id"] if isinstance(item, dict) else item): index
                for index, item in enumerate(signers)
            }

    def signer_target(self, rel_path):
        if self.signer_to_id is None:
            return None
        components = set(str(rel_path).replace("\\", "/").split("/"))
        matches = [name for name in self.signer_to_id if name in components]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous signer components in path: {rel_path}")
        signer_id = self.signer_to_id[matches[0]] if matches else -1
        return torch.tensor([signer_id], dtype=torch.long)

    def sample_id(self, rel_path):
        if self.sample_id_map is None:
            return sample_id_from_path(rel_path)
        return self.sample_id_map.lookup(self.subset, rel_path)

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            fields = path_count_label.split(",")
            if len(fields) not in (4, 5, 6, 8):
                raise ValueError(f"Expected 4, 5, 6, or 8 CSV fields, got {len(fields)}: {path_count_label}")
            dataset_name, rel_path, input_length, token_id = fields[:4]
            cache_path = fields[4] if len(fields) == 5 else None
            if len(fields) == 8:
                cache_path = fields[4]
                auxiliary = tuple(torch.tensor([int(value) for value in field.split()]) for field in fields[5:8])
                syllable = None
            elif len(fields) == 6:
                cache_path = fields[4]
                auxiliary = (None, None, None)
                syllable = torch.tensor([int(value) for value in fields[5].split()])
            else:
                auxiliary = (None, None, None)
                syllable = None
            paths_counts_labels.append((dataset_name, rel_path, int(input_length), torch.tensor([int(_) for _ in token_id.split()]), cache_path, *auxiliary, syllable))
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id, cache_path, onset_id, rime_id, tone_id, syllable_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if self.modality == "video":
            video = load_video(path, cache_path)
            video = self.video_transform(video)
            answer = {"input": video, "target": token_id, "sample_id": self.sample_id(rel_path)}
            if self.emit_transformed_input_sha256:
                answer["transformed_input_sha256"] = transformed_input_sha256(
                    video
                )
            signer_target = self.signer_target(rel_path)
            if signer_target is not None: answer["signer_target"] = signer_target
            if onset_id is not None: answer.update(onset_target=onset_id, rime_target=rime_id, tone_target=tone_id)
            if syllable_id is not None: answer.update(syllable_target=syllable_id)
            return answer
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            answer = {"input": audio, "target": token_id, "sample_id": self.sample_id(rel_path)}
            signer_target = self.signer_target(rel_path)
            if signer_target is not None: answer["signer_target"] = signer_target
            if onset_id is not None: answer.update(onset_target=onset_id, rime_target=rime_id, tone_target=tone_id)
            if syllable_id is not None: answer.update(syllable_target=syllable_id)
            return answer

    def __len__(self):
        return len(self.list)
