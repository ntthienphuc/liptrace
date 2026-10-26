import json
import math
import os
import random

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .transforms import AudioTransform, VideoTransform
from pavir.artifacts import CanonicalIDMap


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        if data_type == "sample_id":
            batch_out["sample_ids"] = [sample[data_type] for sample in batch]
            continue
        if data_type == "transformed_input_sha256":
            batch_out["transformed_input_sha256s"] = [
                sample[data_type] for sample in batch
            ]
            continue
        pad_val = -1 if data_type.endswith("target") else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def _batch_by_token_count(idx_target_lengths, max_frames, batch_size=None):
    """Batch under the actual padded-frame workload.

    Auto-AVSR pads every sample to the longest sequence in its physical
    batch.  Summing raw lengths can therefore understate GPU work.  The locked
    ``max_frames`` contract is enforced as ``batch_size * padded_length``.
    """
    batches = []
    current_batch = []
    current_max_length = 0
    for idx, target_length in idx_target_lengths:
        candidate_max = max(current_max_length, int(target_length))
        padded_workload = (len(current_batch) + 1) * candidate_max
        if current_batch and (
            padded_workload > max_frames
            or (batch_size and len(current_batch) == batch_size)
        ):
            batches.append(current_batch)
            current_batch = [idx]
            current_max_length = int(target_length)
        else:
            current_batch.append(idx)
            current_max_length = candidate_max

    if current_batch:
        batches.append(current_batch)

    return batches


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, lengths, max_frames, num_buckets, shuffle=False, batch_size=None
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [
            (idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)
        ]
        if shuffle:
            idx_length_buckets = random.sample(
                idx_length_buckets, len(idx_length_buckets)
            )
        else:
            idx_length_buckets = sorted(
                idx_length_buckets, key=lambda x: x[1], reverse=True
            )
        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class CurriculumBatchSampler(torch.utils.data.Sampler):
    """Keep the update budget fixed while expanding easy-to-hard exposure.

    Each epoch emits exactly one full-dataset worth of bucket indices. Before
    the curriculum reaches 100%, batches are sampled with replacement from the
    easiest eligible prefix. At 100%, every batch is visited once in a seeded
    random order. Difficulty is fixed from train-only metadata.
    """

    def __init__(self, batch_scores, fractions, seed=42):
        if not batch_scores:
            raise ValueError("Curriculum requires at least one training batch")
        if not fractions or any(value <= 0 or value > 1 for value in fractions):
            raise ValueError("Curriculum fractions must be in (0, 1]")
        if any(right < left for left, right in zip(fractions, fractions[1:])):
            raise ValueError("Curriculum fractions must be non-decreasing")
        if fractions[-1] != 1.0:
            raise ValueError("Curriculum must finish with the full training set")
        self.sorted_indices = torch.tensor(
            sorted(range(len(batch_scores)), key=lambda index: (batch_scores[index], index)),
            dtype=torch.long,
        )
        self.fractions = tuple(float(value) for value in fractions)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @property
    def current_fraction(self):
        return self.fractions[min(self.epoch, len(self.fractions) - 1)]

    def __iter__(self):
        total = len(self.sorted_indices)
        eligible = max(1, min(total, math.ceil(total * self.current_fraction)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if eligible == total:
            positions = torch.randperm(total, generator=generator)
        else:
            positions = torch.randint(eligible, (total,), generator=generator)
        return iter(self.sorted_indices[positions].tolist())

    def __len__(self):
        return len(self.sorted_indices)


class DataModule(LightningDataModule):
    def __init__(
        self,
        args=None,
        batch_size=None,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=10,
    ):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        sample_id_map = getattr(args, "sample_id_map", None) if args is not None else None
        self.sample_id_map = (
            CanonicalIDMap.load(sample_id_map) if sample_id_map else None
        )
        self.emit_transformed_input_sha256 = bool(
            getattr(args, "prescreen_evidence_journal", None)
        ) or os.environ.get("PAVIR_PRESCREEN_BATCH_EVIDENCE", "0") == "1"

    def train_dataloader(self):
        dataset = AVDataset(
            root_dir=self.args.root_dir,
            label_path=os.path.join(self.args.root_dir, "labels", self.args.train_file),
            subset="train",
            modality=self.args.modality,
            audio_transform=AudioTransform("train") if self.args.modality == "audio" else None,
            video_transform=VideoTransform("train") if self.args.modality == "video" else None,
            signer_vocab=getattr(self.args, "signer_vocab", None),
            sample_id_map=self.sample_id_map,
            emit_transformed_input_sha256=self.emit_transformed_input_sha256,
        )
        dataset = CustomBucketDataset(
            dataset,
            dataset.input_lengths,
            self.args.max_frames,
            self.train_num_buckets,
            batch_size=self.batch_size,
        )
        sampler = None
        curriculum_scores = getattr(self.args, "curriculum_scores", None)
        if curriculum_scores:
            payload = json.loads(open(curriculum_scores, encoding="utf-8").read())
            scores = payload.get("scores", payload)
            row_scores = []
            missing = []
            for row in dataset.dataset.list:
                sample_id = dataset.dataset.sample_id(row[1])
                if sample_id not in scores:
                    missing.append(sample_id)
                    row_scores.append(0.0)
                else:
                    row_scores.append(float(scores[sample_id]))
            if missing:
                raise ValueError(f"Curriculum scores missing {len(missing)} samples; first={missing[0]}")
            batch_scores = [
                sum(row_scores[index] for index in batch) / len(batch)
                for batch in dataset.batches
            ]
            fractions = [float(value) for value in self.args.curriculum_fractions.split(",")]
            sampler = CurriculumBatchSampler(
                batch_scores, fractions, seed=self.args.curriculum_seed
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collate_pad,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = AVDataset(
            root_dir=self.args.root_dir,
            label_path=os.path.join(self.args.root_dir, "labels", self.args.val_file),
            subset="val",
            modality=self.args.modality,
            audio_transform=AudioTransform("val") if self.args.modality == "audio" else None,
            video_transform=VideoTransform("val") if self.args.modality == "video" else None,
            signer_vocab=getattr(self.args, "signer_vocab", None),
            sample_id_map=self.sample_id_map,
        )
        dataset = CustomBucketDataset(
            dataset, dataset.input_lengths, 1000, 1, batch_size=self.batch_size
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn=collate_pad,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        test_subset = getattr(self.args, "test_subset", "test")
        video_view = getattr(self.args, "video_view", "center")
        dataset = AVDataset(
            root_dir=self.args.root_dir,
            label_path=os.path.join(self.args.root_dir, "labels", self.args.test_file),
            subset=test_subset,
            modality=self.args.modality,
            audio_transform=(
                AudioTransform("test", snr_target=self.args.decode_snr_target)
                if self.args.modality == "audio"
                else None
            ),
            video_transform=(
                VideoTransform("test", spatial_view=video_view)
                if self.args.modality == "video"
                else None
            ),
            signer_vocab=getattr(self.args, "signer_vocab", None),
            sample_id_map=self.sample_id_map,
        )
        limit = getattr(self.args, "limit_test_samples", None)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit_test_samples must be positive")
            dataset = torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
