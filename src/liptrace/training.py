import os
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Repro / utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_text(s: str, mode="legacy-lower-v1") -> str:
    from .contracts import normalize
    return normalize(str(s), mode)


def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(prev + 1, dp[j] + 1, dp[j - 1] + 1)
            prev = cur
    return dp[m]


def cer_score(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def wer_score(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def min_ctc_required_len(text: str) -> int:
    text = str(text)
    if not text:
        return 0
    extra = 0
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            extra += 1
    return len(text) + extra


def uniform_sample_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((k,), dtype=np.int64)
    if n >= k:
        return np.linspace(0, n - 1, num=k).round().astype(np.int64)
    idx = np.linspace(0, n - 1, num=n).round().astype(np.int64)
    pad = np.full((k - n,), n - 1, dtype=np.int64)
    return np.concatenate([idx, pad], axis=0)


from .media import read_video_gray_resize


# ----------------------------
# Data prep
# ----------------------------
def build_charset(texts: List[str]) -> List[str]:
    chars = sorted(set("".join(texts)))
    return chars


def filter_impossible_rows(df: pd.DataFrame, max_frames: int, normalization="legacy-lower-v1") -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df.copy()
    tmp["text"] = tmp["text"].astype(str).map(lambda x: normalize_text(x, normalization))
    tmp["min_ctc_required_len"] = tmp["text"].map(min_ctc_required_len)
    bad = tmp[tmp["min_ctc_required_len"] > max_frames].copy()
    good = tmp[tmp["min_ctc_required_len"] <= max_frames].copy()
    return good.reset_index(drop=True), bad.reset_index(drop=True)


class LipNetCsvDataset(Dataset):
    def __init__(self, csv_path: str, char_to_id: Dict[str, int], max_frames: int = 32, img_size: int = 96, normalization="legacy-lower-v1"):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
        if "text" not in self.df.columns:
            if "label_ascii" in self.df.columns:
                self.df["text"] = self.df["label_ascii"].astype(str)
            elif "label_text" in self.df.columns:
                self.df["text"] = self.df["label_text"].astype(str)
            else:
                raise ValueError("CSV must contain text or label_ascii/label_text.")
        self.df["text"] = self.df["text"].astype(str).map(lambda x: normalize_text(x, normalization))

        self.char_to_id = char_to_id
        self.max_frames = max_frames
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_path = str(row["clip_path"])
        text = str(row["text"])

        video = read_video_gray_resize(clip_path, self.max_frames, self.img_size)

        target = np.array([self.char_to_id[ch] for ch in text], dtype=np.int64)
        return {
            "video": torch.from_numpy(video).float(),
            "target": torch.from_numpy(target).long(),
            "target_len": len(target),
            "input_len": self.max_frames,
            "text": text,
            "clip_path": clip_path,
        }


def collate_fn(batch):
    videos = torch.stack([b["video"] for b in batch], dim=0)  # (B,1,T,H,W)
    target_lens = torch.tensor([b["target_len"] for b in batch], dtype=torch.long)
    input_lens = torch.tensor([b["input_len"] for b in batch], dtype=torch.long)
    targets = torch.cat([b["target"] for b in batch], dim=0) if batch else torch.empty((0,), dtype=torch.long)
    texts = [b["text"] for b in batch]
    clip_paths = [b["clip_path"] for b in batch]
    return {
        "video": videos,
        "targets": targets,
        "target_lens": target_lens,
        "input_lens": input_lens,
        "texts": texts,
        "clip_paths": clip_paths,
    }


# ----------------------------
# Model: LipNet-style
# ----------------------------
from .model import LipNetBackbone


# ----------------------------
# Decode / eval
# ----------------------------
def greedy_decode(log_probs_tbc: torch.Tensor, blank_index: int, id_to_char: Dict[int, str]) -> List[str]:
    """
    log_probs_tbc: (T, B, C)
    """
    pred = log_probs_tbc.argmax(dim=-1).transpose(0, 1).detach().cpu().numpy()  # (B,T)
    out = []
    for seq in pred:
        chars = []
        prev = None
        for idx in seq:
            idx = int(idx)
            if idx == blank_index:
                prev = idx
                continue
            if prev == idx:
                prev = idx
                continue
            chars.append(id_to_char.get(idx, ""))
            prev = idx
        out.append("".join(chars))
    return out


@torch.no_grad()
def evaluate_model(model, loader, device, ctc_loss, blank_index, id_to_char, out_pred_csv="", normalization="legacy-lower-v1"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rows = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        video = batch["video"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)
        target_lens = batch["target_lens"].to(device, non_blocking=True)

        logits = model(video)  # (B,T,C)
        log_probs = logits.log_softmax(dim=-1)
        log_probs_tbc = log_probs.transpose(0, 1)  # (T,B,C)

        loss = ctc_loss(log_probs_tbc, targets, input_lens, target_lens)
        bs = video.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        hyps = greedy_decode(log_probs_tbc, blank_index, id_to_char)
        for ref, hyp, clip in zip(batch["texts"], hyps, batch["clip_paths"]):
            ref = normalize_text(ref, normalization)
            hyp = normalize_text(hyp, normalization)
            rows.append({
                "clip_path": clip,
                "ref_text": ref,
                "hyp_text": hyp,
                "cer": cer_score(ref, hyp),
                "wer": wer_score(ref, hyp),
            })

    df = pd.DataFrame(rows)
    if out_pred_csv:
        df.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")

    metrics = {
        "rows": int(len(df)),
        "loss": float(total_loss / max(total_samples, 1)),
        "cer": float(df["cer"].mean()) if len(df) else None,
        "wer": float(df["wer"].mean()) if len(df) else None,
        "exact_match": float((df["ref_text"] == df["hyp_text"]).mean()) if len(df) else None,
    }
    return metrics


def train_one_epoch(model, loader, optimizer, scaler, device, ctc_loss, blank_index, id_to_char, use_amp: bool):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Train", leave=True)
    for batch in pbar:
        video = batch["video"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_lens = batch["input_lens"].to(device, non_blocking=True)
        target_lens = batch["target_lens"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(video)
                log_probs = logits.log_softmax(dim=-1)
                log_probs_tbc = log_probs.transpose(0, 1)
                loss = ctc_loss(log_probs_tbc, targets, input_lens, target_lens)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(video)
            log_probs = logits.log_softmax(dim=-1)
            log_probs_tbc = log_probs.transpose(0, 1)
            loss = ctc_loss(log_probs_tbc, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()

        bs = video.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        pbar.set_postfix(loss=total_loss / max(total_samples, 1))

    return float(total_loss / max(total_samples, 1))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train LipNet-style PyTorch CTC model on balanced Vietnamese mouth clips.")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--rnn_units", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--normalization", choices=["legacy-lower-v1", "nfc-lower-v1"], default="legacy-lower-v1")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        ap.error("epochs and batch_size must be positive")
    if Path(args.out_dir).exists():
        ap.error("out_dir already exists; choose a new experiment directory")
    from .contracts import profile_from_checkpoint
    profile_from_checkpoint({'args': vars(args), 'charset': ['a'], 'blank_index': 1}, args.normalization)

    from .dataset import audit_dataset
    from .contracts import write_json
    report = audit_dataset([args.train_csv, args.val_csv, args.test_csv], args.max_frames, args.normalization)
    ensure_dir(args.out_dir)
    write_json(Path(args.out_dir) / "dataset_audit.json", report)
    if not report['valid']:
        codes = sorted({i['code'] for i in report['issues'] if i['severity'] == 'error'})
        suffix = ' (unseen training characters)' if 'unseen_character' in codes else ''
        raise ValueError(f"Dataset audit failed: {codes}{suffix}; see dataset_audit.json")
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    if device.type == "cuda":
        print(f"[INFO] cuda = {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(args.train_csv, encoding="utf-8-sig", keep_default_na=False)
    val_df = pd.read_csv(args.val_csv, encoding="utf-8-sig", keep_default_na=False)
    test_df = pd.read_csv(args.test_csv, encoding="utf-8-sig", keep_default_na=False)

    for df, manifest in zip([train_df, val_df, test_df], [args.train_csv, args.val_csv, args.test_csv]):
        df['clip_path'] = [str((Path(x) if Path(x).is_absolute() else Path(manifest).resolve().parent / x).resolve()) for x in df['clip_path']]
        if "text" not in df.columns:
            if "label_ascii" in df.columns:
                df["text"] = df["label_ascii"].astype(str)
            elif "label_text" in df.columns:
                df["text"] = df["label_text"].astype(str)
            else:
                raise ValueError("CSV must contain text or label_ascii/label_text.")

    train_df, train_bad = filter_impossible_rows(train_df, args.max_frames, args.normalization)
    val_df, val_bad = filter_impossible_rows(val_df, args.max_frames, args.normalization)
    test_df, test_bad = filter_impossible_rows(test_df, args.max_frames, args.normalization)

    train_bad.to_csv(Path(args.out_dir) / "train_impossible_ctc.csv", index=False, encoding="utf-8-sig")
    val_bad.to_csv(Path(args.out_dir) / "val_impossible_ctc.csv", index=False, encoding="utf-8-sig")
    test_bad.to_csv(Path(args.out_dir) / "test_impossible_ctc.csv", index=False, encoding="utf-8-sig")

    all_texts = train_df["text"].astype(str).map(lambda x: normalize_text(x, args.normalization)).tolist()
    charset = build_charset(all_texts)
    if not charset:
        raise ValueError("Empty charset.")

    char_to_id = {ch: i for i, ch in enumerate(charset)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    for split_name, df in [('validation', val_df), ('test', test_df)]:
        unknown = set(''.join(df['text'].tolist())) - set(charset)
        if unknown:
            raise ValueError(f'{split_name} contains unseen training characters: {sorted(unknown)}')
    blank_index = len(charset)
    num_classes = len(charset) + 1

    with open(Path(args.out_dir) / "charset.json", "w", encoding="utf-8") as f:
        json.dump({
            "charset": charset,
            "char_to_id": char_to_id,
            "blank_index": blank_index,
        }, f, ensure_ascii=False, indent=2)

    # Save filtered csvs actually used for training.
    train_used_csv = Path(args.out_dir) / "train_used.csv"
    val_used_csv = Path(args.out_dir) / "val_used.csv"
    test_used_csv = Path(args.out_dir) / "test_used.csv"
    train_df.to_csv(train_used_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_used_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_used_csv, index=False, encoding="utf-8-sig")

    train_ds = LipNetCsvDataset(str(train_used_csv), char_to_id, max_frames=args.max_frames, img_size=args.img_size, normalization=args.normalization)
    val_ds = LipNetCsvDataset(str(val_used_csv), char_to_id, max_frames=args.max_frames, img_size=args.img_size, normalization=args.normalization)
    test_ds = LipNetCsvDataset(str(test_used_csv), char_to_id, max_frames=args.max_frames, img_size=args.img_size, normalization=args.normalization)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = LipNetBackbone(
        num_classes=num_classes,
        img_size=args.img_size,
        max_frames=args.max_frames,
        rnn_units=args.rnn_units,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    ctc_loss = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_cer = float("inf")
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, ctc_loss, blank_index, id_to_char, args.amp)
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            ctc_loss=ctc_loss,
            blank_index=blank_index,
            id_to_char=id_to_char,
            normalization=args.normalization,
            out_pred_csv=str(Path(args.out_dir) / f"val_predictions_epoch{epoch:03d}.csv"),
        )
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_cer": val_metrics["cer"],
            "val_wer": val_metrics["wer"],
            "val_exact": val_metrics["exact_match"],
            "lr": optimizer.param_groups[0]["lr"],
            "minutes": (time.time() - t0) / 60.0,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(Path(args.out_dir) / "train_log.csv", index=False, encoding="utf-8-sig")

        print(json.dumps(row, ensure_ascii=False))

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "charset": charset,
            "blank_index": blank_index,
            "args": vars(args),
        }
        torch.save(ckpt, Path(args.out_dir) / "last.pt")

        if val_metrics["cer"] is not None and val_metrics["cer"] < best_val_cer:
            best_val_cer = val_metrics["cer"]
            torch.save(ckpt, Path(args.out_dir) / "best.pt")

    best_ckpt = torch.load(Path(args.out_dir) / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])

    val_metrics = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        ctc_loss=ctc_loss,
        blank_index=blank_index,
        id_to_char=id_to_char,
        normalization=args.normalization,
        out_pred_csv=str(Path(args.out_dir) / "val_predictions.csv"),
    )
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        ctc_loss=ctc_loss,
        blank_index=blank_index,
        id_to_char=id_to_char,
        normalization=args.normalization,
        out_pred_csv=str(Path(args.out_dir) / "test_predictions.csv"),
    )

    summary = {
        "software_version": "0.1.0",
        "historical_reproduction": False,
        "charset_source": "training_only",
        "metrics_aggregation": "sample_mean",
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "dropped_impossible": {
            "train": int(len(train_bad)),
            "val": int(len(val_bad)),
            "test": int(len(test_bad)),
        },
        "charset_size": int(len(charset)),
        "blank_index": int(blank_index),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "img_size": int(args.img_size),
        "max_frames": int(args.max_frames),
        "rnn_units": int(args.rnn_units),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(Path(args.out_dir) / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    from .bundle import create_bundle
    identity = create_bundle(Path(args.out_dir) / 'best.pt', Path(args.out_dir) / 'bundle', args.normalization)
    summary['bundle'] = identity
    write_json(Path(args.out_dir) / 'summary_metrics.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
