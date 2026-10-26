import logging
from argparse import ArgumentParser

import torch
import torchaudio
from datamodule.data_module import DataModule
from pytorch_lightning import Trainer, seed_everything


# Set environment variables and logger level
logging.basicConfig(level=logging.WARNING)


def get_trainer(args):
    return Trainer(num_nodes=1, devices=1, accelerator="gpu", precision="bf16-mixed", logger=False)


def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        help="Type of input modality",
        required=True,
        choices=["audio", "video"],
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory of preprocessed dataset",
        required=True,
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of testing label list. (Default: lrs3_test_transcript_lengths_seg16s.csv)",
        required=True,
    )
    parser.add_argument(
        "--test-subset",
        choices=["train", "val", "valid", "test"],
        default="test",
        help=(
            "Logical split used for canonical-ID lookup. Use val when decoding "
            "VC-calib/VC-promote files drawn from the frozen validation pool."
        ),
    )
    parser.add_argument(
        "--video-view",
        choices=["center", "left", "right"],
        default="center",
        help="Frozen deterministic 88x88 crop for spatial-stability audits.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to the pre-trained model",
        default=None,
    )
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--ctc-weight", type=float, default=0.1)
    parser.add_argument(
        "--word-lm-path",
        type=str,
        default=None,
        help="Pickled Stupid-Backoff word N-gram for in-beam shallow fusion.",
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=0.0,
        help="Weight of the word LM score inside beam search.",
    )
    parser.add_argument(
        "--word-lm-sp-model",
        type=str,
        default=None,
        help="SentencePiece model used to recover completed words for the word LM.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.0,
        help="Per-output-token beam-search length bonus. Positive values favor longer hypotheses.",
    )
    parser.add_argument("--onset-vocab", type=str, default=None)
    parser.add_argument("--rime-vocab", type=str, default=None)
    parser.add_argument("--aux-layer", type=int, choices=[6, 12], default=6)
    parser.add_argument("--onset-weight", type=float, default=0.0)
    parser.add_argument("--rime-weight", type=float, default=0.0)
    parser.add_argument("--syllable-vocab", type=str, default=None)
    parser.add_argument("--syllable-weight", type=float, default=0.0)
    parser.add_argument("--syllable-ctc-weight", type=float, default=0.1)
    parser.add_argument("--syllable-decoder-blocks", type=int, default=3)
    parser.add_argument("--syllable-mode", choices=["unified", "rhyme_first"], default="unified")
    parser.add_argument("--count-weight", type=float, default=0.0)
    parser.add_argument("--count-bins", type=int, default=33)
    parser.add_argument("--count-hidden-dim", type=int, default=256)
    parser.add_argument("--count-dropout", type=float, default=0.1)
    parser.add_argument("--interctc-layer", type=int, choices=[6], default=6)
    parser.add_argument("--interctc-text-weight", type=float, default=0.0)
    parser.add_argument("--interctc-toneless-weight", type=float, default=0.0)
    parser.add_argument("--interctc-shared-weight", type=float, default=0.0)
    parser.add_argument("--signer-vocab", type=str, default=None)
    parser.add_argument("--signer-classes", type=int, default=11)
    parser.add_argument("--signer-layer", type=int, choices=[6], default=6)
    parser.add_argument("--signer-hidden-dim", type=int, default=256)
    parser.add_argument("--signer-dropout", type=float, default=0.1)
    parser.add_argument("--signer-adv-weight", type=float, default=0.0)
    parser.add_argument("--signer-kd-weight", type=float, default=0.0)
    parser.add_argument("--signer-teacher-checkpoint", type=str, default=None)
    parser.add_argument("--signer-ramp-updates", type=int, default=491)
    parser.add_argument("--xbridge-self-weight", type=float, default=0.0)
    parser.add_argument("--xbridge-cross-weight", type=float, default=0.0)
    parser.add_argument("--xbridge-middle-layer", type=int, choices=[6], default=6)
    parser.add_argument("--xbridge-rank", type=int, default=192)
    parser.add_argument("--xbridge-teacher-root", type=str, default=None)
    parser.add_argument("--xbridge-freeze-base", action="store_true")
    parser.add_argument("--xbridge-ctc-only", action="store_true")
    parser.add_argument(
        "--xbridge-disable-fusion",
        action="store_true",
        help="Inference-only ablation that decodes the checkpoint's native final representation.",
    )
    parser.add_argument("--mdbridge-mode", choices=["none", "pointwise", "difference"], default="none")
    parser.add_argument("--mdbridge-middle-layer", type=int, choices=[6], default=6)
    parser.add_argument("--mdbridge-rank", type=int, default=192)
    parser.add_argument("--mdbridge-lags", type=str, default="1,2,4")
    parser.add_argument("--mdbridge-freeze-base", action="store_true")
    parser.add_argument("--mdbridge-disable-fusion", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-lightning-checkpoint", type=str, default=None)
    parser.add_argument(
        "--sample-id-map",
        type=str,
        default=None,
        help="Optional canonical ID map shared with matched training arms.",
    )
    parser.add_argument("--emit-structured-evidence", action="store_true")
    parser.add_argument(
        "--emit-ctc-uncertainty",
        action="store_true",
        help="Export reference-free frame-level CTC uncertainty statistics.",
    )
    parser.add_argument(
        "--emit-perturbation-stability",
        action="store_true",
        help="Export CTC posterior stability under deterministic mild video perturbations.",
    )
    parser.add_argument("--limit-test-samples", type=int, default=None, help="Deterministic prefix limit for smoke tests only")
    parser.add_argument(
        "--skip-text-decoding",
        action="store_true",
        help="Export structured CTC evidence without the expensive attention beam search.",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default=999999,
        help="Level of signal-to-noise ratio (SNR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    seed_everything(args.seed, workers=True)
    modelmodule = get_lightning_module(args)
    if hasattr(modelmodule.model, "xbridge_disable_fusion"):
        modelmodule.model.xbridge_disable_fusion = args.xbridge_disable_fusion
    if hasattr(modelmodule.model, "md_disable_fusion"):
        modelmodule.model.md_disable_fusion = args.mdbridge_disable_fusion
    datamodule = DataModule(args)
    trainer = get_trainer(args)
    trainer.test(model=modelmodule, datamodule=datamodule, ckpt_path=args.checkpoint_path)


if __name__ == "__main__":
    cli_main()
