# Utils/main.py

import os
import argparse
from typing import Optional

from Utils.config import get_config, DataConfig, ModelConfig, TrainingConfig
from Utils.train import train as train_pipeline
from preprocess import download as dl


def cmd_download(args: argparse.Namespace) -> None:
    """
    Explicit 'download' command: build/refresh cache for the training JSON.
    """
    cfg = get_config()
    data_cfg: DataConfig = cfg["data"]

    # Optional overrides
    if args.data_dir is not None:
        data_cfg.data_dir = args.data_dir
    if args.json is not None:
        data_cfg.json_path = args.json
    if args.cache_dir is not None:
        data_cfg.cache_dir = args.cache_dir

    print("=== Download + landmark extraction (with cache; videos deleted) ===")
    print(f"data_dir  = {data_cfg.data_dir}")
    print(f"json_path = {data_cfg.json_path}")
    print(f"cache_dir = {data_cfg.cache_dir}")

    dl.build_cache(data_config=data_cfg, max_videos=args.max_videos)


def cmd_train(args: argparse.Namespace) -> None:
    """
    'train' command: always run downloader (idempotent) then train.
    """
    cfg = get_config()
    data_cfg: DataConfig = cfg["data"]
    model_cfg: ModelConfig = cfg["model"]
    train_cfg: TrainingConfig = cfg["training"]

    # Optional overrides
    if args.data_dir is not None:
        data_cfg.data_dir = args.data_dir
    if args.train_json is not None:
        data_cfg.json_path = args.train_json
    if args.val_json is not None:
        data_cfg.val_json_path = args.val_json
    if args.cache_dir is not None:
        data_cfg.cache_dir = args.cache_dir

    if args.model_type is not None:
        model_cfg.model_type = args.model_type

    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.no_augment:
        train_cfg.augment = False

    # 1) Always run the downloader first (idempotent).
    print("=== Step 1: Download / build feature cache ===")
    print(f"data_dir  = {data_cfg.data_dir}")
    print(f"json_path = {data_cfg.json_path}")
    print(f"cache_dir = {data_cfg.cache_dir}")
    dl.build_cache(data_config=data_cfg, max_videos=args.max_videos)

    # 2) Then run the existing training pipeline.
    print("\n=== Step 2: Training ===")
    best_acc = train_pipeline(
        data_cfg,
        model_cfg,
        train_cfg,
        resume_from=args.resume,
    )
    print(f"\nBest validation accuracy: {best_acc:.4f}")


def cmd_infer(args: argparse.Namespace) -> None:
    """
    'infer' command: run live MediaPipe + model inference using ASLInference.
    """
    # Lazy import so train/download do NOT depend on mediapipe.solutions
    from Utils.inference import ASLInference

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("=== Live ASL inference ===")
    print(f"model     = {model_path}")
    print(f"camera_id = {args.camera}")
    print(f"threshold = {args.threshold}")
    print(f"window    = {args.window}")

    engine = ASLInference(
        model_path=model_path,
        camera_id=args.camera,
        confidence_threshold=args.threshold,
        prediction_window=args.window,
        num_hands=args.num_hands,
        show_landmarks=not args.no_landmarks,
    )
    engine.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-level ASL project runner (download, train, infer)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download command
    p_dl = subparsers.add_parser(
        "download",
        help="Download MS-ASL clips, extract landmarks, cache, delete videos.",
    )
    p_dl.add_argument("--data_dir", type=str, default=None)
    p_dl.add_argument("--json", type=str, default=None, help="Train annotation JSON")
    p_dl.add_argument("--cache_dir", type=str, default=None)
    p_dl.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional cap on number of videos to process.",
    )
    p_dl.set_defaults(func=cmd_download)

    # train command
    p_tr = subparsers.add_parser(
        "train",
        help="Download/build cache (idempotent), then train model.",
    )
    p_tr.add_argument("--data_dir", type=str, default=None)
    p_tr.add_argument("--train_json", type=str, default=None)
    p_tr.add_argument("--val_json", type=str, default=None)
    p_tr.add_argument("--cache_dir", type=str, default=None)
    p_tr.add_argument(
        "--model_type",
        type=str,
        choices=["lstm", "transformer", "tcn"],
        default=None,
    )
    p_tr.add_argument("--batch_size", type=int, default=None)
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--lr", type=float, default=None)
    p_tr.add_argument("--resume", type=str, default=None)
    p_tr.add_argument("--seed", type=int, default=None)
    p_tr.add_argument("--no_augment", action="store_true")
    p_tr.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional cap on number of videos to cache before training.",
    )
    p_tr.set_defaults(func=cmd_train)

    # infer command
    p_inf = subparsers.add_parser(
        "infer",
        help="Run real-time webcam inference using ASLInference.",
    )
    p_inf.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    p_inf.add_argument("--camera", type=int, default=0)
    p_inf.add_argument("--threshold", type=float, default=0.5)
    p_inf.add_argument("--window", type=int, default=30)
    p_inf.add_argument("--num_hands", type=int, default=2)
    p_inf.add_argument("--no_landmarks", action="store_true")
    p_inf.set_defaults(func=cmd_infer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
