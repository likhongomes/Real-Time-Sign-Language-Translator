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

    dl.build_cache(
        data_config=data_cfg,
        max_videos=args.max_videos,
    )


def _maybe_build_cache_for_train_and_val(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    max_videos: Optional[int],
    no_cache: bool,
) -> None:
    """
    Centralized logic for cache building before training.

    - If no_cache is True -> skip everything, rely only on existing cache.
    - If max_videos == 0   -> also skip, same behavior.
    - Else                 -> call the cache builder, which handles train/val internally.
    """
    if no_cache or (max_videos is not None and max_videos == 0):
        print("=== Step 1: Download / build feature cache ===")
        print("Skipping cache build (no_cache or max_videos=0); using existing landmark_cache only.")
        return

    print("=== Step 1: Download / build feature cache ===")
    print(f"data_dir  = {data_cfg.data_dir}")
    print(f"json_path = {data_cfg.json_path}")
    print(f"val_json  = {data_cfg.val_json_path}")
    print(f"cache_dir = {data_cfg.cache_dir}")

    dl.build_cache(
        data_config=data_cfg,
        max_videos=max_videos,
    )


def _run_single_model(
    model_type: str,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    args: argparse.Namespace,
) -> float:
    """
    Helper to train a single model_type and return best validation accuracy.
    """
    model_cfg.model_type = model_type
    print(f"\nRunning single model_type: {model_type}")

    _maybe_build_cache_for_train_and_val(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        max_videos=args.max_videos,
        no_cache=args.no_cache,
    )

    print("\n=== Step 2: Training ===")
    best_acc = train_pipeline(
        data_cfg,
        model_cfg,
        train_cfg,
        resume_from=args.resume,
    )
    print(f"\nBest validation accuracy ({model_type}): {best_acc:.4f}")
    return best_acc


def cmd_train(args: argparse.Namespace) -> None:
    """
    'train' command:

    - Optionally (idempotently) build cache for train/val unless --no_cache or max_videos=0.
    - Then train either:
        * a single model_type, or
        * all three: transformer, lstm, tcn (if --run3 is set).
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

    # Multi-model run: transformer -> lstm -> tcn
    if args.run3:
        all_models = ["transformer", "lstm", "tcn"]
        results = {}

        # One cache pass for all three (unless skipped)
        _maybe_build_cache_for_train_and_val(
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            max_videos=args.max_videos,
            no_cache=args.no_cache,
        )

        for m in all_models:
            model_cfg.model_type = m
            print(f"\n=== Step 2: Training model_type={m} ===")
            best_acc = train_pipeline(
                data_cfg,
                model_cfg,
                train_cfg,
                resume_from=args.resume,
            )
            results[m] = best_acc
            print(f"\nBest validation accuracy ({m}): {best_acc:.4f}")

        print("\n=== Summary over 3 models ===")
        for m, acc in results.items():
            print(f"{m:12s}: {acc:.4f}")

        # Pick "best" as the max over the three
        best_model = max(results.items(), key=lambda kv: kv[1])
        print(f"\nBest overall: {best_model[0]} with val acc {best_model[1]:.4f}")
        return

    # Single-model run
    if args.model_type is not None:
        model_cfg.model_type = args.model_type

    _run_single_model(
        model_type=model_cfg.model_type,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        args=args,
    )


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
        help="(Optionally) build cache, then train model(s).",
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
        help="Which model to run (ignored if --run3 is set).",
    )
    p_tr.add_argument(
        "--run3",
        action="store_true",
        help="Train/evaluate all three models (transformer, lstm, tcn) in sequence.",
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
        help=(
            "Cap on number of videos to cache. "
            "If set to 0, cache building is entirely skipped."
        ),
    )
    p_tr.add_argument(
        "--no_cache",
        action="store_true",
        help="Skip all cache building and rely only on existing landmark_cache.",
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
