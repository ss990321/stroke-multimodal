#!/usr/bin/env python3
"""Stage 4: score recordings with a trained fold."""

from __future__ import annotations

import argparse

import pandas as pd

from stroke_multimodal.config import Config, parse_overrides
from stroke_multimodal.dataset import splits as split_module
from stroke_multimodal.dataset.folds import RecordingStore, build_fold
from stroke_multimodal.evaluation import score, youden_threshold
from stroke_multimodal.inference import load_fold_models, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append",
                        metavar="KEY=VALUE",
                        help="override any config setting")
    parser.add_argument("--fold", type=int, required=True,
                        help="which trained fold to score with")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"],
                        help="which split of the fold to score")
    parser.add_argument("--threshold-from-validation", action="store_true",
                        help="also write a decision at the fold's Youden point")
    args = parser.parse_args()

    cfg = Config.load(args.config, **parse_overrides(args.overrides))

    import torch
    if cfg.training.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.training.device = "cpu"

    checkpoint_dir = cfg.paths.outputs / "checkpoints" / f"fold{args.fold}"
    if not checkpoint_dir.exists():
        raise SystemExit(f"no checkpoints at {checkpoint_dir}; run 03_train.py first")

    table = pd.read_csv(cfg.paths.features_csv)
    folds = split_module.load(cfg.paths.splits_pkl)
    fold = next((f for f in folds if f.index == args.fold), None)
    if fold is None:
        raise SystemExit(f"fold {args.fold} is not in {cfg.paths.splits_pkl}")

    cache = cfg.paths.recording_cache
    if cache is not None and cache.exists():
        store = RecordingStore.from_npz(cache)
    else:
        store = RecordingStore.from_directory(
            cfg.paths.ecg_dir, table["file_name"].astype(str),
            n_leads=cfg.signal.n_leads, n_samples=cfg.signal.n_samples)

    data = build_fold(fold, table, store)
    models = load_fold_models(checkpoint_dir, data, cfg)
    predictions = predict(models, data, split=args.split)

    threshold = None
    if args.threshold_from_validation:
        validation = predict(models, data, split="val")
        threshold = youden_threshold(validation["label"], validation["hybrid_prob"])
        predictions["hybrid_decision"] = (
            predictions["hybrid_prob"] >= threshold).astype(int)
        print(f"operating point from the validation split: {threshold:.4f}")

    out = cfg.paths.outputs / f"predictions_fold{args.fold}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"{len(predictions):,} records scored with fold {args.fold}")
    if threshold is not None and predictions["label"].nunique() > 1:
        metrics = score(predictions["label"], predictions["hybrid_prob"], threshold)
        print("  " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
