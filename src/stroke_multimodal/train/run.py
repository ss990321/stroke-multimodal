"""Stage 3: running the experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..dataset.folds import FoldData, RecordingStore, build_fold
from ..dataset.splits import Fold
from ..evaluation import METRICS, score_with_validation_threshold, summarise
from ..models.hybrid_fusion import fuse
from .branches import build_branches

BRANCH_KEYS = ("signal", "feature", "interaction")
BRANCH_LABELS = {"signal": "Signal-Only", "feature": "Feature-Only",
                 "interaction": "Interaction Early Fusion",
                 "hybrid": "Hybrid Fusion"}


@dataclass
class FoldResult:
    index: int
    predictions: dict
    metrics: pd.DataFrame
    epochs: dict
    checkpoints: dict


def run_fold(data: FoldData, cfg: Config, save_checkpoints_to: Path | None = None
             ) -> FoldResult:
    """Train every branch on one fold, fuse them, and score the result."""
    branches = build_branches(cfg.training)

    probabilities: dict = {"val": {}, "test": {}}
    epochs, checkpoints = {}, {}
    for key in BRANCH_KEYS:
        result = branches[key].fit(data, seed=cfg.training.seed)
        for split in ("val", "test"):
            probabilities[split][key] = result.predictions[split]
        epochs[key] = result.epochs_run
        checkpoints[key] = result.state_dict

    for split in ("val", "test"):
        probabilities[split]["hybrid"] = fuse(
            {k: probabilities[split][k] for k in BRANCH_KEYS})

    frames = {}
    for split in ("val", "test"):
        arrays = data.split(split)
        frames[split] = pd.DataFrame({
            "fold": data.index,
            "file_name": arrays.file_name,
            "label": arrays.y,
            **{f"{k}_prob": probabilities[split][k]
               for k in (*BRANCH_KEYS, "hybrid")},
        })

    rows = []
    for key in (*BRANCH_KEYS, "hybrid"):
        rows.append({
            "fold": data.index,
            "model": BRANCH_LABELS[key],
            "epochs": epochs.get(key, np.nan),
            **score_with_validation_threshold(
                data.val.y, probabilities["val"][key],
                data.test.y, probabilities["test"][key]),
        })

    if save_checkpoints_to is not None:
        directory = save_checkpoints_to / f"fold{data.index}"
        directory.mkdir(parents=True, exist_ok=True)
        for key, state in checkpoints.items():
            torch.save(state, directory / f"{key}.pt")
        (directory / "feature_names.json").write_text(
            json.dumps(data.feature_names, ensure_ascii=False, indent=2),
            encoding="utf-8")

    return FoldResult(index=data.index, predictions=frames,
                      metrics=pd.DataFrame(rows), epochs=epochs,
                      checkpoints=checkpoints)


def run_cross_validation(folds: list[Fold], table: pd.DataFrame,
                         store: RecordingStore, cfg: Config,
                         save_checkpoints: bool = True,
                         progress: bool = True) -> dict:
    """Every fold, with the pooled out-of-fold predictions."""
    checkpoint_root = cfg.paths.outputs / "checkpoints" if save_checkpoints else None

    per_fold, pooled, val_frames = [], [], []
    for fold in folds:
        data = build_fold(fold, table, store)
        result = run_fold(data, cfg, save_checkpoints_to=checkpoint_root)
        per_fold.append(result.metrics)
        pooled.append(result.predictions["test"])
        val_frames.append(result.predictions["val"])
        if progress:
            line = "  ".join(
                f"{row['model']} {row['auc']:.4f}"
                for _, row in result.metrics.iterrows())
            print(f"fold {fold.index}: {line}", flush=True)

    metrics = pd.concat(per_fold, ignore_index=True)
    pooled_predictions = pd.concat(pooled, ignore_index=True)

    duplicates = len(pooled_predictions) - pooled_predictions["file_name"].nunique()
    if duplicates:
        print(f"warning: {duplicates} duplicate record(s) in the pooled "
              "predictions", flush=True)

    return {
        "per_fold": metrics,
        "summary": summarise(metrics, metrics=METRICS + ["fpr_at_target_sensitivity"]),
        "pooled_test": pooled_predictions,
        "val_predictions": pd.concat(val_frames, ignore_index=True),
    }
