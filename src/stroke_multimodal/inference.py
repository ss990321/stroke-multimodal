"""Stage 4: scoring recordings with a trained fold."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .config import Config
from .dataset.folds import FoldData
from .models.hybrid_fusion import fuse
from .train.branches import build_branches
from .train.run import BRANCH_KEYS


def load_fold_models(checkpoint_dir: Path, data: FoldData, cfg: Config) -> dict:
    """Rebuild each branch of a trained fold from its saved weights."""
    names_path = checkpoint_dir / "feature_names.json"
    if names_path.exists():
        saved = json.loads(names_path.read_text(encoding="utf-8"))
        if list(saved) != list(data.feature_names):
            raise SystemExit(
                "the checkpoint was trained on a different feature set:\n"
                f"  trained on {len(saved)} features, given "
                f"{len(data.feature_names)}")

    branches = build_branches(cfg.training)
    models = {}
    for key, branch in branches.items():
        model = branch.build_model(data).to(cfg.training.device)
        model.load_state_dict(torch.load(checkpoint_dir / f"{key}.pt",
                                         map_location=cfg.training.device))
        model.eval()
        models[key] = (branch, model)
    return models


def predict(models: dict, data: FoldData, split: str = "test") -> pd.DataFrame:
    """Score one split with already-trained branch models."""
    arrays = data.split(split)
    probabilities = {key: branch.predict(model, data, arrays)
                     for key, (branch, model) in models.items()}
    probabilities["hybrid"] = fuse({k: probabilities[k] for k in BRANCH_KEYS})
    return pd.DataFrame({
        "file_name": arrays.file_name,
        "label": arrays.y,
        **{f"{k}_prob": probabilities[k] for k in (*BRANCH_KEYS, "hybrid")},
    })
