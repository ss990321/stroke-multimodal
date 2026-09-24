"""How a branch is trained, and what makes each of the three different."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..config import BranchSpec, TrainingSpec
from ..dataset.folds import FoldData, SplitArrays
from ..models.feature_only import FeatureOnlyNet
from ..models.interaction_early_fusion import InteractionEarlyFusion
from ..models.signal_only import SignalOnlyNet

EVAL_BATCH = 128


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def iter_batches(n: int, size: int, generator=None, shuffle: bool = False):
    order = (torch.randperm(n, generator=generator).numpy() if shuffle
             else np.arange(n))
    for start in range(0, n, size):
        batch = order[start:start + size]
        if len(batch) >= 2 or not shuffle:
            yield batch


@dataclass
class TrainingResult:
    predictions: dict
    epochs_run: int
    best_val_loss: float
    n_parameters: int
    state_dict: dict


class Branch:
    """A trainable branch. Subclasses provide `build_model` and `inputs`."""

    name = "branch"

    def __init__(self, spec: BranchSpec, device: str = "cpu",
                 deterministic: bool = False):
        self.spec = spec
        self.device = device
        self.deterministic = deterministic

    def build_model(self, data: FoldData) -> nn.Module:
        raise NotImplementedError

    def inputs(self, data: FoldData, split: SplitArrays, rows) -> tuple:
        """The tensors this branch's forward pass takes, for the given rows."""
        raise NotImplementedError

    def _loss(self, y_train: np.ndarray) -> nn.Module:
        positives = int((y_train == 1).sum())
        negatives = int((y_train == 0).sum())
        ratio = negatives / max(positives, 1)
        if self.spec.pos_weight_cap is not None:
            ratio = min(ratio, self.spec.pos_weight_cap)
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([ratio], dtype=torch.float32,
                                    device=self.device))

    def _validation_loss(self, model, data, split, criterion) -> float:
        model.eval()
        y = torch.tensor(split.y, dtype=torch.float32, device=self.device)
        total = 0.0
        with torch.no_grad():
            for rows in iter_batches(len(split), EVAL_BATCH):
                logits = model(*self.inputs(data, split, rows))
                total += float(criterion(logits, y[rows])) * len(rows)
        return total / len(split)

    def predict(self, model, data: FoldData, split: SplitArrays) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for rows in iter_batches(len(split), EVAL_BATCH):
                logits = model(*self.inputs(data, split, rows))
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)

    def fit(self, data: FoldData, seed: int = 42,
            predict_splits=("val", "test")) -> TrainingResult:
        set_seed(seed, self.deterministic)
        model = self.build_model(data).to(self.device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.spec.lr,
                                      weight_decay=self.spec.weight_decay)
        generator = torch.Generator().manual_seed(seed)

        train, val = data.train, data.val
        criterion = self._loss(train.y)
        y_train = torch.tensor(train.y, dtype=torch.float32, device=self.device)

        best_loss, best_state, bad = np.inf, None, 0
        epochs_run = self.spec.max_epochs
        for epoch in range(1, self.spec.max_epochs + 1):
            model.train()
            for rows in iter_batches(len(train), self.spec.batch_size,
                                     generator, shuffle=True):
                optimiser.zero_grad(set_to_none=True)
                logits = model(*self.inputs(data, train, rows))
                criterion(logits, y_train[rows]).backward()
                optimiser.step()

            val_loss = self._validation_loss(model, data, val, criterion)
            if val_loss < best_loss - self.spec.min_delta:
                best_loss, bad = val_loss, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.spec.patience:
                    epochs_run = epoch
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        predictions = {s: self.predict(model, data, data.split(s))
                       for s in predict_splits}
        result = TrainingResult(
            predictions=predictions,
            epochs_run=epochs_run,
            best_val_loss=float(best_loss),
            n_parameters=sum(p.numel() for p in model.parameters()
                             if p.requires_grad),
            state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        )
        del model
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return result


class SignalOnlyBranch(Branch):
    """Reads the recording."""

    name = "Signal-Only"

    def build_model(self, data: FoldData):
        return SignalOnlyNet(pretrained=True)

    def inputs(self, data: FoldData, split: SplitArrays, rows):
        recordings = data.recordings[split.recording_rows[rows]]
        return (torch.tensor(recordings, device=self.device),)


class FeatureOnlyBranch(Branch):
    """Reads the derived measurements."""

    name = "Feature-Only"

    def build_model(self, data: FoldData):
        return FeatureOnlyNet(in_dim=data.n_features)

    def inputs(self, data: FoldData, split: SplitArrays, rows):
        return (torch.tensor(split.features[rows], device=self.device),)


class InteractionEarlyFusionBranch(Branch):
    """Reads both, and lets them condition one another before classification."""

    name = "Interaction Early Fusion"

    def build_model(self, data: FoldData):
        n = data.n_features
        return InteractionEarlyFusion(
            in_ch_signal=data.recordings.shape[1], tab_dim=n,
            signal_emb_dim=512, tab_emb_dim=128, raw_emb_dim=128,
            fusion_dim=128, dropout=0.2, raw_nhead=4, raw_num_layers=2,
            raw_ff_dim=256, raw_feat_ch=n, raw_downsample_stride=25,
            fusion_type="concat", pretrained=True)

    def inputs(self, data: FoldData, split: SplitArrays, rows):
        recordings = data.recordings[split.recording_rows[rows]]
        return (torch.tensor(recordings, device=self.device),
                torch.tensor(split.features[rows], device=self.device))


BRANCH_CLASSES = {
    "signal": SignalOnlyBranch,
    "feature": FeatureOnlyBranch,
    "interaction": InteractionEarlyFusionBranch,
}


def build_branches(spec: TrainingSpec) -> dict:
    """One configured branch per key, in the order the pipeline runs them."""
    return {
        key: BRANCH_CLASSES[key](getattr(spec, key), device=spec.device,
                                 deterministic=spec.deterministic)
        for key in ("signal", "feature", "interaction")
    }
