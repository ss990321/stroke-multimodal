"""Assembling one fold's arrays, with everything fitted on training records only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import MODEL_FEATURES, check_table
from .splits import Fold


@dataclass
class SplitArrays:
    """One split of one fold."""

    file_name: np.ndarray
    y: np.ndarray
    features: np.ndarray
    recording_rows: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class FoldData:
    index: int
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    feature_names: list
    recordings: np.ndarray

    def split(self, name: str) -> SplitArrays:
        return getattr(self, name)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


class RecordingStore:
    """Every recording in one array, addressed by file name."""

    def __init__(self, recordings: np.ndarray, index: dict):
        self.recordings = recordings
        self.index = index

    @classmethod
    def from_npz(cls, path: Path) -> "RecordingStore":
        data = np.load(path, allow_pickle=True)
        key = "ecg" if "ecg" in data.files else "X"
        recordings = np.asarray(data[key], dtype=np.float32)
        names = np.asarray(data["file_name"]).astype(str).tolist()
        index = {name: i for i, name in enumerate(names)}
        if len(index) != len(names):
            raise SystemExit(f"{path} holds duplicate file names")
        return cls(recordings, index)

    @classmethod
    def from_directory(cls, directory: Path, file_names, n_leads: int,
                       n_samples: int) -> "RecordingStore":
        from ..feature_extraction.extracted.signal import read_ecg_csv

        file_names = [str(f) for f in file_names]
        recordings = np.zeros((len(file_names), n_leads, n_samples),
                              dtype=np.float32)
        for i, name in enumerate(file_names):
            frame = read_ecg_csv(directory / f"{name}.csv")
            values = frame.to_numpy(dtype=np.float32).T
            leads = min(n_leads, values.shape[0])
            samples = min(n_samples, values.shape[1])
            recordings[i, :leads, :samples] = values[:leads, :samples]
        return cls(recordings, {n: i for i, n in enumerate(file_names)})

    def rows_for(self, file_names) -> np.ndarray:
        missing = [f for f in file_names if f not in self.index]
        if missing:
            raise SystemExit(
                f"{len(missing)} recording(s) are in the split but not in the "
                f"store, e.g. {missing[:3]}")
        return np.array([self.index[f] for f in file_names], dtype=int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        names = [None] * len(self.index)
        for name, i in self.index.items():
            names[i] = name
        np.savez_compressed(path, ecg=self.recordings,
                            file_name=np.array(names, dtype=object))


def build_fold(fold: Fold, table: pd.DataFrame, store: RecordingStore,
               feature_names: list | None = None) -> FoldData:
    """Impute, standardise and index one fold."""
    check_table(table.columns)
    names = list(feature_names if feature_names is not None else MODEL_FEATURES)

    indexed = table.set_index(table["file_name"].astype(str))
    train_rows = indexed.loc[fold.files("train")]

    medians = train_rows[names].median(numeric_only=True)

    raw = {}
    for split in ("train", "val", "test"):
        rows = indexed.loc[fold.files(split)]
        raw[split] = rows[names].fillna(medians).to_numpy(dtype=np.float32)

    mean = raw["train"].mean(axis=0)
    sd = raw["train"].std(axis=0)
    sd[sd < 1e-12] = 1.0

    arrays = {}
    for split in ("train", "val", "test"):
        files = [str(f) for f in fold.files(split)]
        standardised = (raw[split] - mean) / sd
        standardised[~np.isfinite(standardised)] = 0.0
        arrays[split] = SplitArrays(
            file_name=np.array(files, dtype=object),
            y=indexed.loc[files, "label"].to_numpy().astype(np.int64),
            features=standardised.astype(np.float32),
            recording_rows=store.rows_for(files),
        )

    return FoldData(index=fold.index, train=arrays["train"], val=arrays["val"],
                    test=arrays["test"], feature_names=names,
                    recordings=store.recordings)
