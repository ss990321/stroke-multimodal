"""Stage 2: patient-level cross-validation splits."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class Fold:
    index: int
    seed: int
    train_patients: list = field(default_factory=list)
    val_patients: list = field(default_factory=list)
    test_patients: list = field(default_factory=list)
    train_files: list = field(default_factory=list)
    val_files: list = field(default_factory=list)
    test_files: list = field(default_factory=list)

    def files(self, split: str) -> list:
        return getattr(self, f"{split}_files")

    def patients(self, split: str) -> list:
        return getattr(self, f"{split}_patients")


def build_folds(table: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                stratify: bool = True) -> list[Fold]:
    """Partition patients, then halve each held-out fold into test and val."""
    labels_per_patient = table.groupby("PatientID")["label"].nunique()
    if (labels_per_patient > 1).any():
        raise SystemExit(
            "a patient carries more than one label; stratifying patients is "
            "not well defined until that is resolved")

    patients = table.groupby("PatientID")["label"].first().sort_index()
    ids = patients.index.to_numpy()
    labels = patients.to_numpy().astype(int)

    files_of_patient = (table.groupby("PatientID")["file_name"]
                        .apply(lambda s: s.astype(str).tolist()).to_dict())

    def files_for(patient_ids):
        return [f for pid in patient_ids for f in files_of_patient[pid]]

    splitter = (StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                if stratify
                else KFold(n_splits=n_folds, shuffle=True, random_state=seed))

    folds = []
    for k, (_, held) in enumerate(splitter.split(ids, labels)):
        held_ids, held_labels = ids[held], labels[held]
        test_ids, val_ids = train_test_split(
            held_ids, test_size=0.5, random_state=seed,
            stratify=held_labels if stratify else None)
        train_ids = np.setdiff1d(ids, held_ids)

        folds.append(Fold(
            index=k, seed=seed,
            train_patients=sorted(int(x) for x in train_ids),
            val_patients=sorted(int(x) for x in val_ids),
            test_patients=sorted(int(x) for x in test_ids),
            train_files=files_for(train_ids),
            val_files=files_for(val_ids),
            test_files=files_for(test_ids),
        ))
    return folds


def audit(folds: list[Fold], table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-fold sizes and balance, plus the design-wide leakage checks."""
    label_of = dict(zip(table["file_name"].astype(str), table["label"].astype(int)))

    rows = []
    for fold in folds:
        row = {"fold": fold.index}
        for split in ("train", "val", "test"):
            files = fold.files(split)
            row[f"n_{split}"] = len(files)
            row[f"positive_{split}"] = (
                round(float(np.mean([label_of[f] for f in files])), 4)
                if files else np.nan)
        sets = [set(fold.patients(s)) for s in ("train", "val", "test")]
        row["patient_leakage"] = sum(len(a & b) for a, b in combinations(sets, 2))
        rows.append(row)

    test_patients = [set(f.test_patients) for f in folds]
    all_patients = set(table["PatientID"].astype(int))
    held_out = set().union(*test_patients) | set().union(
        *[set(f.val_patients) for f in folds])
    pooled = [f for fold in folds for f in fold.test_files]

    checks = {
        "patients": len(all_patients),
        "patients_ever_tested": len(set().union(*test_patients)),
        "patients_never_held_out": len(all_patients - held_out),
        "test_patient_overlaps": [len(a & b)
                                  for a, b in combinations(test_patients, 2)],
        "pooled_test_rows": len(pooled),
        "pooled_test_distinct": len(set(pooled)),
        "total_patient_leakage": int(sum(r["patient_leakage"] for r in rows)),
    }
    return pd.DataFrame(rows), checks


def assert_clean(checks: dict) -> None:
    """Stop if the split does not have the properties it is meant to have."""
    problems = []
    if checks["total_patient_leakage"]:
        problems.append(
            f"{checks['total_patient_leakage']} patient(s) appear in more than "
            "one split of the same fold")
    if any(checks["test_patient_overlaps"]):
        problems.append(
            f"test sets overlap: {checks['test_patient_overlaps']}")
    if checks["pooled_test_rows"] != checks["pooled_test_distinct"]:
        duplicates = checks["pooled_test_rows"] - checks["pooled_test_distinct"]
        problems.append(
            f"{duplicates} duplicate row(s) in the pooled test predictions")
    if problems:
        raise SystemExit("the split is not clean:\n  - " + "\n  - ".join(problems))


def save(folds: list[Fold], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(folds, handle)


def load(path: Path) -> list[Fold]:
    with open(path, "rb") as handle:
        return pickle.load(handle)
