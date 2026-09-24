"""Checks on the properties that fail quietly.

These are not accuracy tests - synthetic data cannot say whether the models are
any good. They test the things that would otherwise break without anyone
noticing: that the splits do not leak, that what is fitted on training data
stays fitted on training data, that reloading a checkpoint reproduces the
predictions it was saved from, and that the fusion does what it claims.

    python -m pytest tests/ -q
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


from stroke_multimodal.config import Config
from stroke_multimodal.dataset import splits as split_module
from stroke_multimodal.dataset.folds import RecordingStore, build_fold
from stroke_multimodal.schema import (
    ALL_FEATURES,
    MISSINGNESS_SOURCES,
    MODEL_FEATURES,
)
from stroke_multimodal.models.hybrid_fusion import fuse, to_logit, to_probability
from stroke_multimodal.inference import load_fold_models, predict
from stroke_multimodal.train.run import run_fold

N_RECORDS = 80
N_LEADS = 12
N_SAMPLES = 500


@pytest.fixture(scope="module")
def cohort():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, N_RECORDS)
    table = pd.DataFrame({
        "file_name": [f"r{i:03d}" for i in range(N_RECORDS)],
        "PatientID": np.arange(N_RECORDS),
        "label": labels,
    })
    for column in ALL_FEATURES:
        if not column.endswith("_missing"):
            table[column] = rng.normal(0, 1, N_RECORDS) + 0.4 * labels
    for source in MISSINGNESS_SOURCES:
        absent = rng.random(N_RECORDS) < 0.15
        table.loc[absent, source] = np.nan
        table[f"{source}_missing"] = absent.astype(int)
    table = table[["file_name", "PatientID", "label"] + ALL_FEATURES]

    recordings = rng.normal(0, 1, (N_RECORDS, N_LEADS, N_SAMPLES)).astype(np.float32)
    store = RecordingStore(recordings,
                           {n: i for i, n in enumerate(table["file_name"])})
    return table, store


@pytest.fixture(scope="module")
def folds(cohort):
    table, _ = cohort
    return split_module.build_folds(table, n_folds=5, seed=42)


# --------------------------------------------------------------------- splits
def test_split_has_no_leakage(folds, cohort):
    table, _ = cohort
    _, checks = split_module.audit(folds, table)
    split_module.assert_clean(checks)
    assert checks["total_patient_leakage"] == 0
    assert not any(checks["test_patient_overlaps"])


def test_every_patient_is_held_out_exactly_once(folds, cohort):
    table, _ = cohort
    held = [p for fold in folds
            for p in fold.test_patients + fold.val_patients]
    assert len(held) == len(set(held)), "a patient is held out by two folds"
    assert set(held) == set(table["PatientID"]), "a patient is never held out"


def test_pooled_predictions_have_one_row_per_record(folds):
    pooled = [f for fold in folds for f in fold.test_files]
    assert len(pooled) == len(set(pooled))


def test_a_patients_recordings_never_straddle_a_split(cohort):
    """Two recordings of one patient must land on the same side."""
    table, _ = cohort
    doubled = table.copy()
    doubled["PatientID"] = doubled["PatientID"] // 2      # pair them up
    # A patient's label is a property of the patient, so the paired recordings
    # have to agree; the splitter refuses a cohort where they do not.
    doubled["label"] = doubled.groupby("PatientID")["label"].transform("first")
    built = split_module.build_folds(doubled, n_folds=5, seed=42)
    patient_of = dict(zip(doubled["file_name"], doubled["PatientID"]))
    for fold in built:
        sides = {}
        for split in ("train", "val", "test"):
            for f in fold.files(split):
                sides.setdefault(patient_of[f], set()).add(split)
        assert all(len(s) == 1 for s in sides.values())


# ----------------------------------------------------------- fold assembly
def test_imputation_and_scaling_are_fitted_on_training_only(folds, cohort):
    """Changing a test value must not change any training feature."""
    table, store = cohort
    data = build_fold(folds[0], table, store)

    altered = table.copy()
    test_files = set(folds[0].test_files)
    mask = altered["file_name"].isin(test_files)
    for column in MODEL_FEATURES:
        if not column.endswith("_missing"):
            altered.loc[mask, column] = altered.loc[mask, column] * 1000 + 500

    shifted = build_fold(folds[0], altered, store)
    np.testing.assert_allclose(data.train.features, shifted.train.features,
                               rtol=1e-6, atol=1e-6)


def test_dropped_features_are_absent(folds, cohort):
    table, store = cohort
    data = build_fold(folds[0], table, store)
    assert "QTcFrederica_missing" not in data.feature_names
    assert "LF/HF" not in data.feature_names
    assert len(data.feature_names) == len(MODEL_FEATURES)


def test_no_missing_values_reach_the_model(folds, cohort):
    table, store = cohort
    data = build_fold(folds[0], table, store)
    for split in ("train", "val", "test"):
        assert np.isfinite(data.split(split).features).all()


# --------------------------------------------------------------------- fusion
def test_equal_weight_fusion_is_the_mean_of_logits():
    a = np.array([0.2, 0.6, 0.9])
    b = np.array([0.4, 0.5, 0.1])
    c = np.array([0.7, 0.3, 0.5])
    expected = to_probability((to_logit(a) + to_logit(b) + to_logit(c)) / 3)
    np.testing.assert_allclose(fuse({"a": a, "b": b, "c": c}), expected)


def test_fusion_weights_are_normalised():
    a = np.array([0.2, 0.8])
    b = np.array([0.6, 0.4])
    np.testing.assert_allclose(fuse({"a": a, "b": b}, {"a": 2, "b": 2}),
                               fuse({"a": a, "b": b}))


def test_fusion_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fuse({"a": np.array([0.5]), "b": np.array([0.5, 0.5])})


# ------------------------------------------------------- train then reload
def test_reloaded_checkpoints_reproduce_their_predictions(folds, cohort, tmp_path):
    table, store = cohort
    cfg = Config.load()
    cfg.paths.outputs = tmp_path
    cfg.training.device = "cpu"
    for branch in (cfg.training.signal, cfg.training.feature,
                   cfg.training.interaction):
        branch.max_epochs = 1
        branch.patience = 1

    data = build_fold(folds[0], table, store)
    trained = run_fold(data, cfg, save_checkpoints_to=tmp_path / "checkpoints")

    models = load_fold_models(tmp_path / "checkpoints" / "fold0", data, cfg)
    reloaded = predict(models, data, split="test")

    for column in ("signal_prob", "feature_prob", "interaction_prob",
                   "hybrid_prob"):
        np.testing.assert_allclose(
            trained.predictions["test"][column].to_numpy(),
            reloaded[column].to_numpy(), rtol=1e-5, atol=1e-6,
            err_msg=f"{column} changed when the checkpoint was reloaded")


def test_loading_a_checkpoint_with_the_wrong_feature_set_is_refused(
        folds, cohort, tmp_path):
    table, store = cohort
    cfg = Config.load()
    cfg.training.device = "cpu"
    for branch in (cfg.training.signal, cfg.training.feature,
                   cfg.training.interaction):
        branch.max_epochs = 1
        branch.patience = 1

    data = build_fold(folds[0], table, store)
    run_fold(data, cfg, save_checkpoints_to=tmp_path / "checkpoints")

    fewer = build_fold(folds[0], table, store,
                       feature_names=MODEL_FEATURES[:-5])
    with pytest.raises(SystemExit):
        load_fold_models(tmp_path / "checkpoints" / "fold0", fewer, cfg)


def test_mixed_labels_within_a_patient_are_refused(cohort):
    """The guard the previous test works around: it must actually fire."""
    table, _ = cohort
    mixed = table.copy()
    mixed["PatientID"] = 0
    with pytest.raises(SystemExit):
        split_module.build_folds(mixed, n_folds=5, seed=42)
