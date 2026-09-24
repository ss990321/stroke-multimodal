"""Stage 1: turn a cohort of recordings into the feature table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .extracted import hrv_frequency, intervals, measures, signal
from .muse import read_muse_xml
from ..schema import (
    ALL_FEATURES,
    MUSE_FEATURES,
    ID_COLUMNS,
    MISSINGNESS_SOURCES,
)

DIAGNOSTIC_FIELDS = ["beats_total", "beats_valid", "qrs_from_delineator"]


def extract_record(file_name: str, cfg: Config) -> tuple[dict, dict]:
    """One recording: its features, and how the measurement went."""
    row: dict = {"file_name": file_name}

    xml_path = cfg.paths.muse_xml_dir / f"{file_name}.xml"
    if xml_path.exists():
        row.update(read_muse_xml(xml_path))
    else:
        row.update({c: np.nan for c in MUSE_FEATURES})
        row["Gender"] = np.nan

    rate_hz = cfg.signal.sampling_rate_hz
    raw = signal.load_lead(cfg.paths.ecg_dir / f"{file_name}.csv",
                           n_samples=cfg.signal.n_samples)
    denoised = signal.denoise(raw)

    row.update(measures.shape_features(denoised))

    rate, rr_ms = measures.rate_and_amplitude_features(denoised, rate_hz)
    row.update(rate)
    row.update(measures.hrv_time_features(rr_ms))
    row.update(_frequency_features(denoised, rate_hz))

    measured = intervals.delineate_intervals(raw, rate_hz)
    for column in ("QRS (ms)", "ST (ms)", "QT (ms)"):
        row[column] = measured[column]
    row["QTc (ms)"] = measures.corrected_qt(row["QT (ms)"], row["RR-I (ms)"])

    diagnostics = {"file_name": file_name,
                   **{k: measured[k] for k in DIAGNOSTIC_FIELDS}}
    return row, diagnostics


def _frequency_features(denoised, sampling_rate: int) -> dict:
    """Band powers; these need beat positions rather than RR intervals."""
    rpeaks = signal.detect_rpeaks(denoised, sampling_rate)
    if rpeaks.size < 3:
        return dict(hrv_frequency.EMPTY)
    return hrv_frequency.hrv_frequency_features(rpeaks, sampling_rate)


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """One indicator per column that can be absent, read off the NaN pattern."""
    out = df.copy()
    for source in MISSINGNESS_SOURCES:
        out[f"{source}_missing"] = out[source].isna().astype(int)
    return out


def summarise_extraction(diagnostics: pd.DataFrame, table: pd.DataFrame) -> dict:
    """How well the waveform measurement went."""
    out = {"records": len(table)}
    if len(diagnostics):
        out["records_with_no_beats"] = int((diagnostics["beats_total"] < 2).sum())
        out["median_beats"] = float(diagnostics["beats_total"].median())
        out["median_valid_beats"] = float(diagnostics["beats_valid"].median())
        out["qrs_from_delineator"] = int(diagnostics["qrs_from_delineator"].sum())
    for column in ("QRS (ms)", "ST (ms)", "QT (ms)", "QTc (ms)"):
        out[f"missing_{column}"] = int(table[column].isna().sum())
    return out


def build_table(cfg: Config, cohort: pd.DataFrame | None = None,
                        progress: bool = True) -> tuple[pd.DataFrame, dict]:
    """Extract every recording named in the cohort file and assemble the table."""
    if cohort is None:
        cohort = pd.read_csv(cfg.paths.cohort_csv)
    for column in ID_COLUMNS:
        if column not in cohort.columns:
            raise SystemExit(f"the cohort file needs a '{column}' column")

    rows, notes, failures = [], [], []
    total = len(cohort)
    for i, file_name in enumerate(cohort["file_name"].astype(str), start=1):
        try:
            row, note = extract_record(file_name, cfg)
            rows.append(row)
            notes.append(note)
        except Exception as exc:
            failures.append({"file_name": file_name, "error": repr(exc)})
        if progress and (i % 100 == 0 or i == total):
            print(f"  {i}/{total} recordings", flush=True)

    if not rows:
        raise SystemExit("no recording could be read; check paths.ecg_dir")

    features = add_missingness_indicators(pd.DataFrame(rows))
    table = cohort[ID_COLUMNS].merge(features, on="file_name", how="inner")
    for column in ALL_FEATURES:
        if column not in table.columns:
            table[column] = np.nan
    table = table[ID_COLUMNS + ALL_FEATURES]

    report = summarise_extraction(pd.DataFrame(notes), table)
    if failures:
        path = cfg.paths.features_csv.with_name("extraction_failures.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(path, index=False, encoding="utf-8-sig")
        report["unreadable_recordings"] = len(failures)
        report["failures_written_to"] = str(path)

    return table, report


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")
