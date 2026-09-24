"""The feature table the models consume: which columns, and where each is from."""

from __future__ import annotations

CLINICAL_FEATURES = ["Gender"]

MUSE_FEATURES = [
    "VentricularRate", "AtrialRate", "PRInterval", "QRSDuration", "QTInterval",
    "QTCorrected", "PAxis", "RAxis", "TAxis", "QRSCount", "QOnset", "QOffset",
    "POnset", "POffset", "TOffset", "QTcFrederica", "GlobalRR",
    "PharmaRRinterval", "PharmaPPinterval",
]

EXTRACTED_FEATURES = [
    "QTc (ms)", "QT (ms)", "QRS (ms)", "ST (ms)", "Impulse factor", "PRQ (ms)",
    "Kurtosis", "R-H (mV)", "Crest factor", "pNN50 (%)", "P-H (mV)",
    "RMSSD (ms)", "SDSD (ms)", "Skewness", "Peak value", "HR (bpm)",
    "RR-I (ms)", "LF/HF", "LF (ms²)", "VLF (ms²)", "HF (ms²)",
]

MISSINGNESS_SOURCES = [
    "PRInterval", "POnset", "POffset", "PAxis", "QTcFrederica",
    "LF/HF", "LF (ms²)", "VLF (ms²)", "HF (ms²)",
]
MISSINGNESS_INDICATORS = [f"{c}_missing" for c in MISSINGNESS_SOURCES]

ALL_FEATURES = (CLINICAL_FEATURES + MUSE_FEATURES + MISSINGNESS_INDICATORS
                + EXTRACTED_FEATURES)

ID_COLUMNS = ["file_name", "PatientID", "label"]

# Section 4.3.1: its missingness tracked the acquisition year.
YEAR_CONFOUNDED = ["QTcFrederica_missing"]

# Section 4.3.4: a 10 s window cannot resolve the VLF or LF band.
UNRESOLVABLE_HRV = [
    "LF/HF", "LF (ms²)", "VLF (ms²)", "HF (ms²)",
    "LF/HF_missing", "LF (ms²)_missing", "VLF (ms²)_missing",
    "HF (ms²)_missing",
]

DROPPED_FEATURES = YEAR_CONFOUNDED + UNRESOLVABLE_HRV

MODEL_FEATURES = [c for c in ALL_FEATURES if c not in set(DROPPED_FEATURES)]


def check_table(columns) -> None:
    """Fail early and by name if a table is missing something the models need."""
    have = set(columns)
    missing = [c for c in ID_COLUMNS + ALL_FEATURES if c not in have]
    if missing:
        raise SystemExit(
            f"the feature table is missing {len(missing)} column(s): "
            + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
