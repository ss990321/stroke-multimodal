"""Getting a recording off disk and into a state worth measuring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

LEAD_II_NAMES = ["II", " II", "LeadII", "lead_II", "DII", "LEAD_II", "Lead II",
                 "Einthoven_II", "MLII", "M_LII", "Lead2", "lead2", "II_lead"]
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6",
                  "LeadI", "LeadII", "LeadIII",
                  "V_1", "V_2", "V_3", "V_4", "V_5", "V_6"]

FULL_THRESHOLD = 1.0
INTERVAL_THRESHOLD = 0.75

RPEAK_METHOD = "pantompkins1985"


def read_ecg_csv(path) -> pd.DataFrame:
    """Numeric columns of a recording, tolerating a semicolon-separated file."""
    df = pd.read_csv(path)
    numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not numeric:
        df = pd.read_csv(path, sep=";")
        numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not numeric:
        raise ValueError(f"no numeric ECG column in {path}")
    return df[numeric]


def pick_analysis_lead(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Lead II if it can be identified, otherwise the first plausible lead."""
    columns = set(df.columns)
    for name in LEAD_II_NAMES:
        if name in columns:
            return df[name].to_numpy(), name
    for c in df.columns:
        low = c.lower().strip()
        if low == "ii" or "lead ii" in low or "lead2" in low:
            return df[c].to_numpy(), c
    for name in STANDARD_LEADS:
        if name in columns:
            return df[name].to_numpy(), name
    names = list(df.columns)
    if names:
        pick = names[1] if len(names) > 1 else names[0]
        return df[pick].to_numpy(), pick
    raise ValueError("no numeric ECG column found")


def denoise(x, wavelet: str = "db8", level: int = 4,
            scale: float = FULL_THRESHOLD) -> np.ndarray:
    """Soft universal-threshold denoising of the detail coefficients."""
    x = np.asarray(x, dtype=float)
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric", level=level)

    kept = [coeffs[0]]
    n = len(x)
    for detail in coeffs[1:]:
        if detail.size == 0:
            kept.append(detail)
            continue
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
        threshold = (sigma * np.sqrt(2 * np.log(n))) * scale if sigma > 0 else 0.0
        kept.append(pywt.threshold(detail, value=threshold, mode="soft"))

    return pywt.waverec(kept, wavelet, mode="symmetric")[:len(x)]


def detect_rpeaks(x, sampling_rate: int, method: str = RPEAK_METHOD) -> np.ndarray:
    """Beat positions, or an empty array if the recording cannot be read."""
    import neurokit2 as nk

    try:
        _, info = nk.ecg_peaks(x, sampling_rate=sampling_rate, method=method)
        return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception:
        return np.array([], dtype=int)


def load_lead(path, n_samples: int | None = None) -> np.ndarray:
    """Read a recording and return its analysis lead, truncated to length."""
    values, _ = pick_analysis_lead(read_ecg_csv(path))
    values = np.asarray(values, dtype=float)
    return values[:n_samples] if n_samples and len(values) >= n_samples else values
