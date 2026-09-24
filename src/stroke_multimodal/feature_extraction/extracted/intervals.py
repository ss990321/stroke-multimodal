"""QRS, ST and QT, measured beat by beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from . import signal as ecg_signal

QRS_RANGE = (60, 180)
ST_RANGE = (40, 300)
QT_RANGE = (250, 600)

def qrs_bounds_from_derivative(sig, rpeaks, sampling_rate: int,
                               left_ms: int = 120, right_ms: int = 120,
                               quiet_ms: int = 6, threshold_fraction: float = 0.12):
    """QRS onset and offset from where the first derivative goes quiet."""
    x = np.asarray(sig, dtype=float)
    derivative = np.abs(np.diff(scipy_signal.savgol_filter(x, 11, 3)))

    left = int(left_ms * sampling_rate / 1000)
    right = int(right_ms * sampling_rate / 1000)
    quiet = max(1, int(quiet_ms * sampling_rate / 1000))

    onsets, offsets = [], []
    for r in np.asarray(rpeaks, dtype=int):
        lo = max(1, r - left)
        hi = min(len(derivative) - 1, r + right - 1)
        if r <= 1 or hi <= lo:
            onsets.append(np.nan)
            offsets.append(np.nan)
            continue

        before, after = derivative[lo:r - 1], derivative[r - 1:hi]
        if len(before) < quiet or len(after) < quiet:
            onsets.append(np.nan)
            offsets.append(np.nan)
            continue

        threshold_before = float(np.max(before) * threshold_fraction)
        threshold_after = float(np.max(after) * threshold_fraction)

        onsets.append(lo + _last_quiet_run(before, threshold_before, quiet))
        found = _first_quiet_run(after, threshold_after, quiet)
        offsets.append(r - 1 + found if np.isfinite(found) else np.nan)

    return np.asarray(onsets, dtype=float), np.asarray(offsets, dtype=float)


def _last_quiet_run(segment, threshold, length) -> float:
    run = 0
    for i in range(len(segment) - 1, -1, -1):
        run = run + 1 if segment[i] < threshold else 0
        if run >= length:
            return float(i - length + 1)
    return np.nan


def _first_quiet_run(segment, threshold, length) -> float:
    run = 0
    for i in range(len(segment)):
        run = run + 1 if segment[i] < threshold else 0
        if run >= length:
            return float(i)
    return np.nan


def _as_array(waves: dict, key: str, n: int) -> np.ndarray:
    """A delineation series as floats of length n, NaN where absent."""
    values = waves.get(key)
    if values is None:
        return np.full(n, np.nan, dtype=float)
    out = np.full(n, np.nan, dtype=float)
    series = np.asarray(values, dtype=float)
    out[:min(n, series.size)] = series[:n]
    return out


def _to_ms(start, end, sampling_rate) -> float:
    if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
        return np.nan
    return (end - start) * 1000.0 / sampling_rate


def delineate_intervals(signal_raw, sampling_rate: int = 500) -> dict:
    """Per-beat QRS, ST and QT, and the recording-level summary."""
    import neurokit2 as nk

    empty = {"QRS (ms)": np.nan, "ST (ms)": np.nan, "QT (ms)": np.nan,
             "beats_total": 0, "beats_valid": 0, "qrs_from_delineator": 0}

    denoised = ecg_signal.denoise(signal_raw,
                                  scale=ecg_signal.INTERVAL_THRESHOLD)
    rpeaks = ecg_signal.detect_rpeaks(denoised, sampling_rate)
    if rpeaks.size < 2:
        return empty

    try:
        _, waves = nk.ecg_delineate(denoised, rpeaks=rpeaks,
                                    sampling_rate=sampling_rate, method="dwt")
    except Exception:
        waves = {}

    n = rpeaks.size
    q_on = _as_array(waves, "ECG_Q_Onsets", n)
    s_off = _as_array(waves, "ECG_S_Offsets", n)
    t_on = _as_array(waves, "ECG_T_Onsets", n)
    t_off = _as_array(waves, "ECG_T_Offsets", n)

    from_delineator = int(np.isfinite(q_on).sum() + np.isfinite(s_off).sum())

    fallback_on, fallback_off = qrs_bounds_from_derivative(
        denoised, rpeaks, sampling_rate)
    q_on = np.where(np.isfinite(q_on), q_on, fallback_on)
    s_off = np.where(np.isfinite(s_off), s_off, fallback_off)

    per_beat = pd.DataFrame({
        "QRS (ms)": [_to_ms(q_on[i], s_off[i], sampling_rate) for i in range(n)],
        "ST (ms)": [_to_ms(s_off[i], t_on[i], sampling_rate) for i in range(n)],
        "QT (ms)": [_to_ms(q_on[i], t_off[i], sampling_rate) for i in range(n)],
    })

    keep = (per_beat["QRS (ms)"].between(*QRS_RANGE, inclusive="both")
            & per_beat["ST (ms)"].between(*ST_RANGE, inclusive="both")
            & per_beat["QT (ms)"].between(*QT_RANGE, inclusive="both"))

    out = dict(empty)
    out["beats_total"] = n
    out["beats_valid"] = int(keep.sum())
    out["qrs_from_delineator"] = from_delineator
    if keep.any():
        for column in ("QRS (ms)", "ST (ms)", "QT (ms)"):
            out[column] = float(np.nanmedian(per_beat.loc[keep, column]))
    return out
