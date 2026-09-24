"""What is measured from the waveform."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from . import signal as ecg_signal


def shape_features(x) -> dict:
    """Amplitude-distribution measures of the whole strip."""
    x = np.asarray(x, dtype=float)
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x))))
    return {
        "Kurtosis": float(kurtosis(x, fisher=False, bias=False)),
        "Skewness": float(skew(x, bias=False)),
        "Peak value": peak,
        "Impulse factor": peak / mean_abs if mean_abs > 0 else np.nan,
        "Crest factor": peak / rms if rms > 0 else np.nan,
    }


def hrv_time_features(rr_ms) -> dict:
    """Time-domain variability of the RR series."""
    out = {"RMSSD (ms)": np.nan, "SDSD (ms)": np.nan, "pNN50 (%)": np.nan}
    if rr_ms is None or len(rr_ms) < 3:
        return out
    diff = np.diff(np.asarray(rr_ms, dtype=float))
    out["RMSSD (ms)"] = float(np.sqrt(np.mean(diff ** 2)))
    if len(diff) > 1:
        out["SDSD (ms)"] = float(np.std(diff, ddof=1))
    out["pNN50 (%)"] = float(np.sum(np.abs(diff) > 50.0) / len(diff) * 100.0)
    return out


def rate_and_amplitude_features(signal, sampling_rate: int = 500
                                ) -> tuple[dict, np.ndarray | None]:
    """Rate, PR interval and wave amplitudes from the delineated waveform."""
    import neurokit2 as nk

    out = {"RR-I (ms)": np.nan, "HR (bpm)": np.nan, "PRQ (ms)": np.nan,
           "P-H (mV)": np.nan, "R-H (mV)": np.nan}
    values = np.asarray(signal, dtype=float)
    try:
        rpeaks = ecg_signal.detect_rpeaks(values, sampling_rate)
        if rpeaks.size < 2:
            return out, None

        rr = np.diff(rpeaks) / sampling_rate * 1000.0
        out["RR-I (ms)"] = float(np.mean(rr))
        out["HR (bpm)"] = float(60_000.0 / np.mean(rr))
        out["R-H (mV)"] = float(np.mean(values[rpeaks]))

        _, waves = nk.ecg_delineate(values, rpeaks=rpeaks,
                                    sampling_rate=sampling_rate, method="dwt")

        p_onsets = _index_array(waves.get("ECG_P_Onsets"), rpeaks.size)
        spans = [(r - p) * 1000.0 / sampling_rate
                 for p, r in zip(p_onsets, rpeaks) if np.isfinite(p) and r > p]
        if spans:
            out["PRQ (ms)"] = float(np.mean(spans))

        p_peaks = _index_array(waves.get("ECG_P_Peaks"), rpeaks.size)
        usable = p_peaks[np.isfinite(p_peaks)].astype(int)
        if usable.size:
            out["P-H (mV)"] = float(np.mean(values[usable]))

        return out, rr
    except Exception:
        return out, None


def _index_array(values, n: int) -> np.ndarray:
    """A delineation series as floats of length n, NaN where absent."""
    out = np.full(n, np.nan, dtype=float)
    if values is None:
        return out
    series = np.asarray(values, dtype=float)
    out[:min(n, series.size)] = series[:n]
    return out


def corrected_qt(qt_ms: float, rr_ms: float) -> float:
    """Bazett's correction."""
    if not np.isfinite(qt_ms) or not np.isfinite(rr_ms) or rr_ms <= 0:
        return np.nan
    return float(qt_ms / np.sqrt(rr_ms / 1000.0))

