"""Frequency-domain heart-rate variability from a short recording."""

from __future__ import annotations

import numpy as np
from scipy import interpolate

RESAMPLE_HZ = 4.0
FMAX_HZ = 1.0
NFFT = 4096

VLF_BAND = (0.0033, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
RELAXED_LF_BAND = (0.07, 0.20)
RELAXED_HF_BAND = (0.20, 0.40)

_integrate = getattr(np, "trapezoid", None) or np.trapz
EMPTY = {"VLF (ms²)": np.nan, "LF (ms²)": np.nan, "HF (ms²)": np.nan,
         "LF/HF": np.nan}


def _even_rr_series(rpeaks, sampling_rate=500.0, resample_hz=RESAMPLE_HZ):
    """RR intervals on a uniform time grid, mean-removed."""
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 3:
        return None, None
    beat_times = rpeaks / float(sampling_rate)
    rr = np.diff(beat_times)
    if np.any(rr <= 0) or len(rr) < 2:
        return None, None
    rr_times = beat_times[1:]
    grid = np.arange(rr_times[0], rr_times[-1], 1.0 / resample_hz)
    if len(grid) < 16:
        return None, None
    spline = interpolate.interp1d(rr_times, rr, kind="cubic",
                                  fill_value="extrapolate", assume_sorted=True)
    even = spline(grid)
    return grid, even - np.mean(even)


def _levinson_durbin(autocorr, order):
    """Solve the Yule-Walker equations; returns the AR coefficients and the"""
    a = np.zeros(order + 1, dtype=float)
    error = autocorr[0]
    if error <= 0 or not np.isfinite(error):
        return None, None
    for k in range(1, order + 1):
        acc = autocorr[k]
        for j in range(1, k):
            acc += a[j] * autocorr[k - j]
        gamma = -acc / error
        previous = a.copy()
        a[k] = gamma
        for j in range(1, k):
            a[j] = previous[j] + gamma * previous[k - j]
        error *= (1.0 - gamma * gamma)
        if error <= 0 or not np.isfinite(error):
            return a[1:k + 1], None
    return a[1:], error


def _ar_spectrum(x, fs, order, nfft=NFFT, fmax=FMAX_HZ):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    if len(x) < order + 2 or not np.all(np.isfinite(x)):
        return None, None
    autocorr = np.array(
        [np.sum(x[:len(x) - k] * x[k:]) / (len(x) - k) for k in range(order + 1)],
        dtype=float)
    ar, noise_var = _levinson_durbin(autocorr, order)
    if ar is None or noise_var is None or not np.isfinite(noise_var):
        return None, None

    n_points = max(256, int(nfft * (fmax / (fs / 2))))
    freqs = np.linspace(0.0, fmax, n_points)
    omega = 2 * np.pi * freqs / fs
    denominator = np.ones_like(freqs, dtype=complex)
    for k, coefficient in enumerate(ar, start=1):
        denominator += coefficient * np.exp(-1j * omega * k)
    psd = (noise_var / (np.abs(denominator) ** 2)).real.clip(min=0)
    return freqs, psd


def _band_power(freqs, psd, low, high) -> float:
    inside = np.where((freqs >= low) & (freqs < high))[0]
    if inside.size < 2:
        return np.nan
    return float(_integrate(psd[inside], freqs[inside]))


def hrv_frequency_features(rpeaks, sampling_rate: float = 500.0,
                           use_relaxed_bands: bool = True) -> dict:
    """Band powers and their ratio; NaN where the spectrum could not be fitted."""
    _, even = _even_rr_series(rpeaks, sampling_rate=sampling_rate)
    if even is None:
        return dict(EMPTY)

    order = int(min(10, max(2, len(even) // 8)))
    freqs, psd = _ar_spectrum(even, fs=RESAMPLE_HZ, order=order)
    if freqs is None:
        return dict(EMPTY)

    vlf = _band_power(freqs, psd, *VLF_BAND)
    lf = _band_power(freqs, psd, *LF_BAND)
    hf = _band_power(freqs, psd, *HF_BAND)

    if use_relaxed_bands and (not np.isfinite(lf) or not np.isfinite(hf)
                              or hf == 0.0):
        lf2 = _band_power(freqs, psd, *RELAXED_LF_BAND)
        hf2 = _band_power(freqs, psd, *RELAXED_HF_BAND)
        if np.isfinite(lf2) and np.isfinite(hf2) and hf2 > 0:
            lf, hf = lf2, hf2

    ratio = (lf / hf) if (np.isfinite(lf) and np.isfinite(hf) and hf > 0) else np.nan
    return {"VLF (ms²)": float(vlf), "LF (ms²)": float(lf),
            "HF (ms²)": float(hf), "LF/HF": float(ratio)}
