"""Checks on the waveform measurement.

The interval extractor is the part of this pipeline most exposed to a
dependency change: it asks neurokit2 for QRS boundary keys that neurokit2 does
not return, and relies on its own derivative estimator instead. If a future
version starts returning those keys, the numbers change and nothing else would
say so. These tests pin that behaviour down.

Simulated recordings are used, so the assertions are physiological ranges rather
than exact values - the point is that the measurement produces something real
rather than a column of NaN.

    python -m pytest tests/test_features.py -q
"""

from __future__ import annotations


import numpy as np
import pytest


nk = pytest.importorskip("neurokit2", reason="stage 1 only")

from stroke_multimodal.feature_extraction.extracted import (
    hrv_frequency,
    intervals,
    measures,
    signal,
)

SAMPLING_RATE = 500


@pytest.fixture(scope="module")
def strip():
    return nk.ecg_simulate(duration=10, sampling_rate=SAMPLING_RATE,
                           heart_rate=72, random_state=1)


def test_intervals_are_physiological(strip):
    out = intervals.delineate_intervals(strip, SAMPLING_RATE)
    assert intervals.QRS_RANGE[0] <= out["QRS (ms)"] <= intervals.QRS_RANGE[1]
    assert intervals.ST_RANGE[0] <= out["ST (ms)"] <= intervals.ST_RANGE[1]
    assert intervals.QT_RANGE[0] <= out["QT (ms)"] <= intervals.QT_RANGE[1]
    assert out["beats_valid"] >= 5


def test_the_derivative_estimator_is_what_measures_qrs(strip):
    """neurokit2 does not supply the QRS boundaries, so every beat falls back.

    If this fails, the installed neurokit2 has started returning
    ECG_Q_Onsets/ECG_S_Offsets. That is not a bug in this package, but the QRS,
    ST and QT columns it then produces are not comparable with the study's, and
    the change has to be reported rather than absorbed silently.
    """
    out = intervals.delineate_intervals(strip, SAMPLING_RATE)
    assert out["qrs_from_delineator"] == 0, (
        "this neurokit2 returns QRS boundary keys the study's version did not; "
        "QRS, ST and QT are no longer comparable with the published table")


def test_a_flat_recording_yields_no_intervals():
    flat = np.zeros(10 * SAMPLING_RATE)
    out = intervals.delineate_intervals(flat, SAMPLING_RATE)
    assert np.isnan(out["QRS (ms)"])
    assert out["beats_valid"] == 0


def test_rate_and_amplitudes_are_plausible(strip):
    denoised = signal.denoise(strip)
    out, rr = measures.rate_and_amplitude_features(denoised, SAMPLING_RATE)
    assert rr is not None and len(rr) >= 5
    assert 60 <= out["HR (bpm)"] <= 85          # simulated at 72
    assert 700 <= out["RR-I (ms)"] <= 1000
    assert out["R-H (mV)"] > out["P-H (mV)"]    # R towers over P


def test_rate_features_survive_an_unsegmentable_recording():
    out, rr = measures.rate_and_amplitude_features(
        np.zeros(10 * SAMPLING_RATE), SAMPLING_RATE)
    assert rr is None
    assert all(np.isnan(v) for v in out.values())


def test_bazett_correction():
    # At RR = 1000 ms the correction is the identity.
    assert measures.corrected_qt(400.0, 1000.0) == pytest.approx(400.0)
    # Faster rate lengthens the corrected value.
    assert measures.corrected_qt(400.0, 600.0) > 400.0
    assert np.isnan(measures.corrected_qt(np.nan, 800.0))
    assert np.isnan(measures.corrected_qt(400.0, 0.0))


def test_shape_features_need_no_fiducial_points():
    """They must work on a recording the delineator cannot segment."""
    out = measures.shape_features(np.random.default_rng(0).normal(0, 1, 5000))
    assert all(np.isfinite(v) for v in out.values())


def test_frequency_features_are_nan_without_enough_beats():
    out = hrv_frequency.hrv_frequency_features(np.array([100, 600]), SAMPLING_RATE)
    assert all(np.isnan(v) for v in out.values())


def test_frequency_features_return_the_four_columns(strip):
    _, info = nk.ecg_peaks(strip, sampling_rate=SAMPLING_RATE)
    out = hrv_frequency.hrv_frequency_features(info["ECG_R_Peaks"], SAMPLING_RATE)
    assert set(out) == {"VLF (ms²)", "LF (ms²)", "HF (ms²)", "LF/HF"}


def test_lead_two_is_preferred_when_present():
    import pandas as pd

    frame = pd.DataFrame({"I": [1.0, 2.0], "II": [3.0, 4.0], "V1": [5.0, 6.0]})
    values, name = signal.pick_analysis_lead(frame)
    assert name == "II"
    np.testing.assert_array_equal(values, [3.0, 4.0])


def test_zero_valued_muse_measurements_become_missing(tmp_path):
    """The cart writes 0 for a measurement it did not make."""
    from stroke_multimodal.feature_extraction.muse import read_muse_xml

    xml = tmp_path / "r.xml"
    xml.write_text(
        "<RestingECG><RestingECGMeasurements>"
        "<PRInterval>0</PRInterval><QRSDuration>92</QRSDuration>"
        "<VentricularRate>0</VentricularRate>"
        "</RestingECGMeasurements>"
        "<PatientDemographics><Gender>FEMALE</Gender></PatientDemographics>"
        "</RestingECG>", encoding="utf-8")

    out = read_muse_xml(xml)
    assert np.isnan(out["PRInterval"]), "a PR interval of 0 is absent, not short"
    assert out["QRSDuration"] == 92
    assert out["Gender"] == 0
    # A ventricular rate of 0 is not in the zero-means-missing list, because
    # zero there would be a genuine (if alarming) reading.
    assert out["VentricularRate"] == 0
