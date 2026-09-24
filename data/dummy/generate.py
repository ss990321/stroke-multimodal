#!/usr/bin/env python3
"""Build a synthetic cohort in the layout stage 1 expects."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]

# Recordings are written as integer microvolts, as the acquisition cart does.
R_AMPLITUDE_UV = 900.0

# Mean and standard deviation of each MUSE measurement in the study cohort, by
# class, with the proportion the cart did not measure. These are aggregates
# over 4,147 records; no individual record can be recovered from them.
#                     control (mean, sd)    stroke (mean, sd)    missing
MUSE_STATS = {
    "VentricularRate":  ((76.6, 18.9), (75.8, 17.8), 0.000),
    "AtrialRate":       ((84.2, 46.3), (86.1, 52.7), 0.000),
    "PRInterval":       ((162.3, 27.8), (167.3, 27.7), 0.137),
    "QRSDuration":      ((93.5, 18.1), (92.3, 15.8), 0.000),
    "QTInterval":       ((396.4, 45.9), (404.9, 44.2), 0.000),
    "QTCorrected":      ((439.6, 39.1), (446.9, 30.8), 0.000),
    "PAxis":            ((49.1, 24.7), (45.3, 25.5), 0.119),
    "RAxis":            ((37.0, 42.6), (27.1, 37.8), 0.000),
    "TAxis":            ((48.4, 48.1), (46.3, 46.2), 0.000),
    "QRSCount":         ((12.6, 3.1), (12.4, 3.0), 0.000),
    "QOnset":           ((218.6, 6.8), (219.2, 6.6), 0.000),
    "QOffset":          ((265.4, 8.8), (265.3, 8.7), 0.000),
    "POnset":           ((137.9, 15.2), (135.6, 15.7), 0.127),
    "POffset":          ((188.9, 14.7), (185.6, 16.0), 0.127),
    "TOffset":          ((416.8, 23.3), (421.7, 22.8), 0.000),
    "QTcFrederica":     ((424.3, 35.6), (432.2, 28.8), 0.023),
    "GlobalRR":         ((824.4, 177.6), (831.4, 181.8), 0.000),
    "PharmaRRinterval": ((781.1, 259.0), (693.9, 351.4), 0.000),
    "PharmaPPinterval": ((764.9, 282.9), (675.9, 370.9), 0.000),
}

# Proportion of male records, and the simulated heart rate, by class.
MALE_SHARE = {0: 0.548, 1: 0.584}
HEART_RATE = {0: (76.6, 18.9), 1: (75.8, 17.8)}

# A simulated waveform carries no disease signal, so a small amplitude shift is
# added to give the signal branch something to learn. It is an artefact of the
# simulation, not a property of stroke.
LABEL_AMPLITUDE_SHIFT = 0.06


def draw(rng, stats, label: int) -> float:
    """One measurement, or NaN at the rate the cart failed to produce it."""
    control, stroke, missing_rate = stats
    mean, sd = stroke if label else control
    if rng.random() < missing_rate:
        return np.nan
    return float(rng.normal(mean, sd))


def simulate_recording(rng, heart_rate: float, label: int, seconds: int,
                       sampling_rate: int) -> np.ndarray:
    """A 12-lead recording in microvolts, shaped (leads, samples)."""
    try:
        import neurokit2 as nk

        base = nk.ecg_simulate(duration=seconds, sampling_rate=sampling_rate,
                               heart_rate=heart_rate,
                               random_state=int(rng.integers(0, 2**31 - 1)))
    except Exception:
        base = rng.normal(0, 0.1, seconds * sampling_rate)

    base = np.asarray(base) * R_AMPLITUDE_UV * (1 + LABEL_AMPLITUDE_SHIFT * label)
    # Each lead is the same rhythm at a different projection, which is enough
    # here without modelling real lead geometry.
    projections = np.array([0.35, 1.00, 0.65, -0.68, -0.15, 0.83,
                            -0.55, -0.40, 0.30, 0.75, 0.90, 0.70])
    recording = np.outer(projections, base)
    recording += rng.normal(0, 12.0, recording.shape)
    return recording


def write_muse_xml(path: Path, rng, label: int) -> None:
    root = ET.Element("RestingECG")
    measurements = ET.SubElement(root, "RestingECGMeasurements")
    for tag, stats in MUSE_STATS.items():
        value = draw(rng, stats, label)
        # The cart writes 0 for a measurement it did not make.
        ET.SubElement(measurements, tag).text = (
            "0" if np.isnan(value) else f"{value:.1f}")

    demographics = ET.SubElement(root, "PatientDemographics")
    ET.SubElement(demographics, "Gender").text = (
        "MALE" if rng.random() < MALE_SHARE[label] else "FEMALE")
    ET.ElementTree(root).write(path, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=24)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--sampling-rate", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ecg_dir = args.out / "ecg"
    xml_dir = args.out / "muse"
    ecg_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.records):
        name = f"dummy{i:04d}"
        label = int(i % 2)

        mean, sd = HEART_RATE[label]
        heart_rate = float(np.clip(rng.normal(mean, sd), 45.0, 120.0))

        recording = simulate_recording(rng, heart_rate, label, args.seconds,
                                       args.sampling_rate)
        pd.DataFrame(np.rint(recording.T).astype(np.int32),
                     columns=LEADS).to_csv(ecg_dir / f"{name}.csv", index=False)
        write_muse_xml(xml_dir / f"{name}.xml", rng, label)

        rows.append({"file_name": name, "PatientID": i, "label": label})

    cohort = pd.DataFrame(rows)
    cohort.to_csv(args.out / "cohort.csv", index=False)

    size_mb = sum(f.stat().st_size for f in args.out.rglob("*")
                  if f.is_file()) / 1e6
    print(f"{len(cohort)} records, {args.seconds}s at {args.sampling_rate} Hz")
    print(f"  stroke {int(cohort['label'].sum())} / "
          f"control {int((1 - cohort['label']).sum())}")
    print(f"written to {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
