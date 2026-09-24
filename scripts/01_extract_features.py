#!/usr/bin/env python3
"""Stage 1: recordings and MUSE exports -> the feature table."""

from __future__ import annotations

import argparse

from stroke_multimodal.config import Config, parse_overrides
from stroke_multimodal.feature_extraction.build_table import build_table, write_table
from stroke_multimodal.schema import ALL_FEATURES, MISSINGNESS_INDICATORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None,
                        help="JSON or YAML config; defaults are used otherwise")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="override any config setting, e.g. paths.ecg_dir=/data/ecg")
    args = parser.parse_args()

    cfg = Config.load(args.config, **parse_overrides(args.overrides))
    print(f"cohort      {cfg.paths.cohort_csv}")
    print(f"recordings  {cfg.paths.ecg_dir}")
    print(f"MUSE XML    {cfg.paths.muse_xml_dir}\n")

    table, report = build_table(cfg)
    write_table(table, cfg.paths.features_csv)

    print(f"\n{len(table):,} recordings x {len(ALL_FEATURES)} features")
    print(f"written to {cfg.paths.features_csv}")

    print("\nwaveform measurement:")
    for key, value in report.items():
        print(f"  {key:<26} {value}")
    missing = table[ALL_FEATURES].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print("\nmissing values per column (imputed later, inside each fold):")
        for column, count in missing.items():
            print(f"  {column:<24} {count:>6,}  ({100 * count / len(table):.1f}%)")
    print("\nmissingness indicators written: " + ", ".join(MISSINGNESS_INDICATORS))


if __name__ == "__main__":
    main()
