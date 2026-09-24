#!/usr/bin/env python3
"""Stage 2: the feature table -> patient-level cross-validation folds."""

from __future__ import annotations

import argparse

import pandas as pd

from stroke_multimodal.config import Config, parse_overrides
from stroke_multimodal.dataset import splits as split_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="override any config setting, e.g. split.seed=43")
    parser.add_argument("--allow-unclean", action="store_true",
                        help="write the split even if the audit fails")
    args = parser.parse_args()

    cfg = Config.load(args.config, **parse_overrides(args.overrides))
    table = pd.read_csv(cfg.paths.features_csv)
    folds = split_module.build_folds(table, n_folds=cfg.split.n_folds,
                                     seed=cfg.split.seed, stratify=cfg.split.stratify)
    per_fold, checks = split_module.audit(folds, table)

    pd.set_option("display.width", 200)
    print(per_fold.to_string(index=False))
    print()
    print(f"patients                        {checks['patients']:,}")
    print(f"  ever in a test set            {checks['patients_ever_tested']:,}")
    print(f"  never held out                {checks['patients_never_held_out']:,}")
    print(f"test-set patient overlaps       {checks['test_patient_overlaps']}")
    print(f"pooled test predictions         {checks['pooled_test_rows']:,} "
          f"({checks['pooled_test_distinct']:,} distinct)")
    print(f"patients in two splits of one fold  {checks['total_patient_leakage']}")

    if not args.allow_unclean:
        split_module.assert_clean(checks)
        print("\naudit passed: the test sets are disjoint and nothing leaks.")

    split_module.save(folds, cfg.paths.splits_pkl)
    audit_path = cfg.paths.splits_pkl.with_name("split_audit.csv")
    per_fold.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"\nwritten to {cfg.paths.splits_pkl}\n         and {audit_path}")


if __name__ == "__main__":
    main()
