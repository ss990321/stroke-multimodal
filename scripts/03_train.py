#!/usr/bin/env python3
"""Stage 3: train the three branches on every fold and fuse them."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from stroke_multimodal.config import Config, parse_overrides
from stroke_multimodal.dataset import splits as split_module
from stroke_multimodal.dataset.folds import RecordingStore
from stroke_multimodal.evaluation import format_table
from stroke_multimodal.train.run import run_cross_validation


def load_store(cfg: Config, table: pd.DataFrame) -> RecordingStore:
    """Read every recording once, from the cache when one has been written."""
    cache = cfg.paths.recording_cache
    if cache is not None and cache.exists():
        print(f"recordings  {cache} (cached)")
        return RecordingStore.from_npz(cache)

    print(f"recordings  {cfg.paths.ecg_dir} (reading {len(table):,} files)")
    store = RecordingStore.from_directory(
        cfg.paths.ecg_dir, table["file_name"].astype(str),
        n_leads=cfg.signal.n_leads, n_samples=cfg.signal.n_samples)
    if cache is not None:
        store.save(cache)
        print(f"            cached to {cache}")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append",
                        metavar="KEY=VALUE",
                        help="override any config setting, e.g. training.seed=43")
    parser.add_argument("--no-checkpoints", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config, **parse_overrides(args.overrides))

    import torch
    if cfg.training.device.startswith("cuda") and not torch.cuda.is_available():
        print("no CUDA device available; falling back to CPU")
        cfg.training.device = "cpu"

    table = pd.read_csv(cfg.paths.features_csv)
    folds = split_module.load(cfg.paths.splits_pkl)
    store = load_store(cfg, table)

    print(f"device      {cfg.training.device}")
    print(f"seed        {cfg.training.seed}"
          + ("  (deterministic)" if cfg.training.deterministic else ""))
    print(f"folds       {len(folds)}\n")

    started = time.perf_counter()
    results = run_cross_validation(folds, table, store, cfg,
                                   save_checkpoints=not args.no_checkpoints)
    elapsed = (time.perf_counter() - started) / 60

    out = cfg.paths.outputs
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("per_fold_metrics", results["per_fold"]),
                        ("summary_metrics", results["summary"]),
                        ("pooled_out_of_fold", results["pooled_test"]),
                        ("validation_predictions", results["val_predictions"])):
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n" + "=" * 96)
    print(f"mean over {len(folds)} folds (95% confidence interval)")
    print("=" * 96)
    print(format_table(results["summary"]).to_string(index=False))
    pooled = results["pooled_test"]
    print(f"\npooled out-of-fold predictions: {len(pooled):,} rows, "
          f"{pooled['file_name'].nunique():,} distinct records")
    print(f"finished in {elapsed:.1f} min; written to {out}")


if __name__ == "__main__":
    main()
