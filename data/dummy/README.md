# dummy

A synthetic cohort in the exact layout stage 1 expects. Simulated recordings,
so no patient data and no meaningful accuracy - the point is that the pipeline
runs end to end on a fresh clone.

```bash
python scripts/01_extract_features.py --config configs/dummy.yaml
python scripts/02_make_splits.py      --config configs/dummy.yaml
python scripts/03_train.py            --config configs/dummy.yaml
python scripts/04_predict.py          --config configs/dummy.yaml --fold 0
```

`generate.py` rebuilds this folder, and takes a size:

```bash
python data/dummy/generate.py --records 60 --seconds 10
```

The committed set is deliberately small (a few hundred kilobytes). Sixty or more
records makes the five folds large enough to be worth looking at.

## One thing that looks like a bug but is not

**The metrics are meaningless.** Twenty records over five folds leaves two test
records per fold, so accuracy moves in steps of 50% and the confidence intervals
are degenerate. This set exists to prove the code path runs end to end, not to
show performance. For numbers that behave like numbers:

```bash
python data/dummy/generate.py --records 120 --seconds 10
```

which is not committed, because it would be about 38 MB.

## Why 10 s

The same length as the recordings the study used, so the extraction runs against
the window it was written for: the delineator sees a dozen beats, and the
autoregressive HRV estimator has a long enough RR series to fit. A shorter
cohort still runs, but the frequency-domain columns come back empty because the
estimator declines a tachogram below its minimum length.
