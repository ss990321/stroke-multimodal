# Hybrid Multimodal Fusion of Raw ECG Signals and Derived Measurements for Stroke Classification

Reference implementation for the paper of the same name.

A single 10-second 12-lead ECG carries two kinds of information: the waveform
itself, and the quantitative measurements derived from it. Measurements discard
waveform detail; a model given only the raw signal may not learn the
quantitative structure from a limited cohort. This framework uses both, and
couples them at more than one level.

| Branch | Reads | Architecture |
|---|---|---|
| Signal-Only | the 12 × 5000 recording | EfficientNet-B0, single-channel stem |
| Feature-Only | 41 derived measurements | MLP 128-128-64-32 |
| Interaction Early Fusion | both, coupled before encoding | EfficientNet-B0 + raw interaction branch |
| **Hybrid Fusion** | the three branch logits | equal-weight average |

On 4,147 records from 4,020 patients, Hybrid Fusion reached an AUC of 0.832, an
F1-score of 77.6% and a Brier score of 0.169 — 0.034 AUC above the best
unimodal model.

## Pipeline

Four stages. Each writes a file the next one reads, so any stage can be run on
its own or replaced.

```
recordings + MUSE XML
        │  01_extract_features.py
        ▼
   feature table  ──────────────────┐
        │  02_make_splits.py        │
        ▼                           │
  cross-validation folds            │
        │                           │
        │  03_train.py  ◄───────────┘
        ▼
  checkpoints + metrics + out-of-fold predictions
        │  04_predict.py
        ▼
   predictions for new recordings
```

```bash
python scripts/01_extract_features.py --config configs/default.yaml
python scripts/02_make_splits.py      --config configs/default.yaml
python scripts/03_train.py            --config configs/default.yaml
python scripts/04_predict.py          --config configs/default.yaml --fold 0
```

`configs/default.yaml` expects your recordings under `data/`; see **Data
layout** below. To run before you have any, use `configs/dummy.yaml`, which
points at the synthetic cohort in this repository.

Already have a feature table? Skip stage 1 and point `paths.features_csv` at it.
Stage 3 needs only the columns listed in `src/stroke_multimodal/schema.py`.

Every setting lives in the config file. To vary one without editing it:

```bash
python scripts/03_train.py --config configs/default.yaml --set training.seed=43
```

## Install

```bash
pip install -e ".[extraction]"     # everything, including stage 1
pip install -e .                   # stages 2-4 only
```

`neurokit2` and `PyWavelets` are needed only by stage 1.

## Run it without clinical data

A synthetic cohort of 100 recordings is committed, so a fresh clone works
immediately:

```bash
python scripts/01_extract_features.py --config configs/dummy.yaml
python scripts/02_make_splits.py      --config configs/dummy.yaml
python scripts/03_train.py            --config configs/dummy.yaml
python scripts/04_predict.py          --config configs/dummy.yaml --fold 0

python -m pytest tests/ -q
```

The MUSE measurements are drawn from the study cohort's per-class means,
standard deviations and missing rates, so the feature table looks like the real
one. The waveforms are simulated and carry no disease signal, so the accuracy
figures mean nothing. What this checks is that every stage runs, that the splits
do not leak, and that a reloaded checkpoint reproduces the predictions it was
saved from. `data/dummy/generate.py` rebuilds the cohort at any size. See
`data/dummy/README.md`.

## Data layout

Stage 1 expects one waveform CSV and one MUSE XML per recording, plus a cohort
file naming them:

```
data/
  ecg/<file_name>.csv        one recording per file; columns are leads
  muse/<file_name>.xml       the cart's RestingECGMeasurements export
  cohort.csv                 file_name, PatientID, label - you supply this
```

The study's recordings came from a GE MUSE system: 10 s, 12 leads, 500 Hz.

No clinical data is distributed here. Real locations belong in
`configs/local.yaml`, which is not tracked; `configs/default.yaml` holds only
relative paths, and a test fails if an absolute path reaches a tracked config.

## The feature table

50 columns in three groups, defined in `schema.py`:

- **19 MUSE measurements** plus sex, read from the XML. The cart writes `0`
  for a measurement it did not make; that is converted to missing at read time.
- **21 extracted measurements** computed from the waveform: wavelet denoising,
  Pan-Tompkins beat detection, then intervals, amplitudes, time-domain
  variability and an autoregressive spectrum.
- **9 missingness indicators**, one per column that can be absent.

Nine are excluded from the models, leaving **41**:

| Excluded | Why |
|---|---|
| `QTcFrederica_missing` | its missingness tracked the acquisition year, not the disease state |
| `VLF`, `LF`, `HF`, `LF/HF` and their four indicators | a 10 s window cannot resolve those bands |

The second exclusion is worth stating plainly. A window of *T* seconds resolves
frequencies no more finely than about 1/*T* — 0.1 Hz here. The VLF band is
0.037 Hz wide and its lower edge needs a 303 s cycle. An autoregressive model
still returns a number, because it extrapolates a fitted model below the
resolution limit; that number is not a measurement. The variables are still
computed, so the exclusion can be audited rather than taken on trust.

## Validation design

Patients — not recordings — are partitioned into five disjoint folds. Each fold
is held out in turn and split equally into a test and a validation set, the
remaining four folds forming the training set. This gives the 80:10:10 ratio the
paper reports, with every patient held out exactly once and the five test sets
disjoint, so the pooled out-of-fold predictions hold each record once and can be
analysed as a single held-out set. All records from one patient stay together.

`02_make_splits.py` audits this and refuses to write a split that leaks.

Two quantities are estimated from data, and both are fitted inside each fold on
its training split alone: the medians used to fill missing features, and the
mean and standard deviation used to standardise them. `tests/test_pipeline.py`
asserts this by perturbing the test split and checking that no training feature
moves.

## Reproducibility

`neurokit2` is pinned to the version the study used, because its beat detector
and wave delineator determine the extracted measurements. The other
dependencies are given as minimum versions; the published runs used PyTorch
2.8.0 on an NVIDIA RTX A5000.

Training uses a fixed seed. For bit-reproducible training, at some cost in
speed:

```bash
python scripts/03_train.py --config configs/default.yaml     --set training.deterministic=true
```

## Layout

```
configs/
  default.yaml                    every setting, with the published values
  dummy.yaml                      the synthetic cohort
  local.yaml.example              template for real, untracked paths
scripts/
  01_extract_features.py          recordings + XML -> feature table
  02_make_splits.py               feature table   -> folds, with an audit
  03_train.py                     folds           -> checkpoints and metrics
  04_predict.py                   checkpoints     -> predictions
src/stroke_multimodal/
  config.py                       what this run is
  schema.py                       what the feature table is, and what was dropped
  feature_extraction/
    muse/reader.py                the cart's export, and its zero-means-missing convention
    extracted/
      signal.py                   reading, lead selection, denoising, beat detection
      measures.py                 rate, amplitudes, time-domain variability, QTc
      intervals.py                QRS, ST and QT, measured beat by beat
      hrv_frequency.py            autoregressive band powers
    build_table.py                assembling one row per recording
  dataset/
    splits.py                     patient-level folds and the leakage audit
    folds.py                      per-fold imputation and scaling; the recording store
  models/
    signal_only.py  feature_only.py  interaction_early_fusion.py  hybrid_fusion.py
  train/
    branches.py                   the training loop, and what makes each branch differ
    run.py                        one fold, and the whole cross-validation
  inference.py                    scoring with a trained fold
  evaluation.py                   metrics and operating points
tests/
  test_pipeline.py                leakage, fitting, checkpoint round-trip, fusion
  test_features.py                waveform measurement and the MUSE conventions
  test_repository.py              what the repository would publish
data/dummy/                       a synthetic cohort, and the script that made it
```

## Licence

MIT. See `LICENSE`.

## Citation

Im S-Y et al. *Hybrid Multimodal Fusion of Raw ECG Signals and Derived
Measurements for Stroke Classification.* Bioengineering.
