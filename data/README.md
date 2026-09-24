# data

Nothing here is distributed with the repository except `dummy/`. The recordings
this study used are clinical and cannot be shared.

## What goes where

```
ecg/<file_name>.csv        one recording per file; columns are leads
muse/<file_name>.xml       the acquisition cart's RestingECGMeasurements export
cohort.csv                 file_name, PatientID, label - you supply this
```

Create these yourself, or point `configs/local.yaml` at wherever they already
live. Everything the pipeline writes goes to `outputs/`.

`cohort.csv` is the one input you must produce yourself: it says which
recordings are in the study and what their outcome is.

## Pointing at data that lives elsewhere

Clinical data usually sits outside the repository. Put its real location in
`configs/local.yaml`, which is not tracked by git:

```yaml
paths:
  root: "/absolute/path/to/your/project"
  ecg_dir: "/absolute/path/to/ecg"
  muse_xml_dir: "/absolute/path/to/muse"
  cohort_csv: "/absolute/path/to/cohort.csv"
```

then run with `--config configs/local.yaml`. `configs/default.yaml` keeps only
relative paths so that no real location is ever committed.

## dummy/

A small synthetic cohort, committed so the pipeline can be run immediately
without any clinical data. The recordings are simulated, so the metrics mean
nothing; what they demonstrate is that the code path works. See `dummy/README.md`.
