"""Checks on what the repository would publish.

Clinical data and the paths that point at it must not reach a commit. Reviewing
that by eye before every push does not scale and is exactly the kind of thing
that slips through once.

    python -m pytest tests/test_repository.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Shapes that mean someone's machine rather than a relative location.
ABSOLUTE_PATH_MARKERS = ["/home/", "/Users/", "C:\\", "C:/", "/mnt/", "/data/"]


def committed_configs():
    """Configs that are tracked; local.yaml is ignored by git and may hold
    anything."""
    return [p for p in (ROOT / "configs").glob("*.yaml")
            if not p.name.endswith("local.yaml")]


def test_there_are_configs_to_check():
    assert committed_configs(), "no config files found to check"


@pytest.mark.parametrize("path", committed_configs(), ids=lambda p: p.name)
def test_committed_configs_hold_no_absolute_paths(path):
    """A real location in a tracked config is a data leak waiting to happen."""
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):          # commented examples are fine
            continue
        for marker in ABSOLUTE_PATH_MARKERS:
            assert marker not in stripped, (
                f"{path.name} names an absolute path ({marker}); real locations "
                "belong in configs/local.yaml, which is not tracked")


def test_the_local_config_is_ignored():
    """The escape hatch for real paths must actually be untracked."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "configs/local.yaml" in ignore


def test_only_the_synthetic_cohort_is_committed_under_data():
    """data/ holds clinical recordings; only dummy/ may be published."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/*" in ignore
    assert "!data/dummy/" in ignore


def test_the_synthetic_cohort_is_present_and_runnable():
    """A fresh clone should be able to run the pipeline immediately."""
    dummy = ROOT / "data" / "dummy"
    assert (dummy / "cohort.csv").exists(), "the dummy cohort file is missing"
    assert (dummy / "generate.py").exists(), "the dummy generator is missing"
    recordings = list((dummy / "ecg").glob("*.csv"))
    assert len(recordings) >= 10, (
        f"only {len(recordings)} dummy recordings; the five folds need more")
    exports = list((dummy / "muse").glob("*.xml"))
    assert len(exports) == len(recordings), (
        "every dummy recording needs a matching MUSE export")


def test_the_synthetic_cohort_stays_small():
    """It is committed, so it has to stay a reasonable size for a clone."""
    total = sum(f.stat().st_size for f in (ROOT / "data" / "dummy").rglob("*")
                if f.is_file())
    assert total < 30e6, f"the dummy cohort is {total / 1e6:.1f} MB; regenerate it smaller"
