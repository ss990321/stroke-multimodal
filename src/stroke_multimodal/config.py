"""Every setting the pipeline takes, in one place."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Paths:
    """Where the pipeline reads and writes."""

    root: Path = Path(".")
    ecg_dir: Path = Path("data/ecg")
    muse_xml_dir: Path = Path("data/muse")
    cohort_csv: Path = Path("data/cohort.csv")
    features_csv: Path = Path("data/features/feature_table.csv")
    splits_pkl: Path = Path("data/splits/cv5_patient.pkl")
    outputs: Path = Path("outputs")
    recording_cache: Path = Path("data/recordings.npz")

    def resolve(self) -> "Paths":
        root = self.root.expanduser().resolve()
        return Paths(
            root=root,
            **{f: root / getattr(self, f) if not getattr(self, f).is_absolute()
               else getattr(self, f)
               for f in ("ecg_dir", "muse_xml_dir", "cohort_csv", "features_csv",
                         "splits_pkl", "outputs", "recording_cache")},
        )


@dataclass
class SignalSpec:
    """Acquisition geometry of the recordings the models expect."""

    sampling_rate_hz: int = 500
    duration_s: int = 10
    n_leads: int = 12

    @property
    def n_samples(self) -> int:
        return self.sampling_rate_hz * self.duration_s


@dataclass
class SplitSpec:
    """Patient-level 5-fold cross-validation."""

    n_folds: int = 5
    seed: int = 42
    stratify: bool = True


@dataclass
class BranchSpec:
    """Optimiser and early-stopping settings for one branch."""

    max_epochs: int
    patience: int
    batch_size: int
    lr: float
    weight_decay: float
    min_delta: float = 0.0
    pos_weight_cap: float | None = None


@dataclass
class TrainingSpec:
    seed: int = 42
    signal: BranchSpec = field(default_factory=lambda: BranchSpec(
        max_epochs=30, patience=5, batch_size=64, lr=1e-3,
        weight_decay=1e-2, pos_weight_cap=20.0))
    feature: BranchSpec = field(default_factory=lambda: BranchSpec(
        max_epochs=200, patience=5, batch_size=64, lr=1e-3,
        weight_decay=1e-3, min_delta=1e-6))
    interaction: BranchSpec = field(default_factory=lambda: BranchSpec(
        max_epochs=30, patience=5, batch_size=64, lr=1e-3, weight_decay=1e-4))
    device: str = "cuda:0"
    deterministic: bool = False


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    signal: SignalSpec = field(default_factory=SignalSpec)
    split: SplitSpec = field(default_factory=SplitSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "Config":
        """Read a JSON or YAML config; keys absent from the file keep the value"""
        cfg = cls()
        if path is not None:
            raw = _read_mapping(Path(path))
            cfg = _merge(cfg, raw)
        if overrides:
            cfg = _merge(cfg, overrides)
        cfg.paths = cfg.paths.resolve()
        return cfg

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=str))


def parse_overrides(assignments) -> dict:
    """Turn `["training.seed=43", "split.n_folds=10"]` into a nested mapping."""
    out: dict = {}
    for item in assignments or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw.strip())
    return out


def _coerce(text: str):
    """Read a command-line value as the type it looks like."""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _read_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise SystemExit(
                "PyYAML is needed to read a .yaml config; install it or pass a "
                ".json file instead") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _merge(cfg, raw: dict):
    """Apply a nested mapping onto a dataclass tree, leaving absent keys alone."""
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown configuration key: {key}")
        current = getattr(cfg, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            setattr(cfg, key, _merge(current, value))
        elif isinstance(current, Path):
            setattr(cfg, key, Path(value))
        else:
            setattr(cfg, key, value)
    return cfg
