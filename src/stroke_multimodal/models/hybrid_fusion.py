"""Hybrid Fusion: combining the three branch predictions."""

from __future__ import annotations

import numpy as np

EPS = 1e-7


def to_logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def to_probability(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fuse(branch_probabilities: dict, weights: dict | None = None) -> np.ndarray:
    """Weighted average of branch logits, returned as a probability."""
    names = list(branch_probabilities)
    if not names:
        raise ValueError("no branch predictions to fuse")

    if weights is None:
        w = {name: 1.0 / len(names) for name in names}
    else:
        missing = [n for n in names if n not in weights]
        if missing:
            raise ValueError(f"no weight given for {missing}")
        total = sum(weights[n] for n in names)
        if total <= 0:
            raise ValueError("weights must sum to something positive")
        w = {n: weights[n] / total for n in names}

    lengths = {len(np.asarray(branch_probabilities[n])) for n in names}
    if len(lengths) != 1:
        raise ValueError(f"branches disagree on length: {lengths}")

    stacked = sum(w[n] * to_logit(branch_probabilities[n]) for n in names)
    return to_probability(stacked)
