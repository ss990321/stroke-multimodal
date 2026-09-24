"""Measurements the acquisition cart reports, read from its XML export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ...schema import MUSE_FEATURES, MISSINGNESS_SOURCES

ZERO_MEANS_MISSING = [c for c in MISSINGNESS_SOURCES if c in MUSE_FEATURES]

SEX_TAG = "Gender"
SEX_ENCODING = {"MALE": 1, "FEMALE": 0}


def _first_text(root: ET.Element, tag: str) -> str | None:
    """Value of the first element with this tag, anywhere in the document."""
    node = root.find(f".//{tag}")
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None


def read_muse_xml(path: str | Path) -> dict:
    """The MUSE measurements of one recording, missing where not measured."""
    root = ET.parse(Path(path)).getroot()

    out: dict[str, float] = {}
    for tag in MUSE_FEATURES:
        text = _first_text(root, tag)
        try:
            value = float(text) if text is not None else np.nan
        except ValueError:
            value = np.nan
        if tag in ZERO_MEANS_MISSING and value == 0:
            value = np.nan
        out[tag] = value

    sex = _first_text(root, SEX_TAG)
    out["Gender"] = float(SEX_ENCODING.get((sex or "").upper(), np.nan))
    return out
