"""Stage 2: splitting the cohort and assembling a fold."""

from .folds import FoldData, RecordingStore, build_fold
from .splits import Fold, build_folds

__all__ = ["Fold", "build_folds", "FoldData", "RecordingStore", "build_fold"]
