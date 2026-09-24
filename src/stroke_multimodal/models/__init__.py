"""The four models of the framework."""

from .feature_only import FeatureOnlyNet
from .hybrid_fusion import fuse
from .interaction_early_fusion import InteractionEarlyFusion
from .signal_only import SignalOnlyNet

__all__ = ["SignalOnlyNet", "FeatureOnlyNet", "InteractionEarlyFusion", "fuse"]
