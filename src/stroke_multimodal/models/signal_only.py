"""Signal-Only branch: EfficientNet-B0 over the recording as a single image."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class SignalOnlyNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        old = backbone.features[0][0]
        stem = nn.Conv2d(1, old.out_channels, old.kernel_size,
                         stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            if pretrained:
                stem.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(stem.weight, mode="fan_out",
                                        nonlinearity="relu")
        backbone.features[0][0] = stem
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.backbone(x).reshape(-1)
