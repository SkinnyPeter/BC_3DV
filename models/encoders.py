from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchvision.models as tvm


class SmallCNNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        base = tvm.resnet18(weights=None)
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(nn.Linear(dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


def build_mlp(in_dim: int, hidden_dims: List[int], out_dim: int | None = None, final_activation: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
        if final_activation:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def build_image_encoder(name: str, in_channels: int, out_dim: int) -> nn.Module:
    if name == "small_cnn":
        return SmallCNNEncoder(in_channels, out_dim)
    if name == "resnet18":
        return ResNet18Encoder(in_channels, out_dim)
    raise ValueError(f"Unsupported image encoder: {name}")
