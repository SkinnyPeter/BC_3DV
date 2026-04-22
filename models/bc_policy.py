from __future__ import annotations

import torch
import torch.nn as nn

from models.encoders import build_image_encoder, build_mlp


class BCPolicy(nn.Module):
    def __init__(
        self,
        image_channels: int,
        proprio_dim: int,
        action_dim: int,
        image_encoder_name: str = "small_cnn",
        image_feature_dim: int = 256,
        proprio_hidden: list[int] | None = None,
        fusion_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        proprio_hidden = proprio_hidden or [128, 128]
        fusion_hidden = fusion_hidden or [512, 256]

        self.image_encoder = build_image_encoder(image_encoder_name, image_channels, image_feature_dim)
        self.proprio_encoder = build_mlp(proprio_dim, proprio_hidden)

        fusion_in = image_feature_dim + proprio_hidden[-1]
        self.head = build_mlp(fusion_in, fusion_hidden, out_dim=action_dim)

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        prop_feat = self.proprio_encoder(proprio)
        fused = torch.cat([image_feat, prop_feat], dim=-1)
        return self.head(fused)
