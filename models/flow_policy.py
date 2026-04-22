from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.encoders import build_image_encoder, build_mlp


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, math.log(10000), steps=half, device=t.device, dtype=t.dtype) * (-1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        image_channels: int,
        proprio_dim: int,
        action_dim: int,
        image_encoder_name: str = "small_cnn",
        image_feature_dim: int = 256,
        proprio_hidden: list[int] | None = None,
        cond_hidden: list[int] | None = None,
        flow_hidden: list[int] | None = None,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        proprio_hidden = proprio_hidden or [128, 128]
        cond_hidden = cond_hidden or [512, 256]
        flow_hidden = flow_hidden or [512, 512]

        self.image_encoder = build_image_encoder(image_encoder_name, image_channels, image_feature_dim)
        self.proprio_encoder = build_mlp(proprio_dim, proprio_hidden)

        cond_in = image_feature_dim + proprio_hidden[-1]
        self.cond_mlp = build_mlp(cond_in, cond_hidden)

        self.time_emb = SinusoidalTimeEmbedding(time_embed_dim)
        vf_in = action_dim + cond_hidden[-1] + time_embed_dim
        self.vector_field = build_mlp(vf_in, flow_hidden, out_dim=action_dim)

    def condition(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        i = self.image_encoder(image)
        p = self.proprio_encoder(proprio)
        return self.cond_mlp(torch.cat([i, p], dim=-1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = torch.cat([x_t, t_emb, cond], dim=-1)
        return self.vector_field(h)

    @torch.no_grad()
    def sample(self, image: torch.Tensor, proprio: torch.Tensor, steps: int = 20, sigma: float = 1.0) -> torch.Tensor:
        cond = self.condition(image, proprio)
        x = sigma * torch.randn(image.shape[0], self.vector_field[-1].out_features, device=image.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), float(i) / steps, device=image.device)
            v = self.forward(x, t, cond)
            x = x + dt * v
        return x
