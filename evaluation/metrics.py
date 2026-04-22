from __future__ import annotations

from typing import Dict

import torch


def action_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def endpoint_error(pred: torch.Tensor, target: torch.Tensor, step_dim: int) -> torch.Tensor:
    p_end = pred.view(pred.size(0), -1, step_dim)[:, -1]
    t_end = target.view(target.size(0), -1, step_dim)[:, -1]
    return torch.norm(p_end - t_end, dim=-1).mean()


def per_dim_mse(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    m = ((pred - target) ** 2).mean(dim=0)
    return {f"mse_dim_{i}": float(v.item()) for i, v in enumerate(m)}
