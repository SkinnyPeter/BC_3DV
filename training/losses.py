from __future__ import annotations

import torch
import torch.nn.functional as F


def bc_loss(pred: torch.Tensor, target: torch.Tensor, smoothness_weight: float = 0.0, action_step_dim: int | None = None) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if smoothness_weight <= 0 or action_step_dim is None:
        return mse

    B, D = pred.shape
    if D % action_step_dim != 0:
        return mse

    T = D // action_step_dim
    p = pred.view(B, T, action_step_dim)
    smooth = ((p[:, 1:] - p[:, :-1]) ** 2).mean() if T > 1 else torch.tensor(0.0, device=pred.device)
    return mse + smoothness_weight * smooth


def flow_matching_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v_pred, v_target)
