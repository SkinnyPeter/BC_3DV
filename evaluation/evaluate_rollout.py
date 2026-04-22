from __future__ import annotations

import argparse

import numpy as np
import torch

from models.bc_policy import BCPolicy
from models.flow_policy import FlowMatchingPolicy
from sim.isaac_wrapper import IsaacSimWrapper
from utils.config_utils import load_yaml


def preprocess_obs(obs, image_hw=(128, 128)):
    """TODO(project-specific): align Isaac observation format with training tensors."""
    raise NotImplementedError("TODO(project-specific): implement observation preprocessing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy with Isaac Sim rollouts.")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--policy", type=str, choices=["bc", "flow"], required=True)
    parser.add_argument("--horizon", type=int, default=200)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg["evaluation"]["device"] if torch.cuda.is_available() else "cpu")
    sim = IsaacSimWrapper(config={})

    # TODO(project-specific): instantiate model with correct dimensions from config/metadata.
    if args.policy == "bc":
        policy = BCPolicy(image_channels=3, proprio_dim=7, action_dim=56).to(device)
        ckpt = torch.load(cfg["checkpoints"]["bc_checkpoint"], map_location=device)
        policy.load_state_dict(ckpt["model"])
        policy.eval()

        def policy_fn(obs):
            image, proprio = preprocess_obs(obs)
            with torch.no_grad():
                act = policy(image.to(device), proprio.to(device))
            return act.squeeze(0).cpu().numpy()

    else:
        policy = FlowMatchingPolicy(image_channels=3, proprio_dim=7, action_dim=56).to(device)
        ckpt = torch.load(cfg["checkpoints"]["flow_checkpoint"], map_location=device)
        policy.load_state_dict(ckpt["model"])
        policy.eval()

        def policy_fn(obs):
            image, proprio = preprocess_obs(obs)
            with torch.no_grad():
                act = policy.sample(image.to(device), proprio.to(device), steps=20)
            return act.squeeze(0).cpu().numpy()

    traj = sim.run_closed_loop_rollout(policy_fn=policy_fn, horizon=args.horizon)
    print(f"[evaluate_rollout] collected {len(traj['actions'])} actions")


if __name__ == "__main__":
    main()
