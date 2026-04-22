from __future__ import annotations

from typing import Any, Dict

import numpy as np


class IsaacSimWrapper:
    """Project-facing interface for Isaac Sim integration.

    TODO(project-specific): wire these methods to actual Isaac Sim scene, articulation,
    camera, and reset APIs used in your setup.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def reset_episode(self, episode_id: str | None = None) -> Dict[str, Any]:
        """Reset simulator state and return initial observation."""
        raise NotImplementedError("TODO(project-specific): implement simulator reset")

    def set_state_from_demo(self, demo_state: np.ndarray) -> None:
        """Set robot (and possibly object) state from demonstration state vector."""
        raise NotImplementedError("TODO(project-specific): implement state-setting logic")

    def apply_action(self, action: np.ndarray) -> None:
        """Apply an action vector to the simulator controller."""
        raise NotImplementedError("TODO(project-specific): implement action application")

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation dict with image + proprio to match training interface."""
        raise NotImplementedError("TODO(project-specific): implement observation extraction")

    def step(self) -> None:
        """Advance simulation by one control step."""
        raise NotImplementedError("TODO(project-specific): implement stepping")

    def run_closed_loop_rollout(self, policy_fn, horizon: int) -> Dict[str, Any]:
        """Run a rollout by repeatedly querying the policy and stepping sim."""
        obs = self.reset_episode()
        traj = {"obs": [], "actions": []}
        for _ in range(horizon):
            action = policy_fn(obs)
            self.apply_action(action)
            self.step()
            obs = self.get_observation()
            traj["obs"].append(obs)
            traj["actions"].append(action)
        return traj
