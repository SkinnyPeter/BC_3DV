from __future__ import annotations

import argparse
import h5py
import numpy as np

from sim.isaac_wrapper import IsaacSimWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a demonstration episode in Isaac Sim.")
    parser.add_argument("--episode_h5", type=str, required=True)
    parser.add_argument("--state_key", type=str, default="observations/qpos")
    parser.add_argument("--action_key", type=str, default="actions")
    args = parser.parse_args()

    sim = IsaacSimWrapper(config={})

    with h5py.File(args.episode_h5, "r") as f:
        states = np.asarray(f[args.state_key])
        actions = np.asarray(f[args.action_key]) if args.action_key in f else None

    sim.reset_episode()
    for t in range(len(states)):
        sim.set_state_from_demo(states[t])
        if actions is not None:
            sim.apply_action(actions[t])
        sim.step()

    print("[replay_episode] finished replay.")


if __name__ == "__main__":
    main()
