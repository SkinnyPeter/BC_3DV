from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import h5py

from datasets.utils import load_episode_filepaths


def split_episodes(episode_names: List[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    names = episode_names[:]
    rng.shuffle(names)

    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = names[:n_train]
    val = names[n_train : n_train + n_val]
    test = names[n_train + n_val : n_train + n_val + n_test]

    return {"train": train, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess H5 dataset into metadata + splits.")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--glob", type=str, default="*.h5")
    parser.add_argument("--proprio_key", type=str, default="observations/qpos")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--write_index_cache", action="store_true")
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--action_stride", type=int, default=1)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = load_episode_filepaths(args.raw_dir, args.glob)
    if not files:
        raise FileNotFoundError(f"No files found in {args.raw_dir} ({args.glob})")

    metadata = {"episodes": []}
    for p in files:
        with h5py.File(p, "r") as f:
            T = int(f[args.proprio_key].shape[0])
            metadata["episodes"].append(
                {
                    "episode_name": p.name,
                    "filepath": str(p),
                    "timesteps": T,
                }
            )

    splits = split_episodes([p.name for p in files], args.train_ratio, args.val_ratio, args.seed)

    metadata_path = processed_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    splits_path = processed_dir / "splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    if args.write_index_cache:
        index_cache = {}
        for split_name, ep_names in splits.items():
            rows = []
            for ep_i, ep_name in enumerate(ep_names):
                ep_path = Path(args.raw_dir) / ep_name
                with h5py.File(ep_path, "r") as f:
                    T = int(f[args.proprio_key].shape[0])
                min_t = args.frame_stack - 1
                max_t = T - 1 - (args.action_chunk - 1) * args.action_stride
                for t in range(min_t, max_t + 1):
                    rows.append({"episode_idx": ep_i, "t": t, "episode_name": ep_name})
            index_cache[split_name] = rows

        index_path = processed_dir / "index_cache.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index_cache, f)
        print(f"[preprocess] wrote {index_path}")

    print(f"[preprocess] wrote {metadata_path}")
    print(f"[preprocess] wrote {splits_path}")
    print(f"[preprocess] episodes: {len(files)}")


if __name__ == "__main__":
    main()
