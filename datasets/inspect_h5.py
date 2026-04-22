from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import h5py
import pandas as pd

from datasets.utils import classify_key, first_dim_timesteps, load_episode_filepaths, walk_h5_datasets


def inspect_file(h5_path: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "file": str(h5_path),
        "datasets": [],
        "timesteps_candidates": [],
        "likely_image_keys": [],
        "likely_state_keys": [],
        "likely_action_keys": [],
    }

    with h5py.File(h5_path, "r") as f:
        for node in walk_h5_datasets(f):
            cls = classify_key(node.key)
            item = {
                "key": node.key,
                "shape": list(node.shape),
                "dtype": node.dtype,
                "class": cls,
            }
            ts = first_dim_timesteps(node.shape)
            if ts is not None:
                item["first_dim"] = ts
                record["timesteps_candidates"].append(ts)

            if cls == "image":
                record["likely_image_keys"].append(node.key)
            elif cls == "state":
                record["likely_state_keys"].append(node.key)
            elif cls == "action":
                record["likely_action_keys"].append(node.key)

            record["datasets"].append(item)

    if record["timesteps_candidates"]:
        record["episode_len_estimate"] = Counter(record["timesteps_candidates"]).most_common(1)[0][0]
    else:
        record["episode_len_estimate"] = None

    record["has_action_key"] = len(record["likely_action_keys"]) > 0
    return record


def aggregate_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    key_stats = defaultdict(lambda: {"count": 0, "shapes": Counter(), "dtypes": Counter()})
    episode_lengths = []

    for rec in records:
        if rec["episode_len_estimate"] is not None:
            episode_lengths.append(rec["episode_len_estimate"])
        for d in rec["datasets"]:
            ks = key_stats[d["key"]]
            ks["count"] += 1
            ks["shapes"][tuple(d["shape"])] += 1
            ks["dtypes"][d["dtype"]] += 1

    keys_summary = []
    for key, v in sorted(key_stats.items(), key=lambda x: x[0]):
        keys_summary.append(
            {
                "key": key,
                "count": v["count"],
                "common_shapes": [{"shape": list(s), "count": c} for s, c in v["shapes"].most_common(3)],
                "common_dtypes": [{"dtype": d, "count": c} for d, c in v["dtypes"].most_common(3)],
                "class": classify_key(key),
            }
        )

    all_image_keys = sorted(set(k for rec in records for k in rec["likely_image_keys"]))
    all_state_keys = sorted(set(k for rec in records for k in rec["likely_state_keys"]))
    all_action_keys = sorted(set(k for rec in records for k in rec["likely_action_keys"]))

    return {
        "num_files": len(records),
        "episode_length_stats": {
            "num_with_estimate": len(episode_lengths),
            "min": min(episode_lengths) if episode_lengths else None,
            "max": max(episode_lengths) if episode_lengths else None,
            "mean": float(sum(episode_lengths) / len(episode_lengths)) if episode_lengths else None,
        },
        "likely_image_keys": all_image_keys,
        "likely_state_keys": all_state_keys,
        "likely_action_keys": all_action_keys,
        "has_any_action_key": len(all_action_keys) > 0,
        "keys_summary": keys_summary,
        "recommended_defaults": {
            "image_key": all_image_keys[0] if all_image_keys else None,
            "proprio_key": all_state_keys[0] if all_state_keys else None,
            "action_key": all_action_keys[0] if all_action_keys else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a directory of H5 files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--glob", type=str, default="*.h5")
    parser.add_argument("--output_json", type=str, default="data/processed/h5_inspection.json")
    parser.add_argument("--output_csv", type=str, default="data/processed/h5_dataset_summary.csv")
    args = parser.parse_args()

    files = load_episode_filepaths(args.input_dir, args.glob)
    if not files:
        raise FileNotFoundError(f"No H5 files found in {args.input_dir} with glob {args.glob}")

    records = [inspect_file(f) for f in files]
    summary = aggregate_summary(records)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"files": records, "aggregate": summary}, f, indent=2)

    rows = []
    for rec in records:
        for d in rec["datasets"]:
            rows.append(
                {
                    "file": rec["file"],
                    "episode_len_estimate": rec["episode_len_estimate"],
                    "key": d["key"],
                    "shape": d["shape"],
                    "dtype": d["dtype"],
                    "class": d["class"],
                }
            )
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    print(f"[inspect_h5] scanned files: {len(files)}")
    print(f"[inspect_h5] likely image keys: {summary['likely_image_keys']}")
    print(f"[inspect_h5] likely state keys: {summary['likely_state_keys']}")
    print(f"[inspect_h5] likely action keys: {summary['likely_action_keys']}")
    print(f"[inspect_h5] has_any_action_key: {summary['has_any_action_key']}")
    print(f"[inspect_h5] wrote JSON: {output_json}")
    print(f"[inspect_h5] wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
