from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


def _load_array(h5_file: h5py.File, key: str) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"Key '{key}' not found in H5 file.")
    return np.asarray(h5_file[key])


def _vector_stats(arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.reshape(arr.shape[0], -1)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "global_min": float(flat.min()),
        "global_max": float(flat.max()),
        "global_mean": float(flat.mean()),
        "global_std": float(flat.std()),
        "per_dim_mean": [float(x) for x in flat.mean(axis=0)],
        "per_dim_std": [float(x) for x in flat.std(axis=0)],
        "first_rows": flat[:3].tolist(),
    }


def _quat_stats(arr: np.ndarray) -> Dict[str, Any] | None:
    flat = arr.reshape(arr.shape[0], -1)
    if flat.shape[1] != 7:
        return None

    quat = flat[:, 3:7]
    norms = np.linalg.norm(quat, axis=1)
    return {
        "quat_norm_mean": float(norms.mean()),
        "quat_norm_std": float(norms.std()),
        "quat_norm_min": float(norms.min()),
        "quat_norm_max": float(norms.max()),
        "frac_norm_within_0p01_of_1": float(np.mean(np.abs(norms - 1.0) < 0.01)),
        "frac_norm_within_0p05_of_1": float(np.mean(np.abs(norms - 1.0) < 0.05)),
        "frac_norm_within_0p10_of_1": float(np.mean(np.abs(norms - 1.0) < 0.10)),
    }


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _quat_alignment(a: np.ndarray, b: np.ndarray) -> Dict[str, float] | None:
    if a.shape[1] != 7 or b.shape[1] != 7:
        return None
    qa = a[:, 3:7]
    qb = b[:, 3:7]
    qa_norm = np.linalg.norm(qa, axis=1, keepdims=True)
    qb_norm = np.linalg.norm(qb, axis=1, keepdims=True)
    if np.any(qa_norm == 0.0) or np.any(qb_norm == 0.0):
        return None

    qa_unit = qa / qa_norm
    qb_unit = qb / qb_norm
    dots = np.abs(np.sum(qa_unit * qb_unit, axis=1))
    return {
        "quat_abs_dot_mean": float(dots.mean()),
        "quat_abs_dot_min": float(dots.min()),
        "frac_quat_abs_dot_gt_0p95": float(np.mean(dots > 0.95)),
        "frac_quat_abs_dot_gt_0p99": float(np.mean(dots > 0.99)),
    }


def _paired_metrics(action: np.ndarray, obs: np.ndarray) -> Dict[str, Any]:
    a = action.reshape(action.shape[0], -1)
    o = obs.reshape(obs.shape[0], -1)
    T = min(len(a), len(o))
    a = a[:T]
    o = o[:T]

    result: Dict[str, Any] = {
        "same_timestep_mse": _mse(a, o),
        "same_timestep_mae": _mae(a, o),
    }

    quat_same = _quat_alignment(a, o)
    if quat_same is not None:
        result["same_timestep_quat_alignment"] = quat_same

    if T > 1:
        o_next = o[1:]
        a_prev = a[:-1]
        o_prev = o[:-1]
        o_delta = o_next - o_prev

        result.update(
            {
                "vs_next_obs_mse": _mse(a_prev, o_next),
                "vs_next_obs_mae": _mae(a_prev, o_next),
                "vs_obs_delta_mse": _mse(a_prev, o_delta),
                "vs_obs_delta_mae": _mae(a_prev, o_delta),
            }
        )

        quat_next = _quat_alignment(a_prev, o_next)
        if quat_next is not None:
            result["vs_next_obs_quat_alignment"] = quat_next

    return result


def _heuristic_summary(action: np.ndarray, obs: np.ndarray | None) -> List[str]:
    notes: List[str] = []
    flat = action.reshape(action.shape[0], -1)

    if flat.shape[1] == 7:
        quat = _quat_stats(flat)
        assert quat is not None
        if quat["frac_norm_within_0p05_of_1"] > 0.8:
            notes.append("Action looks quaternion-like in dims 3:7, so this may be an EE pose or pose-like command.")
        else:
            notes.append("Action is 7D but the last 4 dims do not look strongly unit-quaternion-like.")
    else:
        notes.append(f"Action is {flat.shape[1]}D, so it is not a simple 3D position + quaternion pose vector.")

    if obs is not None:
        paired = _paired_metrics(action, obs)
        next_mse = paired.get("vs_next_obs_mse")
        delta_mse = paired.get("vs_obs_delta_mse")
        same_mse = paired.get("same_timestep_mse")

        if next_mse is not None and delta_mse is not None:
            if next_mse < delta_mse and next_mse < same_mse:
                notes.append("Action is numerically closer to the next observation than to an observation delta.")
            elif delta_mse < next_mse and delta_mse < same_mse:
                notes.append("Action is numerically closer to an observation delta than to an absolute observation.")
            else:
                notes.append("Action does not clearly match same-step obs, next-step obs, or simple obs delta.")

    notes.append("This script is heuristic: it can suggest semantics, but key names and controller code are still the final source of truth.")
    return notes


def inspect_action_key(h5_path: Path, action_key: str, obs_key: str | None) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5_file:
        action = _load_array(h5_file, action_key)
        if action.ndim == 1:
            action = action[:, None]
        report: Dict[str, Any] = {
            "action_key": action_key,
            "action_stats": _vector_stats(action),
        }

        quat = _quat_stats(action)
        if quat is not None:
            report["action_quaternion_stats"] = quat

        obs = None
        if obs_key is not None:
            obs = _load_array(h5_file, obs_key)
            if obs.ndim == 1:
                obs = obs[:, None]
            report["observation_key"] = obs_key
            report["observation_stats"] = _vector_stats(obs)
            obs_quat = _quat_stats(obs)
            if obs_quat is not None:
                report["observation_quaternion_stats"] = obs_quat

            if action.shape[1:] == obs.shape[1:]:
                report["paired_metrics"] = _paired_metrics(action, obs)
            else:
                report["paired_metrics_warning"] = (
                    "Skipped paired action/observation comparison because the trailing dimensions differ."
                )

        report["summary"] = _heuristic_summary(action, obs)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect whether H5 action tensors look like joint or EE pose commands.")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to a single H5 episode.")
    parser.add_argument(
        "--action_keys",
        nargs="+",
        required=True,
        help="Action dataset keys to inspect, e.g. actions_arm_left actions_arm_right",
    )
    parser.add_argument(
        "--observation_keys",
        nargs="*",
        default=None,
        help="Optional observation keys aligned with action_keys, e.g. EE pose observations for comparison.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/processed/action_semantics.json",
        help="Where to write the full JSON report.",
    )
    args = parser.parse_args()

    if args.observation_keys is not None and len(args.observation_keys) not in (0, len(args.action_keys)):
        raise ValueError("observation_keys must be omitted or have the same length as action_keys.")

    observation_keys = args.observation_keys or [None] * len(args.action_keys)
    reports = []
    for action_key, obs_key in zip(args.action_keys, observation_keys):
        reports.append(inspect_action_key(Path(args.input_h5), action_key, obs_key))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_h5": args.input_h5,
                "reports": reports,
            },
            f,
            indent=2,
        )

    print(f"[inspect_action_semantics] wrote JSON: {output_path}")
    for report in reports:
        print("")
        print(f"=== {report['action_key']} ===")
        print(f"shape: {report['action_stats']['shape']}, dtype: {report['action_stats']['dtype']}")
        if "observation_key" in report:
            print(f"paired observation: {report['observation_key']}")
        if "action_quaternion_stats" in report:
            quat = report["action_quaternion_stats"]
            print(
                "quat norm mean/std:"
                f" {quat['quat_norm_mean']:.4f} / {quat['quat_norm_std']:.4f}"
                f" | frac within 0.05 of 1: {quat['frac_norm_within_0p05_of_1']:.3f}"
            )
        paired = report.get("paired_metrics")
        if paired is not None:
            print(
                "paired mse:"
                f" same={paired['same_timestep_mse']:.6f}"
                f", next={paired.get('vs_next_obs_mse', float('nan')):.6f}"
                f", delta={paired.get('vs_obs_delta_mse', float('nan')):.6f}"
            )
        for note in report["summary"]:
            print(f"- {note}")


if __name__ == "__main__":
    main()
