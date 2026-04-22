from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


IMAGE_HINTS = ("image", "rgb", "camera", "cam")
STATE_HINTS = ("qpos", "joint", "state", "proprio", "robot")
ACTION_HINTS = ("action", "control", "cmd")


@dataclass
class H5NodeInfo:
    key: str
    shape: Tuple[int, ...]
    dtype: str


def walk_h5_datasets(handle: h5py.Group, prefix: str = "") -> Iterable[H5NodeInfo]:
    for key, item in handle.items():
        node_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            yield H5NodeInfo(key=node_key, shape=tuple(item.shape), dtype=str(item.dtype))
        elif isinstance(item, h5py.Group):
            yield from walk_h5_datasets(item, prefix=node_key)


def classify_key(key: str) -> str:
    lower = key.lower()
    if any(h in lower for h in IMAGE_HINTS):
        return "image"
    if any(h in lower for h in ACTION_HINTS):
        return "action"
    if any(h in lower for h in STATE_HINTS):
        return "state"
    return "other"


def first_dim_timesteps(shape: Tuple[int, ...]) -> int | None:
    if len(shape) == 0:
        return None
    return int(shape[0])


def load_episode_filepaths(raw_dir: str | Path, file_glob: str = "*.h5") -> List[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(file_glob))
    return [f for f in files if f.is_file()]


def safe_read_dataset(h5_file: h5py.File, key: str) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"Key '{key}' not found in H5 file. Available keys are nested; inspect file first.")
    return np.asarray(h5_file[key])
