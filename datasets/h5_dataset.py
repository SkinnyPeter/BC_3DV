from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from datasets.utils import load_episode_filepaths


def _normalize_image(img: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    return (img - mean_t) / std_t


def _to_chw_float(image_np: np.ndarray) -> torch.Tensor:
    img = torch.from_numpy(image_np)
    if img.ndim == 2:
        img = img.unsqueeze(-1)
    if img.shape[-1] in (1, 3):
        img = img.permute(2, 0, 1)
    img = img.float()
    if img.max() > 1.0:
        img = img / 255.0
    return img


def derive_surrogate_actions_from_states(states: np.ndarray) -> np.ndarray:
    """Derive surrogate action as next-step state delta.

    TODO(project-specific): replace with true action derivation if robot dynamics are known.
    """
    deltas = np.zeros_like(states)
    deltas[:-1] = states[1:] - states[:-1]
    deltas[-1] = deltas[-2] if len(states) > 1 else 0.0
    return deltas


class H5EpisodeDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        split_file: str,
        split: str,
        image_key: str,
        proprio_key: str,
        action_key: str = "actions",
        file_glob: str = "*.h5",
        frame_stack: int = 1,
        action_chunk: int = 1,
        action_stride: int = 1,
        resize_hw: Sequence[int] = (128, 128),
        normalize_images: bool = True,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        derive_action_if_missing: bool = True,
        index_cache_file: str | None = None,
    ) -> None:
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.frame_stack = frame_stack
        self.action_chunk = action_chunk
        self.action_stride = action_stride
        self.resize_hw = tuple(resize_hw)
        self.normalize_images = normalize_images
        self.image_mean = image_mean
        self.image_std = image_std
        self.derive_action_if_missing = derive_action_if_missing

        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        split_eps = set(splits[split])

        all_files = load_episode_filepaths(self.raw_dir, file_glob=file_glob)
        self.episode_files = [p for p in all_files if p.name in split_eps]
        if not self.episode_files:
            raise RuntimeError(f"No episodes found for split={split}. Check split file and raw_dir.")

        self.samples: List[Dict[str, Any]] = self._build_or_load_index(index_cache_file, split)

    def _build_or_load_index(self, index_cache_file: str | None, split: str) -> List[Dict[str, Any]]:
        if index_cache_file is not None and Path(index_cache_file).exists():
            with open(index_cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if split in cache:
                return cache[split]

        samples: List[Dict[str, Any]] = []
        for ep_i, path in enumerate(self.episode_files):
            with h5py.File(path, "r") as f:
                T = int(f[self.proprio_key].shape[0])

            min_t = self.frame_stack - 1
            max_t = T - 1 - (self.action_chunk - 1) * self.action_stride
            for t in range(min_t, max_t + 1):
                samples.append({"episode_idx": ep_i, "t": t, "episode_name": path.name})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frame_stack(self, f: h5py.File, t: int) -> torch.Tensor:
        frames = []
        for i in range(self.frame_stack):
            ti = t - (self.frame_stack - 1 - i)
            img_np = np.asarray(f[self.image_key][ti])
            img = _to_chw_float(img_np)
            img = F.interpolate(img.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
            if self.normalize_images:
                img = _normalize_image(img, self.image_mean, self.image_std)
            frames.append(img)
        return torch.cat(frames, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        ep_path = self.episode_files[s["episode_idx"]]
        t = s["t"]

        with h5py.File(ep_path, "r") as f:
            image = self._read_frame_stack(f, t)
            proprio = torch.from_numpy(np.asarray(f[self.proprio_key][t])).float()

            if self.action_key in f:
                action_all = np.asarray(f[self.action_key])
                action_is_derived = False
            else:
                if not self.derive_action_if_missing:
                    raise KeyError(f"Missing action key '{self.action_key}' in {ep_path}")
                states = np.asarray(f[self.proprio_key])
                action_all = derive_surrogate_actions_from_states(states)
                action_is_derived = True

            idxs = [t + i * self.action_stride for i in range(self.action_chunk)]
            action_chunk = torch.from_numpy(np.asarray(action_all[idxs])).float().reshape(-1)

        return {
            "image": image,
            "proprio": proprio,
            "action": action_chunk,
            "episode_id": s["episode_name"],
            "timestep": t,
            "action_is_derived": action_is_derived,
        }
