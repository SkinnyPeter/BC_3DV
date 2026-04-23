"""Hugging Face H5 Dataset - streams data without full download."""
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import huggingface_hub


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
    """Derive surrogate action as next-step state delta."""
    deltas = np.zeros_like(states)
    deltas[:-1] = states[1:] - states[:-1]
    deltas[-1] = deltas[-2] if len(states) > 1 else 0.0
    return deltas


def _coerce_keys(keys: Sequence[str] | None, key: str | None, name: str) -> list[str]:
    if keys is not None:
        if len(keys) == 0:
            raise ValueError(f"{name} must not be empty.")
        return list(keys)
    if key is not None:
        return [key]
    raise ValueError(f"Either {name} or {name[:-1]} must be provided.")


class HFH5EpisodeDataset(Dataset):
    """Dataset that streams H5 files from Hugging Face."""

    def __init__(
        self,
        hf_repo_id: str,
        split_file: str,  # local file with train/val splits
        split: str,
        image_keys: list[str],
        proprio_key: str | None = None,
        action_key: str | None = "actions",
        proprio_keys: Sequence[str] | None = None,
        action_keys: Sequence[str] | None = None,
        frame_stack: int = 1,
        action_chunk: int = 8,
        action_stride: int = 1,
        resize_hw: Sequence[int] = (128, 128),
        normalize_images: bool = True,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        derive_action_if_missing: bool = True,
    ) -> None:
        super().__init__()
        self.hf_repo_id = hf_repo_id
        self.image_keys = image_keys
        self.proprio_keys = _coerce_keys(proprio_keys, proprio_key, "proprio_keys")
        self.action_keys = _coerce_keys(action_keys, action_key, "action_keys")
        if derive_action_if_missing and len(self.action_keys) != len(self.proprio_keys):
            raise ValueError("action_keys and proprio_keys must have the same length when deriving actions.")
        self.frame_stack = frame_stack
        self.action_chunk = action_chunk
        self.action_stride = action_stride
        self.resize_hw = tuple(resize_hw)
        self.normalize_images = normalize_images
        self.image_mean = image_mean
        self.image_std = image_std
        self.derive_action_if_missing = derive_action_if_missing

        # Load splits from local file
        with open(split_file, "r") as f:
            splits = json.load(f)
        self.episode_names = splits[split]

        # Get file list from HF
        self.api = huggingface_hub.HfApi()
        self.file_list = self._get_file_list()

        # Build sample index
        self.samples = self._build_index()

    def _get_file_list(self) -> Dict[str, str]:
        """Get list of H5 files from repo."""
        files = self.api.list_repo_files(self.hf_repo_id)
        h5_files = [f for f in files if f.endswith(".h5")]
        return {Path(f).name: f for f in h5_files}

    def _build_index(self) -> List[Dict[str, Any]]:
        """Build index of all samples."""
        samples = []
        for ep_name in self.episode_names:
            if ep_name not in self.file_list:
                print(f"Warning: {ep_name} not found in HF repo")
                continue

            # Get episode length - we need to stream to know this
            # For now, we'll try to get it from metadata or assume a fixed length
            # You may need to adjust this based on your data
            T = 200  # placeholder - adjust based on your data

            min_t = self.frame_stack - 1
            max_t = T - 1 - (self.action_chunk - 1) * self.action_stride
            for t in range(min_t, max_t + 1):
                samples.append({"episode_name": ep_name, "t": t})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frame_stack(self, f: h5py.File, t: int, image_key: str) -> torch.Tensor:
        """Read frame stack for a single camera."""
        frames = []
        for i in range(self.frame_stack):
            ti = t - (self.frame_stack - 1 - i)
            img_np = np.asarray(f[image_key][ti])
            img = _to_chw_float(img_np)
            img = F.interpolate(img.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
            if self.normalize_images:
                img = _normalize_image(img, self.image_mean, self.image_std)
            frames.append(img)
        return torch.cat(frames, dim=0)

    def _read_proprio(self, f: h5py.File, t: int) -> torch.Tensor:
        parts = [torch.from_numpy(np.asarray(f[key][t])).float().reshape(-1) for key in self.proprio_keys]
        return torch.cat(parts, dim=0)

    def _read_action_chunk(self, f: h5py.File, idxs: list[int], ep_name: str) -> tuple[torch.Tensor, bool]:
        chunks = []
        action_is_derived = False
        for i, action_key in enumerate(self.action_keys):
            if action_key in f:
                action_all = np.asarray(f[action_key])
            else:
                if not self.derive_action_if_missing:
                    raise KeyError(f"Missing action key '{action_key}' in {ep_name}")
                states = np.asarray(f[self.proprio_keys[i]])
                action_all = derive_surrogate_actions_from_states(states)
                action_is_derived = True
            chunks.append(torch.from_numpy(np.asarray(action_all[idxs])).float())
        return torch.cat(chunks, dim=-1).reshape(-1), action_is_derived

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        ep_name = s["episode_name"]
        t = s["t"]

        # Stream the specific episode file from HF
        remote_path = self.file_list[ep_name]

        # Download only the needed file (streams, doesn't download everything)
        local_path = huggingface_hub.hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=remote_path
        )

        with h5py.File(local_path, "r") as f:
            # Read multiple cameras and concatenate
            images = []
            for image_key in self.image_keys:
                img = self._read_frame_stack(f, t, image_key)
                images.append(img)
            image = torch.cat(images, dim=0)  # (num_cameras * C, H, W)

            idxs = [t + i * self.action_stride for i in range(self.action_chunk)]
            proprio = self._read_proprio(f, t)
            action_chunk, action_is_derived = self._read_action_chunk(f, idxs, ep_name)

        return {
            "image": image,
            "proprio": proprio,
            "action": action_chunk,
            "episode_id": ep_name,
            "timestep": t,
            "action_is_derived": action_is_derived,
        }
