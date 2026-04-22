from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.h5_dataset import H5EpisodeDataset
from evaluation.metrics import action_mse, endpoint_error, per_dim_mse
from models.bc_policy import BCPolicy
from models.flow_policy import FlowMatchingPolicy
from training.engine import move_batch_to_device
from utils.config_utils import load_yaml


def build_loader(dataset_cfg, split: str, batch_size: int):
    ds = H5EpisodeDataset(
        raw_dir=dataset_cfg["paths"]["raw_dir"],
        split_file=dataset_cfg["paths"]["splits_file"],
        split=split,
        image_key=dataset_cfg["keys"]["image_key"],
        proprio_key=dataset_cfg["keys"]["proprio_key"],
        action_key=dataset_cfg["keys"]["action_key"],
        file_glob=dataset_cfg["h5"]["file_glob"],
        frame_stack=dataset_cfg["sequence"]["frame_stack"],
        action_chunk=dataset_cfg["sequence"]["action_chunk"],
        action_stride=dataset_cfg["sequence"]["action_stride"],
        resize_hw=dataset_cfg["image"]["resize_hw"],
        normalize_images=dataset_cfg["image"]["normalize"],
        image_mean=dataset_cfg["image"]["mean"],
        image_std=dataset_cfg["image"]["std"],
        derive_action_if_missing=dataset_cfg["fallback"]["derive_action_if_missing"],
        index_cache_file=dataset_cfg["paths"].get("index_cache_file"),
    )
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def load_bc(ckpt_path, sample, cfg):
    model = BCPolicy(
        image_channels=sample["image"].shape[0],
        proprio_dim=sample["proprio"].numel(),
        action_dim=sample["action"].numel(),
        image_encoder_name=cfg["model"]["image_encoder"],
        image_feature_dim=cfg["model"]["image_feature_dim"],
        proprio_hidden=cfg["model"]["proprio_hidden"],
        fusion_hidden=cfg["model"]["fusion_hidden"],
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


def load_flow(ckpt_path, sample, cfg):
    model = FlowMatchingPolicy(
        image_channels=sample["image"].shape[0],
        proprio_dim=sample["proprio"].numel(),
        action_dim=sample["action"].numel(),
        image_encoder_name=cfg["model"]["image_encoder"],
        image_feature_dim=cfg["model"]["image_feature_dim"],
        proprio_hidden=cfg["model"]["proprio_hidden"],
        cond_hidden=cfg["model"]["cond_hidden"],
        flow_hidden=cfg["model"]["flow_hidden"],
        time_embed_dim=cfg["model"]["time_embed_dim"],
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


def eval_model(name, model, loader, device, step_dim, is_flow=False):
    model = model.to(device).eval()
    sum_mse, sum_epe, n = 0.0, 0.0, 0
    per_dim_acc = None

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = model.sample(batch["image"], batch["proprio"], steps=20) if is_flow else model(batch["image"], batch["proprio"])
            gt = batch["action"]

            bsz = gt.size(0)
            sum_mse += action_mse(pred, gt).item() * bsz
            sum_epe += endpoint_error(pred, gt, step_dim).item() * bsz
            n += bsz

            pd = per_dim_mse(pred, gt)
            if per_dim_acc is None:
                per_dim_acc = {k: 0.0 for k in pd.keys()}
            for k, v in pd.items():
                per_dim_acc[k] += v * bsz

    result = {"model": name, "action_mse": sum_mse / n, "endpoint_error": sum_epe / n}
    if per_dim_acc:
        result.update({k: v / n for k, v in per_dim_acc.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    eval_cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(eval_cfg["dataset_config"])
    bc_cfg = load_yaml("configs/bc.yaml")
    flow_cfg = load_yaml("configs/flow.yaml")

    ds, loader = build_loader(dataset_cfg, eval_cfg["evaluation"]["split"], eval_cfg["evaluation"]["batch_size"])
    sample = ds[0]
    step_dim = sample["proprio"].numel()

    bc_model = load_bc(eval_cfg["checkpoints"]["bc_checkpoint"], sample, bc_cfg)
    flow_model = load_flow(eval_cfg["checkpoints"]["flow_checkpoint"], sample, flow_cfg)

    device = torch.device(eval_cfg["evaluation"]["device"] if torch.cuda.is_available() else "cpu")
    results = [
        eval_model("bc", bc_model, loader, device, step_dim, is_flow=False),
        eval_model("flow", flow_model, loader, device, step_dim, is_flow=True),
    ]

    out_dir = Path(eval_cfg["evaluation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "offline_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[evaluate_offline] results:")
    for r in results:
        print(r)
    print(f"[evaluate_offline] wrote {out_path}")


if __name__ == "__main__":
    main()
