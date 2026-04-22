from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.h5_dataset import H5EpisodeDataset
from models.flow_policy import FlowMatchingPolicy
from training.engine import move_batch_to_device
from training.losses import flow_matching_loss
from utils.config_utils import load_yaml
from utils.logging_utils import build_logger, log_scalars
from utils.seed import set_seed


def build_dataloader(dataset_cfg, split: str, batch_size: int):
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
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=dataset_cfg["loader"]["num_workers"],
        pin_memory=dataset_cfg["loader"]["pin_memory"],
        persistent_workers=dataset_cfg["loader"]["persistent_workers"],
    )
    return ds, dl


def compute_flow_loss(model, batch, sigma):
    y = batch["action"]
    bsz = y.size(0)
    t = torch.rand(bsz, device=y.device)
    x0 = sigma * torch.randn_like(y)
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * y
    v_target = y - x0

    cond = model.condition(batch["image"], batch["proprio"])
    v_pred = model(x_t, t, cond)
    return flow_matching_loss(v_pred, v_target)


def evaluate(model, loader, device, sigma):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss = compute_flow_loss(model, batch, sigma)
            total += loss.item() * batch["image"].size(0)
            n += batch["image"].size(0)
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/flow.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(cfg["dataset_config"])
    set_seed(cfg["seed"])

    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(cfg["logging"]["backend"], str(out_dir / "tb"))

    train_ds, train_loader = build_dataloader(dataset_cfg, "train", cfg["training"]["batch_size"])
    _, val_loader = build_dataloader(dataset_cfg, "val", cfg["training"]["batch_size"])
    sample = train_ds[0]

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )

    best_val = float("inf")
    step = 0
    sigma = cfg["flow"]["sigma"]
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for batch in train_loader:
            step += 1
            batch = move_batch_to_device(batch, device)
            loss = compute_flow_loss(model, batch, sigma)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()

            if step % cfg["logging"]["log_every_steps"] == 0:
                log_scalars(logger, {"train/loss": loss.item()}, step)

        if epoch % cfg["training"]["val_every"] == 0:
            val_loss = evaluate(model, val_loader, device, sigma)
            log_scalars(logger, {"val/loss": val_loss}, epoch)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "checkpoint_best.pt")

        if epoch % cfg["training"]["save_every"] == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / f"checkpoint_{epoch:04d}.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "checkpoint_last.pt")
    logger.close()


if __name__ == "__main__":
    main()
