"""Training script for Hugging Face H5 dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.hf_h5_dataset import HFH5EpisodeDataset
from models.bc_policy import BCPolicy
from training.engine import move_batch_to_device
from training.losses import bc_loss
from utils.config_utils import load_yaml
from utils.logging_utils import build_logger, log_scalars
from utils.seed import set_seed


def build_dataloader(dataset_cfg, split: str, batch_size: int, hf_repo_id: str):
    key_cfg = dataset_cfg["keys"]
    ds = HFH5EpisodeDataset(
        hf_repo_id=hf_repo_id,
        split_file=dataset_cfg["paths"]["splits_file"],
        split=split,
        image_keys=key_cfg["image_keys"],
        proprio_key=key_cfg.get("proprio_key"),
        action_key=key_cfg.get("action_key"),
        proprio_keys=key_cfg.get("proprio_keys"),
        action_keys=key_cfg.get("action_keys"),
        frame_stack=dataset_cfg["sequence"]["frame_stack"],
        action_chunk=dataset_cfg["sequence"]["action_chunk"],
        action_stride=dataset_cfg["sequence"]["action_stride"],
        resize_hw=dataset_cfg["image"]["resize_hw"],
        normalize_images=dataset_cfg["image"]["normalize"],
        image_mean=dataset_cfg["image"]["mean"],
        image_std=dataset_cfg["image"]["std"],
        derive_action_if_missing=dataset_cfg["fallback"]["derive_action_if_missing"],
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


def evaluate(model, loader, device, smoothness_weight, action_step_dim):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = model(batch["image"], batch["proprio"])
            loss = bc_loss(pred, batch["action"], smoothness_weight, action_step_dim)
            total += loss.item() * batch["image"].size(0)
            n += batch["image"].size(0)
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bc.yaml")
    parser.add_argument("--hf_repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., username/dataset)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(cfg["dataset_config"])
    set_seed(cfg["seed"])

    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(cfg["logging"]["backend"], str(out_dir / "tb"))

    print(f"Loading data from Hugging Face: {args.hf_repo_id}")
    train_ds, train_loader = build_dataloader(dataset_cfg, "train", cfg["training"]["batch_size"], args.hf_repo_id)
    _, val_loader = build_dataloader(dataset_cfg, "val", cfg["training"]["batch_size"], args.hf_repo_id)

    sample = train_ds[0]
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    print(f"Sample shapes:")
    print(f"  image: {sample['image'].shape}")
    print(f"  proprio: {sample['proprio'].shape}")
    print(f"  action: {sample['action'].shape}")

    model = BCPolicy(
        image_channels=sample["image"].shape[0],
        proprio_dim=sample["proprio"].numel(),
        action_dim=sample["action"].numel(),
        image_encoder_name=cfg["model"]["image_encoder"],
        image_feature_dim=cfg["model"]["image_feature_dim"],
        proprio_hidden=cfg["model"]["proprio_hidden"],
        fusion_hidden=cfg["model"]["fusion_hidden"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )

    step = 0
    best_val = float("inf")
    action_step_dim = sample["action"].numel() // dataset_cfg["sequence"]["action_chunk"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            pred = model(batch["image"], batch["proprio"])
            loss = bc_loss(pred, batch["action"], cfg["loss"].get("smoothness_weight", 0.0), action_step_dim)
            loss.backward()

            if cfg["training"].get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])

            optimizer.step()

            if step % cfg["logging"]["log_every_steps"] == 0:
                log_scalars(logger, {"train/loss": loss.item()}, step)

            step += 1

        # Validation
        val_loss = evaluate(model, val_loader, device, cfg["loss"].get("smoothness_weight", 0.0), action_step_dim)
        log_scalars(logger, {"val/loss": val_loss}, step)

        print(f"Epoch {epoch}: train loss = {loss.item():.4f}, val loss = {val_loss:.4f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "best_model.pt")
            print(f"  -> Saved best model (val loss: {best_val:.4f})")

        # Save checkpoint
        if epoch % cfg["training"].get("save_every", 10) == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / f"model_epoch_{epoch}.pt")

    # Save final model
    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "final_model.pt")
    logger.close()
    print(f"Training complete! Final model saved to {out_dir}")


if __name__ == "__main__":
    main()
