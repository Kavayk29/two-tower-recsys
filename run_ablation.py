"""
Place at: run_ablation.py (repo root)

Trains a short comparison run with the item tower's text-embedding
columns (item_text_emb_0..383) removed, leaving only genre/year/
popularity features. Uses a SEPARATE artifacts folder and MLflow run
name so it never touches your main training run or its checkpoint.

This does NOT modify any existing files -- it's a standalone diagnostic.

Usage:
    python run_ablation.py --epochs 20 --mode no_text
    python run_ablation.py --epochs 20 --mode full      # baseline, same epoch budget, for a fair comparison

Compare the printed recall_at_50 between the two runs directly -- same
epoch budget both times is what makes the comparison fair, since more
epochs alone can raise recall regardless of features used.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.models.two_tower import TwoTowerModel
from src.training.dataset import InteractionDataset
from src.training.evaluate import evaluate_model
from src.training.trainer import train_one_epoch, compute_val_loss


def load_config(path="configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_item_feat_cols(item_features: pd.DataFrame, mode: str) -> list:
    """
    Columns that feed the ITEM TOWER's own input (the target item's
    scoring features) -- NOT the history mechanism, which always needs
    item_text_emb_* columns present in item_features regardless of mode.
    """
    base_exclude = ["movieId", "year"]
    text_cols = [c for c in item_features.columns if c.startswith("item_text_emb_")]

    if mode == "full":
        return [c for c in item_features.columns if c not in base_exclude]
    elif mode == "no_text":
        print(f"Excluding {len(text_cols)} title-text-embedding columns from "
              f"the item tower's own input (history still uses them, as it must).")
        return [
            c for c in item_features.columns
            if c not in base_exclude and c not in text_cols
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "no_text"], default="no_text")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])

    # Separate artifacts dir per mode -- never overwrites your main run.
    artifacts_dir = Path("artifacts") / f"ablation_{args.mode}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Mode: {args.mode} | Device: {device} | Epochs: {args.epochs}")
    print(f"Artifacts: {artifacts_dir.resolve()}")

    train_interactions = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_interactions = pd.read_parquet(processed_dir / "val_interactions.parquet")
    user_features = pd.read_parquet(processed_dir / "user_features.parquet")
    # Full item_features kept intact -- history construction needs the
    # item_text_emb_* columns regardless of ablation mode. Only the
    # columns fed into the item tower's own input are restricted below.
    item_features = pd.read_parquet(processed_dir / "item_features.parquet")
    item_feat_cols = get_item_feat_cols(item_features, args.mode)

    history_embed_dim = config["features"]["text_embed_dim"]
    max_history_len = config["features"]["max_history_len"]

    train_dataset = InteractionDataset(
        train_interactions, user_features, item_features,
        history_embed_dim=history_embed_dim, max_history_len=max_history_len,
        causal=True, item_feat_cols=item_feat_cols,
    )
    val_dataset = InteractionDataset(
        val_interactions, user_features, item_features,
        history_embed_dim=history_embed_dim, max_history_len=max_history_len,
        causal=False, item_feat_cols=item_feat_cols,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0,
    )

    model = TwoTowerModel(
        history_embed_dim=history_embed_dim,
        scalar_feature_dim=train_dataset.scalar_dim,
        item_input_dim=train_dataset.item_dim,
        user_hidden_dims=config["model"]["user_hidden_dims"],
        item_hidden_dims=config["model"]["item_hidden_dims"],
        embedding_dim=config["model"]["embedding_dim"],
        num_attention_heads=config["model"]["attention_heads"],
        num_attention_layers=config["model"]["attention_layers"],
        max_history_len=max_history_len,
        dropout=config["model"]["dropout"],
        temperature=config["model"]["temperature"],
        logq_correction=config["training"]["logq_correction"],
    ).to(device)

    print(f"Item feature dim for this run: {train_dataset.item_dim} "
          f"(full mode uses {len([c for c in item_features.columns if c not in ['movieId','year']])})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    warmup_epochs = min(config["training"]["warmup_epochs"], max(1, args.epochs // 4))
    cosine_epochs = max(1, args.epochs - warmup_epochs)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs),
        ],
        milestones=[warmup_epochs],
    )

    best_recall_at_50 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        val_metrics = {}
        if epoch % 2 == 0 or epoch == args.epochs:
            val_loss = compute_val_loss(model, val_loader, device)
            val_metrics = evaluate_model(
                model, val_interactions, item_features, user_features, device,
                history_embed_dim=history_embed_dim, max_history_len=max_history_len,
                k_values=[10, 50], item_feat_cols=item_feat_cols,
            )
            print(
                f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"recall@10 {val_metrics['recall_at_10']:.4f} | recall@50 {val_metrics['recall_at_50']:.4f} | "
                f"hit@10 {val_metrics['hit_rate_at_10']:.4f}"
            )
            if val_metrics["recall_at_50"] > best_recall_at_50:
                best_recall_at_50 = val_metrics["recall_at_50"]
                torch.save(model.state_dict(), artifacts_dir / "best_model.pt")
        else:
            print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f}")

    print(f"\n[{args.mode}] Best recall@50 over {args.epochs} epochs: {best_recall_at_50:.4f}")


if __name__ == "__main__":
    main()