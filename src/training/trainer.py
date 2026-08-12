# src/training/trainer.py
#
# Changes vs. previous version:
#   1. setup_mlflow_tracking(): tries the remote DagsHub tracking URI from
#      config.yaml; if auth/network fails (e.g. the 403 you hit — missing
#      MLFLOW_TRACKING_USERNAME/PASSWORD for a fresh machine), falls back
#      to local file-based tracking at ./mlruns instead of crashing the
#      whole run.
#   2. Best-checkpoint selection now uses recall_at_50 (matches the metric
#      you actually want to track) instead of hit_rate_at_10.
#   3. mlflow.log_metrics / log_artifact / pytorch.log_model calls are
#      wrapped so a later transient tracking failure can't kill an
#      in-progress multi-epoch run.

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import mlflow
from tqdm import tqdm
import time

from src.models.two_tower import TwoTowerModel
from src.training.dataset import InteractionDataset
from src.training.evaluate import evaluate_model
from src.utils import get_artifacts_dir

try:
    import mlflow.pytorch
    MLFLOW_PYTORCH_AVAILABLE = True
except ImportError:
    MLFLOW_PYTORCH_AVAILABLE = False


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_mlflow_tracking(config: dict) -> None:
    """
    Try the remote (DagsHub) tracking URI from config.yaml. If it's
    unreachable or unauthenticated, fall back to local file-based
    tracking instead of letting the whole training run crash on a
    tracking-server problem.
    """
    remote_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    try:
        mlflow.set_tracking_uri(remote_uri)
        mlflow.set_experiment(experiment_name)
        # Cheap call that actually exercises auth, so we find out now
        # rather than mid-training.
        mlflow.search_experiments(max_results=1)
        print(f"MLflow tracking -> remote: {remote_uri}")
        return
    except Exception as e:
        print(f"Could not use remote MLflow tracking ({remote_uri}): {e}")
        print(
            "Falling back to local tracking at ./mlflow.db (sqlite). To "
            "use the hosted DagsHub tracker, set MLFLOW_TRACKING_USERNAME "
            "(your DagsHub username) and MLFLOW_TRACKING_PASSWORD "
            "(a DagsHub access token, from Settings -> Tokens) as "
            "environment variables and rerun."
        )
        # Plain "file:./mlruns" is maintenance-mode-only on current MLflow
        # versions and raises unless MLFLOW_ALLOW_FILE_STORE=true is set.
        # sqlite is the supported local backend and needs no extra flags.
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)


def safe_log_metrics(log_dict: dict, step: int) -> None:
    try:
        mlflow.log_metrics(log_dict, step=step)
    except Exception as e:
        print(f" [mlflow] log_metrics failed, continuing without it: {e}")


def safe_log_artifact(path: str) -> None:
    try:
        mlflow.log_artifact(path)
    except Exception as e:
        print(f" [mlflow] log_artifact failed, continuing without it: {e}")


def safe_log_model(model) -> None:
    if not MLFLOW_PYTORCH_AVAILABLE:
        return
    try:
        mlflow.pytorch.log_model(
            model, "two-tower-model", registered_model_name="TwoTowerRetriever"
        )
    except Exception as e:
        # Model Registry needs a database-backed store; the local file
        # fallback above doesn't support it, so this is expected to skip
        # when running locally. Not fatal either way.
        print(f" [mlflow] log_model/registry failed, continuing without it: {e}")


def train_one_epoch(
    model: TwoTowerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for hist, scalar, item_feat, sampling_probs in pbar:
        hist = hist.to(device)
        scalar = scalar.to(device)
        item_feat = item_feat.to(device)
        sampling_probs = sampling_probs.to(device)

        optimizer.zero_grad()
        loss = model(hist, scalar, item_feat, sampling_probs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def compute_val_loss(
    model: TwoTowerModel,
    loader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for hist, scalar, item_feat, sampling_probs in loader:
            hist = hist.to(device)
            scalar = scalar.to(device)
            item_feat = item_feat.to(device)
            sampling_probs = sampling_probs.to(device)

            loss = model(hist, scalar, item_feat, sampling_probs)
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def run_training(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)

    processed_dir = Path(config["data"]["processed_dir"])
    artifacts_dir = get_artifacts_dir(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Artifacts will be saved to: {artifacts_dir.resolve()}")

    print("Loading data...")
    train_interactions = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_interactions = pd.read_parquet(processed_dir / "val_interactions.parquet")
    user_features = pd.read_parquet(processed_dir / "user_features.parquet")
    item_features = pd.read_parquet(processed_dir / "item_features.parquet")

    history_embed_dim = config["features"]["text_embed_dim"]
    max_history_len = config["features"]["max_history_len"]

    train_dataset = InteractionDataset(
        train_interactions,
        user_features,
        item_features,
        history_embed_dim=history_embed_dim,
        max_history_len=max_history_len,
        causal=True,
    )

    val_dataset = InteractionDataset(
        val_interactions,
        user_features,
        item_features,
        history_embed_dim=history_embed_dim,
        max_history_len=max_history_len,
        causal=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0
    )

    scalar_dim = train_dataset.scalar_dim
    item_dim = train_dataset.item_dim

    print(f"History shape:   ({max_history_len}, {history_embed_dim})")
    print(f"Scalar feature dim: {scalar_dim}")
    print(f"Item feature dim:  {item_dim}")

    setup_mlflow_tracking(config)

    run_name = (
        f"emb{config['model']['embedding_dim']}"
        f"_lr{config['training']['learning_rate']}"
        f"_bs{config['training']['batch_size']}"
        f"_attn{config['model']['attention_heads']}x{config['model']['attention_layers']}"
        f"_temp{config['model']['temperature']}"
        f"_causal-fix"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "embedding_dim": config["model"]["embedding_dim"],
            "user_hidden_dims": str(config["model"]["user_hidden_dims"]),
            "item_hidden_dims": str(config["model"]["item_hidden_dims"]),
            "attention_heads": config["model"]["attention_heads"],
            "attention_layers": config["model"]["attention_layers"],
            "dropout": config["model"]["dropout"],
            "temperature": config["model"]["temperature"],
            "learning_rate": config["training"]["learning_rate"],
            "warmup_epochs": config["training"]["warmup_epochs"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "logq_correction": config["training"]["logq_correction"],
            "max_history_len": max_history_len,
            "history_embed_dim": history_embed_dim,
            "scalar_dim": scalar_dim,
            "item_dim": item_dim,
            "train_size": len(train_interactions),
            "device": str(device),
            "causal_training_history": True,
            "checkpoint_metric": "recall_at_50",
        })

        model = TwoTowerModel(
            history_embed_dim=history_embed_dim,
            scalar_feature_dim=scalar_dim,
            item_input_dim=item_dim,
            user_hidden_dims=config["model"]["user_hidden_dims"],
            item_hidden_dims=config["model"]["item_hidden_dims"],
            embedding_dim=config["model"]["embedding_dim"],
            num_attention_heads=config["model"]["attention_heads"],
            num_attention_layers=config["model"]["attention_layers"],
            max_history_len=max_history_len,
            dropout=config["model"]["dropout"],
            temperature=config["model"]["temperature"],
            logq_correction=config["training"]["logq_correction"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        warmup_epochs = config["training"]["warmup_epochs"]
        cosine_epochs = config["training"]["epochs"] - warmup_epochs

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        best_recall_at_50 = 0.0
        best_model_path = artifacts_dir / "best_model.pt"

        for epoch in range(1, config["training"]["epochs"] + 1):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch
            )

            val_loss = None
            val_metrics = {}

            if epoch % 2 == 0 or epoch == config["training"]["epochs"]:
                print(f"\nEvaluating epoch {epoch}...")

                val_loss = compute_val_loss(model, val_loader, device)

                val_metrics = evaluate_model(
                    model,
                    val_interactions,
                    item_features,
                    user_features,
                    device,
                    history_embed_dim=history_embed_dim,
                    max_history_len=max_history_len,
                    k_values=[10, 50]
                )

            scheduler.step()
            elapsed = time.time() - t0

            log_dict = {
                "train_loss": train_loss,
                "epoch_time_sec": elapsed
            }

            if val_loss is not None:
                log_dict["val_loss"] = val_loss

            log_dict.update(val_metrics)

            log_dict = {
                k: v for k, v in log_dict.items()
                if not (isinstance(v, float) and np.isnan(v))
            }

            safe_log_metrics(log_dict, step=epoch)

            print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | {elapsed:.1f}s")

            if val_loss is not None:
                print(f" val_loss: {val_loss:.4f}")

            for k, v in val_metrics.items():
                print(f" {k}: {v:.4f}")

            if val_metrics.get("recall_at_50", 0) > best_recall_at_50:
                best_recall_at_50 = val_metrics["recall_at_50"]
                torch.save(model.state_dict(), best_model_path)
                print(f" New best Recall@50: {best_recall_at_50:.4f} - saved")

        safe_log_metrics({"best_recall_at_50": best_recall_at_50}, step=config["training"]["epochs"])
        safe_log_artifact(str(best_model_path))
        safe_log_model(model)

        print(f"\nTraining complete. Best Recall@50: {best_recall_at_50:.4f}")


if __name__ == "__main__":
    run_training()