import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time

from src.models.two_tower import TwoTowerModel
from src.training.dataset import InteractionDataset
from src.training.evaluate import evaluate_model


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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


def run_training(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    processed_dir = Path(config["data"]["processed_dir"])
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print("Loading data...")
    train_interactions = pd.read_parquet(
        processed_dir / "train_interactions.parquet"
    )
    val_interactions = pd.read_parquet(
        processed_dir / "val_interactions.parquet"
    )
    user_features = pd.read_parquet(processed_dir / "user_features.parquet")
    item_features = pd.read_parquet(processed_dir / "item_features.parquet")

    dataset = InteractionDataset(
        train_interactions, user_features, item_features,
        history_embed_dim=config["features"]["text_embed_dim"],
        max_history_len=config["features"]["max_history_len"]
    )

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    history_embed_dim = config["features"]["text_embed_dim"]
    scalar_dim = dataset.scalar_dim
    item_dim = dataset.item_dim

    print(f"History embed dim: {history_embed_dim}")
    print(f"Scalar feature dim: {scalar_dim}")
    print(f"Item feature dim: {item_dim}")
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_name = (
        f"emb{config['model']['embedding_dim']}"
        f"_lr{config['training']['learning_rate']}"
        f"_bs{config['training']['batch_size']}"
        f"_attn{config['model']['attention_heads']}"
        f"_logq{config['training']['logq_correction']}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "embedding_dim": config["model"]["embedding_dim"],
            "hidden_dims": str(config["model"]["user_hidden_dims"]),
            "attention_heads": config["model"]["attention_heads"],
            "dropout": config["model"]["dropout"],
            "temperature": config["model"]["temperature"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "logq_correction": config["training"]["logq_correction"],
            "history_embed_dim": history_embed_dim,
            "scalar_dim": scalar_dim,
            "item_dim": item_dim,
            "train_size": len(train_interactions),
            "device": str(device)
        })

        model = TwoTowerModel(
            history_embed_dim=history_embed_dim,
            scalar_feature_dim=scalar_dim,
            item_input_dim=item_dim,
            hidden_dims=config["model"]["user_hidden_dims"],
            embedding_dim=config["model"]["embedding_dim"],
            num_attention_heads=config["model"]["attention_heads"],
            dropout=config["model"]["dropout"],
            temperature=config["model"]["temperature"],
            logq_correction=config["training"]["logq_correction"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        )

        best_ndcg = 0.0
        best_model_path = artifacts_dir / "best_model.pt"

        for epoch in range(1, config["training"]["epochs"] + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, loader, optimizer, device, epoch
            )

            val_metrics = {}
            if epoch % 2 == 0 or epoch == config["training"]["epochs"]:
                print(f"\nEvaluating epoch {epoch}...")
                val_metrics = evaluate_model(
                    model, val_interactions, item_features,
                    user_features, device,
                    history_embed_dim=history_embed_dim,
                    k_values=[10, 50]
                )

            scheduler.step()
            elapsed = time.time() - t0

            log_dict = {"train_loss": train_loss, "epoch_time_sec": elapsed}
            log_dict.update(val_metrics)

            log_dict = {k: v for k,v in log_dict.items() if not (isinstance(v,float) and np.isnan(v))}
            mlflow.log_metrics(log_dict, step=epoch)

            print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | {elapsed:.1f}s")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")

            if val_metrics.get("ndcg_at_10", 0) > best_ndcg:
                best_ndcg = val_metrics["ndcg_at_10"]
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best NDCG@10: {best_ndcg:.4f} — saved")

        mlflow.log_metric("best_ndcg_at_10", best_ndcg)
        mlflow.log_artifact(str(best_model_path))
        mlflow.pytorch.log_model(
            model, "two-tower-model",
            registered_model_name="TwoTowerRetriever"
        )

        print(f"\nTraining complete. Best NDCG@10: {best_ndcg:.4f}")


if __name__ == "__main__":
    run_training()