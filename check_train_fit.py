"""
Place at: check_train_fit.py (repo root)

Diagnostic-only script. Loads the current best checkpoint and computes
recall@k on a TRAIN-set leave-one-out probe using the SAME causal history
construction training used (via InteractionDataset(causal=True)), so it's
a fair comparison to val recall from evaluate.py -- not inflated by the
static full-history leakage evaluate.py would otherwise introduce if you
just pointed it at train_interactions directly.

Protocol: for a random sample of users, take their LAST train interaction
as a single held-out probe item; build history from everything strictly
before it (identical to what training saw for that row); rank the full
catalog; check if the probe item lands in top-k. This is a standard
leave-one-out eval used across recsys literature (e.g. NCF), and it's
the fairest apples-to-apples read against your val recall.

Run while training is still going, ideally on CPU to avoid GPU
contention with the training process:
    python check_train_fit.py --device cpu
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

from src.models.two_tower import TwoTowerModel
from src.training.dataset import InteractionDataset
from src.training.evaluate import (
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_hit_rate_at_k,
    evaluate_model,
)
from src.utils import get_artifacts_dir


def load_config(path="configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config, scalar_dim, item_dim, history_embed_dim, max_history_len):
    return TwoTowerModel(
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
        logq_correction=False,
    )


def causal_train_recall(
    model, train_interactions, item_features, user_features, device,
    history_embed_dim, max_history_len, k_values, max_users
):
    """Leave-one-out recall on TRAIN, using the same causal history
    construction as training (via InteractionDataset(causal=True))."""

    ds = InteractionDataset(
        train_interactions, user_features, item_features,
        history_embed_dim=history_embed_dim,
        max_history_len=max_history_len,
        causal=True,
    )

    # Last row per user (in ds's sorted userId/timestamp order) = most
    # recent interaction = the leave-one-out probe target.
    df = pd.DataFrame({"user_id": ds.user_ids, "movie_id": ds.movie_ids})
    df["row_idx"] = np.arange(len(df))
    last_rows = df.groupby("user_id").tail(1)

    eligible_users = sorted(last_rows["user_id"].unique().tolist())
    sampled_users = random.Random(42).sample(
        eligible_users, min(max_users, len(eligible_users))
    )
    last_rows = last_rows[last_rows["user_id"].isin(sampled_users)]

    item_feat_cols = [
        c for c in item_features.columns if c not in ["movieId", "year"]
    ]
    item_feat_idx = item_features.set_index("movieId")
    item_feat_idx.index = item_feat_idx.index.astype(int)
    movie_ids = item_features["movieId"].astype(int).values

    all_item_feats = torch.tensor(
        item_feat_idx[item_feat_cols].values.astype(np.float32)
    ).to(device)

    model.eval()
    with torch.no_grad():
        all_item_embs = model.item_tower(all_item_feats)

    metric_names = (
        [f"ndcg_at_{k}" for k in k_values]
        + [f"hit_rate_at_{k}" for k in k_values]
        + [f"recall_at_{k}" for k in k_values]
    )
    metrics = {name: [] for name in metric_names}

    with torch.no_grad():
        for _, row in last_rows.iterrows():
            idx = row["row_idx"]
            hist, scalar, _, _ = ds[idx]
            hist = hist.unsqueeze(0).to(device)
            scalar = scalar.unsqueeze(0).to(device)

            user_emb = model.user_tower(hist, scalar)
            scores = (user_emb @ all_item_embs.T).squeeze(0).cpu().numpy()
            ranked = movie_ids[np.argsort(-scores)].tolist()
            relevant = {int(row["movie_id"])}

            for k in k_values:
                metrics[f"ndcg_at_{k}"].append(compute_ndcg_at_k(relevant, ranked, k))
                metrics[f"hit_rate_at_{k}"].append(compute_hit_rate_at_k(relevant, ranked, k))
                metrics[f"recall_at_{k}"].append(compute_recall_at_k(relevant, ranked, k))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cpu or cuda; default: cuda if available")
    parser.add_argument("--max_users", type=int, default=300)
    args = parser.parse_args()

    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    artifacts_dir = get_artifacts_dir(config)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    train_interactions = pd.read_parquet(processed_dir / "train_interactions.parquet")
    val_interactions = pd.read_parquet(processed_dir / "val_interactions.parquet")
    user_features = pd.read_parquet(processed_dir / "user_features.parquet")
    item_features = pd.read_parquet(processed_dir / "item_features.parquet")

    scalar_cols = [
        c for c in user_features.columns if c not in ["userId", "user_history_embs"]
    ]
    item_feat_cols = [
        c for c in item_features.columns if c not in ["movieId", "year"]
    ]
    history_embed_dim = config["features"]["text_embed_dim"]
    max_history_len = config["features"]["max_history_len"]

    model = build_model(
        config, len(scalar_cols), len(item_feat_cols),
        history_embed_dim, max_history_len
    ).to(device)

    ckpt_path = artifacts_dir / "best_model.pt"
    print(f"Loading checkpoint: {ckpt_path.resolve()}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\nComputing TRAIN recall (causal leave-one-out probe)...")
    train_metrics = causal_train_recall(
        model, train_interactions, item_features, user_features, device,
        history_embed_dim, max_history_len, k_values=[10, 50],
        max_users=args.max_users,
    )

    print("Computing VAL recall (same protocol evaluate.py uses)...")
    val_metrics = evaluate_model(
        model, val_interactions, item_features, user_features, device,
        history_embed_dim=history_embed_dim, max_history_len=max_history_len,
        k_values=[10, 50], max_users=args.max_users,
    )

    print("\n" + "=" * 50)
    print(f"{'Metric':<20}{'Train':>12}{'Val':>12}")
    print("=" * 50)
    for name in ["recall_at_10", "recall_at_50", "hit_rate_at_10", "hit_rate_at_50", "ndcg_at_10", "ndcg_at_50"]:
        print(f"{name:<20}{train_metrics[name]:>12.4f}{val_metrics[name]:>12.4f}")
    print("=" * 50)

    gap = train_metrics["recall_at_50"] - val_metrics["recall_at_50"]
    print(f"\nrecall_at_50 gap (train - val): {gap:+.4f}")
    if gap < 0.05:
        print("-> Small gap: model is likely UNDERFITTING. Consider training "
              "longer, reducing dropout/weight_decay, or the feature-quality "
              "issue (title embedding dominating item features) discussed earlier.")
    else:
        print("-> Larger gap: model is likely OVERFITTING the causal-history "
              "training signal relative to val. Consider more regularization, "
              "or that the train/val distributions differ more than expected.")


if __name__ == "__main__":
    main()