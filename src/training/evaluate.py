import torch
import numpy as np
import pandas as pd
from typing import Dict


def compute_ndcg_at_k(relevant: set, ranked: list, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(r + 2)
        for r, item in enumerate(ranked[:k])
        if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_hit_rate_at_k(relevant: set, ranked: list, k: int) -> float:
    return float(bool(set(ranked[:k]) & relevant))


def evaluate_model(
    model,
    val_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    device: torch.device,
    history_embed_dim: int = 384,
    k_values: list = [10, 50],
    max_users: int = 300
) -> Dict[str, float]:
    model.eval()

    hist_cols = [
        c for c in user_features.columns
        if c.startswith("user_hist_emb_")
    ]
    scalar_cols = [
        c for c in user_features.columns
        if c != "userId" and not c.startswith("user_hist_emb_")
    ]
    item_feat_cols = [
        c for c in item_features.columns
        if c not in ["movieId", "year"]
    ]

    user_feat_idx = user_features.set_index("userId")
    item_feat_idx = item_features.set_index("movieId")
    
    user_feat_idx.index = user_feat_idx.index.astype(int)
    item_feat_idx.index = item_feat_idx.index.astype(int)

    movie_ids = item_features["movieId"].astype(int).values

    # Precompute all item embeddings
    print("  Precomputing item embeddings...")
    all_item_feats = torch.tensor(
        item_feat_idx[item_feat_cols].values.astype(np.float32),
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        all_item_embs = model.item_tower(all_item_feats)  # (N, D)

    user_val_items = (
        val_interactions.groupby("userId")["movieId"]
        .apply(set).to_dict()
    )

    val_users = [
        u for u in list(user_val_items.keys())
        if u in user_feat_idx.index
    ][:max_users]

    if not val_users:
        print("No validation users found in user features.skipping.")
        return {f"ndcg@{k}":0.0 for k in k_values } | {f"hit_rate@{k}":0.0 for k in k_values}

    metrics = {f"ndcg@{k}": [] for k in k_values}
    metrics.update({f"hit_rate@{k}": [] for k in k_values})

    for user_id in val_users:
        hist = torch.tensor(
            user_feat_idx.loc[user_id, hist_cols].values.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 384)

        scalar = torch.tensor(
            user_feat_idx.loc[user_id, scalar_cols].values.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(0).to(device)  # (1, scalar_dim)

        with torch.no_grad():
            user_emb = model.user_tower(hist, scalar)  # (1, D)

        scores = (user_emb @ all_item_embs.T).squeeze(0).cpu().numpy()
        ranked = movie_ids[np.argsort(-scores)].tolist()
        relevant = user_val_items[user_id]

        for k in k_values:
            metrics[f"ndcg@{k}"].append(compute_ndcg_at_k(relevant, ranked, k))
            metrics[f"hit_rate@{k}"].append(
                compute_hit_rate_at_k(relevant, ranked, k)
            )

    return {key: float(np.mean(vals)) for key, vals in metrics.items()}