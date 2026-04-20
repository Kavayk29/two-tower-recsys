import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class InteractionDataset(Dataset):
    """
    Each sample: one positive (user, item) interaction.
    Returns:
      - history_embeddings: (seq_len, text_embed_dim) — for self-attention
      - scalar_features: (scalar_dim,) — profile + behavioural scalars
      - item_features: (item_dim,)
      - sampling_prob: float — for logQ correction
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        history_embed_dim: int = 384,
        max_history_len: int = 50
    ):
        self.history_embed_dim = history_embed_dim
        self.max_history_len = max_history_len

        valid_users = set(user_features["userId"].values)
        valid_items = set(item_features["movieId"].values)

        self.interactions = interactions[
            interactions["userId"].isin(valid_users) &
            interactions["movieId"].isin(valid_items)
        ].reset_index(drop=True)

        print(f"Dataset: {len(self.interactions):,} valid interactions")

        self.user_features = user_features.set_index("userId")
        self.item_features = item_features.set_index("movieId")

        # Split user feature columns into two groups
        all_user_cols = [c for c in user_features.columns if c != "userId"]

        # History embedding columns: user_hist_emb_0 ... user_hist_emb_383
        self.hist_cols = [
            c for c in all_user_cols if c.startswith("user_hist_emb_")
        ]

        # Scalar columns: everything else (genre affinity + profile + behavioural)
        self.scalar_cols = [
            c for c in all_user_cols if not c.startswith("user_hist_emb_")
        ]

        # Item feature columns
        self.item_feat_cols = [
            c for c in item_features.columns
            if c not in ["movieId", "year"]
        ]

        # Sampling probabilities for logQ correction
        item_counts = self.interactions["movieId"].value_counts()
        total = item_counts.sum()
        self.item_sampling_probs = (item_counts / total).to_dict()

        print(f"  History embedding cols: {len(self.hist_cols)}")
        print(f"  Scalar feature cols: {len(self.scalar_cols)}")
        print(f"  Item feature cols: {len(self.item_feat_cols)}")

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int):
        row = self.interactions.iloc[idx]
        user_id = row["userId"]
        movie_id = row["movieId"]

        # History embeddings: reshape to (seq_len=1, embed_dim)
        # We store mean-pooled history as a single vector.
        # The attention layer will treat it as seq_len=1 here.
        # For true sequence attention, extend this to store full history.
        hist = self.user_features.loc[user_id, self.hist_cols].values.astype(np.float32)
        hist_tensor = torch.tensor(hist, dtype=torch.float32).unsqueeze(0)
        # Shape: (1, 384) — single pooled vector treated as sequence of length 1

        scalar = self.user_features.loc[
            user_id, self.scalar_cols
        ].values.astype(np.float32)

        item_feat = self.item_features.loc[
            movie_id, self.item_feat_cols
        ].values.astype(np.float32)

        sampling_prob = np.float32(
            self.item_sampling_probs.get(movie_id, 1e-9)
        )

        return (
            hist_tensor,
            torch.tensor(scalar, dtype=torch.float32),
            torch.tensor(item_feat, dtype=torch.float32),
            torch.tensor(sampling_prob, dtype=torch.float32)
        )

    @property
    def scalar_dim(self) -> int:
        return len(self.scalar_cols)

    @property
    def item_dim(self) -> int:
        return len(self.item_feat_cols)