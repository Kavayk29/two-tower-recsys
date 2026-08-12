"""
Place at: src/training/dataset.py

Fixes vs. the original version:

1. Training examples use a CAUSAL, leave-one-out SEQUENCE history (only
   interactions strictly before the target interaction, per user)
   instead of reusing the same static full-history vector for every
   training row of a given user.

2. Training examples ALSO use causally-scoped SCALAR features now
   (genre affinity, activity level, recency) computed from the same
   prior-items window, instead of the static per-user scalar vector
   from user_features.parquet (which reflects the user's ENTIRE train
   history -- including interactions that happen after the row being
   predicted). Demographic fields (gender/age/occupation) stay static
   since they don't change over the training window.

The static full-history / full-scalar path (from user_features.parquet)
is kept for val-loss computation and serving, where using "everything
up to now" is correct and matches evaluate.py / the serving API.

Root cause this fixes: with fixed static features and no causal
truncation, information about a training row's future (including the
target item's own embedding, and the user's eventual total activity
level) leaks into that row's input -- a shortcut that inflates
train-time contrastive accuracy without improving generalization, so
real (val/test) ranking metrics come out far worse than training loss
suggests.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        history_embed_dim: int = 384,
        max_history_len: int = 50,
        causal: bool = False,
        item_feat_cols: list = None,
    ):
        """
        causal=True:  build a leave-one-out sequence history AND
                      leave-one-out scalar features per training example,
                      from `interactions` itself (use for the TRAIN split).
        causal=False: use the static precomputed history/scalars in
                      `user_features` (use for VAL/eval — matches how
                      serving and evaluate.py compute features: everything
                      known "as of now", with no future item excluded
                      because there's no leakage risk there).
        item_feat_cols: optional override for which item_features columns
                      feed the ITEM TOWER's own input (the target item's
                      scoring features). Defaults to all columns except
                      movieId/year. Independent of history construction --
                      history always uses item_text_emb_* columns
                      regardless of this override, since the user-tower
                      history mechanism needs a per-item embedding for
                      each sequence slot. Use this to run feature
                      ablations (e.g. excluding the text embedding from
                      the item tower's own input) without breaking the
                      history mechanism.
        """
        self.history_embed_dim = history_embed_dim
        self.max_history_len = max_history_len
        self.causal = causal

        valid_users = set(user_features["userId"].values)
        valid_items = set(item_features["movieId"].values)

        interactions = interactions[
            interactions["userId"].isin(valid_users)
            & interactions["movieId"].isin(valid_items)
        ].reset_index(drop=True)

        print(f"Dataset: {len(interactions):,} valid interactions (causal={causal})")

        self.user_features = user_features.set_index("userId")
        self.item_features = item_features.set_index("movieId")

        all_user_cols = [c for c in user_features.columns if c != "userId"]
        self.hist_col = "user_history_embs"
        self.scalar_cols = [c for c in all_user_cols if c != self.hist_col]

        if item_feat_cols is not None:
            self.item_feat_cols = item_feat_cols
        else:
            self.item_feat_cols = [
                c for c in item_features.columns if c not in ["movieId", "year"]
            ]

        if causal:
            # --- Sequence history setup (unchanged from before) ---
            embed_cols = [
                c for c in item_features.columns if c.startswith("item_text_emb_")
            ]
            if len(embed_cols) != history_embed_dim:
                raise ValueError(
                    f"Found {len(embed_cols)} item_text_emb_* columns but "
                    f"history_embed_dim={history_embed_dim}. These must match."
                )

            item_idx = item_features.set_index("movieId")
            self._movie_to_row = {mid: i for i, mid in enumerate(item_idx.index)}
            self._item_embed_matrix = (
                item_idx[embed_cols].values.astype(np.float32)
            )

            # --- Scalar feature setup (new) ---
            # Genre one-hot columns on the ITEM side (e.g. "genre_action"),
            # used to recompute genre affinity from a causal window.
            item_genre_cols = [
                c for c in item_features.columns if c.startswith("genre_")
            ]
            self._item_genre_matrix = (
                item_idx[item_genre_cols].values.astype(np.float32)
                if item_genre_cols else None
            )

            # Map from the USER-side scalar column name back to which
            # item_genre_cols index it corresponds to, e.g.
            # "user_genre_action" -> index of "genre_action".
            self._genre_scalar_positions = {}
            for i, col in enumerate(self.scalar_cols):
                for g_idx, g_col in enumerate(item_genre_cols):
                    if col == f"user_{g_col}":
                        self._genre_scalar_positions[i] = g_idx

            # The three engineered activity/recency scalar cols, if present.
            self._activity_scalar_idx = {
                name: self.scalar_cols.index(name)
                for name in (
                    "user_total_interactions",
                    "user_interactions_per_month",
                    "user_days_since_last",
                )
                if name in self.scalar_cols
            }

            interactions = interactions.sort_values(
                ["userId", "timestamp"]
            ).reset_index(drop=True)
            interactions["_pos_in_user_seq"] = interactions.groupby(
                "userId"
            ).cumcount()

            self._user_seq: dict = {
                uid: grp["movieId"].tolist()
                for uid, grp in interactions.groupby("userId")
            }
            self._user_timestamps: dict = {
                uid: grp["timestamp"].tolist()
                for uid, grp in interactions.groupby("userId")
            }
            self._positions = interactions["_pos_in_user_seq"].values
            self._timestamps = interactions["timestamp"].values

        self.user_ids = interactions["userId"].values
        self.movie_ids = interactions["movieId"].values

        item_counts = pd.Series(self.movie_ids).value_counts()
        total = item_counts.sum()
        self.item_sampling_probs = (item_counts / total).to_dict()

        print(f" Scalar feature cols: {len(self.scalar_cols)}")
        print(f" Item feature cols:  {len(self.item_feat_cols)}")
        print(f" History shape:    ({max_history_len}, {history_embed_dim})")

    def __len__(self) -> int:
        return len(self.user_ids)

    def _prior_movie_ids(self, user_id, pos: int) -> list:
        seq = self._user_seq[user_id]
        start = max(0, pos - self.max_history_len)
        return seq[start:pos]  # strictly before this position

    def _causal_history(self, prior_movie_ids: list) -> torch.Tensor:
        hist_tensor = torch.zeros(
            self.max_history_len, self.history_embed_dim, dtype=torch.float32
        )
        if prior_movie_ids:
            rows = [self._movie_to_row[m] for m in prior_movie_ids]
            embs = self._item_embed_matrix[rows]
            hist_tensor[-len(embs):] = torch.from_numpy(embs)
        return hist_tensor

    def _static_history(self, user_id) -> torch.Tensor:
        hist_flat = np.array(
            self.user_features.loc[user_id, self.hist_col], dtype=np.float32
        )
        return torch.from_numpy(hist_flat).reshape(
            self.max_history_len, self.history_embed_dim
        )

    def _causal_scalar(self, user_id, pos: int, prior_movie_ids: list, current_ts) -> np.ndarray:
        """
        Overrides the genre-affinity + activity/recency entries of the
        static scalar vector with values computed only from interactions
        strictly before this row. Demographic entries (gender/age/
        occupation) are left as-is, since they're static by nature.
        """
        scalar = self.user_features.loc[
            user_id, self.scalar_cols
        ].values.astype(np.float32).copy()

        # Genre affinity: mean of prior items' genre one-hots (same
        # window as the sequence history).
        if self._item_genre_matrix is not None and self._genre_scalar_positions:
            if prior_movie_ids:
                rows = [self._movie_to_row[m] for m in prior_movie_ids]
                genre_affinity = self._item_genre_matrix[rows].mean(axis=0)
            else:
                genre_affinity = np.zeros(self._item_genre_matrix.shape[1], dtype=np.float32)

            for scalar_idx, genre_idx in self._genre_scalar_positions.items():
                scalar[scalar_idx] = genre_affinity[genre_idx]

        # Activity / recency: uncapped count of prior interactions (not
        # windowed to max_history_len -- this reflects overall activity
        # level up to this point, same intent as the original feature,
        # just correctly scoped in time).
        if self._activity_scalar_idx:
            timestamps = self._user_timestamps[user_id]

            total_prior = pos  # count of interactions strictly before this one

            if pos > 0:
                first_ts = timestamps[0]
                date_range_days = max((current_ts - first_ts) / np.timedelta64(1, "D"), 1)
                interactions_per_month = total_prior / (date_range_days / 30 + 1)
                # Recency gap: days since the user's PREVIOUS interaction.
                # (Different from the original static feature, which measured
                # staleness relative to the end of the whole train set -- that
                # reference point doesn't exist yet mid-sequence. Gap-since-
                # last-interaction is the standard causal analogue.)
                prev_ts = timestamps[pos - 1]
                days_since_last = max((current_ts - prev_ts) / np.timedelta64(1, "D"), 0)
            else:
                interactions_per_month = 0.0
                days_since_last = 0.0

            if "user_total_interactions" in self._activity_scalar_idx:
                scalar[self._activity_scalar_idx["user_total_interactions"]] = np.log1p(total_prior)
            if "user_interactions_per_month" in self._activity_scalar_idx:
                scalar[self._activity_scalar_idx["user_interactions_per_month"]] = np.log1p(interactions_per_month)
            if "user_days_since_last" in self._activity_scalar_idx:
                scalar[self._activity_scalar_idx["user_days_since_last"]] = np.log1p(days_since_last)

        return scalar

    def __getitem__(self, idx: int):
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]

        if self.causal:
            pos = self._positions[idx]
            prior_movie_ids = self._prior_movie_ids(user_id, pos)
            hist_tensor = self._causal_history(prior_movie_ids)
            scalar = self._causal_scalar(
                user_id, pos, prior_movie_ids, self._timestamps[idx]
            )
        else:
            hist_tensor = self._static_history(user_id)
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
            torch.tensor(sampling_prob, dtype=torch.float32),
        )

    @property
    def scalar_dim(self) -> int:
        return len(self.scalar_cols)

    @property
    def item_dim(self) -> int:
        return len(self.item_feat_cols)