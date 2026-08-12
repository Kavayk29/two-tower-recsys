"""
Place at: tests/test_dataset.py

Regression test for the training-time leakage bug: with causal=True,
each training example's history must contain ONLY interactions strictly
before it in that user's timeline, and must never include the target
item itself. This is what was silently broken before (a single static
full-history vector was reused for every row of a user, including rows
whose target item was itself part of that "history").
"""

import numpy as np
import pandas as pd
import torch

from src.training.dataset import InteractionDataset

HISTORY_EMBED_DIM = 4
MAX_HISTORY_LEN = 2


def _make_item_features(movie_ids):
    rng = np.random.default_rng(0)
    rows = []
    for i, mid in enumerate(movie_ids):
        row = {"movieId": mid, "year": 2000.0 + i}
        # Give each item a distinct, identifiable embedding (all entries
        # equal to the movieId) so we can assert on *which* items show up
        # in a history vector, not just its shape.
        for d in range(HISTORY_EMBED_DIM):
            row[f"item_text_emb_{d}"] = float(mid)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_user_features(user_ids):
    rows = []
    for uid in user_ids:
        row = {
            "userId": uid,
            "user_foo": 0.5,
            "user_history_embs": [0.0] * (MAX_HISTORY_LEN * HISTORY_EMBED_DIM),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_causal_history_excludes_target_and_future_items():
    # user 1 watches movies 10, 20, 30 in that order
    interactions = pd.DataFrame({
        "userId": [1, 1, 1],
        "movieId": [10, 20, 30],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    })
    item_features = _make_item_features([10, 20, 30])
    user_features = _make_user_features([1])

    ds = InteractionDataset(
        interactions,
        user_features,
        item_features,
        history_embed_dim=HISTORY_EMBED_DIM,
        max_history_len=MAX_HISTORY_LEN,
        causal=True,
    )

    # Row order follows the sorted (userId, timestamp) order inside the dataset.
    by_target = {}
    for i in range(len(ds)):
        hist, _, _, _ = ds[i]
        target_movie = ds.movie_ids[i]
        by_target[target_movie] = hist

    # Target = 10 (first interaction): no prior items -> history is all zeros
    assert torch.allclose(by_target[10], torch.zeros(MAX_HISTORY_LEN, HISTORY_EMBED_DIM))

    # Target = 20: only movie 10 precedes it. History must contain 10's
    # embedding (all 10.0) and must NOT contain 20's own embedding anywhere.
    hist_20 = by_target[20]
    assert torch.any(hist_20 == 10.0)
    assert not torch.any(hist_20 == 20.0)

    # Target = 30: movies 10 and 20 precede it (max_history_len=2 fits both).
    # Must NOT contain 30's own embedding.
    hist_30 = by_target[30]
    assert torch.any(hist_30 == 10.0)
    assert torch.any(hist_30 == 20.0)
    assert not torch.any(hist_30 == 30.0)


def test_noncausal_mode_uses_static_history_unchanged():
    """
    causal=False (val/eval path) should behave exactly as before: pull the
    precomputed static history straight from user_features, regardless of
    which item is the target.
    """
    interactions = pd.DataFrame({
        "userId": [1, 1],
        "movieId": [10, 20],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
    })
    item_features = _make_item_features([10, 20])
    user_features = _make_user_features([1])

    ds = InteractionDataset(
        interactions,
        user_features,
        item_features,
        history_embed_dim=HISTORY_EMBED_DIM,
        max_history_len=MAX_HISTORY_LEN,
        causal=False,
    )

    hist_a, _, _, _ = ds[0]
    hist_b, _, _, _ = ds[1]

    # Both rows pull the same static (all-zero, in this fixture) history.
    assert torch.allclose(hist_a, hist_b)