"""
Place at: tests/test_model.py

Pure shape/forward-pass tests -- no data files or downloads needed.
Run with: pytest tests/test_model.py -v
"""

import torch

from src.models.two_tower import TwoTowerModel


def _make_model(
    history_embed_dim=16,
    scalar_feature_dim=5,
    item_input_dim=10,
    embedding_dim=8,
    max_history_len=6,
):
    return TwoTowerModel(
        history_embed_dim=history_embed_dim,
        scalar_feature_dim=scalar_feature_dim,
        item_input_dim=item_input_dim,
        user_hidden_dims=[32, 16],
        item_hidden_dims=[32, 16],
        embedding_dim=embedding_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=max_history_len,
        dropout=0.0,
        temperature=0.1,
        logq_correction=False,
    )


def test_forward_returns_scalar_loss():
    torch.manual_seed(0)
    batch_size = 4
    model = _make_model()

    history = torch.randn(batch_size, 6, 16)
    scalars = torch.randn(batch_size, 5)
    item_feats = torch.randn(batch_size, 10)

    loss = model(history, scalars, item_feats)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_user_embedding_shape_and_normalization():
    model = _make_model(embedding_dim=8)
    model.eval()

    history = torch.randn(3, 6, 16)
    scalars = torch.randn(3, 5)

    emb = model.get_user_embedding(history, scalars)

    assert emb.shape == (3, 8)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_item_embedding_shape_and_normalization():
    model = _make_model(embedding_dim=8)
    model.eval()

    item_feats = torch.randn(5, 10)
    emb = model.get_item_embedding(item_feats)

    assert emb.shape == (5, 8)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_padded_history_positions_are_ignored_in_pooling():
    """
    UserTower mean-pools only over non-zero (non-padding) history rows.
    A fully-padded (all-zero) history should still produce a finite,
    well-formed embedding rather than NaNs from a divide-by-zero.
    """
    model = _make_model(embedding_dim=8)
    model.eval()

    empty_history = torch.zeros(2, 6, 16)
    scalars = torch.randn(2, 5)

    emb = model.get_user_embedding(empty_history, scalars)

    assert torch.isfinite(emb).all()
    assert emb.shape == (2, 8)