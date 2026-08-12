"""
Place at: tests/test_api.py

These tests bypass the real startup event (which needs a trained model +
FAISS index on disk) and instead inject a lightweight fake state directly,
so the API's request/response handling can be tested without artifacts.

Run with: pytest tests/test_api.py -v
"""

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from src.serving import api


class _FakeIndex:
    """Minimal stand-in for a faiss index: returns fixed neighbour ids."""
    def search(self, query, k):
        n = min(k, 5)
        indices = np.arange(n).reshape(1, -1)
        distances = np.zeros((1, n), dtype=np.float32)
        return distances, indices


class _FakeModel:
    def get_user_embedding(self, history, scalars):
        return torch.zeros(history.shape[0], 8)


def _install_fake_state():
    api._state.clear()
    api._state.update(
        model=_FakeModel(),
        device=torch.device("cpu"),
        index=_FakeIndex(),
        movie_ids=np.array([101, 102, 103, 104, 105]),
        user_features=pd.DataFrame(
            {
                "user_history_embs": [[0.0] * (2 * 4)],
                "user_foo": [0.5],
            },
            index=pd.Index([1], name="userId"),
        ),
        scalar_cols=["user_foo"],
        history_embed_dim=4,
        max_history_len=2,
        top_k_candidates=50,
    )


def test_health_ok_when_model_loaded():
    _install_fake_state()
    client = TestClient(api.app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_recommend_returns_movie_ids_for_known_user():
    _install_fake_state()
    client = TestClient(api.app)

    resp = client.get("/recommend/1", params={"k": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 1
    assert len(body["movie_ids"]) == 3
    assert set(body["movie_ids"]).issubset({101, 102, 103, 104, 105})


def test_recommend_404_for_unknown_user():
    _install_fake_state()
    client = TestClient(api.app)

    resp = client.get("/recommend/9999")

    assert resp.status_code == 404


def test_recommend_400_for_k_out_of_range():
    _install_fake_state()
    client = TestClient(api.app)

    resp = client.get("/recommend/1", params={"k": 999})

    assert resp.status_code == 400