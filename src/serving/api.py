"""
Place at: src/serving/api.py

Minimal FastAPI serving layer: loads the trained model, FAISS index, and
feature tables once at startup, then exposes:
  GET /health              -> liveness/readiness check
  GET /recommend/{user_id} -> top-k movie ids for a known user

State is held in a module-level dict (`_state`) rather than FastAPI's
app.state so tests can inject a fake model/index/features without needing
a real checkpoint or FAISS index on disk -- see tests/test_api.py, which
calls `api._state.update(...)` directly and never triggers the startup
event (TestClient only runs startup/shutdown handlers when used as a
context manager).
"""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException

from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import load_index
from src.utils import get_artifacts_dir

_state: dict = {}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup()
    yield
    _state.clear()


app = FastAPI(title="two-tower-recsys", lifespan=lifespan)


def _startup() -> None:
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    artifacts_dir = get_artifacts_dir(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_features = pd.read_parquet(
        processed_dir / "user_features.parquet"
    ).set_index("userId")
    scalar_cols = [c for c in user_features.columns if c != "user_history_embs"]

    item_features = pd.read_parquet(processed_dir / "item_features.parquet")
    item_feat_cols = [
        c for c in item_features.columns if c not in ["movieId", "year"]
    ]

    history_embed_dim = config["features"]["text_embed_dim"]
    max_history_len = config["features"]["max_history_len"]

    model = TwoTowerModel(
        history_embed_dim=history_embed_dim,
        scalar_feature_dim=len(scalar_cols),
        item_input_dim=len(item_feat_cols),
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
    model.load_state_dict(
        torch.load(artifacts_dir / "best_model.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    index, movie_ids = load_index(artifacts_dir, nprobe=config["retrieval"]["nprobe"])

    _state.clear()
    _state.update(
        model=model,
        device=device,
        index=index,
        movie_ids=movie_ids,
        user_features=user_features,
        scalar_cols=scalar_cols,
        history_embed_dim=history_embed_dim,
        max_history_len=max_history_len,
        top_k_candidates=config["retrieval"]["top_k_candidates"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    top_k_candidates = _state["top_k_candidates"]
    if not (1 <= k <= top_k_candidates):
        raise HTTPException(
            status_code=400,
            detail=f"k must be between 1 and {top_k_candidates}",
        )

    user_features = _state["user_features"]
    if user_id not in user_features.index:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_id}")

    device = _state["device"]
    max_history_len = _state["max_history_len"]
    history_embed_dim = _state["history_embed_dim"]

    hist_flat = np.array(
        user_features.loc[user_id, "user_history_embs"], dtype=np.float32
    )
    hist = torch.tensor(hist_flat).reshape(
        1, max_history_len, history_embed_dim
    ).to(device)

    scalar = torch.tensor(
        user_features.loc[user_id, _state["scalar_cols"]].values.astype(np.float32),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        user_emb = _state["model"].get_user_embedding(hist, scalar)

    query = np.ascontiguousarray(user_emb.cpu().numpy().astype(np.float32))
    _, indices = _state["index"].search(query, k)

    hits = indices[0]
    hits = hits[hits >= 0]  # drop FAISS's -1 "not enough candidates" slots
    movie_ids = _state["movie_ids"][hits].tolist()

    return {"user_id": user_id, "movie_ids": movie_ids}
