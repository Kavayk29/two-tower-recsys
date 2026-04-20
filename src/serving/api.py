import torch
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import yaml

from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import load_index, retrieve_top_k


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


app = FastAPI(title="Two-Tower Recommender API")

_model = None
_faiss_index = None
_movie_ids = None
_user_features = None
_item_features = None
_config = None
_device = None
_hist_cols = None
_scalar_cols = None
_item_feat_cols = None


@app.on_event("startup")
def load_artifacts():
    global _model, _faiss_index, _movie_ids
    global _user_features, _item_features, _config
    global _device, _hist_cols, _scalar_cols, _item_feat_cols

    _config = load_config()
    _device = torch.device("cpu")
    processed_dir = Path(_config["data"]["processed_dir"])

    print("Loading features...")
    _user_features = pd.read_parquet(
        processed_dir / "user_features.parquet"
    ).set_index("userId")
    _item_features = pd.read_parquet(
        processed_dir / "item_features.parquet"
    )

    all_user_cols = list(_user_features.columns)
    _hist_cols = [c for c in all_user_cols if c.startswith("user_hist_emb_")]
    _scalar_cols = [
        c for c in all_user_cols if not c.startswith("user_hist_emb_")
    ]
    _item_feat_cols = [
        c for c in _item_features.columns
        if c not in ["movieId", "year"]
    ]

    print("Loading FAISS index...")
    _faiss_index, _movie_ids = load_index(Path("artifacts"))

    history_embed_dim = len(_hist_cols)
    scalar_dim = len(_scalar_cols)
    item_dim = len(_item_feat_cols)

    print("Loading model...")
    _model = TwoTowerModel(
        history_embed_dim=history_embed_dim,
        scalar_feature_dim=scalar_dim,
        item_input_dim=item_dim,
        hidden_dims=_config["model"]["user_hidden_dims"],
        embedding_dim=_config["model"]["embedding_dim"],
        num_attention_heads=_config["model"]["attention_heads"],
        dropout=_config["model"]["dropout"],
        temperature=_config["model"]["temperature"],
        logq_correction=False
    ).to(_device)

    _model.load_state_dict(
        torch.load("artifacts/best_model.pt", map_location=_device)
    )
    _model.eval()
    print("API ready.")


class RecommendResponse(BaseModel):
    user_id: int
    recommended_movie_ids: List[int]
    message: str


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = 10):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    item_feat_idx = _item_features.set_index("movieId")

    # Cold start fallback
    if user_id not in _user_features.index:
        popular = (
            _item_features
            .nlargest(k * 5, "item_num_ratings_log")["movieId"]
            .head(k).tolist()
        )
        return RecommendResponse(
            user_id=user_id,
            recommended_movie_ids=popular,
            message="cold_start_fallback"
        )

    hist = torch.tensor(
        _user_features.loc[user_id, _hist_cols].values.astype(np.float32),
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(_device)

    scalar = torch.tensor(
        _user_features.loc[user_id, _scalar_cols].values.astype(np.float32),
        dtype=torch.float32
    ).unsqueeze(0).to(_device)

    with torch.no_grad():
        user_emb = _model.user_tower(hist, scalar).cpu().numpy()

    top_k_ids = retrieve_top_k(
        _faiss_index, _movie_ids, user_emb,
        k=_config["retrieval"]["top_k_candidates"]
    )

    return RecommendResponse(
        user_id=user_id,
        recommended_movie_ids=top_k_ids[:k],
        message="two_tower_retrieval"
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}