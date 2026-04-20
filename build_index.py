import torch
import pandas as pd
from pathlib import Path
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import build_and_save_index
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item_features = pd.read_parquet(processed_dir / "item_features.parquet")
    user_features = pd.read_parquet(processed_dir / "user_features.parquet")

    hist_cols = [
        c for c in user_features.columns if c.startswith("user_hist_emb_")
    ]
    scalar_cols = [
        c for c in user_features.columns
        if c != "userId" and not c.startswith("user_hist_emb_")
    ]
    item_feat_cols = [
        c for c in item_features.columns if c not in ["movieId", "year"]
    ]

    model = TwoTowerModel(
        history_embed_dim=len(hist_cols),
        scalar_feature_dim=len(scalar_cols),
        item_input_dim=len(item_feat_cols),
        hidden_dims=config["model"]["user_hidden_dims"],
        embedding_dim=config["model"]["embedding_dim"],
        num_attention_heads=config["model"]["attention_heads"],
        dropout=config["model"]["dropout"],
        temperature=config["model"]["temperature"],
        logq_correction=False
    )

    model.load_state_dict(
        torch.load("artifacts/best_model.pt", map_location=device)
    )
    model.to(device)

    build_and_save_index(
        model=model,
        item_features=item_features,
        save_dir=Path("artifacts"),
        device=device,
        index_type=config["retrieval"]["faiss_index_type"],
        nlist=config["retrieval"]["nlist"]
    )


if __name__ == "__main__":
    main()