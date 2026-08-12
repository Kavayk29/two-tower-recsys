# build_index.py — Pass item_hidden_dims separately
#
# Change vs. original: artifacts_dir now resolved via
# src.utils.get_artifacts_dir(), matching trainer.py exactly (previously
# the Kaggle-path logic here used a relative "kaggle/working" string that,
# combined with a cwd already at /kaggle/working, resolved to a doubled
# /kaggle/working/kaggle/working/artifacts path -- never where trainer.py
# actually saved the model).

import torch
import pandas as pd
from pathlib import Path
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import build_and_save_index
from src.utils import get_artifacts_dir
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    artifacts_dir = get_artifacts_dir(config)
    processed_dir = Path(config["data"]["processed_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item_features = pd.read_parquet(processed_dir / "item_features.parquet")
    user_features = pd.read_parquet(processed_dir / "user_features.parquet")

    scalar_cols = [
        c for c in user_features.columns
        if c not in ["userId", "user_history_embs"]
    ]

    item_feat_cols = [
        c for c in item_features.columns
        if c not in ["movieId", "year"]
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
        logq_correction=False
    )

    model_path = artifacts_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {model_path.resolve()}. "
            "Run training (src.training.trainer) first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    print("model loaded successfully")

    build_and_save_index(
        model=model,
        item_features=item_features,
        save_dir=artifacts_dir,
        device=device,
        index_type=config["retrieval"]["faiss_index_type"],
        nlist=config["retrieval"]["nlist"],
        nprobe=config["retrieval"]["nprobe"]
    )
    print(f"Index built and saved to {artifacts_dir}")


if __name__ == "__main__":
    main()