import torch
import os
import json

from src.data.preprocess import (
    load_ratings, filter_data, sort_by_time, 
    split_by_user_time, build_sequences, build_eval_data
)
from src.data.features import process_user_features, process_item_features

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    CONFIG = {
        "ratings_path": "data/raw/ratings.dat",
        "users_path": "data/raw/users.dat",
        "movies_path": "data/raw/movies.dat",
        "min_rating": 4.0,
        "min_interactions": 5,
        "val_frac": 0.1,
        "test_frac": 0.1,
        "max_history_len": 50,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
    }

    ensure_dir("data/processed")

    print("Processing user features...")
    process_user_features(CONFIG["users_path"], "data/processed/user_features.json")

    print("Processing item features...")
    process_item_features(CONFIG["movies_path"], "data/processed/item_features.json", "data/processed/title_embeddings.pt", CONFIG["embed_model"])

    print("Loading ratings data...")
    df = load_ratings(CONFIG["ratings_path"])
    
    print(f"Loaded data size: {len(df)}")

    print("Filtering...")
    df = filter_data(df, CONFIG["min_rating"], CONFIG["min_interactions"])
    print(f"After filtering: {len(df)}")

    print("Sorting...")
    df = sort_by_time(df)

    print("Splitting...")
    train_df, val_df, test_df = split_by_user_time(df, CONFIG["val_frac"], CONFIG["test_frac"])
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("Building sequences...")
    train_sequences = build_sequences(train_df, CONFIG["max_history_len"])
    val_data = build_eval_data(train_df, val_df)
    test_data = build_eval_data(train_df, test_df)

    print("Saving files...")
    save_json(train_sequences, "data/processed/train.json")
    save_json(val_data, "data/processed/val.json")
    save_json(test_data, "data/processed/test.json")

    print("Done!")