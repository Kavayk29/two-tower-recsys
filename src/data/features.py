import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import yaml
from tqdm import tqdm


ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

OCCUPATION_MAP = {
    0: "other", 1: "academic", 2: "artist", 3: "clerical",
    4: "college", 5: "customer_service", 6: "doctor", 7: "executive",
    8: "farmer", 9: "homemaker", 10: "k12student", 11: "lawyer",
    12: "programmer", 13: "retired", 14: "sales", 15: "scientist",
    16: "self_employed", 17: "technician", 18: "tradesman",
    19: "unemployed", 20: "writer"
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_item_features(
    movies: pd.DataFrame,
    text_model: SentenceTransformer,
    embed_dim: int = 384
) -> pd.DataFrame:
    print("Building item features...")

    if "rich_text" in movies.columns:
        text_col = "rich_text"
        print(" Using TMDB-enriched text (title + overview + cast + director)")
    else:
        text_col = "title_clean"
        print(" Using title only (run TMDB enrichment for richer features)")

    for genre in ALL_GENRES:
        col = f"genre_{genre.lower().replace('-', '_')}"
        movies[col] = movies["genres"].str.contains(
            genre, regex=False
        ).astype(np.float32)

    def year_to_era(year):
        if pd.isna(year):
            return 2
        y = int(year)
        if y < 1980:
            return 0
        elif y < 1990:
            return 1
        elif y < 2000:
            return 2
        elif y < 2010:
            return 3
        else:
            return 4

    movies["release_era"] = movies["year"].apply(year_to_era).astype(np.float32)
    movies["year_norm"] = (
        (movies["year"].fillna(movies["year"].median()) - 1900) / 120
    ).astype(np.float32)

    print(f" Encoding {len(movies):,} movie texts...")
    texts = movies[text_col].fillna("").tolist()

    embeddings = text_model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    embed_cols = [f"item_text_emb_{i}" for i in range(embed_dim)]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)

    movies = pd.concat([movies.reset_index(drop=True), embed_df], axis=1)

    movies = movies.drop(
        columns=[
            "title", "genres", "title_clean", "rich_text",
            "overview", "cast", "director"
        ],
        errors="ignore"
    )

    print(f" Item features shape: {movies.shape}")
    return movies


def build_popularity_features(
    train_interactions: pd.DataFrame,
    item_features: pd.DataFrame
) -> pd.DataFrame:
    print("Adding popularity features to items...")

    stats = train_interactions.groupby("movieId").agg(
        item_num_ratings=("rating", "count"),
        item_avg_rating=("rating", "mean"),
    ).reset_index()

    stats["item_num_ratings_log"] = np.log1p(
        stats["item_num_ratings"]
    ).astype(np.float32)

    stats["item_avg_rating"] = stats["item_avg_rating"].astype(np.float32)

    raw_counts = train_interactions.groupby("movieId").size()
    quantiles = raw_counts.quantile([0.25, 0.5, 0.75])

    def assign_tier(count):
        if count <= quantiles[0.25]:
            return 0.0
        elif count <= quantiles[0.5]:
            return 1.0
        elif count <= quantiles[0.75]:
            return 2.0
        else:
            return 3.0

    tier_series = raw_counts.apply(assign_tier).reset_index()
    tier_series.columns = ["movieId", "item_popularity_tier"]

    stats = stats.merge(tier_series, on="movieId")
    item_features = item_features.merge(stats, on="movieId", how="left")

    item_features["item_num_ratings_log"] = item_features[
        "item_num_ratings_log"
    ].fillna(0.0).astype(np.float32)

    item_features["item_avg_rating"] = item_features[
        "item_avg_rating"
    ].fillna(item_features["item_avg_rating"].median()).astype(np.float32)

    item_features["item_popularity_tier"] = item_features[
        "item_popularity_tier"
    ].fillna(0.0).astype(np.float32)

    item_features = item_features.drop(
        columns=["item_num_ratings"], errors="ignore"
    )

    return item_features


def build_user_features(
    train_interactions: pd.DataFrame,
    users: pd.DataFrame,
    item_features: pd.DataFrame,
    max_history_len: int = 50,
    embed_dim: int = 384
) -> pd.DataFrame:
    print("Building user features...")

    genre_cols = [c for c in item_features.columns if c.startswith("genre_")]
    embed_cols = [c for c in item_features.columns if c.startswith("item_text_emb_")]

    merged = train_interactions.merge(
        item_features[["movieId"] + genre_cols + embed_cols],
        on="movieId", how="left"
    )

    user_records = []
    user_ids = merged["userId"].unique()
    users_indexed = users.set_index("userId")

    print(f" Computing features for {len(user_ids):,} users...")

    for user_id in tqdm(user_ids, desc="Users"):
        user_data = merged[merged["userId"] == user_id].sort_values("timestamp")
        history = user_data.tail(max_history_len)

        genre_affinity = history[genre_cols].mean().values.astype(np.float32)

        history_embeds = history[embed_cols].values.astype(np.float32)
        padded = np.zeros((max_history_len, embed_dim), dtype=np.float32)

        actual_len = min(len(history_embeds), max_history_len)
        padded[:actual_len] = history_embeds[-actual_len:]

        history_flat = padded.flatten().tolist()

        total_interactions = len(user_data)
        date_range = max(
            (user_data["timestamp"].max() - user_data["timestamp"].min()).days, 1
        )

        interactions_per_month = total_interactions / (date_range / 30 + 1)

        days_since_last = (
            train_interactions["timestamp"].max() - user_data["timestamp"].max()
        ).days

        record = {"userId": user_id}

        for i, col in enumerate(genre_cols):
            record[f"user_{col}"] = float(genre_affinity[i])

        record["user_history_embs"] = history_flat
        record["user_total_interactions"] = float(np.log1p(total_interactions))
        record["user_interactions_per_month"] = float(np.log1p(interactions_per_month))
        record["user_days_since_last"] = float(np.log1p(days_since_last))

        if user_id in users_indexed.index:
            profile = users_indexed.loc[user_id]
            record["user_gender"] = float(profile["gender_encoded"])
            record["user_age_norm"] = float(profile["age_norm"])
            record["user_occupation_norm"] = float(profile["occupation_norm"])
        else:
            record["user_gender"] = 0.5
            record["user_age_norm"] = 0.5
            record["user_occupation_norm"] = 0.5

        user_records.append(record)

    user_features = pd.DataFrame(user_records)
    print(f" User features shape: {user_features.shape}")

    return user_features


def run_feature_engineering(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)

    processed_dir = Path(config["data"]["processed_dir"])
    embed_dim = config["features"]["text_embed_dim"]
    max_history = config["features"]["max_history_len"]

    print("Loading processed data...")

    enriched_path = processed_dir / "movies_enriched.parquet"
    movies_path = processed_dir / "movies.parquet"

    movies = pd.read_parquet(
        enriched_path if enriched_path.exists() else movies_path
    )

    users = pd.read_parquet(processed_dir / "users.parquet")
    train = pd.read_parquet(processed_dir / "train_interactions.parquet")

    print(f"Loading text model: {config['features']['text_embed_model']}")
    text_model = SentenceTransformer(config["features"]["text_embed_model"])

    item_features = build_item_features(movies, text_model, embed_dim)
    item_features = build_popularity_features(train, item_features)

    item_features.to_parquet(processed_dir / "item_features.parquet", index=False)
    print(f"Saved item_features.parquet — shape: {item_features.shape}")

    user_features = build_user_features(
        train, users, item_features, max_history, embed_dim
    )

    user_features.to_parquet(processed_dir / "user_features.parquet", index=False)
    print(f"Saved user_features.parquet — shape: {user_features.shape}")

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    run_feature_engineering()